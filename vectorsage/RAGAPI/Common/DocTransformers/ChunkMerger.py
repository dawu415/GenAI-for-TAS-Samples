from Common.DocTransformers.DocTransformer import DocTransformer, Document, DocChunk
from itertools import groupby, count
from typing import List, Dict
import numpy as np


class ChunkMerger(DocTransformer):

    def __init__(self, sort_results_by_cosine_similarity: bool = True):
        self.sort_results_by_cosine_similarity = sort_results_by_cosine_similarity
        
    def _merge_dicts(self, a: Dict, b: Dict):
        updated_dict = a.copy()

        for key, value in b.items():
            if key in updated_dict:
                if type(updated_dict[key]) == type(value):
                    if type(value) == list:
                        updated_dict[key] = list(set(value + updated_dict[key]))
                    elif type(value) == dict:
                        updated_dict[key] = self._merge_dicts(updated_dict[key], value)
                    # No op on other types
                else:
                    updated_dict[f"{key}_{type(value)}"] = value
            else:
                updated_dict[key] = value

        return updated_dict

    def _gaussian_weight(self, distance, sigma=1.0):
        """
        Compute the Gaussian weight for a given distance.
        
        :param distance: The distance from the reference point.
        :param sigma: The standard deviation of the Gaussian function.
        :return: The weight computed using the Gaussian function.
        """
        return np.exp(-0.5 * (distance / sigma) ** 2)

    def __call__(self, parent:Document, texts: List[DocChunk]) -> List[DocChunk]:
        def group_key(result, c=count()):
            return (result.metadata["title"], result.metadata["id"] - next(c))

        if not texts:
            return []

        merged = []
        position_value = {text.metadata["id"]: idx for idx, text in enumerate(texts)}

        texts.sort(key=lambda x: (x.metadata["title"], x.metadata["id"]))

        for _, chunkGroup in groupby(texts, key=group_key):
            chunkGroup = list(chunkGroup)
            
            if len(chunkGroup) == 1:
                merged.append(chunkGroup[0])
            else:
                # Determine the overlap amount between two consecutive contents
                overlap_count_list = []
                for u, v in zip(range(len(chunkGroup) - 1), range(1, len(chunkGroup))):
                    max_overlap = 0
                    for i in range(1, min(len(chunkGroup[u].content), len(chunkGroup[v].content)) + 1):
                        if chunkGroup[u].content[-i:] == chunkGroup[v].content[:i]:
                            max_overlap = i
                    overlap_count_list.append(max_overlap)

                # Use the overlap amount and do a merge of chunkGroup items
                content = ""
                metadata = {}
                embedding = None
                total_weight = 0.0
                weighted_similarity_sum = 0.0

                # Identify the chunk with the highest cosine similarity
                max_similarity_chunk = max(chunkGroup, key=lambda x: x.metadata["cosine_similarity"])
                max_similarity_position = position_value[max_similarity_chunk.metadata["id"]]
                sigma = 1.0  # Standard deviation for the Gaussian function

                # Gaussian weighting is used for recomputing the cosine similarity 
                # It is used to assign weights to each chunk in a group relative to the highest 
                # scoring chunk. i.e. Highest scroing chunk has highest weighting. Adjoining chunks
                # start having lower weight as it goes further from the highest scoring chunk. 
                for idx, chunk in enumerate(chunkGroup):
                    distance = position_value[chunk.metadata["id"]] - max_similarity_position
                    weight = self._gaussian_weight(distance, sigma)
                    total_weight += weight
                    weighted_similarity_sum += chunk.metadata["cosine_similarity"] * weight

                    offset = overlap_count_list[idx] if idx < len(overlap_count_list) else 0
                    content += chunk.content[:len(chunk.content) - offset]

                    metadata = self._merge_dicts(metadata, chunk.metadata)

                    if embedding is not None and chunk.embedding is not None:
                        embedding = np.vstack((embedding, chunk.embedding))
                    elif embedding is None and chunk.embedding is not None:
                        embedding = chunk.embedding
                    elif embedding is not None and chunk.embedding is None:
                        chunk_embedding = np.zeros_like(embedding)
                        embedding = np.vstack((embedding, chunk_embedding))
                    else:
                        embedding = None

                similarity = weighted_similarity_sum / total_weight if total_weight != 0 else 0
                metadata["cosine_similarity"] = float(similarity)

                merged.append(DocChunk(content=content, embedding=embedding, metadata=metadata))

        if self.sort_results_by_cosine_similarity:
            merged.sort(key=lambda x: x.metadata["cosine_similarity"], reverse=True)

        return merged

