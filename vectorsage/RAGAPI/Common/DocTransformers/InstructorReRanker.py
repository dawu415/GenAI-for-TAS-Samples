from Common.DocTransformers.DocTransformer import DocTransformer, Document, DocChunk
from Common.OpenAIProviders.OpenAIEmbeddingProvider import OpenAIEmbeddingProvider
from typing import List
from itertools import chain

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class InstructorReRanker(DocTransformer):

    def __init__(self, embedding_provider: OpenAIEmbeddingProvider,  query: str, topic_domain: str, cluster_count: int=5):
        self.embedding_provider = embedding_provider
        self.input_query = query
        self.topic_domain = topic_domain
        self.cluster_count = cluster_count
  
    def _compute_relevance_scores(self, query_embedding, doc_embeddings):
        similarities = cosine_similarity([query_embedding], doc_embeddings).flatten()
        return similarities

    def __call__(self, parent:Document, texts: List[DocChunk]) -> List[DocChunk]:
        
        # Assume that the texts have computed cluster embeddings
        # Convert the input query into a clustering embedding
        # Convert user query into an embedding with the instruction
        instruction = f"Represent this {self.topic_domain} question for clustering:"
        query_embedding = self.embedding_provider.get_embeddings_with_instructions(instruction, self.input_query)        

        embeddings = np.array([text.embedding for text in texts])
        # Perform K-means clustering
        # Cluster count is set to the cluster_count value otherwise, the size of our input / 2
        n_cluster = self.cluster_count if len(embeddings) >= self.cluster_count else int(round(len(embeddings)/2.0))
        if n_cluster == 0:
            return texts

        clusters = KMeans(n_clusters=n_cluster, random_state=0).fit_predict(embeddings)

        # Compute cosine similarity between query and each document
        similarity_scores = cosine_similarity([query_embedding], embeddings)[0]

        # Calculate average relevance score for each cluster
        cluster_stats = {}
        max_cluster_id = None
        max_cosine_similarity = -9999
        text_array = np.array(texts)
        for cluster_id in range(self.cluster_count):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if cluster_indices.size == 0:
                cluster_stats[cluster_id] = None
                continue
            cluster_similarity_scores = similarity_scores[cluster_indices]

            # Update metadata scores and add in the delta score for reference
            for text, score in zip(text_array[cluster_indices], cluster_similarity_scores):
                text.metadata["delta_score"] = text.metadata["cosine_similarity"] - score
                text.metadata["cosine_similarity"] = score

            
            cluster_stats[cluster_id] = {
                                            "cluster_size": len(cluster_similarity_scores),
                                            "min_cluster_score": np.min(cluster_similarity_scores),
                                            "max_cluster_score": np.max(cluster_similarity_scores),
                                            "avg_cluster_score": np.mean(cluster_similarity_scores),
                                            "texts": text_array[cluster_indices]
                                         }
            
            # print(f"Cluster ID {cluster_id} - ({len(cluster_similarity_scores)}, {np.min(cluster_similarity_scores)}, {np.max(cluster_similarity_scores)},{np.mean(cluster_similarity_scores)})")
            # print(f"Scores {cluster_id} - ({cluster_similarity_scores})")
                       
            if cluster_stats[cluster_id]["avg_cluster_score"] > max_cosine_similarity:
                max_cosine_similarity = cluster_stats[cluster_id]["avg_cluster_score"]
                max_cluster_id = cluster_id

        # Get the closest cluster and compute a minimum score as a filter for the
        # other clusters
        min_filter_score = (cluster_stats[max_cluster_id]["avg_cluster_score"] + cluster_stats[max_cluster_id]["min_cluster_score"])/2.0

        # Collected all the texts 
        filtered_texts = []

        # Collect texts from other clusters, filtered by cosine similarity
        other_cluster_texts = chain.from_iterable(
            cluster_stats[cluster_id]["texts"]
                for cluster_id in range(self.cluster_count)
                   if cluster_stats[cluster_id] != None
        )

        # Filter texts based on the minimum filter score
        filtered_texts.extend(
            text for text in other_cluster_texts if text.metadata["cosine_similarity"] >= min_filter_score
            )

        filtered_texts.sort(key=lambda x: x.metadata["cosine_similarity"], reverse=True)

        return filtered_texts 