from Common.DocTransformers.DocTransformer import DocTransformer, Document, DocChunk
from typing import List, Dict, Any

class ChunkReOrder(DocTransformer):

    def __init__(self):
        pass

    def __call__(self, parent:Document, texts: List[DocChunk]) -> List[DocChunk]:
        # When larger number of chunks start coming in, LLMs tend to prioritize content
        # at the beggingin and and near the end - rather than the middle.  
           
        # It is assumed that texts is sorted in descending order based on cosine similarity score 
        shuffled_result = []
        for i, value in enumerate(texts[::-1]):
            if (i + 1) % 2 == 1:
                shuffled_result.append(value)
            else:
                shuffled_result.insert(0, value)
        return shuffled_result