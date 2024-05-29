from Common.DocTransformers.DocTransformer import DocTransformer, Document, DocChunk
from Common.OpenAIProviders.OpenAIEmbeddingProvider import OpenAIEmbeddingProvider
from typing import List

class EmbeddingGenerator(DocTransformer):

    def __init__(self, embedding_provider: OpenAIEmbeddingProvider, use_metadata_with_embeddings: bool = True):
        self.embedding_provider = embedding_provider
        self.use_metadata_with_embeddings = use_metadata_with_embeddings

    def _dict_to_markdown(self,metadata):
        markdown_str = ""
        for key, value in metadata.items():
            markdown_str += f"**{key.capitalize()}:** {value}\n\n"
        return markdown_str
        
    def generate_embeddings(self, input_text_list):
        return self.embedding_provider.get_embeddings(input_text_list)

    def __call__(self, parent:Document, texts: List[DocChunk]) -> List[DocChunk]:
 
        chunk_list = []
        if self.use_metadata_with_embeddings:
            chunk_list = [f"{self._dict_to_markdown(chunk.metadata)} {chunk.content}" for chunk in texts]
        else:
            chunk_list = [chunk.content for chunk in texts]
    
        embeddings = self.generate_embeddings(chunk_list)

        for text_chunk, embedding in zip(texts, embeddings):
            text_chunk.embedding = embedding

        return texts 

class InstructorEmbeddingGenerator(EmbeddingGenerator):

    def __init__(self, embedding_provider: OpenAIEmbeddingProvider, 
                 embedding_instruction: str,
                 use_metadata_with_embeddings: bool = True):
        super().__init__(embedding_provider=embedding_provider, use_metadata_with_embeddings=use_metadata_with_embeddings)
        self.embedding_instruction = embedding_instruction

    def __call__(self, parent:Document, texts: List[DocChunk]) -> List[DocChunk]:
 
        chunk_list = []
        if self.use_metadata_with_embeddings:
            chunk_list = [[self.embedding_instruction, f"{self._dict_to_markdown(chunk.metadata)} {chunk.content}"] for chunk in texts]
        else:
            chunk_list = [[self.embedding_instruction, chunk.content] for chunk in texts]
    
        embeddings = self.generate_embeddings(chunk_list)

        for text_chunk, embedding in zip(texts, embeddings):
            text_chunk.embedding = embedding

        return texts 