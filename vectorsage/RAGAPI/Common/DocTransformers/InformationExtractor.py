from Common.DocTransformers.DocTransformer import DocTransformer, Document, DocChunk
from Common.OpenAIProviders.OpenAILLMProvider import OpenAIProvider
from typing import List, Dict, Any

class InformationExtractor(DocTransformer):

    def __init__(self, llm_provider: OpenAIProvider, query: str, additional_context: List[Dict[str, Any]]):
        self.llm_provider = llm_provider
        self.input_query = query
        self.additional_context = additional_context

    def __call__(self, parent:Document, texts: List[DocChunk]) -> List[DocChunk]:
        
        # For each Chunk run this against the LLM and get a concise breakdown of
        # context
        prompt="Assess the relevance of the context to the query. \
            If the Context is not relevant, respond with 'IRRELEVANT'. \
            Otherwise, respond with 'RELEVANT'.\
            'RELEVANT' content should also include extracted information in dot points in relation to the query: \
            ---     \
            Context: \
            {} \
            ---\
            Query: {}\
            ---\
            In your response, you will not include any phrases referencing the context itself, such as 'The context describes', 'In the context', 'Based on the context', or 'the context provided' or similar."
        for text in texts:
            specific_prompt = prompt.format(text.content.replace("\n"," "), self.input_query)
            message  = self.additional_context + [{"role": "user", "content": specific_prompt}]
            response = self.llm_provider.chat_completion(message)

            if "RELEVANT:\n\n" in response:
                text.content = response.replace(" RELEVANT:\n\n","")

        return texts 