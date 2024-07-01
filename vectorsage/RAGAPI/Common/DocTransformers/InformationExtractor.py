from Common.DocTransformers.DocTransformer import DocTransformer, Document, DocChunk
from Common.OpenAIProviders.OpenAILLMProvider import OpenAIProvider
from typing import List, Dict, Any

class InformationExtractor(DocTransformer):

    def __init__(self, llm_provider: OpenAIProvider, query: str, additional_context: List[Dict[str, Any]] = None):
        self.llm_provider = llm_provider
        self.input_query = query
        self.additional_context = additional_context if additional_context else []

    def __call__(self, parent:Document, texts: List[DocChunk]) -> List[DocChunk]:
        
        # For each Chunk run this against the LLM and get a concise breakdown of
        # context
        prompt="Assess the query and determine which context are relevant to answering the query. \
            ---     \
            Context: \
            {} \
            ---\
            Query: {}\
            ---\
            If irrelevant, mark it with IRRELEVANT. Do not include any phrases referencing the context itself, such as 'The context describes'. Return only the determined relevant context without changes to the Context content."

        retained_text = [] 
        for text in texts:
            specific_prompt = prompt.format(text.content.replace("\n"," "), self.input_query)
            message  = self.additional_context + [{"role": "user", "content": specific_prompt}]
            response = self.llm_provider.chat_completion(message)

            if "IRRELEVANT" not in response:
              retained_text.append(text)

        return retained_text 