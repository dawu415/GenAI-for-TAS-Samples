from Common.DocTransformers.DocTransformer import DocTransformer, Document, DocChunk

import en_core_web_lg
from transformers import AutoTokenizer
from typing import List
import numpy as np
import os
import re


class GatherStatistics(DocTransformer):
    def __init__(self, model_tokenizer_path, query: str, prompt_template: str, supported_template_variables: List[str]):

        try:
            self.nlp = en_core_web_lg.load()
        except Exception as e:
            raise RuntimeError(f"Failed to load Spacy model: {e}")
        
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_path,truncation=True)
        self.query = query
        self.prompt_template = prompt_template
        self.supported_template_variables = supported_template_variables
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def __call__(self, parent:Document, texts: List[DocChunk]) -> List[DocChunk]:

        total_characters = 0
        total_words = 0
        total_tokens = 0
        token_count_per_chunk = []
        word_count_per_chunk = []
        for text in texts:
            word_count= len([token for token in self.nlp(text.content) if not token.is_punct and not token.is_space])
            word_count_per_chunk.append(word_count)
            total_words += word_count
            total_characters += len(text.content)
            tokenized_text = self.model_tokenizer(text.content, truncation=True, return_tensors="np", padding=False)["input_ids"]
            total_tokens += tokenized_text.shape[1]
            token_count_per_chunk.append(tokenized_text.shape[1])
            
        # Use the found content and form a prompt
        context = " ".join([re.sub(r'\n+', ' ', kbchunk.content) for kbchunk in texts])

        # Create prompt variables
        # These must match what is available at get_supported_prompt_template_variables
        # Could not see a clearer way to do this than below...
        # Otherwise introspection via vars() could be done, but that's confusing magic.
        prompt_variables = {
            "query": self.query,
            "context": context
        }

        # Verify the prompt variables
        unsupported_keys = [k for k in prompt_variables.keys() if k not in self.supported_template_variables]
        if unsupported_keys:
            raise Exception(f"Unsupported keys in prompt variables: {unsupported_keys}")

        # Format the prompt template
        prompt = self.prompt_template.format_map(prompt_variables)

        query_word_count = len([token for token in self.nlp(self.query) if not token.is_punct and not token.is_space])
        prompt_word_count = len([token for token in self.nlp(prompt) if not token.is_punct and not token.is_space])

        total_words += query_word_count + prompt_word_count
        total_characters+= len(self.query) + len(prompt)
        query_token_count = self.model_tokenizer(self.query, truncation=True, return_tensors="np", padding=False)["input_ids"].shape[1]
        estimated_token_per_response = self.model_tokenizer(prompt, truncation=True, return_tensors="np", padding=False)["input_ids"].shape[1]

        return {
            "Total_Characters": total_characters,
            "Total_Tokens": total_tokens,
            "Total_Words": total_words,
            "Token_Count_Per_Chunk": token_count_per_chunk,
            "Word_Count_Per_Chunk": word_count_per_chunk,
            "Query_Word_Count": query_token_count,
            "Query_Token_Count": query_token_count,
            "Estimated_Token_Per_Response": estimated_token_per_response,
            "Estimated_Words_Per_Response":prompt_word_count
        }