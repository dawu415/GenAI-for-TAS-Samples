from Common.OpenAIProviders.OpenAIProvider import OpenAIProvider
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from typing import Iterable
import httpx

from transformers import AutoTokenizer

class ModelTokenAdaptor():
    pass

class HuggingfaceModelTokenAdaptor(ModelTokenAdaptor):
    
    def __init__(self,
                 llm_model_name: str,
                 huggingface_token: str=""):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, token=huggingface_token)

    def apply_chat_template(self, messages: Iterable[ChatCompletionMessageParam]):
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

class OpenAILegacyAPILLMProvider(OpenAIProvider):
    model_name: str = ""
    def __init__(self, 
                 api_base: str,
                 api_key:str, 
                 llm_model_name: str,
                 model_token_adaptor:ModelTokenAdaptor,
                 temperature: float = 1.0,
                ):
        super().__init__(model_name = llm_model_name,
                         api_base=api_base,
                         api_key=api_key,
                         http_client=httpx.Client(verify=False)
                        )
        self.temperature = temperature
        self.model_token_adaptor = model_token_adaptor
        self.is_using_legacy_chat_api = True

        if self.model_token_adaptor is None:
            raise ValueError("Model Token Adaptor must not be None")
        
    def chat_completion(self, user_assistant_messages: Iterable[ChatCompletionMessageParam]):
        response = self.oai_client.completions.create(
                                                    model = self.model_name,
                                                    prompt=self.model_token_adaptor.apply_chat_template(user_assistant_messages),
                                                    temperature = self.temperature,
                                                    max_tokens=4096,
                                                    stream=False
                                                )
        return response.choices[0].text
    
    def stream_chat_completion(self, user_assistant_messages: Iterable[ChatCompletionMessageParam]):
        stream = self.oai_client.completions.create(
                                                    model = self.model_name,
                                                    prompt=self.model_token_adaptor.apply_chat_template(user_assistant_messages),
                                                    temperature = self.temperature,
                                                    max_tokens=4096,
                                                    stream=True
                                                )
        return stream

