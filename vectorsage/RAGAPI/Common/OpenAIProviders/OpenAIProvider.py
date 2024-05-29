import httpx
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from typing import Iterable

class OpenAIProvider:
    oai_client: OpenAI

    def __init__(self, 
                 model_name: str,
                 api_base: str, 
                 api_key: str, 
                 http_client: httpx.Client
                ):
        
        self.model_name = model_name
        self.oai_client = OpenAI( base_url=api_base, 
                                  api_key=api_key, 
                                  http_client=http_client
                                )
        self.is_using_legacy_chat_api = False
        
    
    def chat_completion(self, user_assistant_messages: Iterable[ChatCompletionMessageParam]):
        return ""
    
    def stream_chat_completion(self, user_assistant_messages: Iterable[ChatCompletionMessageParam]):
        return None
