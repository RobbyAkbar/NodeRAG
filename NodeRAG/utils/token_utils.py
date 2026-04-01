import tiktoken
from typing import Protocol, List
# from transformers import AutoTokenizer

class token_counter(Protocol):
    
    def __init__(self,model_name:str):
        self.model_name = model_name
    
    def __call__(self, text:str) -> int:
        ...
    
class tiktoken_counter(token_counter):
    
    def __init__(self,model_name:str):
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.token_limit_bound = 128000
    

    def encode(self, text:str) -> List[int]:
        return self.tokenizer.encode(text)
    
    def token_limit(self, text:str) -> bool:
                
        return len(self.encode(text)) > self.token_limit_bound
    


    def __call__(self, text: str) -> int:
        return len(self.encode(text))
    
# class deepseek_counter(token_counter):
    
#     def __init__(self,model_name:str):
#         self.tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3')
#         self.token_limit_bound = 640000
    

#     def encode(self, text:str) -> List[int]:
#         return self.tokenizer.encode(text)
    
#     def token_limit(self, text:str) -> bool:
#         return len(self.encode(text)) > self.token_limit_bound
    
#     def __call__(self, text:str) -> int:
#         return len(self.encode(text))
    
    
        



def get_token_counter(model_name:str) -> token_counter:

    model_name = model_name.lower()

    # OpenRouter slugs are "provider/model-name" — strip the prefix
    if '/' in model_name:
        model_name = model_name.split('/', 1)[1]

    if 'gpt' in model_name:
        return tiktoken_counter(model_name)
    elif 'gemini' in model_name:
        token = tiktoken_counter('gpt-4o')
        token.token_limit_bound = 1280000
        return token
    elif 'claude' in model_name:
        # Claude models use the same token budget as GPT-4o for estimation
        token = tiktoken_counter('gpt-4o')
        token.token_limit_bound = 200000
        return token
    elif 'llama' in model_name or 'mistral' in model_name or 'mixtral' in model_name:
        token = tiktoken_counter('gpt-4o')
        token.token_limit_bound = 128000
        return token
    # elif 'deepseek' in model_name:
    #     return deepseek_counter(model_name)
    else:
        # Fallback: use gpt-4o tokenizer as a reasonable approximation
        return tiktoken_counter('gpt-4o')


    
        
    