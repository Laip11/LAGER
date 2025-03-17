from openai import OpenAI
from tqdm import tqdm

def batch_generator(data_list: list, batch_size: int = 64):
    # data_list is a list    
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i+batch_size]

class OpenAIClient:
    def __init__(
        self,
        api_key=None, 
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini"
    ):
        self.client = OpenAI(
            api_key=api_key, 
            base_url=base_url
        )
        self.model = model

    def get_response_chat(
        self,
        messages,
        max_tokens=8, 
        temperature=0,
        logprobs=True,  
        top_logprobs=20,
    ):
        # no batch
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs
        )
        
        return completion

    def get_response_completion(
        self,
        prompts: list,
        max_tokens=8, 
        temperature=0,
        batch_size=32
    ):
        responses = []
        for batch in tqdm(batch_generator(prompts, batch_size), total=(len(prompts)//batch_size)+1):
            completions = self.client.completions.create(
                model=self.model,
                prompt=batch,
                max_tokens=max_tokens,
                temperature=temperature
            )
            responses_batch = []
            for choice in completions.choices:
                responses_batch.append(choice.text)
            responses += responses_batch
        return responses
