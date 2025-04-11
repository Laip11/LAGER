from email import message
import pandas as pd
import numpy as np
from tqdm import tqdm,trange
import os
import json
import argparse
import numpy as np
import warnings
from scipy.stats import spearmanr, pearsonr
warnings.filterwarnings("ignore")
import time
from openai import OpenAI



def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
    max_tokens=5,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=True,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
    top_logprobs=20,
    ) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion


import numpy as np

def calculate_expected_score(logprob_data):
    # 初始化为负无穷的 numpy 数组，表示1-5分的logprob
    score_map = np.full(5, -np.inf)

    # 遍历 logprob_data
    for top_lp in logprob_data:
        token = top_lp.token
        if token in ['1', '2', '3', '4', '5']:
            score_map[int(token) - 1] = top_lp.logprob  # 按索引存储 logprob
        else:
            continue
    if np.all(score_map == -np.inf):
        return 0, 0
    
    max_logit_index = np.argmax(score_map)  
    max_logit_score = max_logit_index + 1  

    
    exp_logprobs = np.exp(score_map)  
    probabilities = exp_logprobs / np.sum(exp_logprobs)

    expectation = np.dot(probabilities, np.arange(1, 6))  
    return expectation, max_logit_score




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--save_dir', type=str,default='results')
    argparser.add_argument('--points', type=int, default=5)
    argparser.add_argument('--input_file', type=str,default='/nfsdata/laip/datasets/BiGGen-Bench-human-eval.json')
    argparser.add_argument('--with_feedback', type=int, default=1)
    args = argparser.parse_args()
    API_SECRET_KEY= 'sk-t2pDyZtkt2XoCGez89425d07205d412591FbC4927d3b0a3c'
    BASE_URL = "https://api.ai-gaochao.cn/v1"
    ## vllm base_url = "http://localhost:8000/v1"
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)

    if 'flask' in (args.input_file).lower():
        data_name = 'flask'
    elif 'valid' in (args.input_file).lower():
        data_name = 'valid'
    elif 'biggen' in (args.input_file).lower():
        data_name = 'biggen'
    elif 'helpsteer' in (args.input_file).lower():
        data_name = 'helpsteer'
    



    all_prompts = []
    humanscore_ls = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                continue
            if args.with_feedback:
                all_prompts.append(data['user_prompt_feedback'])
            else:
                all_prompts.append(data['user_prompt'])
            humanscore_ls.append(sum(data['score'])/len(data['score']) if type(data['score'])==list else data['score'])
            if data_name == 'helpsteer':
                if len(all_prompts)>2000:
                    break
    
    score_ls = ['1','2','3','4','5']
    messages = [[{"role": "user", "content":prompt}] for prompt in all_prompts]
    direct_score_ls = []
    weighted_score_ls = []
    save_ls = []


    for i in trange(len(messages[:10])):
        message = messages[i]
        #try:
        completion = get_completion(
            messages=message,
            model="gpt-4o-mini",
            max_tokens=256,
            temperature=0,
            stop=None,
            seed=123,
            tools=None,
            logprobs=True,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.
            top_logprobs=10,
        )
        print(completion.choices[0].message.content)


        for idx in range(len(completion.choices[0].logprobs.content)-1,-1,-1):

            if completion.choices[0].logprobs.content[idx].token in score_ls:
                score_idx = idx
                break
        try:
            expected_score,direct_score = calculate_expected_score(completion.choices[0].logprobs.content[score_idx].top_logprobs)
            save_ls.append(i)
        except:
            continue
        direct_score_ls.append(direct_score)
        weighted_score_ls.append(expected_score)
        # except:
        #     continue
    humanscore_ls = [humanscore_ls[j] for j in save_ls]
    print(len(humanscore_ls),len(direct_score_ls),len(weighted_score_ls))

    print(f"{data_name}_{args.with_feedback*'_with_feedback'}")
    print(f'spearmanr_direct: {spearmanr(humanscore_ls,direct_score_ls)[0]}')
    print(f'pearsonr_direct: {pearsonr(humanscore_ls,direct_score_ls)[0]}')
    print(f'spearmanr_weighted: {spearmanr(humanscore_ls,weighted_score_ls)[0]}')
    print(f'pearsonr_weighted: {pearsonr(humanscore_ls,weighted_score_ls)[0]}')
    
    with open(f"{data_name}_{args.with_feedback*'_with_feedback'}_gpt-4o-mini.txt",'w') as f:
        f.write(f'spearmanr_direct: {spearmanr(humanscore_ls,direct_score_ls)[0]}\n')
        f.write(f'pearsonr_direct: {pearsonr(humanscore_ls,direct_score_ls)[0]}\n')
        f.write(f'spearmanr_weighted: {spearmanr(humanscore_ls,weighted_score_ls)[0]}\n')
        f.write(f'pearsonr_weighted: {pearsonr(humanscore_ls,weighted_score_ls)[0]}\n')


    