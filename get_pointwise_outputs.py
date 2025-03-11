from utils import get_batch_inputs,get_layer_outputs
from transformers import AutoModelForCausalLM,AutoTokenizer
import pandas as pd
from tqdm import tqdm 
import os
import json
import torch
import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name_or_path', type=str)
    argparser.add_argument('--save_dir', type=str,default='results')
    argparser.add_argument('--points', type=int, default=5)
    argparser.add_argument('--batch_size', type=int, default=4)
    argparser.add_argument('--max_new_tokens', type=int, default=256)
    argparser.add_argument('--input_file', type=str)
    argparser.add_argument('--with_feedback', type=int, default=0)
    argparser.add_argument('--dtype', type=str, default='bfloat16')
    argparser.add_argument('--max_length', type=int, default=2048)
    argparser.add_argument('--with_cot', type=int, default=0)
    argparser.add_argument('--temperature', type=float, default=0)
    args = argparser.parse_args()

    cuda_n = torch.cuda.device_count()
    if 'flask' in (args.input_file).lower():
        data_name = 'flask'
    elif 'valid' in (args.input_file).lower():
        data_name = 'valid'
    elif 'helpsteer' in (args.input_file).lower():
        data_name = 'helpsteer'
    else:
        data_name = args.input_file.split('/')[-1].split('.')[0]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map = 'auto',
        torch_dtype = args.dtype,
        trust_remote_code=True).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,trust_remote_code=True)


    model_name = args.model_name_or_path.split('/')[-1]

    points_ids_list = [ tokenizer.convert_tokens_to_ids([str(i)])[0] for i in range(1,args.points+1)]

    all_prompts = []
    score_ls = []

    # load dataset
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
            score_ls.append(sum(data['score'])/len(data['score']) if type(data['score'])==list else data['score'])
            if data_name == 'helpsteer':
                if len(all_prompts)>2000:
                    break
    print(len(all_prompts))

    tokenized_inputs,save_idx_ls = get_batch_inputs(all_prompts,
                                                    tokenizer,
                                                    with_cot = args.with_cot,
                                                    batch_size = args.batch_size,
                                                    device = f'cuda:{cuda_n-1}',
                                                    max_length=args.max_length)
    
    all_res = get_layer_outputs(model,
                                tokenized_inputs,
                                max_new_tokens = args.max_new_tokens,
                                tokenizer=tokenizer,
                                points_ids_list=points_ids_list,
                                temperature = args.temperature)


    processed_res = []
    for i in range(len(all_res)):
        res = all_res[i]
        idx = save_idx_ls[i]
        score = score_ls[idx]
        prompt = res['prompt']
        res1 = res['res']
        direct_socre = res1.iloc[-1]['direct_score']
        weighted_socre = res1.iloc[-1]['weighted_score']
        weighted_direct_socre = res1['direct_score'].mean().item()
        #internalscore = res1['weighted_score'].mean()

        dict1_res = {
                'idx':idx,
                'prompt':prompt,
                'human_score':score,
                'direct_socre':float(direct_socre),
                'weighted_socre':float(weighted_socre),
                'weighted_direct_socre':float(weighted_direct_socre),
                #'internalscore':float(internalscore),
                'df':res1.to_dict()}
        processed_res.append(dict1_res)

    if not os.path.exists(f'{args.save_dir}/{data_name}'):
            os.makedirs(f'{args.save_dir}/{data_name}')

    out_dir = f"{args.save_dir}/{data_name}/{model_name}{'_with_feedback'*args.with_feedback}_logits.json"
    with open(out_dir, 'w') as f:
        json.dump(processed_res, f, indent=4)

