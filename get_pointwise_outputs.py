

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


def get_batch_inputs(prompts, tokenizer, batch_size = 8, padding = 'max_length',device = 'cuda',max_length = 2048,with_cot:int=0):

    tokenized_inputs = []
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if with_cot:
        messages = [{"role": "user", "content": prompt+'Please think through it step by step' } for prompt in prompts]
    else:
        messages = [{"role": "user", "content": prompt } for prompt in prompts]

    all_text = [tokenizer.apply_chat_template([message],tokenize=False,add_generation_prompt = True) for message in messages]

    save_idx_ls,post_del_text = [],[]
    for i in range(len(all_text)):
        text = all_text[i]
        if len(tokenizer(text)['input_ids']) <= max_length:
            save_idx_ls.append(i)
            post_del_text.append(text)

    for i in range(0,len(post_del_text),batch_size):
        text_batch = post_del_text[i:(i+batch_size)]
        batch_prompts = [prompts[idx] for idx in save_idx_ls[i:(i+batch_size)]]
        batch_inputs = tokenizer(text_batch, padding=padding, max_length = max_length, return_tensors="pt",padding_side = 'left').to(device)
        tokenized_inputs.append({'batch_prompts':batch_prompts, 'batch_inputs':batch_inputs,'text_batch':text_batch})
    return (tokenized_inputs,save_idx_ls)


def get_layer_outputs(model, tokenized_inputs,tokenizer,max_new_tokens,points_ids_list,temperature = 0):

    all_res = []
    if temperature != 0:
        do_sample = True
    else:
        do_sample = False
    similarity_matrix = []
    for block_inputs in tqdm(tokenized_inputs):

        prompts = block_inputs['batch_prompts']
        outputs = model.generate(
                    **block_inputs['batch_inputs'],
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    do_sample = do_sample,
                    temperature = temperature,
                    top_k = 5,
                    top_p = 0.9,
                    max_new_tokens = max_new_tokens,
                    )
        responses_ids = outputs.sequences[:,block_inputs['batch_inputs']['input_ids'].shape[-1]:]
        columns = ['layer_n','direct_score','weighted_score','probs']
        responses = tokenizer.batch_decode(responses_ids,skip_special_tokens=True)
        print(responses)
        
        for i in range(responses_ids.shape[0]):
            score_idxs  = None
            for j in range(responses_ids.shape[1]-1,-1,-1):
                if responses_ids[i,j] in points_ids_list:
                    score_idxs = j
                    break
            if score_idxs == None:
                res = pd.DataFrame([[-1]*len(columns)],columns=columns)
                all_res.append({'prompt':prompts[i],'res':res})
                continue
            
            score_hidden_state = outputs.hidden_states[score_idxs]
            res = pd.DataFrame(columns=columns)

            logits_list = []
            for layer_n,layer_hidden_state in enumerate(score_hidden_state):
                
                last_token_hidden_state = layer_hidden_state[i,-1,:]
                
                lm_head = model.lm_head

                logits = lm_head(last_token_hidden_state)
                logits_list.append(logits[points_ids_list].to(torch.float32).to('cpu').tolist())

                probs = logits[points_ids_list].softmax(dim=-1)
                probs_dict = { p+1: probs[p].item() for p in range(len(points_ids_list)) }

                direct_score  = probs.argmax(dim=-1).item()+1
                weight_score = sum([key*value for key,value in zip(probs_dict.keys(),probs_dict.values())])

                probs = probs.tolist()
                layer_res = pd.DataFrame([[layer_n,direct_score,weight_score,probs]],columns=columns)
                res = pd.concat([res,layer_res],axis=0,ignore_index=True)
    
            res['logits'] = logits_list
            all_res.append({'prompt':prompts[i],'res':res})
    return all_res 



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

