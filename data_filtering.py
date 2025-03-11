from my_utils.setup_logger import setup_logger
import json
from transformers import AutoTokenizer,AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import warnings
import argparse  
# conda activate llm
#     python data_filtering.py --aspect
sft_aspects = ['answer_accuracy', 'logical_consistency', 'relevance', 'fluency_and_clarity', 'length_appropriateness', 'diversity', 'instruction_difficulty']
warnings.filterwarnings("ignore")


def get_batch_inputs(prompts, tokenizer, batch_size = 8, padding = 'max_length',device = 'cuda',max_length = 2048):
    from tqdm import tqdm
    tokenized_inputs = []
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    messages = [{"role": "user", "content": prompt } for prompt in prompts]

    all_text = [tokenizer.apply_chat_template([message],tokenize=False,add_generation_prompt = True) for message in messages]

    save_idx_ls,post_del_text = [],[]
    for i in tqdm(range(len(all_text))):
        text = all_text[i]
        if len(tokenizer(text)['input_ids']) <= max_length:
            save_idx_ls.append(i)
            post_del_text.append(text)

    for i in tqdm(range(0,len(post_del_text),batch_size)):
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
        columns = ['layer_n','direct_score','weighted_score','probs','ratio']
        
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
            h_list= []
            logits_list = []
            for layer_n,layer_hidden_state in enumerate(score_hidden_state):
                
                last_token_hidden_state = layer_hidden_state[i,-1,:]
                
                lm_head = model.lm_head

                logits = lm_head(last_token_hidden_state)
                logits_list.append(logits[points_ids_list].to(torch.float32).to('cpu').tolist())
                h_list.append(last_token_hidden_state.to(torch.float32).to('cpu').tolist())
                soft_max_logits = logits.softmax(dim=-1)
                max_logits = soft_max_logits.max(dim=-1).values.item()
                point_max_logits = soft_max_logits[points_ids_list].max(dim=-1).values.item()
                ratio = point_max_logits/max_logits
                probs = logits[points_ids_list].softmax(dim=-1)

                direct_score  = probs.argmax(dim=-1).item()+1
                probs_dict = { p+1: probs[p].item() for p in range(len(points_ids_list)) }
                weight_score = sum([key*value for key,value in zip(probs_dict.keys(),probs_dict.values())])

                probs = probs.tolist()
                layer_res = pd.DataFrame([[layer_n,direct_score,weight_score,probs,ratio]],columns=columns)
                res = pd.concat([res,layer_res],axis=0,ignore_index=True)


            res['logits'] = logits_list
            all_res.append({'prompt':prompts[i],'res':res})
        
    return all_res 


if __name__ == '__main__':
    import torch
    args = argparse.ArgumentParser()
    args.add_argument('--aspect', type=str,required=True)
    args.add_argument('--model_name_or_path', required=True)
    args.add_argument('--batch_size', type=int,default=16)
    args = args.parse_args()

    if args.aspect not in sft_aspects:
        raise ValueError(f'Aspect {args.aspect} is not in the list of valid aspects: {sft_aspects}')

    logger = setup_logger(__name__,log_file=f"data_filering_{args.aspect}.log")

    logger.info(f'Aspect: {args.aspect}')

    logger.info('Start loading model and tokenizer...')
    # load model and tokenizer
    model_path = args.model_path 

    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype = torch.bfloat16,device_map = 'cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    points_ids_list = [ tokenizer.convert_tokens_to_ids([str(i)])[0] for i in range(1,10)]
    logger.info('Model and tokenizer loaded.')


    # load data
    data_path = '/nfsdata/laip/datasets/sft_prompt_7type.jsonl'
    all_data = pd.read_json(data_path,lines=True)
    all_prompt = all_data[f'prompt_{args.aspect}'].tolist()
    logger.info('Data loaded,starting to tokenize data...')

    tokenized_inputs,save_idx_ls = get_batch_inputs(all_prompt,tokenizer,batch_size = args.batch_size,device = 'cuda',max_length = 1600)
    
    logger.info(f'Data tokenized.Total number of data: {len(save_idx_ls)}')
    logger.info('Start filtering data...')

    all_res = get_layer_outputs(model, tokenized_inputs,tokenizer,max_new_tokens = 15,points_ids_list = points_ids_list)
    
    logger.info('Data filtering finished.')
    logger.info('Start saving results...')
    weights = [0.003437180072069168,
 0.0034694664645940065,
 0.003370967460796237,
 0.0033762697130441666,
 0.0032019256614148617,
 0.003022323828190565,
 0.0027004529256373644,
 0.0031398916617035866,
 0.00311932316981256,
 0.0032164095900952816,
 0.0035408996045589447,
 0.003587346524000168,
 0.003944995813071728,
 0.004196827299892902,
 0.0036134175024926662,
 0.003482070518657565,
 0.0035765226930379868,
 0.002972847782075405,
 0.002817249856889248,
 0.0024364066775888205,
 0.0024376907385885715,
 0.0024486437905579805,
 0.0036508042830973864,
 0.8593869209289551,
 0.0031657542567700148,
 0.003469533985480666,
 0.0035897952038794756,
 0.009388254024088383,
 0.005958248395472765,
 0.005170912481844425,
 0.005272462032735348,
 0.0053072962909936905,
 0.024530891329050064]
    direct_score_ls,weighted_score_ls,weighted_direct_score_ls,internalscore_ls = [],[],[],[]
    pred_score_ls1,pred_score_ls2,pred_score_ls3 = [],[],[]
    for res in all_res:
        df = pd.DataFrame(res['res'])
        if df['direct_score'].iloc[-1] == -1:
            direct_score_ls.append(-1)
            weighted_score_ls.append(-1)
            weighted_direct_score_ls.append(-1)
            internalscore_ls.append(-1)
            pred_score_ls1.append(-1)
            pred_score_ls2.append(-1)
            pred_score_ls3.append(-1)
            continue
        logits = df['logits'].apply(lambda x:torch.tensor(x,dtype=torch.float32))
        # 加权
        distribution1 = torch.softmax((logits*weights).sum(),dim=-1)
        pre_score1 = (distribution1*torch.tensor([1,2,3,4,5,6,7,8,9],dtype=torch.float32)).sum().item()
        # 不加权
        distribution2 = torch.softmax(logits.sum(),dim=-1)
        pre_score2 = (distribution2*torch.tensor([1,2,3,4,5,6,7,8,9],dtype=torch.float32)).sum().item()
        distribution3 = df['logits'].apply(lambda x:torch.tensor(x,dtype=torch.float32).softmax(dim=-1))
        pre_score3 = ((distribution3.apply(lambda x:(x*torch.tensor([1,2,3,4,5,6,7,8,9],dtype=torch.float32)).sum()))*weights).sum().item()
        direct_score_ls.append(res['res'].iloc[-1]['direct_score'])
        weighted_score_ls.append(res['res'].iloc[-1]['weighted_score'])
        weighted_direct_score_ls.append(res['res']['direct_score'].mean().item())
        internalscore_ls.append(res['res']['weighted_score'].mean())
        pred_score_ls1.append(pre_score1)
        pred_score_ls2.append(pre_score2)
        pred_score_ls3.append(pre_score3)
    final_df = all_data[['instruction','input','output',f'prompt_{args.aspect}']].iloc[save_idx_ls]
    final_df[f'{args.aspect}_direct_score'] = direct_score_ls
    final_df[f'{args.aspect}_weighted_score'] = weighted_score_ls
    final_df[f'{args.aspect}_weighted_direct_score'] = weighted_direct_score_ls
    final_df[f'{args.aspect}_internal_score'] = internalscore_ls
    final_df[f'{args.aspect}_pred_score1'] = pred_score_ls1
    final_df[f'{args.aspect}_pred_score2'] = pred_score_ls2
    final_df[f'{args.aspect}_pred_score3'] = pred_score_ls3

    save_path = f'/nfsdata/laip/datasets/sft_data_{args.aspect}_filtered.jsonl'
    final_df.to_json(save_path,orient='records',lines=True)
    logger.info(f'Results saved to {save_path}')


