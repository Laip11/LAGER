import json
from transformers import AutoTokenizer,AutoModelForCausalLM
import pandas as pd
import warnings
import argparse  
from utils import get_layer_outputs,get_batch_inputs,setup_logger


judge_aspects = ['answer_accuracy', 
               'logical_consistency', 
               'relevance', 
               'fluency_and_clarity', 
               'length_appropriateness', 
               'diversity', 
               'instruction_difficulty']

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    import torch
    args = argparse.ArgumentParser()
    args.add_argument('--aspect', type=str,required=True)
    args.add_argument('--model_name_or_path', required=True)
    args.add_argument('--batch_size', type=int,default=16)
    args = args.parse_args()

    if args.aspect not in judge_aspects:
        raise ValueError(f'Aspect {args.aspect} is not in the list of valid aspects: {judge_aspects}')

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
    direct_score_ls,weighted_score_ls,weighted_direct_score_ls,avg_weighted_score_ls = [],[],[],[]
    palmscore_w_ls,palmscore_wo_ls = [],[]
    for res in all_res:
        df = pd.DataFrame(res['res'])
        if df['direct_score'].iloc[-1] == -1:
            direct_score_ls.append(-1)
            weighted_score_ls.append(-1)
            weighted_direct_score_ls.append(-1)
            avg_weighted_score_ls.append(-1)
            palmscore_w_ls.append(-1)
            palmscore_wo_ls.append(-1)
            continue
        logits = df['logits'].apply(lambda x:torch.tensor(x,dtype=torch.float32))
        # 加权 palmscore(w tuning)
        distribution1 = torch.softmax((logits*weights).sum(),dim=-1)
        pre_score1 = (distribution1*torch.tensor([1,2,3,4,5,6,7,8,9],dtype=torch.float32)).sum().item()

        # 不加权 palmscore(w/o tuning)
        distribution2 = torch.softmax(logits.sum(),dim=-1)
        pre_score2 = (distribution2*torch.tensor([1,2,3,4,5,6,7,8,9],dtype=torch.float32)).sum().item()

        direct_score_ls.append(res['res'].iloc[-1]['direct_score'])
        weighted_score_ls.append(res['res'].iloc[-1]['weighted_score'])
        weighted_direct_score_ls.append(res['res']['direct_score'].mean().item())
        avg_weighted_score_ls.append(res['res']['weighted_score'].mean())

        palmscore_w_ls.append(pre_score1)
        palmscore_wo_ls.append(pre_score2)
    final_df = all_data[['instruction','input','output',f'prompt_{args.aspect}']].iloc[save_idx_ls]
    final_df[f'{args.aspect}_direct_score'] = direct_score_ls
    final_df[f'{args.aspect}_weighted_score'] = weighted_score_ls
    final_df[f'{args.aspect}_weighted_direct_score'] = weighted_direct_score_ls
    final_df[f'{args.aspect}_internal_score'] = avg_weighted_score_ls
    final_df[f'{args.aspect}_palmscore_w'] = palmscore_w_ls
    final_df[f'{args.aspect}_palmscore_wo'] = palmscore_wo_ls


    save_path = f'results/sft_data_{args.aspect}_filtered.jsonl'
    final_df.to_json(save_path,orient='records',lines=True)
    logger.info(f'Results saved to {save_path}')


