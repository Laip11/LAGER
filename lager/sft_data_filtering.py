import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import pandas as pd
import warnings
import argparse  
import torch.nn as nn
from utils import *
from lager.optimize_layer_weights import optimize_layer_weights
warnings.filterwarnings("ignore")

judge_aspects = [
    'answer_accuracy', 
    'logical_consistency', 
    'relevance', 
    'fluency_and_clarity', 
    'length_appropriateness', 
    'diversity', 
    'instruction_difficulty'
    ]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aspect', type=str,required=True)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--batch_size', type=int,default=16)
    parser.add_argument('--valid_data_path', type=str,required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    validate_model_and_data_consistency(args.model_name_or_path,args.valid_data_path)
    weights = optimize_layer_weights(data_path = args.valid_data_path, 
                                      loss_fn = nn.CrossEntropyLoss(),
                                      num_epochs=1, 
                                      lr=0.01,
                                      batch_size = 8,
                                      seed = 42)

    if args.aspect not in judge_aspects:
        raise ValueError(f'Aspect {args.aspect} is not in the list of judge aspects: {judge_aspects}')

    logger = setup_logger(__name__,log_file=f"data_filering_{args.aspect}.log")

    logger.info(f'Aspect: {args.aspect}')

    logger.info('Start loading model and tokenizer...')
    # load model and tokenizer
    model,tokenizer = load_model_and_tokenizer(args.model_path)
    
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
    
    direct_score_ls,weighted_score_ls,weighted_direct_score_ls,avg_weighted_score_ls = [],[],[],[]
    lager_w_ls,lager_wo_ls = [],[]
    for res in all_res:
        df = pd.DataFrame(res['res'])
        if df['direct_score'].iloc[-1] == -1:
            direct_score_ls.append(-1)
            weighted_score_ls.append(-1)
            weighted_direct_score_ls.append(-1)
            avg_weighted_score_ls.append(-1)
            lager_w_ls.append(-1)
            lager_wo_ls.append(-1)
            continue

        logits = df['logits'].apply(lambda x:torch.tensor(x,dtype=torch.float32))
        #  lager(w tuning)
        distribution1 = torch.softmax((logits*weights).sum(),dim=-1)
        pre_score1 = (distribution1*torch.tensor([1,2,3,4,5,6,7,8,9],dtype=torch.float32)).sum().item()

        # lager(w/o tuning)
        distribution2 = torch.softmax((logits/logits/weights.shape[0]).sum(),dim=-1)
        pre_score2 = (distribution2*torch.tensor([1,2,3,4,5,6,7,8,9],dtype=torch.float32)).sum().item()

        direct_score_ls.append(res['res'].iloc[-1]['direct_score'])
        weighted_score_ls.append(res['res'].iloc[-1]['weighted_score'])
        weighted_direct_score_ls.append(res['res']['direct_score'].mean().item())
        avg_weighted_score_ls.append(res['res']['weighted_score'].mean())

        lager_w_ls.append(pre_score1)
        lager_wo_ls.append(pre_score2)
    final_df = all_data[['instruction','input','output',f'prompt_{args.aspect}']].iloc[save_idx_ls]
    final_df[f'{args.aspect}_direct_score'] = direct_score_ls
    final_df[f'{args.aspect}_weighted_score'] = weighted_score_ls
    final_df[f'{args.aspect}_weighted_direct_score'] = weighted_direct_score_ls
    final_df[f'{args.aspect}_internal_score'] = avg_weighted_score_ls
    final_df[f'{args.aspect}_lager_w'] = lager_w_ls
    final_df[f'{args.aspect}_lager_wo'] = lager_wo_ls


    save_path = f'results/sft_data_{args.aspect}_filtered.jsonl'
    final_df.to_json(save_path,orient='records',lines=True)
    logger.info(f'Results saved to {save_path}')
if __name__ == '__main__':
    main()


