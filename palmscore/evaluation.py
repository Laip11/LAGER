import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import os
import json
import pandas as pd
import torch
import argparse
import torch.nn as nn
from utils import validate_data_consistency
from palmscore.optimize_layer_weights import optimize_layer_weights
from scipy.stats import spearmanr,pearsonr
from prettytable import PrettyTable
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--valid_data_path', type=str, required=True)
    args = parser.parse_args()
    return args


def calc_corr(pred_score, score_type,human_score,type_r = 'pearson' ):
    r_ls = ['pearson','spearman']

    if type_r not in r_ls:
        raise ValueError('type_r must be one of {}'.format(r_ls))
    elif type_r == 'pearson':
        r = pearsonr(pred_score, human_score)[0]
    elif type_r == 'spearman':
        r = spearmanr(pred_score, human_score)[0]
    return r

def print_correlations(all_score_dict,human_score):
    metrics = ['pearson','spearman']
    scores = ['direct_score','weighted_score','palmscore_wo','palmscore_w']
    # scores = ['palmscore_w',
    #           'prob_weighted_agg_e_score',
    #           'logits_agg_weighted_max_score',
    #           'prob_weighted_agg_max_score',
    #           'palmscore_wo',
    #           'prob_agg_e_score',
    #           'logits_agg_max_score',
    #           'prob_agg_max_score',
    #           'e_score',
    #           'direct_score']
    table = PrettyTable(['score_type']+metrics)
    for score in scores:
        add_row = [score] +[round(calc_corr(all_score_dict[score],score_type=score,human_score=human_score,type_r = 'pearson'),3),
                            round(calc_corr(all_score_dict[score],score_type=score,human_score=human_score,type_r = 'spearman'),3)]
        table.add_row(add_row)
    print(table)

def main():
    args = get_args()
    #model_name = validate_data_consistency(args.data_path, args.valid_data_path)

    weights = optimize_layer_weights(data_path = args.valid_data_path, 
                                      num_epochs=1, 
                                      lr=0.01,
                                      batch_size = 4,
                                      seed = 42)

    print("learned weights:", weights)
    weights = weights.detach().cpu().numpy()


    all_res = json.load(open(args.data_path))

    all_human_score,direct_score_ls,e_score_ls,logits_agg_max_score_ls,prob_weighted_agg_e_score_ls,prob_weighted_agg_max_score_ls = [],[],[],[],[],[]

    logits_agg_weighted_max_score_ls,palmscore_w_ls,palmscore_wo_ls,prob_agg_max_score_ls,layer_direct_score_weighted_ls,prob_agg_e_score_ls = [],[],[],[],[],[]


    for i in range(len(all_res)):
        res = all_res[i]
        
        if res['weighted_socre'] == -1:
            continue
        all_human_score.append(res['human_score'])
        df = pd.DataFrame(res['df']) 
        logits = df['logits'].apply(lambda x:torch.tensor(x,dtype=torch.float32))

        # logits_agg weighted
        distribution2 = torch.softmax((logits*weights).sum(),dim=-1)
        palmscore_w = ((distribution2*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum()).item()
        logits_agg_weighted_max_score = torch.argmax(distribution2).item()+1

    
        distribution3 = torch.softmax((logits/weights.shape[0]).sum(),dim=-1)
        palmscore_wo = (distribution3*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum().item()
        logits_agg_max_score = torch.argmax(distribution3).item()+1

        # prob_agg_max_score
        distribution4 = logits.apply(lambda x:torch.tensor(x,dtype=torch.float32).softmax(dim=-1)).sum().softmax(dim=-1)
        
        prob_agg_e_score = (distribution4*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum().item()
        prob_agg_max_score = torch.argmax(distribution4).item()+1

        # prob_weighted_agg
        distribution5  = (logits.apply(lambda x:torch.tensor(x,dtype=torch.float32).softmax(dim=-1))*weights).sum().softmax(dim=-1)
        prob_weighted_agg_e_score = (distribution5*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum().item()
        prob_weighted_agg_max_score = torch.argmax(distribution5).item()+1

        direct_score_ls.append(res['direct_socre'])
        e_score_ls.append(res['weighted_socre'])
        palmscore_w_ls.append(palmscore_w)
        palmscore_wo_ls.append(palmscore_wo)
        prob_agg_max_score_ls.append(prob_agg_max_score)
        prob_agg_e_score_ls.append(prob_agg_e_score)
        logits_agg_weighted_max_score_ls.append(logits_agg_weighted_max_score)
        logits_agg_max_score_ls.append(logits_agg_max_score)
        prob_weighted_agg_e_score_ls.append(prob_weighted_agg_e_score)
        prob_weighted_agg_max_score_ls.append(prob_weighted_agg_max_score)
        
    all_score_dict = {'direct_score':direct_score_ls,
                    'e_score':e_score_ls,
                    'palmscore_w':palmscore_w_ls,
                    'palmscore_wo':palmscore_wo_ls,
                    'prob_agg_max_score':prob_agg_max_score_ls,
                    'prob_agg_e_score':prob_agg_e_score_ls,
                    'logits_agg_max_score':logits_agg_max_score_ls,
                    'logits_agg_weighted_max_score':logits_agg_weighted_max_score_ls,
                    'prob_weighted_agg_e_score':prob_weighted_agg_e_score_ls,
                    'prob_weighted_agg_max_score':prob_weighted_agg_max_score_ls

                    }
    
    print_correlations(all_score_dict,all_human_score)
    if os.path.exists('scores') == False:
        os.mkdir('scores')
    with open('scores/direct_score.json','w') as f:
        json.dump(direct_score_ls,f)
    with open('scores/e_score.json','w') as f:
        json.dump(e_score_ls,f)
    with open('scores/palmscore_w.json','w') as f:
        json.dump(palmscore_w_ls,f)
    with open('scores/human_score.json','w') as f:
        json.dump(all_human_score,f)

if __name__ == '__main__':
    main()
