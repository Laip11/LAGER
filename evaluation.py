import json
import numpy as np
import pandas as pd
import torch
import argparse
import torch.nn as nn
from src.utils import optimize_layer_weights
from scipy.stats import spearmanr,pearsonr
from prettytable import PrettyTable
import warnings
import random

random.seed(56)
warnings.filterwarnings("ignore")

def calc_corr(pred_score, score_type,human_score,type_r = 'pearson' ):
    r_ls = ['pearson','spearman']
    with open(f'predscore_{score_type}.json','w') as f:
        json.dump(pred_score,f)
    if type_r not in r_ls:
        raise ValueError('type_r must be one of {}'.format(r_ls))
    elif type_r == 'pearson':
        r = pearsonr(pred_score, human_score)[0]
    elif type_r == 'spearman':
        r = spearmanr(pred_score, human_score)[0]
    return r

def print_correlations(all_score_dict,human_score):
    metrics = ['pearson','spearman']
    scores = ['direct_score','palmscore_w','palmscore_wo']
    table = PrettyTable(['score_type']+metrics)
    for score in scores:
        add_row = [score] +[round(calc_corr(all_score_dict[score],score_type=score,human_score=human_score,type_r = 'pearson'),3),
                            round(calc_corr(all_score_dict[score],score_type='',human_score=human_score,type_r = 'spearman'),3)]
        table.add_row(add_row)
    print(table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--valid_data_path', type=str, required=True)
    args = parser.parse_args()
    if ( ('llama' in (args.data_path).lower() and 'llama' in (args.valid_data_path).lower()) or
        ('internlm' in (args.data_path).lower() and 'internlm' in (args.valid_data_path).lower()) or
        ('mistral' in (args.data_path).lower() and 'mistral' in (args.valid_data_path).lower())):
        pass
    else:
        raise Exception('The models corresponding to data_path and valid_data_path should be the same.')

    weights = optimize_layer_weights(data_path = args.valid_data_path, loss_fn = nn.CrossEntropyLoss(),num_epochs=1, lr=0.01)

    print("learned weights:", weights)
    weights = weights.numpy()

    data_path = args.data_path
    all_res = json.load(open(data_path))

    if 'llama' in data_path.lower():
        model_name = 'Meta-Llama-3___1-8B-Instruct'
    elif 'mistral' in data_path.lower():
        model_name = 'Mistral-7B-Instruct-v0___3'
    elif 'internlm' in data_path.lower():
        model_name = 'internlm3-8b-instruct'


    save_ls,all_human_score,direct_score_ls,weighted_score_ls,avg_direct_score_ls,avg_weighted_score = [],[],[],[],[],[]

    weighted_weighted_score_ls = []
    palmscore_w_ls =[]
    palmscore_wo_ls = []
    avg_logits_weighted_score_ls = []


    score = np.array([[1,2,3,4,5] for i in range(pd.DataFrame(all_res[0]['df']).shape[0])])
    final_score  = pd.DataFrame(score)


    for i in range(len(all_res)):
        res = all_res[i]
        
        if res['weighted_socre'] == -1:
            continue
        all_human_score.append(res['human_score'])
        df = pd.DataFrame(res['df']) 
        score = np.array([1,2,3,4,5])

        # weighed_score 加权
        distribution1 = df['logits'].apply(lambda x:torch.tensor(x,dtype=torch.float32).softmax(dim=-1))
        weighted_weighted_score = ((distribution1.apply(lambda x:(x*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum()))*weights).sum().item()

        # 累积logits 加权
        distribution2 = torch.softmax((logits*weights).sum(),dim=-1)
        palmscore_w = ((distribution2*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum()).item()

        # logits argmax weighted score
        # logits = df['logits'].apply(lambda x:torch.tensor(x,dtype=torch.float32))
        # distribution2 = (logits).apply(lambda x:torch.tensor(x,dtype=torch.float32).argmax(dim=-1))
        # pre_score = ((torch.tensor([ i+1 for i in distribution2]))*weights).sum().item()

        #print(pre_score2)
        #pre_score2 = torch.argmax(distribution2).item()+1

        # 累积logits 不加权    
        distribution3 = torch.softmax((logits/weights.shape[0]).sum(),dim=-1)
        palmscore_wo = (distribution3*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum().item()

        # 平均logits然后直接加权
        # logits 不加权然后weighed  avg_logits_weighted_score
        logits = df['logits'].apply(lambda x:torch.tensor(x,dtype=torch.float32))
        distribution4 = torch.softmax((logits/logits.shape[0]).sum(),dim=-1)
        avg_logits_weighted_score = (distribution4*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum().item()

        prompt = res['prompt']
        direct_score_ls.append(res['direct_socre'])
        weighted_score_ls.append(res['weighted_socre'])
        avg_direct_score_ls.append(res['weighted_direct_socre'])
        avg_weighted_score.append(res['internalscore'])
        weighted_weighted_score_ls.append(weighted_weighted_score)
        palmscore_w_ls.append(palmscore_w)
        palmscore_wo_ls.append(palmscore_wo)
        avg_logits_weighted_score_ls.append(avg_logits_weighted_score)
        
    all_score_dict = {'direct_score':direct_score_ls,
                    'weighted_score':weighted_score_ls,
                    'avg_direct_score':avg_direct_score_ls,
                    'avg_weighted_score':avg_weighted_score,
                    'weighted_weighted_score':weighted_weighted_score_ls,
                    'palmscore_w':palmscore_w_ls,
                    'palmscore_wo_ls':palmscore_wo_ls,
                    'avg_logits_weighted_score_ls':avg_logits_weighted_score_ls
                    }

    print_correlations(all_score_dict,all_human_score)

