import json
from tqdm import tqdm,trange
import numpy as np
import pandas as pd
import torch
from pyexpat import model
import os
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import random
random.seed(56)
from scipy.stats import spearmanr,pearsonr,kendalltau
from prettytable import PrettyTable
import warnings
warnings.filterwarnings("ignore")


def optimize_layer_weights(logits_list, targets, loss_fn, num_epochs=2, lr=0.01,min_lr = 1e-3):
    all_res=  []
    L = len(logits_list[0])  
    random.shuffle(logits_list)
    
    # 初始化权重
    weights = torch.nn.Parameter(torch.ones(L, requires_grad=True))
    optimizer = optim.Adam([weights], lr=lr)

    for epoch in trange(num_epochs):
        total_loss = 0

        for sample_idx in trange(len(logits_list)):  # 遍历所有样本
            logits = logits_list[sample_idx]  
            target = targets[sample_idx]  

            if type(loss_fn) == torch.nn.modules.loss.CrossEntropyLoss:
                target = target - 1
                target = torch.tensor(target,dtype=torch.long)

            normalized_weights = torch.softmax(weights, dim=0)

            # 计算加权和
            weighted_sum = torch.zeros_like(logits[0])  
            for l in range(L):
                if type(loss_fn) == torch.nn.modules.loss.CrossEntropyLoss:
                    weighted_sum += normalized_weights[l] * logits[l]  # logits累积加权
                    predictions = weighted_sum
                else:
                    weighted_sum += normalized_weights[l] * (torch.tensor(logits[l])*torch.tensor([1,2,3,4,5])).sum()  # 加权求和
                    predictions = weighted_sum  # 预测结果


            loss = loss_fn(predictions,target)

            total_loss += loss.item()
            all_res.append(total_loss/(sample_idx+1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #scheduler.step()

        # 每个 epoch 打印损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")

    return torch.softmax(weights, dim=0).detach(),all_res


def calc_corr(pred_score, score_type,human_score,type_r = 'pearson' ):
    r_ls = ['pearson','spearman','kendalltau']
    with open(f'predscore_{score_type}.json','w') as f:
        json.dump(pred_score,f)
    if type_r not in r_ls:
        raise ValueError('type_r must be one of {}'.format(r_ls))
    elif type_r == 'pearson':
        r = pearsonr(pred_score, human_score)[0]
    elif type_r == 'kendalltau':
        r = kendalltau(pred_score, human_score)[0]
    else:
        r = spearmanr(pred_score, human_score)[0]
    return r

def print_correlations(all_score_dict,human_score):
    metrics = ['pearson','spearman','kendalltau']   
    scores = ['direct_score','weighted_score','weighted_direct_score','internalscore','pre_score1','pre_score3','pre_score2','pre_score4']
    table = PrettyTable(['score_type']+metrics)
    for score in scores:
        add_row = [score] +[round(calc_corr(all_score_dict[score],score_type=score,human_score=human_score,type_r = 'pearson'),3),
                            round(calc_corr(all_score_dict[score],score_type='',human_score=human_score,type_r = 'spearman'),3),
                            round(calc_corr(all_score_dict[score],score_type='',human_score=human_score,type_r = 'kendalltau'),3)]
        table.add_row(add_row)
    print(table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/nfsdata/laip/results/valid/Meta-Llama-3___1-8B-Instruct_logits.json', help='data path')
    args = parser.parse_args()

    all_data = json.load(open(args.data_path,'r'))
    human_score_ls = []
    logits_ls = []
    human_score_ls1 = []

    weighted_score_ls = []
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.MSELoss()

    for data in all_data:
        df = pd.DataFrame(data['df'])
        if data['weighted_socre']==-1:
            continue
        _logits = torch.tensor([i for i in df['logits']],dtype=torch.float32)
        #_logits = [i for i in df['logits']]
        human_score_ls1.append(data['human_score'])
        if type(loss_fn) == torch.nn.modules.loss.CrossEntropyLoss:
            human_score = torch.tensor(data['human_score'],dtype=torch.long)
            
        else:
            human_score = torch.tensor(data['human_score'],dtype=torch.float32)

        weighted_score = torch.tensor(df['weighted_score'].to_list(),dtype=torch.float32)
        human_score_ls.append(human_score)
        logits_ls.append(_logits)
        weighted_score_ls.append(weighted_score)

    
    weights,all_loss = optimize_layer_weights(logits_ls, human_score_ls, nn.CrossEntropyLoss(),num_epochs=1, lr=0.01)

    print("学习到的权重：", weights)
    weights = weights.numpy()
    import matplotlib.pyplot as plt

    plt.plot(all_loss)




def calc_corr(pred_score, score_type,human_score,type_r = 'pearson' ):
    r_ls = ['pearson','spearman','kendalltau']
    with open(f'predscore_{score_type}.json','w') as f:
        json.dump(pred_score,f)
    if type_r not in r_ls:
        raise ValueError('type_r must be one of {}'.format(r_ls))
    elif type_r == 'pearson':
        r = pearsonr(pred_score, human_score)[0]
    elif type_r == 'kendalltau':
        r = kendalltau(pred_score, human_score)[0]
    else:
        r = spearmanr(pred_score, human_score)[0]
    return r

def print_correlations(all_score_dict,human_score):
    metrics = ['pearson','spearman','kendalltau']   
    scores = ['direct_score','weighted_score','weighted_direct_score','internalscore','pre_score1','pre_score3','pre_score2','pre_score4']
    table = PrettyTable(['score_type']+metrics)
    for score in scores:
        add_row = [score] +[round(calc_corr(all_score_dict[score],score_type=score,human_score=human_score,type_r = 'pearson'),3),
                            round(calc_corr(all_score_dict[score],score_type='',human_score=human_score,type_r = 'spearman'),3),
                            round(calc_corr(all_score_dict[score],score_type='',human_score=human_score,type_r = 'kendalltau'),3)]
        table.add_row(add_row)
    print(table)

    if 'tulu' in data_path.lower():
        model_name = 'Llama-3___1-Tulu-3-8B'
    elif 'llama' in data_path.lower():
        model_name = 'Meta-Llama-3___1-8B-Instruct'
    elif 'mistral' in data_path.lower():
        model_name = 'Mistral-7B-Instruct-v0___3'
    elif 'internlm' in data_path.lower():
        model_name = 'internlm3-8b-instruct'
    elif 'qwen' in data_path.lower():
        model_name = 'Qwen2___5-7B-Instruct'


    data_path = f'/home/laip/InternalScore/results/flask/{model_name}_logits.json'
    
    all_human_score = []

    all_res = json.load(open(data_path))
    direct_score_ls = []
    weighted_score_ls = []
    weighted_direct_score_ls= []
    internalscore_ls = []
    save_ls = []
    pre_score_ls1 = []
    pre_score_ls2 = []
    pre_score_ls3 = []
    pre_score_ls4 = []
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
        pre_score1 = ((distribution1.apply(lambda x:(x*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum()))*weights).sum().item()

        # 累积logits 加权
        logits = df['logits'].apply(lambda x:torch.tensor(x,dtype=torch.float32))
        distribution2 = (logits).apply(lambda x:torch.tensor(x,dtype=torch.float32).argmax(dim=-1))
        pre_score = ((torch.tensor([ i+1 for i in distribution2]))*weights).sum().item()

        distribution2 = torch.softmax((logits*weights).sum(),dim=-1)
        pre_score2 = ((distribution2*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum()).item()
        #print(pre_score2)
        #pre_score2 = torch.argmax(distribution2).item()+1

        # 累积logits 不加权    
        distribution3 = torch.softmax((logits/weights.shape[0]).sum(),dim=-1)
        pre_score3 = (distribution3*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum().item()

        # ratio + cosine similarity 筛选
    # logits = df[(df['ratio']>ratio)&(df['weights']>cos)]['logits'].apply(lambda x:torch.tensor(x,dtype=torch.float32))
        logits = df['logits'].apply(lambda x:torch.tensor(x,dtype=torch.float32))
        distribution4 = torch.softmax((logits/logits.shape[0]).sum(),dim=-1)
        pre_score4 = (distribution4*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum().item()

        prompt = res['prompt']
        direct_score_ls.append(res['direct_socre'])
        weighted_score_ls.append(res['weighted_socre'])
        weighted_direct_score_ls.append(res['weighted_direct_socre'])
        internalscore_ls.append(res['internalscore'])
        pre_score_ls1.append(pre_score1)
        pre_score_ls2.append(pre_score2)
        pre_score_ls3.append(pre_score3)
        pre_score_ls4.append(pre_score)
        
        #internalscore_ls.append((df['weighted_score']*weights).sum())
    #print(df[df['ratio']>r].shape[0])
    print(len(all_human_score),len(direct_score_ls),len(weighted_score_ls),len(weighted_direct_score_ls),len(internalscore_ls),len(pre_score_ls1))
    all_score_dict = {'direct_score':direct_score_ls,
                    'weighted_score':weighted_score_ls,
                    'weighted_direct_score':weighted_direct_score_ls,
                    'internalscore':internalscore_ls,
                    'pre_score1':pre_score_ls1,
                    'pre_score2':pre_score_ls2,
                    'pre_score3':pre_score_ls3,
                    'pre_score4':pre_score_ls4
                    }


    print_correlations(all_score_dict,all_human_score)

