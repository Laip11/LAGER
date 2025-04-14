import sys
from pathlib import Path

# 将父目录添加到 sys.path
sys.path.append(str(Path(__file__).parent.parent))
import json
import torch
import torch.nn as nn
import argparse
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import pandas as pd
import warnings
from ..utils import *
from .optimize_layer_weights import optimize_layer_weights
warnings.filterwarnings("ignore")


metric_list = ['spearmanr', 'pearsonr']
prompt = '''
Your task is to predict the likely emotional response of a character in this dialogue:

[dialogue]
[End dialogue]

At the end of this dialogue, rate how strongly [human] is likely feeling [emotion_name] on a scale of 1-9.

You only need to output your rating, with no additional commentary:

'''
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,default="Shanghai_AI_Laboratory/internlm3-8b-instruct")
    parser.add_argument('--points', type=int,default=9)
    parser.add_argument('--valid_data_path', type=str,required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model_name = args.model_path.split('/')[-1]

    validate_model_and_data_consistency(args.model_path, args.valid_data_path)


    weights = optimize_layer_weights(
        data_path = args.valid_data_path, 
        loss_fn = nn.CrossEntropyLoss(),
        num_epochs=1, 
        lr=0.01,
        batch_size = 8,
        seed = 42).numpy()
    
    all_data = json.load(open('data/emotional_understanding/emotional_data.json'))

    model,tokenizer = load_model_and_tokenizer(args.model_path)
    
    try:
        points_ids_list = [ 
            tokenizer.convert_tokens_to_ids([str(i)])[0] 
            for i in range(1,args.points+1)
            ]
    except:
        points_ids_list = [ 
            tokenizer.convert_tokens_to_ids([str(i).encode()])[0] 
            for i in range(1,args.points+1)
            ]

    res_score,direct_score_ls,weighted_score_ls,weighted_direct_score_ls,avg_weighed_score_ls,palmscore_w_ls,palmscore_wo_ls= [],[],[],[],[],[],[]

    for data in tqdm(all_data):
        _prompt = prompt.replace('[dialogue]',data['dialogue']).replace('[emotion_name]',data['emotion']).replace('[human]',data['human'])
        message = [{'role':'user','content':_prompt}]
        inputs = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt = True)
        inputs = tokenizer(inputs,return_tensors='pt').to(model.device)
        outputs = model.generate(
                    **inputs,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    do_sample = False,
                    max_new_tokens = 10,
                    )
        responses_ids = outputs.sequences[0][inputs['input_ids'].shape[-1]:]
        # responses = tokenizer.decode(responses_ids,skip_special_tokens=True)
        # print(responses)
        columns = ['layer_n','direct_score','weighted_score','probs','ratio']
        
        score_idxs  = None
        for j in range(responses_ids.shape[0]-1,-1,-1):
            if responses_ids[j] in points_ids_list:
                score_idxs = j
                res_score.append(int(data['emotion_score']))
                break
        if score_idxs == None:
            continue
        
        score_hidden_state = outputs.hidden_states[score_idxs]
        res = pd.DataFrame(columns=columns)
        h_list= []
        logits_list = []
        for layer_n,layer_hidden_state in enumerate(score_hidden_state):
            
            last_token_hidden_state = layer_hidden_state[0,-1,:]
            
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
        logits = res['logits'].apply(lambda x:torch.tensor(x,dtype=torch.float32))
        # palmscore(w. tuning)
        distribution1 = torch.softmax((logits*weights).sum(),dim=-1)
        palmscore_w = (distribution1*torch.tensor([1,2,3,4,5,6,7,8,9],dtype=torch.float32)).sum().item()

        # palmscore(w.o tuning)
        distribution2 = torch.softmax((logits/len(weights)).sum(),dim=-1)
        palmscore_wo = (distribution2*torch.tensor([1,2,3,4,5,6,7,8,9],dtype=torch.float32)).sum().item()

        
        direct_score_ls.append(res.iloc[-1]['direct_score'])
        weighted_score_ls.append(res.iloc[-1]['weighted_score'])
        weighted_direct_score_ls.append(res['direct_score'].mean().item())
        avg_weighed_score_ls.append(res['weighted_score'].mean().item())
        palmscore_w_ls.append(palmscore_w)
        palmscore_wo_ls.append(palmscore_wo)


    calc_p = lambda x,y: pearsonr(x,y)[0]
    calc_s = lambda x,y: spearmanr(x,y)[0]
    print(len(direct_score_ls),len(res_score))
    print(model_name)
    print('direct_score:',round(calc_p(direct_score_ls,res_score),3),round(calc_s(direct_score_ls,res_score),3))
    print('weighted_score:',round(calc_p(weighted_score_ls,res_score),3),round(calc_s(weighted_score_ls,res_score),3))
    print('palmscore_wo:',round(calc_p(palmscore_wo_ls,res_score),3),round(calc_s(palmscore_wo_ls,res_score),3))
    print('palmscore_w:',round(calc_p(palmscore_w_ls,res_score),3),round(calc_s(palmscore_w_ls,res_score),3))

if __name__ == '__main__':
    main()

