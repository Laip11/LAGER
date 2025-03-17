import json
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import numpy as np
import argparse
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
metric_list = ['spearmanr', 'pearsonr']
argparseser = argparse.ArgumentParser()
argparseser.add_argument('--model_path', type=str,default="Shanghai_AI_Laboratory/internlm3-8b-instruct")
argparseser.add_argument('--points', type=int,default=9)
args = argparseser.parse_args()

model_name = args.model_path.split('/')[-1]
all_data = json.load(open('data/sentiment_data.json'))

prompt = '''
Your task is to predict the likely emotional response of a character in this dialogue:

[dialogue]
[End dialogue]

At the end of this dialogue, rate how strongly [human] is likely feeling [emotion_name] on a scale of 1-9.

You only need to output your rating, with no additional commentary:

'''


model = AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype = 'bfloat16',device_map = 'cuda').eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
try:
    points_ids_list = [ tokenizer.convert_tokens_to_ids([str(i)])[0] for i in range(1,args.points+1)]
except:
    points_ids_list = [ tokenizer.convert_tokens_to_ids([str(i).encode()])[0] for i in range(1,args.points+1)]

weights_ls = {'llama':[0.00343718, 0.00346947, 0.00337097, 0.00337627, 0.00320193,
       0.00302232, 0.00270045, 0.00313989, 0.00311932, 0.00321641,
       0.0035409 , 0.00358735, 0.003945  , 0.00419683, 0.00361342,
       0.00348207, 0.00357652, 0.00297285, 0.00281725, 0.00243641,
       0.00243769, 0.00244864, 0.0036508 , 0.8593869 , 0.00316575,
       0.00346953, 0.0035898 , 0.00938825, 0.00595825, 0.00517091,
       0.00527246, 0.0053073 , 0.02453089],
            'tulu':[1.9926482e-03, 2.0288283e-03, 1.9283649e-03, 1.9503274e-03,
       1.8866975e-03, 1.8496471e-03, 1.7655289e-03, 1.8533312e-03,
       1.7974996e-03, 1.7305122e-03, 1.9543676e-03, 2.1507202e-03,
       2.3168905e-03, 2.2129463e-03, 1.7726267e-03, 1.7696656e-03,
       1.4796040e-03, 1.1610090e-03, 1.4760328e-03, 7.6917995e-04,
       7.1918254e-04, 5.7078211e-04, 7.3513773e-04, 9.5099837e-01,
       1.0121432e-03, 1.2787035e-03, 1.0262263e-03, 1.5768759e-03,
       1.1171129e-03, 1.0530197e-03, 1.0068050e-03, 1.0300645e-03,
       2.0291803e-03],
             'mistral':[0.02530812, 0.02529554, 0.02525384, 0.02544024, 0.0254709 ,
       0.02538827, 0.02517533, 0.02481054, 0.02489335, 0.02485125,
       0.02570229, 0.02602397, 0.02684528, 0.02672314, 0.02797128,
       0.0279035 , 0.02839368, 0.02760266, 0.02747969, 0.02890367,
       0.03517389, 0.03830925, 0.04022087, 0.03432938, 0.03526052,
       0.03162143, 0.03453825, 0.05912246, 0.04450823, 0.0135075 ,
       0.0110993 , 0.01106449, 0.08580787],
              'internlm':[0.02532368, 0.02608873, 0.02565252, 0.02568329, 0.02251901,
       0.0214611 , 0.02118479, 0.01968816, 0.02265235, 0.0219109 ,
       0.02135445, 0.01698261, 0.02305569, 0.02254137, 0.02456228,
       0.01938125, 0.01938198, 0.03340582, 0.04947908, 0.03269262,
       0.02046547, 0.01132542, 0.00991966, 0.01287424, 0.00894568,
       0.00456296, 0.00401398, 0.00373798, 0.00364853, 0.00779461,
       0.0107277 , 0.01476849, 0.00996623, 0.0989219 , 0.13831984,
       0.00277205, 0.00235647, 0.00221785, 0.00258766, 0.00203182,
       0.00195165, 0.00179587, 0.00186209, 0.00171047, 0.00193544,
       0.00224081, 0.00464406, 0.00589424, 0.11100519]}

if 'tulu' in (args.model_path).lower():
    weights = weights_ls['tulu']
elif 'llama' in (args.model_path).lower():
    weights = weights_ls['llama']
elif 'mistral' in (args.model_path).lower():
    weights = weights_ls['mistral']
elif 'internlm' in (args.model_path).lower():
    weights = weights_ls['internlm']
res_score = []
direct_score_ls,weighted_score_ls,weighted_direct_score_ls,internalscore_ls,pre_score1_ls,pre_score2_ls,pre_score3_ls= [],[],[],[],[],[],[]
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
    # 加权
    distribution1 = torch.softmax((logits*weights).sum(),dim=-1)
    pre_score1 = (distribution1*torch.tensor([1,2,3,4,5,6,7,8,9],dtype=torch.float32)).sum().item()
    # 不加权
    distribution2 = torch.softmax((logits/len(weights)).sum(),dim=-1)
    pre_score2 = (distribution2*torch.tensor([1,2,3,4,5,6,7,8,9],dtype=torch.float32)).sum().item()
    distribution3 = res['logits'].apply(lambda x:torch.tensor(x,dtype=torch.float32).softmax(dim=-1))
    pre_score3 = ((distribution3.apply(lambda x:(x*torch.tensor([1,2,3,4,5,6,7,8,9],dtype=torch.float32)).sum()))*weights).sum().item()
    direct_score_ls.append(res.iloc[-1]['direct_score'])
    weighted_score_ls.append(res.iloc[-1]['weighted_score'])
    weighted_direct_score_ls.append(res['direct_score'].mean().item())
    internalscore_ls.append(res['weighted_score'].mean().item())
    pre_score1_ls.append(pre_score1)
    pre_score2_ls.append(pre_score2)
    pre_score3_ls.append(pre_score3)


calc_p = lambda x,y: pearsonr(x,y)[0]
calc_s = lambda x,y: spearmanr(x,y)[0]
print(len(direct_score_ls),len(res_score))
print(model_name)
print('direct_score:',round(calc_p(direct_score_ls,res_score),3),round(calc_s(direct_score_ls,res_score),3))
print('weighted_score:',round(calc_p(weighted_score_ls,res_score),3),round(calc_s(weighted_score_ls,res_score),3))
print('weighted_direct_score:',round(calc_p(weighted_direct_score_ls,res_score),3),round(calc_s(weighted_direct_score_ls,res_score),3))
print('internalscore:',round(calc_p(internalscore_ls,res_score),3),round(calc_s(internalscore_ls,res_score),3))
print('pred_score1:',round(calc_p(pre_score1_ls,res_score),3),round(calc_s(pre_score1_ls,res_score),3))
print('pred_score2:',round(calc_p(pre_score2_ls,res_score),3),round(calc_s(pre_score2_ls,res_score),3))
print('pred_score3:',round(calc_p(pre_score3_ls,res_score),3),round(calc_s(pre_score3_ls,res_score),3))