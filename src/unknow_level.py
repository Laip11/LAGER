from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import math
import warnings
import numpy as np

def find_sublist_position(main_list, sublist):
    n = len(sublist)
    for i in range(len(main_list) - n + 1):
        if main_list[i:i + n] == sublist:
            return i
    return -1  

def get_batch_inputs(prompts, tokenizer, batch_size = 8, padding = 'max_length',device = 'cuda',max_length = 2048):
    tokenized_inputs = []
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
    return (tokenized_inputs,)


def get_layer_outputs(model, tokenized_inputs,tokenizer, temperature=0, max_score=5):
    if "Mistral" in args.model_path:
        weights = [0.02530812, 0.02529554, 0.02525384, 0.02544024, 0.0254709 ,
       0.02538827, 0.02517533, 0.02481054, 0.02489335, 0.02485125,
       0.02570229, 0.02602397, 0.02684528, 0.02672314, 0.02797128,
       0.0279035 , 0.02839368, 0.02760266, 0.02747969, 0.02890367,
       0.03517389, 0.03830925, 0.04022087, 0.03432938, 0.03526052,
       0.03162143, 0.03453825, 0.05912246, 0.04450823, 0.0135075 ,
       0.0110993 , 0.01106449, 0.08580787]
    elif "intern" in args.model_path:
        weights = [0.02532368, 0.02608873, 0.02565252, 0.02568329, 0.02251901,
       0.0214611 , 0.02118479, 0.01968816, 0.02265235, 0.0219109 ,
       0.02135445, 0.01698261, 0.02305569, 0.02254137, 0.02456228,
       0.01938125, 0.01938198, 0.03340582, 0.04947908, 0.03269262,
       0.02046547, 0.01132542, 0.00991966, 0.01287424, 0.00894568,
       0.00456296, 0.00401398, 0.00373798, 0.00364853, 0.00779461,
       0.0107277 , 0.01476849, 0.00996623, 0.0989219 , 0.13831984,
       0.00277205, 0.00235647, 0.00221785, 0.00258766, 0.00203182,
       0.00195165, 0.00179587, 0.00186209, 0.00171047, 0.00193544,
       0.00224081, 0.00464406, 0.00589424, 0.11100519]
    elif "Tulu" in args.model_path:
        weights = [1.9926482e-03, 2.0288283e-03, 1.9283649e-03, 1.9503274e-03,
       1.8866975e-03, 1.8496471e-03, 1.7655289e-03, 1.8533312e-03,
       1.7974996e-03, 1.7305122e-03, 1.9543676e-03, 2.1507202e-03,
       2.3168905e-03, 2.2129463e-03, 1.7726267e-03, 1.7696656e-03,
       1.4796040e-03, 1.1610090e-03, 1.4760328e-03, 7.6917995e-04,
       7.1918254e-04, 5.7078211e-04, 7.3513773e-04, 9.5099837e-01,
       1.0121432e-03, 1.2787035e-03, 1.0262263e-03, 1.5768759e-03,
       1.1171129e-03, 1.0530197e-03, 1.0068050e-03, 1.0300645e-03,
       2.0291803e-03]
    elif "Llama" in args.model_path:
        weights = [0.003437180072069168, 0.0034694664645940065, 0.003370967460796237, 0.0033762697130441666,
        0.0032019256614148617, 0.003022323828190565, 0.0027004529256373644, 0.0031398916617035866,
        0.00311932316981256, 0.0032164095900952816, 0.0035408996045589447, 0.003587346524000168,
        0.003944995813071728, 0.004196827299892902, 0.0036134175024926662, 0.003482070518657565,
        0.0035765226930379868, 0.002972847782075405, 0.002817249856889248, 0.0024364066775888205,
        0.0024376907385885715,0.0024486437905579805,0.0036508042830973864, 0.8593869209289551,
        0.0031657542567700148, 0.003469533985480666, 0.0035897952038794756, 0.009388254024088383,
        0.005958248395472765, 0.005170912481844425, 0.005272462032735348, 0.0053072962909936905,
        0.024530891329050064]

    try:
        points_ids_list = [tokenizer.convert_tokens_to_ids([str(i)])[0] for i in range(1, max_score+1)]
    except:
        points_ids_list = [tokenizer.convert_tokens_to_ids([str(i).encode()])[0] for i in range(1, max_score+1)]

    all_res = []
    if temperature != 0:
        do_sample = True
    else:
        do_sample = False
    for block_inputs in tokenized_inputs:

        prompts = block_inputs['batch_prompts']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = model.generate(
                        **block_inputs['batch_inputs'],
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        do_sample = do_sample,
                        temperature = temperature,
                        top_k = 5,
                        top_p = 0.9,
                        max_new_tokens = 16
                        )
        responses_ids = outputs.sequences[:,block_inputs['batch_inputs']['input_ids'].shape[-1]:]
        responses = tokenizer.batch_decode(responses_ids,skip_special_tokens=True)
        
        columns = ['layer_n','direct_score','weighted_score', 'logits', 'probs','entropy','ratio']

        #############################################
        output_ids = responses_ids[0].tolist()
        score_ids = [
            tokenizer.encode("Level:")[-2:],
            tokenizer.encode(" Level:")[-2:],
            tokenizer.encode("level:")[-2:],
            tokenizer.encode(" level:")[-2:],
        ]

        for score_id in score_ids:
            score_pos = find_sublist_position(output_ids, score_id)
            if score_pos != -1:
                break
        if score_pos == -1:
            return "-1", "-1", "-1", "-1", "-1", "-1", "-1",

        pos = []
        for pos_, id_ in enumerate(output_ids[score_pos:]): 
            if tokenizer.decode(id_) in [str(i) for i in range(1, max_score+1)]:
                pos.append(pos_)
        if len(pos) >= 1:
            j = score_pos+pos[0]
        else:
            return "-1", "-1", "-1", "-1", "-1", "-1", "-1",
        # print(tokenizer.decode(responses_ids[0][j]))
        ####################################

        score_hidden_state = outputs.hidden_states[j]
        res = pd.DataFrame(columns=columns)
        for layer_n,layer_hidden_state in enumerate(score_hidden_state):

            last_token_hidden_state = layer_hidden_state[0,-1,:]
            lm_head = model.lm_head

            logits = lm_head(last_token_hidden_state)
            probs = logits[points_ids_list].softmax(dim=-1)

            soft_max_logits = logits.softmax(dim=-1)
            max_logits = soft_max_logits.max(dim=-1).values.item()
            point_max_logits = soft_max_logits[points_ids_list].max(dim=-1).values.item()
            ratio = point_max_logits/max_logits

            direct_score  = probs.argmax(dim=-1).item()+1
            probs_dict = { p+1: probs[p].item() for p in range(len(points_ids_list)) }
            weight_score = sum([key*value/sum(probs_dict.values()) for key,value in zip(probs_dict.keys(),probs_dict.values())])

            probs = probs.tolist()
            layer_res = pd.DataFrame([[
                layer_n, 
                direct_score, weight_score, 
                logits[points_ids_list].to(torch.float32).to('cpu').tolist(), 
                probs, -np.sum(probs * np.log2(probs)), ratio
                ]], columns=columns)
            res = pd.concat([res,layer_res],axis=0,ignore_index=True)
        all_res.append({'prompt':prompts[0],'res':res})

        direct_score = str(all_res[0]['res']['direct_score'].tolist()[-1])
        weighted_score = str(all_res[0]['res']['weighted_score'].tolist()[-1])
        weighted_direct_score = str(all_res[0]['res']['direct_score'].mean())
        weighted_weighted_score = str(all_res[0]['res']['weighted_score'].mean())

        logits = res['logits'].apply(lambda x: torch.tensor(x,dtype=torch.float32))
        # palm score
        distribution1 = torch.softmax((logits*weights).sum(),dim=-1)
        palm_score_wt = (distribution1*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum().item()

        distribution2 = torch.softmax((logits/len(weights)).sum(),dim=-1)
        palm_score_wot = (distribution2*torch.tensor([1,2,3,4,5],dtype=torch.float32)).sum().item()

    return direct_score, weighted_score, weighted_weighted_score, palm_score_wt, palm_score_wot, res



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="")
    parser.add_argument("--in_file", type=str, help="")
    parser.add_argument("--debug", action="store_true", help="")
    args = parser.parse_args()

    # load dataset
    if args.debug:
        df = pd.read_json(args.in_file, lines=True).sample(50)
    else:
        df = pd.read_json(args.in_file, lines=True)
    
    model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if "Mistral" in model_path:
        model_name = "Mistral"
    elif "Tulu" in model_path:
        model_name = "Tulu"
    elif "Qwen" in model_path:
        model_name = "Qwen"
    elif "Llama" in model_path:
        model_name = "Llama"
    elif "intern" in model_path:
        model_name = "intern"
    else:
        print(f"{model_path}is wrong!!!")
    print("now is", model_name)

    # init
    df[f"direct"] = "-1"
    df[f"weighted"] = "-1"
    df[f"internal"] = "-1"
    df[f"palm_score_wt"] = "-1"
    df[f"palm_score_wot"] = "-1"
    df[f'res'] = "None"

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        prompts = [row['unknow_prompt']]

        messages = [
            {
            "role": "user",
            "content": prompts[0]
            } 
        ]

        tokenized_inputs = get_batch_inputs(prompts, tokenizer, batch_size=1)[0]

        # generate scores
        direct_score, weighted_score, weighted_weighted_score, palm_score_wt, palm_score_wot, res = get_layer_outputs(model, tokenized_inputs, tokenizer, temperature=0)

        if type(res) == type("-1"):
            res = "-1"
        else:
            res = res.to_dict()
        assert df.loc[index, 'question'] == row['question'], "index wrong!!"
        df.loc[index, f'direct'] = direct_score
        df.loc[index, f'weighted'] = weighted_score
        df.loc[index, f'internal'] = weighted_weighted_score
        df.loc[index, f'palm_score_wt'] = palm_score_wt
        df.loc[index, f'palm_score_wot'] = palm_score_wot
        df.loc[index, f'res'] = str(res)
        # check
        if index <= 2:
            print(prompts[0])
        print(df.loc[index, f'direct'], "\n", df.loc[index, f'weighted'], "\n", df.loc[index, f'internal'], df.loc[index, f'palm_score_wt'], df.loc[index, f'palm_score_wot'])


    # save
    save_path = f"data/unknow/prediction_{model_name}.json"
    df.to_json(save_path, orient="records", lines=True)