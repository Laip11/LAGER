from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import warnings
import numpy as np
from utils import optimize_layer_weights

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


def get_layer_outputs(model, tokenized_inputs,tokenizer, weights,temperature=0, max_score=5):
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
    parser.add_argument('--valid_data_path', type=str,required=True)
    args = parser.parse_args()

    if ( ('llama' in (args.model_path).lower() and 'llama' in (args.valid_data_path).lower()) or
        ('internlm' in (args.model_path).lower() and 'internlm' in (args.valid_data_path).lower()) or
        ('mistral' in (args.model_path).lower() and 'mistral' in (args.valid_data_path).lower())):
        pass
    else:
        raise Exception('The model must be consistent with the valid_data_path.')


    weights = optimize_layer_weights(data_path=args.valid_data_path,
                                     loss_fn=torch.nn.CrossEntropyLoss(),
                                     num_epochs=2,
                                     lr=0.01)

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
        direct_score, weighted_score, weighted_weighted_score, palm_score_wt, palm_score_wot, res = get_layer_outputs(model, tokenized_inputs, tokenizer, weights,temperature=0)

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