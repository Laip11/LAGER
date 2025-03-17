from vllm import LLM, SamplingParams
import argparse
import logging
import os
import sys
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.generative import process_judgement
from rewardbench.utils import calculate_scores_per_section
import math
import pandas as pd
import warnings
import torch


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--data_path", type=str, default="", help="")
    parser.add_argument("--debug", action="store_true", help="")
    parser.add_argument("--scoring", action="store_true", default=False, help="")
    parser.add_argument("--feedback", action="store_true", default=False, help="")
    args = parser.parse_args()
    return args

def find_sublist_position(main_list, sublist):
    n = len(sublist)
    for i in range(len(main_list) - n + 1):
        if main_list[i:i + n] == sublist:
            return i
    return -1  

def get_batch_inputs(prompts, messages, tokenizer, batch_size = 8, padding = 'max_length',device = 'cuda', max_length=4096):
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


def get_layer_outputs(model, tokenized_inputs, tokenizer, weights, temperature=0, max_new_tokens=16, max_score=5):

    if max_score==5: 
        default_score = 3
    elif max_score==9:
        default_score = 5
        
    try:
        points_ids_list = [tokenizer.convert_tokens_to_ids([str(i)])[0] for i in range(1, max_score+1)]
    except:
        points_ids_list = [tokenizer.convert_tokens_to_ids([str(i).encode()])[0] for i in range(1, max_score+1)]

    all_res = []
    if temperature != 0:
        do_sample = True
    else:
        do_sample = False

    if len(tokenized_inputs)==0:
        return default_score, default_score, default_score, default_score, default_score, default_score, None
    
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
                        max_new_tokens=max_new_tokens
                        )
        responses_ids = outputs.sequences[:,block_inputs['batch_inputs']['input_ids'].shape[-1]:]
        responses = tokenizer.batch_decode(responses_ids,skip_special_tokens=True)
        
        columns = ['layer_n','direct_score','weighted_score', 'logits', 'probs','entropy','ratio']

        #############################################
        output_ids = responses_ids[0].tolist()
        score_ids = [
            tokenizer.encode("score:")[-2:],
            tokenizer.encode(" score:")[-2:],
            tokenizer.encode("Score:")[-2:],
            tokenizer.encode(" Score:")[-2:],
            tokenizer.encode("\nscore:")[-2:],
            tokenizer.encode("\nscore:")[-2:],
            tokenizer.encode("\nScore:")[-2:],
            tokenizer.encode(" \nScore:")[-2:],
        ]

        for score_id in score_ids:
            score_pos = find_sublist_position(output_ids, score_id)
            if score_pos != -1:
                break
        if score_pos == -1:
            return default_score, default_score, default_score, default_score, default_score, default_score, None

        pos = []
        for pos_, id_ in enumerate(output_ids[score_pos:]): 
            if tokenizer.decode(id_) in [str(i) for i in range(1, max_score+1)]:
                pos.append(pos_)
        if len(pos) >= 1:
            j = score_pos+pos[0]
        else:
            print(responses[0])
            return default_score, default_score, default_score, default_score, default_score, default_score, None
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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


def main(template, max_new_tokens, max_score):
    args = get_args()
    feedback_tail = "feedback" * args.feedback
    ###############
    # set moel_name
    ###############
    if "Mistral" in args.model[0]:
        model_name = "Mistral"
        weights = [0.02530812, 0.02529554, 0.02525384, 0.02544024, 0.0254709 ,
       0.02538827, 0.02517533, 0.02481054, 0.02489335, 0.02485125,
       0.02570229, 0.02602397, 0.02684528, 0.02672314, 0.02797128,
       0.0279035 , 0.02839368, 0.02760266, 0.02747969, 0.02890367,
       0.03517389, 0.03830925, 0.04022087, 0.03432938, 0.03526052,
       0.03162143, 0.03453825, 0.05912246, 0.04450823, 0.0135075 ,
       0.0110993 , 0.01106449, 0.08580787]
    elif "intern" in args.model[0]:
        model_name = "intern"
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
    elif "Llama" in args.model[0]:
        model_name = "Llama"
        weights = [0.003437180072069168, 0.0034694664645940065, 0.003370967460796237, 0.0033762697130441666,
        0.0032019256614148617, 0.003022323828190565, 0.0027004529256373644, 0.0031398916617035866,
        0.00311932316981256, 0.0032164095900952816, 0.0035408996045589447, 0.003587346524000168,
        0.003944995813071728, 0.004196827299892902, 0.0036134175024926662, 0.003482070518657565,
        0.0035765226930379868, 0.002972847782075405, 0.002817249856889248, 0.0024364066775888205,
        0.0024376907385885715,0.0024486437905579805,0.0036508042830973864, 0.8593869209289551,
        0.0031657542567700148, 0.003469533985480666, 0.0035897952038794756, 0.009388254024088383,
        0.005958248395472765, 0.005170912481844425, 0.005272462032735348, 0.0053072962909936905,
        0.024530891329050064]
    else:
        model_name = ""
    print(f"now is {model_name}")
    

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ############################
    # Load dataset
    ############################
    dataset = pd.read_json(args.data_path, lines=True)

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.sample(10)

    ############################
    # Run model weights with vllm
    ############################

    def format_judgements(s, chosen=True):
        if chosen:
            input_chosen = template.format(
                question=s['prompt'],
                answer=s['chosen'],
            )
            ####
            messages = [
                {
                    "role": "user", "content": input_chosen,
                },
            ]
            input_chosen = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # print(input_chosen, "\n--------")
            ####
            return input_chosen
        
        elif chosen==False:
            input_rejected = template.format(
                question=s['prompt'],
                answer=s['rejected'],
            )
            ####
            messages = [
                {
                    "role": "user", "content": input_rejected,
                },
            ]
            input_rejected = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # print(input_rejected, "\n--------")
            ####
            return input_rejected

    # fill prompt template
    dataset["input_chosen"] = dataset.apply(lambda s: format_judgements(s, chosen=True), axis=1)
    dataset["input_rejected"] = dataset.apply(lambda s: format_judgements(s, chosen=False), axis=1)
    dataset_prompts = dataset.copy()

    # compare
    def get_final_score(full_prompt):
        prompts = [full_prompt]
        messages = [
            {
            "role": "user",
            "content": prompts[0]
            } 
        ]
        tokenized_inputs = get_batch_inputs(prompts, messages, tokenizer, batch_size=1)[0]
        # generate score
        direct_score, weighted_score, weighted_weighted_score, pre_score1, pre_score2, res = get_layer_outputs(model, tokenized_inputs, tokenizer, weights, temperature=0, max_new_tokens=max_new_tokens, max_score=max_score)
        
        print(direct_score, weighted_score, weighted_weighted_score, pre_score1, pre_score2)

        if type(res) == type(None):
            res = None
        else:
            res = res.to_dict()
        all_res = {
            "model": args.model, 
            "direct": direct_score,
            "weighted": weighted_score,
            "internal": weighted_weighted_score,
            "palm_score_wt": pre_score1,
            "palm_score_wot": pre_score2,
            "layers": str(res)
        }
        return all_res
    
    # 对chosen和rejected评分
    tqdm.pandas()
    dataset_prompts['all_res_chosen'] = dataset_prompts.progress_apply(lambda s: get_final_score(s['input_chosen']), axis=1)
    dataset_prompts['all_res_rejected'] = dataset_prompts.progress_apply(lambda s: get_final_score(s['input_rejected']), axis=1)
    dataset_prompts.to_json(f"data/pair_wise_res_{model_name}.json", orient="records", lines=True)

    direct_chosen = dataset_prompts["all_res_chosen"].apply(lambda x: x['direct']).tolist()
    direct_rejected = dataset_prompts["all_res_rejected"].apply(lambda x: x['direct']).tolist()
    weighted_chosen = dataset_prompts["all_res_chosen"].apply(lambda x: x['weighted']).tolist()
    weighted_rejected = dataset_prompts["all_res_rejected"].apply(lambda x: x['weighted']).tolist()
    internal_chosen = dataset_prompts["all_res_chosen"].apply(lambda x: x['internal']).tolist()
    internal_rejected = dataset_prompts["all_res_rejected"].apply(lambda x: x['internal']).tolist()
    palm_score_wt_chosen = dataset_prompts["all_res_chosen"].apply(lambda x: x['palm_score_wt']).tolist()
    palm_score_wt_rejected = dataset_prompts["all_res_rejected"].apply(lambda x: x['palm_score_wt']).tolist()
    palm_score_wot_chosen = dataset_prompts["all_res_chosen"].apply(lambda x: x['palm_score_wot']).tolist()
    palm_score_wot_rejected = dataset_prompts["all_res_rejected"].apply(lambda x: x['palm_score_wot']).tolist()

    def compare(s1, s2):
        s1 = float(s1)
        s2 = float(s2)
        if s1 > s2:
            return 1
        elif s1 < s2:
            return 0
        elif s1==s2:
            return 0.5  
    results_direct = [compare(s1, s2) for s1,s2 in zip(direct_chosen, direct_rejected)]
    results_weighted = [compare(s1, s2) for s1,s2 in zip(weighted_chosen, weighted_rejected)]
    results_internal = [compare(s1, s2) for s1,s2 in zip(internal_chosen, internal_rejected)]
    results_pre1 = [compare(s1, s2) for s1,s2 in zip(palm_score_wt_chosen, palm_score_wt_rejected)]
    results_pre2 = [compare(s1, s2) for s1,s2 in zip(palm_score_wot_chosen, palm_score_wot_rejected)]
    all_results = {"direct": results_direct, "weighted": results_weighted, "internal": results_internal, 
                    "palm_score_wt": results_pre1, "palm_score_wot": results_pre2,}

    ############################
    # Print & process results
    ############################
    if "helpsteer" or "uf" in args.data_path:
        for score_type, results in all_results.items():
            num_correct = sum(results)
            num_total = len(results)
            acc = num_correct/num_total
            with open('data/pair_wise_info.txt', 'a') as file:
                file.write(f'{args.data_path}\n{args.model}\n{score_type} - {feedback_tail}\nacc: {acc}\n\n-------\n\n')
    elif "reward" in args.data_path:
        for score_type, results in all_results.items():
            out_dataset = dataset.copy()
            out_dataset['results'] = results

            # get core dataset
            results_grouped = {}
            results_grouped["model"] = args.model
            results_grouped["chat_template"] = None

            # print per subset and log into results_grouped file
            present_subsets = out_dataset['subset'].unique()
            for subset in present_subsets:
                subset_dataset = out_dataset[out_dataset["subset"] == subset]
                num_correct = subset_dataset["results"].sum()
                num_total = subset_dataset["results"].shape[0]
                print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
                results_grouped[subset] = num_correct / num_total

            # calculate score
            results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
            final_res = (results_leaderboard['Chat'] + results_leaderboard['Chat Hard'] + results_leaderboard['Safety'] + results_leaderboard['Reasoning'])/4
            
            
            print(args.model)
            with open('data/pair_wise_info.txt', 'a') as file:
                file.write(f'{args.data_path}\n{args.model}\n{score_type} - {feedback_tail}\n{results_leaderboard}\n{final_res}\n\n-------\n\n')
            print(results_leaderboard, "\n", final_res)



if __name__ == "__main__":

    template_prometheus = '''###Task Description:
An instruction (might include an Input inside it) and a response to evaluate are given.
1. Write a detailed feedback that assess the quality of the response.
2. After writing a feedback, write a score that is an integer between 1 and 5.
3. The output format should look as follows: \"Feedback: (write a feedback to quality of the response) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{question}

###Response to evaluate:
{answer}

###Feedback:'''

    template_single_feedback = '''You are an evaluator tasked with assigning a score from 1 to 5 to assess the quality of a response. If you think the response is good, you can give a high score. If you think the response is bad, you can give a low score.

### The answer that needs to be evaluated and its corresponding question:
<Question>
{question}
</Question>

<Response to be evaluated>
{answer}
</Response to be evaluated>

### Output format:
The output format should look as follows: "Feedback: <write your feedback here> Score: <give your score here>"
Remember that you should end with the the score. Please do not generate any other opening, closing, and explainations.

### Feedback:'''

    template_single_no_feedback = '''You are an evaluator tasked with assigning a score from 1 to 5 to assess the quality of a response. If you think the response is good, you can give a high score. If you think the response is bad, you can give a low score.

### The answer that needs to be evaluated and its corresponding question:
<Question>
{question}
</Question>

<Response to be evaluated>
{answer}
</Response to be evaluated>

### Output Format:
Only provide a **single score** from 1 to 5 in format of "score: <the score you give>". Do not generate other content.'''

    args = get_args()
    if args.feedback == True:
        template = template_single_feedback
        max_new_tokens = 1024
    else: 
        template = template_single_no_feedback
        max_new_tokens = 16
    
    main(template, max_new_tokens, max_score=5)
