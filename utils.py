import json
import pandas as pd
from tqdm import tqdm
import torch
import logging
from logging.handlers import RotatingFileHandler
import datetime
import colorlog
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = 'bfloat16',
        device_map = 'auto',
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
        ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
        )
    return model,tokenizer
def validate_model_and_data_consistency(model_path, valid_data_path):

    supported_models = [
        "mistral-7b-instruct-v0.3",
        "internlm3-8b-instruct",
        "llama3.1-8b-instruct",
        "qwen-2.5-14b-instruct",
        "mistral-small-24b-instruct",
        "llama-3.3-70b-instruct"
    ]
    
    model_path_lower = model_path.lower()
    valid_data_path_lower = valid_data_path.lower()
    
    is_consistent = any(
        (model_keyword in model_path_lower and model_keyword in valid_data_path_lower)
        for model_keyword in supported_models
    )
    if not is_consistent:
        raise ValueError("The model corresponding to model_path and valid_data_path should be the same.")
    
def validate_data_consistency(data_path, valid_data_path):
    model_name = None
    supported_models = [
        "mistral-7b-instruct-v0.3",
        "internlm3-8b-instruct",
        "llama3.1-8b-instruct",
        "qwen-2.5-14b-instruct",
        "mistral-small-24b-instruct",
        "llama-3.3-70b-instruct"
    ]
    
    data_path_lower = data_path.lower()
    valid_data_path_lower = valid_data_path.lower()
    
    for model_keyword in supported_models:
        if model_keyword in data_path_lower and model_keyword in valid_data_path_lower:
            model_name = model_keyword
            return model_name

    if not model_name:
        raise ValueError("The model corresponding to data_path and valid_data_path should be the same.")


def load_data(data_path,with_feedback=bool):
    score_ls,all_prompts = [],[]
    data_name = get_data_name(data_path)
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            score_ls.append(sum(data['score'])/len(data['score']) if type(data['score'])==list else data['score'])

            if with_feedback:
                all_prompts.append(data['user_prompt_feedback'])
            else:
                all_prompts.append(data['user_prompt'])

            if data_name == 'helpsteer':
                if len(all_prompts)>2000:
                    break
    return score_ls,all_prompts

def get_data_name(data_path):
    if 'flask' in (data_path).lower():
        data_name = 'flask'
    elif 'valid' in (data_path).lower():
        data_name = 'valid'
    elif 'helpsteer' in (data_path).lower():
        data_name = 'helpsteer'
    elif 'biggen' in (data_path).lower():
        data_name = 'biggen'
    else:
        data_name = data_path.split('/')[-1].split('.')[0]
    return data_name


def get_batch_inputs(prompts, tokenizer, batch_size = 8, padding = 'max_length',device = 'cuda',max_length = 2048,with_cot:int=0):

    tokenized_inputs = []
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if with_cot:
        messages = [{"role": "user", "content": prompt+'Please think through it step by step' } for prompt in prompts]
    else:
        messages = [{"role": "user", "content": prompt } for prompt in prompts]

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
    return (tokenized_inputs,save_idx_ls)


def get_layer_outputs(model, tokenized_inputs,tokenizer,max_new_tokens,points_ids_list,temperature = 0):

    all_res = []
    if temperature != 0:
        do_sample = True
    else:
        do_sample = False
    for block_inputs in tqdm(tokenized_inputs):

        prompts = block_inputs['batch_prompts']
        outputs = model.generate(
                    **block_inputs['batch_inputs'],
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    do_sample = do_sample,
                    temperature = temperature,
                    top_k = 5,
                    top_p = 0.9,
                    max_new_tokens = max_new_tokens,
                    )
        responses_ids = outputs.sequences[:,block_inputs['batch_inputs']['input_ids'].shape[-1]:]
        columns = ['layer_n','direct_score','weighted_score','probs']
        #responses = tokenizer.batch_decode(responses_ids,skip_special_tokens=True)
        #print(responses)
        
        for i in range(responses_ids.shape[0]):
            score_idxs  = None
            for j in range(responses_ids.shape[1]-1,-1,-1):
                if responses_ids[i,j] in points_ids_list:
                    score_idxs = j
                    break
            if score_idxs == None:
                res = pd.DataFrame([[-1]*len(columns)],columns=columns)
                all_res.append({'prompt':prompts[i],'res':res})
                continue
            
            score_hidden_state = outputs.hidden_states[score_idxs]
            res = pd.DataFrame(columns=columns)

            logits_list = []
            for layer_n,layer_hidden_state in enumerate(score_hidden_state):
                
                last_token_hidden_state = layer_hidden_state[i,-1,:]
                
                lm_head = model.lm_head

                logits = lm_head(last_token_hidden_state)
                logits_list.append(logits[points_ids_list].to(torch.float32).to('cpu').tolist())

                probs = logits[points_ids_list].softmax(dim=-1)
                probs_dict = { p+1: probs[p].item() for p in range(len(points_ids_list)) }

                direct_score  = probs.argmax(dim=-1).item()+1
                weight_score = sum([key*value for key,value in zip(probs_dict.keys(),probs_dict.values())])

                probs = probs.tolist()
                layer_res = pd.DataFrame([[layer_n,direct_score,weight_score,probs]],columns=columns)
                res = pd.concat([res,layer_res],axis=0,ignore_index=True)
    
            res['logits'] = logits_list
            all_res.append({'prompt':prompts[i],'res':res})
    return all_res 


log_colors_config = {
    'DEBUG': 'white',
    'INFO': 'white',
    'WARNING': 'blue',
    'ERROR': 'yellow',
    'CRITICAL': 'red'}
def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


# 配置日志
def setup_logger(name, log_file='output.log', level=logging.DEBUG):
    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logging.Formatter.converter = beijing

    # Check if handlers are already added, to avoid duplicate handlers
    if not logger.hasHandlers():
        # Create a rotating file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=1, encoding='utf-8'  # 5MB file size, 1 backup
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s[%(levelname)s]: %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # Create console handler with color
        console_fmt = '%(log_color)s%(asctime)s-%(threadName)s-%(filename)s-[line:%(lineno)d]-%(levelname)s: %(message)s'
        color_config = {
            'DEBUG': 'white',
            'INFO': 'white',
            'WARNING': 'blue',
            'ERROR': 'yellow',
            'CRITICAL': 'red',
        }
        console_formatter = colorlog.ColoredFormatter(fmt=console_fmt, log_colors=color_config)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info("Logger has been successfully configured.")
    return logger


def get_score(
        model, 
        tokenized_inputs, 
        tokenizer, 
        weights, 
        temperature=0, 
        max_new_tokens=16, 
        max_score=5,
        score_token='score'
):
    # set default score
    if max_score==5: 
        default_score = 3
    elif max_score==9:
        default_score = 5
    default_return = (default_score, default_score, default_score, default_score, None)
    
    # convert tokens to ids
    try:
        points_ids_list = [tokenizer.convert_tokens_to_ids([str(i)])[0] for i in range(1, max_score+1)]
    except:
        points_ids_list = [tokenizer.convert_tokens_to_ids([str(i).encode()])[0] for i in range(1, max_score+1)]
    
    all_res = []
    columns = ['layer_n', 'direct_score','weighted_score', 'logits', 'probs']
    if len(tokenized_inputs)==0:
        return default_return
    
    do_sample = True if temperature != 0 else False
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
        responses = tokenizer.batch_decode(responses_ids, skip_special_tokens=True)
        print(responses[0])
        ################ find the position of score token #################
        output_ids = responses_ids[0].tolist()
        if score_token == 'score':
            score_ids = [
                tokenizer.encode("score:")[-2:], tokenizer.encode(" score:")[-2:],
                tokenizer.encode("\nscore:")[-2:], tokenizer.encode("\n\nscore:")[-2:],
                tokenizer.encode("score: ")[-2:], tokenizer.encode(" score: ")[-2:],
                tokenizer.encode("\nscore: ")[-2:], tokenizer.encode("\n\nscore: ")[-2:],
                
                tokenizer.encode("Score:")[-2:], tokenizer.encode(" Score:")[-2:],
                tokenizer.encode("\nScore:")[-2:], tokenizer.encode("\n\nScore:")[-2:],
                tokenizer.encode("Score: ")[-2:], tokenizer.encode(" Score: ")[-2:],
                tokenizer.encode("\nScore: ")[-2:], tokenizer.encode("\n\nScore: ")[-2:],
            ]
        elif score_token == 'level':
            score_ids = [
                tokenizer.encode("level:")[-2:], tokenizer.encode(" level:")[-2:],
                tokenizer.encode("\nlevel:")[-2:], tokenizer.encode("\n\nlevel:")[-2:],
                tokenizer.encode("level: ")[-2:], tokenizer.encode(" level: ")[-2:],
                tokenizer.encode("\nlevel: ")[-2:], tokenizer.encode("\n\nlevel: ")[-2:],
                
                tokenizer.encode("Level:")[-2:], tokenizer.encode(" Level:")[-2:],
                tokenizer.encode("\nLevel:")[-2:], tokenizer.encode("\n\nLevel:")[-2:],
                tokenizer.encode("Level: ")[-2:], tokenizer.encode(" Level: ")[-2:],
                tokenizer.encode("\nLevel: ")[-2:], tokenizer.encode("\n\nLevel: ")[-2:],
            ]


        for score_id in score_ids:
            score_pos = find_sublist_position(output_ids, score_id)
            if score_pos != -1:
                break
        if score_pos == -1:
            return default_return

        pos = []
        for pos_, id_ in enumerate(output_ids[score_pos:]): 
            if tokenizer.decode(id_) in [str(i) for i in range(1, max_score+1)]:
                pos.append(pos_)
        if len(pos) >= 1:
            j = score_pos + pos[0]
        else:
            print(responses[0])
            return default_return
        ####################################

        score_hidden_state = outputs.hidden_states[j]
        res = pd.DataFrame(columns=columns)
        for layer_idx, layer_hidden_state in enumerate(score_hidden_state):
            last_token_hidden_state = layer_hidden_state[0,-1,:]
            lm_head = model.lm_head

            logits = lm_head(last_token_hidden_state)

            probs = logits[points_ids_list].softmax(dim=-1)

            direct_score  = probs.argmax(dim=-1).item()+1
            probs_dict = { p+1: probs[p].item() for p in range(len(points_ids_list)) }
            weight_score = sum(
                [
                    key*value/sum(probs_dict.values()) 
                    for key,value in zip(probs_dict.keys(),probs_dict.values())
                ]
            )

            probs = probs.tolist()
            
            layer_res = pd.DataFrame(
                [[
                    layer_idx, 
                    direct_score, weight_score, 
                    logits[points_ids_list].to(torch.float32).to('cpu').tolist(), 
                    probs,
                ]], 
                columns=columns
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = pd.concat([res, layer_res], axis=0, ignore_index=True)

        all_res.append({'prompt':prompts[0],'res':res})
        
        direct_score = str(all_res[0]['res']['direct_score'].tolist()[-1])
        weighted_score = str(all_res[0]['res']['weighted_score'].tolist()[-1])

        logits = res['logits'].apply(lambda x: torch.tensor(x,dtype=torch.float32))
        # palm score
        distribution1 = torch.softmax((logits*weights).sum(), dim=-1)
        palm_score_wt = (distribution1*torch.tensor([1,2,3,4,5], dtype=torch.float32)).sum().item()

        distribution2 = torch.softmax((logits/len(weights)).sum(), dim=-1)
        palm_score_wot = (distribution2*torch.tensor([1,2,3,4,5], dtype=torch.float32)).sum().item()

    return direct_score, weighted_score, palm_score_wt, palm_score_wot, res

def find_sublist_position(main_list, sublist):
    n = len(sublist)
    for i in range(len(main_list) - n + 1):
        if main_list[i:i + n] == sublist:
            return i
    return -1

def get_final_score(full_prompt, model, tokenizer, weights, max_new_tokens, max_score):
    prompts = [full_prompt]
    messages = [{
        "role": "user", "content": prompts[0]
    }]
    tokenized_inputs = get_batch_inputs(prompts, messages, tokenizer, batch_size=1)[0]
    # generate score
    direct_score, weighted_score, palm_score_wt, palm_score_wot, res = get_layer_outputs(
        model, tokenized_inputs, tokenizer, weights, temperature=0, max_new_tokens=max_new_tokens, max_score=max_score, score_token="level"
    )
    
    print(direct_score, weighted_score, palm_score_wt, palm_score_wot)

    if type(res) == type(None):
        res = None
    else:
        res = res.to_dict()
    all_res = {
        "direct": direct_score,
        "weighted": weighted_score,
        "palm_score_wt": palm_score_wt,
        "palm_score_wot": palm_score_wot,
        "layers": str(res)
    }
    return all_res