import random
import json
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
import matplotlib.pyplot as plt


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
        responses = tokenizer.batch_decode(responses_ids,skip_special_tokens=True)
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


import logging
from logging.handlers import RotatingFileHandler
import datetime
import colorlog

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

class CustomDataset(Dataset):
    def __init__(self, data_path):
        all_data = json.load(open(data_path, 'r'))
        self.human_score_ls = []
        self.logits_ls = []
        self.weighted_score_ls = []

        for data in all_data:
            df = pd.DataFrame(data['df'])
            if data['weighted_socre'] == -1:  # Skip invalid data
                continue

            _logits = torch.tensor([i for i in df['logits']], dtype=torch.float32)
            human_score = torch.tensor(data['human_score'], dtype=torch.long)
            weighted_score = torch.tensor(df['weighted_score'].to_list(), dtype=torch.float32)

            self.human_score_ls.append(human_score)
            self.logits_ls.append(_logits)
            self.weighted_score_ls.append(weighted_score)

    def __len__(self):
        return len(self.logits_ls)

    def __getitem__(self, idx):
        return self.logits_ls[idx], self.human_score_ls[idx]


def optimize_layer_weights(data_path, loss_fn, num_epochs=2, lr=0.01, min_lr=1e-3, batch_size=8,seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Load data
    dataset = CustomDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    L = len(dataset.logits_ls[0])  # Number of layers in logits
    weights = torch.nn.Parameter(torch.cat([torch.zeros(L - 1), torch.tensor([1.0])]), requires_grad=True)
    #weights = torch.nn.Parameter(torch.cat([torch.zeros(L)]), requires_grad=True)

    optimizer = optim.Adam([weights], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=min_lr)

    all_step_losses = []  # Store loss for each step
    global_step = 0  # Global step counter

    for epoch in trange(num_epochs):
        total_loss = 0
        for batch_logits, batch_targets in dataloader:
            batch_logits = batch_logits.to(torch.float32)
            batch_targets = batch_targets.to(torch.long)

            # Adjust target for CrossEntropyLoss
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                batch_targets = batch_targets - 1

            normalized_weights = torch.softmax(weights, dim=0)

            # Compute weighted sum for each sample in the batch
            weighted_sum = torch.zeros_like(batch_logits[:, 0])
            for l in range(L):
                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    weighted_sum += normalized_weights[l] * (batch_logits[:, l])
                else:
                    weighted_sum += normalized_weights[l] * (batch_logits[:, l] * torch.tensor([1, 2, 3, 4, 5])).sum(dim=1)

            predictions = weighted_sum
            loss = loss_fn(predictions, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record the loss for this step
            all_step_losses.append((global_step, loss.item()))
            global_step += 1

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 绘制每个 step 的损失曲线
    steps, losses = zip(*all_step_losses)  # 解压 step 和 loss
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker='.', linestyle='-', color='b', alpha=0.7)
    plt.title("Training Loss Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss-step-figure.png")  # 保存图像
    plt.show()
    return torch.softmax(weights, dim=0).detach()

