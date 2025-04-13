import os
import json
import torch
import argparse
import warnings
from PalmScore.utils import get_batch_inputs,get_layer_outputs,get_data_name,load_data,load_model_and_tokenizer
warnings.filterwarnings("ignore")

def get_args():
    """
    Parse and retrieve command-line arguments.

    Returns:
        args (argparse.Namespace): Parsed arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path',required=True,type=str)
    parser.add_argument('--save_dir',type=str,default='results')
    parser.add_argument('--points',type=int,default=5,help='the max number of points to score.')
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--max_new_tokens',type=int,default=256,help='Maximum number of tokens to generate.')
    parser.add_argument('--input_file',type=str,required=True,help='the path of the input file.')
    parser.add_argument('--with_feedback',action='store_true',help='whether to use reasoning or not.')
    parser.add_argument('--dtype',type=str,default='bfloat16',)
    parser.add_argument('--max_length',type=int,default=2048,help='Maximum sequence length of the input sequence.')
    parser.add_argument('--temperature',type=float,default=0)
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    data_name,model_name = get_data_name(args.input_file),args.model_name_or_path.split('/')[-1]

    if not os.path.exists(f'{args.save_dir}/{data_name}'):
            os.makedirs(f'{args.save_dir}/{data_name}')

    out_dir = f"{args.save_dir}/{data_name}/{model_name}{'_with_feedback'*args.with_feedback}_logits.json"

    if os.path.exists(out_dir):
        print(f'{out_dir} already exists, skip')
        exit()
    else:
        print(f'{out_dir} does not exist, start to generate')

    model,tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    points_ids_list = [
        tokenizer.convert_tokens_to_ids([str(i)])[0] 
        for i in range(1,args.points+1)
        ]

    # load data
    score_ls,all_prompts = load_data(args.input_file,args.with_feedback)


    # biggen bench can not use batch inference
    if data_name == 'biggen':
        tokenized_inputs = []

        messages = [{"role": "user", "content": prompt } for prompt in all_prompts]

        all_text = [tokenizer.apply_chat_template([message],tokenize=False,add_generation_prompt = True) for message in messages]

        tokenized_inputs = []
        for i, text in enumerate(all_text):
            batch_prompts = [all_prompts[i]]
            batch_inputs = tokenizer(text,return_tensors="pt").to(model.device)
            tokenized_inputs.append({'batch_prompts': batch_prompts,'batch_inputs': batch_inputs,'text_batch': text})
    else:
        tokenized_inputs,save_idx_ls = get_batch_inputs(all_prompts,
                                                        tokenizer,
                                                        batch_size = args.batch_size,
                                                        device = f'cuda:{torch.cuda.device_count()-1}',
                                                        max_length=args.max_length)
    
    all_res = get_layer_outputs(model,
                                tokenized_inputs,
                                max_new_tokens = args.max_new_tokens,
                                tokenizer=tokenizer,
                                points_ids_list=points_ids_list,
                                temperature = args.temperature)


    processed_res = []
    for i in range(len(all_res)):
        res = all_res[i]
        idx = save_idx_ls[i]
        score = score_ls[idx]
        prompt = res['prompt']
        res1 = res['res']
        direct_socre = res1.iloc[-1]['direct_score']
        weighted_socre = res1.iloc[-1]['weighted_score']
        weighted_direct_socre = res1['direct_score'].mean().item()
        #internalscore = res1['weighted_score'].mean()

        dict1_res = {
                'idx':idx,
                'prompt':prompt,
                'human_score':score,
                'direct_socre':float(direct_socre),
                'weighted_socre':float(weighted_socre),
                'weighted_direct_socre':float(weighted_direct_socre),
                #'internalscore':float(internalscore),
                'df':res1.to_dict()}
        processed_res.append(dict1_res)

    with open(out_dir, 'w') as f:
        json.dump(processed_res, f, indent=4)

if __name__ == '__main__':
    main()

    

