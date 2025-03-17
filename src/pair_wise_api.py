from vllm import LLM, SamplingParams
import argparse
from tqdm import tqdm
import numpy as np
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
import pandas as pd
from get_response_openai import OpenAIClient

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

def calculate_expected_score(logprob_data, max_score):
    score_map = np.full(max_score, -np.inf)

    for top_lp in logprob_data:
        token = top_lp.token
        if token in [str(i) for i in range(1, max_score+1)]:
            score_map[int(token) - 1] = top_lp.logprob  
        else:
            continue
    if np.all(score_map == -np.inf):
        return 0, 0

    max_logit_index = np.argmax(score_map)  
    max_logit_score = max_logit_index + 1  

    exp_logprobs = np.exp(score_map)  
    probabilities = exp_logprobs / np.sum(exp_logprobs)

    expectation = np.dot(probabilities, np.arange(1, 6))  
    return expectation, max_logit_score

def main(template, max_new_tokens, max_score):
    args = get_args()
    feedback_tail = "feedback" * args.feedback
    model_name = args.model
    print(f"now is {model_name}")
    # load model
    client = OpenAIClient(
        base_url="",
        api_key="",
        model=model_name
    )

    ############################
    # Load dataset
    ############################
    dataset = pd.read_json(args.data_path, lines=True)
    if args.debug:
        dataset = dataset.sample(10)

    def format_judgements(s, scoring=False, chosen=True):
        if scoring and chosen:
            input_chosen = template.format(
                question=s['prompt'],
                answer=s['chosen'],
            )
            return input_chosen
        elif scoring and chosen==False:
            input_rejected = template.format(
                question=s['prompt'],
                answer=s['rejected'],
            )
            return input_rejected

    # fill prompt template
    dataset["input_chosen"] = dataset.apply(lambda s: format_judgements(s, scoring=args.scoring, chosen=True), axis=1)
    dataset["input_rejected"] = dataset.apply(lambda s: format_judgements(s, scoring=args.scoring, chosen=False), axis=1)
    dataset_prompts = dataset.copy()

    # compare
    def get_final_score(full_prompt):
        score_ls = [str(i) for i in range(1, max_score+1)]
        messages = [
            {
            "role": "user",
            "content": full_prompt
            } 
        ]
        try:
            completion = client.get_response_chat(messages, max_tokens=max_new_tokens)
        except:
            all_res = {
                "model": args.model, 
                "direct": 3,
                "weighted": 3,
            }
            return all_res
        for idx in range(len(completion.choices[0].logprobs.content)-1,-1,-1):
            if completion.choices[0].logprobs.content[idx].token in score_ls:
                score_idx = idx
                break
        expected_score, direct_score = calculate_expected_score(completion.choices[0].logprobs.content[score_idx].top_logprobs, max_score=5)
        print(direct_score, expected_score)

        all_res = {
            "model": args.model, 
            "direct": direct_score,
            "weighted": expected_score,
        }
        return all_res
    

    tqdm.pandas()
    dataset_prompts['all_res_chosen'] = dataset_prompts.progress_apply(lambda s: get_final_score(s['input_chosen']), axis=1)
    dataset_prompts['all_res_rejected'] = dataset_prompts.progress_apply(lambda s: get_final_score(s['input_rejected']), axis=1)
    dataset_prompts.to_json(f"data/pair_wise_res_{model_name}.json", orient="records", lines=True)

    direct_chosen = dataset_prompts["all_res_chosen"].apply(lambda x: x['direct']).tolist()
    direct_rejected = dataset_prompts["all_res_rejected"].apply(lambda x: x['direct']).tolist()
    weighted_chosen = dataset_prompts["all_res_chosen"].apply(lambda x: x['weighted']).tolist()
    weighted_rejected = dataset_prompts["all_res_rejected"].apply(lambda x: x['weighted']).tolist()

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
    all_results = {"direct": results_direct, "weighted": results_weighted }


    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    
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
