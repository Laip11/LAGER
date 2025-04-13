import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import logging
from PalmScore.optimize_layer_weights import optimize_layer_weights
from PalmScore.utils import validate_model_and_data_consistency,load_model_and_tokenizer,get_final_score
from sklearn.metrics import f1_score

# Configure logging
def configure_logging(output_path):
    # Create a logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler(output_path)

    # # Set the logging format
    # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

def get_pred_unanswerable(s, score_type, threshold):
    if s[score_type] > threshold:
        return False
    else:
        return True

def get_f1(df, score_type, threshold=0.75):
    gyh = (df[score_type].max() - df[score_type].min()) * threshold + df[score_type].min()
    df['prediction_unanswerable'] = df.apply(lambda s: get_pred_unanswerable(s, score_type, gyh), axis=1)

    y_true = df['unanswerable'].tolist()
    y_pred = df['prediction_unanswerable'].tolist()

    # f1
    f1 = f1_score(y_true, y_pred)
    return f1

def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="")
    parser.add_argument("--in_file", type=str, help="")
    parser.add_argument("--valid_data_path", type=str, help="")
    parser.add_argument("--output_path", type=str, help="")
    parser.add_argument("--debug", action="store_true", help="")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    validate_model_and_data_consistency(args.model, args.valid_data_path)
    model_name = args.model.split("/")[-1]
    weights = optimize_layer_weights(
        data_path = args.valid_data_path, 
        loss_fn = nn.CrossEntropyLoss(),
        num_epochs=1, 
        lr=0.01,
        batch_size = 8,
        seed = 42).numpy()
    
    # set output folder
    output_folder = f"data/{args.output_path}/{model_name}"
    os.makedirs(output_folder, exist_ok=True)
    # set logging
    log_file_path = os.path.join(output_folder, 'output_log.txt')
    logger = configure_logging(log_file_path)

    # load dataset
    if args.debug:
        df = pd.read_json(args.in_file, lines=True).sample(20)
    else:
        df = pd.read_json(args.in_file, lines=True)
    
    # load model
    model,tokenizer = load_model_and_tokenizer(args.model)

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        prompt = row['unknow_prompt']

        # generate scores
        all_res = get_final_score(
            prompt, model, tokenizer, weights, max_new_tokens=8, max_score=5
        )

        df.loc[index, f'direct'] = float(all_res['direct'])
        df.loc[index, f'weighted'] = float(all_res['weighted'])
        df.loc[index, f'palm_score_wt'] = float(all_res['palm_score_wt'])
        df.loc[index, f'palm_score_wot'] = float(all_res["palm_score_wot"])
        # check
        if index <= 2:
            print(prompt)
        # print(df.loc[index, f'direct'], "\n", df.loc[index, f'weighted'], "\n", df.loc[index, f'palm_score_wt'], df.loc[index, f'palm_score_wot'])
    
    # save results
    df.to_json(f"{output_folder}/results.jsonl", orient="records", lines=True)

    # process results
    df = pd.read_json(f"{output_folder}/results.jsonl", lines=True)
    df = df[df['direct'].apply(lambda x: x != "-1" and x != -1)]
    df['unanswerable'] = df['answerable'].apply(lambda x: x == False)

    direct_f1 = get_f1(df, 'direct', 0.75)
    weighted_f1 = get_f1(df, 'weighted', 0.75)
    palm_wt_f1 = get_f1(df, 'palm_score_wt', 0.75)
    palm_wot_f1 = get_f1(df, 'palm_score_wot', 0.75)
    logger.info(f"direct_f1: {direct_f1}")
    logger.info(f"weighted_f1: {weighted_f1}")
    logger.info(f"palm_wt_f1: {palm_wt_f1}")
    logger.info(f"palm_wot_f1: {palm_wot_f1}")
    # print results
    print(f"direct_f1: {direct_f1}, weighted_f1: {weighted_f1}, palm_wt_f1: {palm_wt_f1}, palm_wot_f1: {palm_wot_f1}")

if __name__ == "__main__":
    main()