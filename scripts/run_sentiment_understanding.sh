#CUDA_VISIBLE_DEVICES=3 python src/sentiment_understanding.py --model_path Shanghai_AI_Laboratory/internlm3-8b-instruct --valid_data_path /nfsdata/laip/results/valid/internlm3-8b-instruct_logits.json


CUDA_VISIBLE_DEVICES=3 python src/sentiment_understanding.py \
        --model_path LLM-Research/Meta-Llama-3.1-8B-Instruct \
        --valid_data_path results/valid/Meta-Llama-3___1-8B-Instruct_logits.json 