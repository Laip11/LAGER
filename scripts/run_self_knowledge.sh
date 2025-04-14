CUDA_VISIBLE_DEVICES=0 python palmscore/self-knowledge.py \
    --model LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --valid_data_path results/valid/Meta-Llama-3___1-8B-Instruct_logits.json \
    --in_file data/unknow/prompt_unknow.json