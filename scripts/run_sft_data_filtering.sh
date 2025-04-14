CUDA_VISIBLE_DEVICES=0 python3 palmscore/sft_data_filtering.py \
      --data_path sft_prompt_7type.jsonl\
      --aspect answer_accuracy \
      --batch_size 16 \
      --model_name_or_path LLM-Research/Meta-Llama-3.1-8B-Instruct