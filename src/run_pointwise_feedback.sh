CUDA_VISIBLE_DEVICES=0,1 python3 /get_pointwise_outputs.py \
      --model_name_or_path LLM-Research/Meta-Llama-3.1-8B-Instruct \
      --save_dir results \
      --points 5 \
      --batch_size 8 \
      --max_new_tokens 1024 \
      --with_feedback 1 \
      --dtype bfloat16 \
      --input_file data/flask.json
