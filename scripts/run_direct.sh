# CUDA_VISIBLE_DEVICES=6,7 python3 palmscore/get_pointwise_outputs.py \
#      --model_name_or_path /nfsdata/laip/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#      --save_dir results \
#      --points 5 \
#      --batch_size 2 \
#      --max_new_tokens 512 \
#      --dtype bfloat16 \
#      --with_feedback \
#      --input_file data/main/flask.json

# CUDA_VISIBLE_DEVICES=6,7 python3 palmscore/get_pointwise_outputs.py \
#      --model_name_or_path /nfsdata/laip/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#      --save_dir results \
#      --points 5 \
#      --batch_size 2 \
#      --max_new_tokens 512 \
#      --with_feedback \
#      --dtype bfloat16 \
#      --input_file data/main/helpsteer.json

# CUDA_VISIBLE_DEVICES=4,5 python3 palmscore/get_pointwise_outputs.py \
#      --model_name_or_path /nfsdata/laip/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#      --save_dir results \
#      --points 5 \
#      --batch_size 2 \
#      --max_new_tokens 512 \
#      --with_feedback \
#      --dtype bfloat16 \
#      --input_file data/valid/hs_valid.jsonl


CUDA_VISIBLE_DEVICES=0 python3 palmscore/get_pointwise_outputs.py \
     --model_name_or_path /nfsdata/laip/model/Qwen/Qwen2___5-1___5B-Instruct \
     --save_dir results \
     --points 5 \
     --batch_size 1 \
     --max_new_tokens 10 \
     --dtype bfloat16 \
     --input_file /home/laip/PalmScore/data/main/BiGGen-Bench-human-eval.json

CUDA_VISIBLE_DEVICES=0 python3 palmscore/get_pointwise_outputs.py \
     --model_name_or_path /nfsdata/laip/model/Qwen/Qwen2___5-1___5B-Instruct \
     --save_dir results \
     --points 5 \
     --batch_size 1 \
     --max_new_tokens 256 \
     --with_feedback  \
     --dtype bfloat16 \
     --input_file /home/laip/PalmScore/data/main/BiGGen-Bench-human-eval.json

CUDA_VISIBLE_DEVICES=0 python3 palmscore/get_pointwise_outputs.py \
     --model_name_or_path /nfsdata/laip/model/Qwen/Qwen2___5-0___5B-Instruct \
     --save_dir results \
     --points 5 \
     --batch_size 1 \
     --max_new_tokens 10 \
     --dtype bfloat16 \
     --input_file /home/laip/PalmScore/data/main/BiGGen-Bench-human-eval.json

CUDA_VISIBLE_DEVICES=0 python3 palmscore/get_pointwise_outputs.py \
     --model_name_or_path /nfsdata/laip/model/Qwen/Qwen2___5-0___5B-Instruct \
     --save_dir results \
     --points 5 \
     --batch_size 1 \
     --max_new_tokens 256 \
     --with_feedback  \
     --dtype bfloat16 \
     --input_file /home/laip/PalmScore/data/main/BiGGen-Bench-human-eval.json

# CUDA_VISIBLE_DEVICES=0 python3 palmscore/get_pointwise_outputs.py \
#      --model_name_or_path /nfsdata/laip/model/Qwen/Qwen2___5-0___5B-Instruct \
#      --save_dir results \
#      --points 5 \
#      --batch_size 8 \
#      --max_new_tokens 10 \
#      --dtype bfloat16 \
#      --input_file /home/laip/PalmScore/data/main/flask.json

CUDA_VISIBLE_DEVICES=0 python3 palmscore/get_pointwise_outputs.py \
     --model_name_or_path /nfsdata/laip/model/Qwen/Qwen2___5-0___5B-Instruct \
     --save_dir results \
     --points 5 \
     --batch_size 8 \
     --max_new_tokens 256 \
     --with_feedback  \
     --dtype bfloat16 \
     --input_file /home/laip/PalmScore/data/main/flask.json

CUDA_VISIBLE_DEVICES=0 python3 palmscore/get_pointwise_outputs.py \
     --model_name_or_path /nfsdata/laip/model/Qwen/Qwen2___5-1___5B-Instruct \
     --save_dir results \
     --points 5 \
     --batch_size 8 \
     --max_new_tokens 256 \
     --with_feedback  \
     --dtype bfloat16 \
     --input_file /home/laip/PalmScore/data/main/flask.json

CUDA_VISIBLE_DEVICES=0 python3 palmscore/get_pointwise_outputs.py \
     --model_name_or_path /nfsdata/laip/model/Qwen/Qwen2___5-1___5B-Instruct \
     --save_dir results \
     --points 5 \
     --batch_size 8 \
     --max_new_tokens 256 \
     --with_feedback  \
     --dtype bfloat16 \
     --input_file /home/laip/PalmScore/data/main/helpsteer.json

CUDA_VISIBLE_DEVICES=0 python3 palmscore/get_pointwise_outputs.py \
     --model_name_or_path /nfsdata/laip/model/Qwen/Qwen2___5-7B-Instruct \
     --save_dir results \
     --points 5 \
     --batch_size 8 \
     --max_new_tokens 10 \
     --dtype bfloat16 \
     --input_file /home/laip/PalmScore/data/main/helpsteer.json