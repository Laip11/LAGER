# model_path
# "/data1/laip/model/LLM-Research/Mistral-7B-Instruct-v0___3"
# "/data1/laip/model/Qwen/Qwen2___5-7B-Instruct"
# "/data1/laip/model/LLM-Research/Meta-Llama-3___1-8B-Instruct"
# "/data1/laip/model/LLM-Research/gemma-2-9b-it"
# /data1/laip/model/LLM-Research/Meta-Llama-3___1-70B-Instruct
# /nfsdata/laip/model/models--Skywork--Skywork-Critic-Llama-3.1-8B/snapshots/825f34599593c0145be91644be233d5c634b2380
# /nfsdata/laip/model/shakechen/Llama-2-7b-chat-hf
# /nfsdata/laip/model/AI-ModelScope/Mistral-7B-Instruct-v0___2
cd /home/laip/InternalScore
SESSION_NAME1='helpsteer_direct'
SESSION_NAME2='flask_direct'

mode_name_list=(
                #/nfsdata/laip/model/Shanghai_AI_Laboratory/internlm3-8b-instruct\
                #/nfsdata/laip/model/LLM-Research/Mistral-7B-Instruct-v0___3\
                #/nfsdata/laip/model/LLM-Research/Meta-Llama-3___1-8B-Instruct\
                #/nfsdata/laip/model/LLM-Research/Llama-3___1-Tulu-3-8B
                #/nfsdata/laip/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
                # /nfsdata/laip/model/shakechen/Llama-2-7b-chat-hf\
                # /nfsdata/laip/model/AI-ModelScope/Mistral-7B-Instruct-v0___2
                /nfsdata/laip/model/Qwen/Qwen2___5-0___5B-Instruct \
                /nfsdata/laip/model/Qwen/Qwen2___5-3B-Instruct \
                /nfsdata/laip/model/Qwen/Qwen2___5-7B-Instruct \
                # /nfsdata/laip/model/Qwen/Qwen2___5-14B-Instruct  \
                # /nfsdata/laip/model/Qwen/Qwen2___5-32B-Instruct
                )

for model_name in "${mode_name_list[@]}"; do
  # Set batch size based on the model
  batch_size=4

  # command1="CUDA_VISIBLE_DEVICES=3 python3 /home/laip/InternalScore/test.py \
  #     --model_name_or_path ${model_name} \
  #     --save_dir results \
  #     --points 5 \
  #     --batch_size ${batch_size} \
  #     --max_new_tokens 10 \
  #     --with_feedback 0 \
  #     --dtype bfloat16 \
  #     --temperature 0\
  #     --with_cot 0\
  #     --max_length 2000\
  #     --input_file /home/laip/InternalScore/data/helpsteer.json"

  # # # # # # # Execute the command (you can optionally echo it to see it before running)
  # # # # # # echo "${new_command}"
  # # # # # # # Uncomment the next line to actually run the command
  # screen -dmS "$SESSION_NAME1" bash -c "$command1"
  # eval "$command1"

  command2="CUDA_VISIBLE_DEVICES=0 python3 /home/laip/InternalScore/test.py \
      --model_name_or_path ${model_name} \
      --save_dir results \
      --points 5 \
      --batch_size 4 \
      --max_new_tokens 10 \
      --with_feedback 0 \
      --dtype bfloat16 \
      --with_cot 0\
      --max_length 2000\
      --input_file /home/laip/InternalScore/data/hs_valid.jsonl"

  # # Execute the command (you can optionally echo it to see it before running)
  # #echo "${command2}"
  eval  "$command2"
  # Uncomment the next line to actually run the command
  # screen -dmS "$SESSION_NAME2" bash -c "$command2"
done