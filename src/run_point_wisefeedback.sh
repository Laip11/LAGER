# model_path
# "/data1/laip/model/LLM-Research/Mistral-7B-Instruct-v0___3"
# "/data1/laip/model/Qwen/Qwen2___5-7B-Instruct"
# "/data1/laip/model/LLM-Research/Meta-Llama-3___1-8B-Instruct"
# "/data1/laip/model/LLM-Research/gemma-2-9b-it"
# /data1/laip/model/LLM-Research/Meta-Llama-3___1-70B-Instruct

cd /home/laip/InternalScore
mode_name_list=(
                #/nfsdata/laip/model/LLM-Research/Mistral-7B-Instruct-v0___3\
                #/nfsdata/laip/model/LLM-Research/Meta-Llama-3___1-8B-Instruct\
                #/nfsdata/laip/model/Shanghai_AI_Laboratory/internlm3-8b-instruct\
                #/nfsdata/laip/model/LLM-Research/Llama-3___1-Tulu-3-8B
                #/nfsdata/laip/model/shakechen/Llama-2-7b-chat-hf\
                #/nfsdata/laip/model/AI-ModelScope/Mistral-7B-Instruct-v0___2
                #/nfsdata/laip/model/models--prometheus-eval--prometheus-7b-v2.0/snapshots/66ffb1fc20beebfb60a3964a957d9011723116c5/
                /nfsdata/laip/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
                )

for model_name in "${mode_name_list[@]}"; do
  # Set batch size based on the model
  batch_size=8


  # Update the model path in the command
  # new_command1="CUDA_VISIBLE_DEVICES=7 python3 /home/laip/InternalScore/test.py \
  #     --model_name_or_path ${model_name} \
  #     --save_dir results \
  #     --points 5 \
  #     --batch_size ${batch_size} \
  #     --max_new_tokens 256 \
  #     --with_feedback 1 \
  #     --dtype bfloat16 \
  #     --with_cot 0\
  #     --input_file /home/laip/InternalScore/data/helpsteer.json"

  new_command2="CUDA_VISIBLE_DEVICES=2 python3 /home/laip/InternalScore/test.py \
      --model_name_or_path ${model_name} \
      --save_dir results \
      --points 5 \
      --batch_size ${batch_size} \
      --max_new_tokens 1024 \
      --with_feedback 1 \
      --dtype bfloat16 \
      --with_cot 0\
      --input_file /home/laip/InternalScore/data/hs_valid.jsonl"

  # Execute the command (you can optionally echo it to see it before running)
  # echo "${new_command1}"
  # # Uncomment the next line to actually run the command
  #screen -dmS "helpsteer" bash -c "$new_command1" 
  #eval "${new_command1}"
  eval "${new_command2}"

  #eval "${new_command2}"
  # Uncomment the next line to actually run the command
  #screen -dmS "flask" bash -c "$new_command2" 
done
