list=( \
    "/model/Shanghai_AI_Laboratory/internlm3-8b-instruct" \
    "/model/LLM-Research/Llama-3___1-Tulu-3-8B" \
    "/model/LLM-Research/Meta-Llama-3___1-8B-Instruct" \
    "/model/LLM-Research/Mistral-7B-Instruct-v0___3" \
)

for item in "${list[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python unknow_level.py \
        --model_path $item \
        --in_file data/unknow/prompt_unknow.json
done