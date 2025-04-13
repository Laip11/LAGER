list=( \
    "Shanghai_AI_Laboratory/internlm3-8b-instruct" \
    "/LLM-Research/Llama-3.1-Tulu-3-8B" \
    "/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
    "/LLM-Research/Mistral-7B-Instruct-v0.3" \
)

for item in "${list[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python src/unknow_level.py \
        --model_path $item \
        --in_file data/unknow/prompt_unknow.json
done