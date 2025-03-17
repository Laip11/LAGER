NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" \
CUDA_VISIBLE_DEVICES=0 \
    python src/pair_wise.py \
        --feedback True \
        --model /models/Shanghai_AI_Laboratory/internlm3-8b-instruct  \
        --data_path data/pair_wise/helpsteer_pref.jsonl

