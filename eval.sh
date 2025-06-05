#!/bin/bash

gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


# 1_fps,8_frames,16_frames,32_frames
EVAL_DATASETS="1_fps"

DATASETS_PATH="EOC-Bench/videos/videos"

#  ["qwen2-vl","qwen2.5-vl","internvl2_5","llama-3.2","llava_video","llava_onevision","video_llava","longva","videollama2","videollama3","gpt","videorefer","osprey","vipllava","sphinx_v"]
MODEL_NAME="qwen2.5-vl"

OUTPUT_PATH="eval_results/qwen2.5vl-7b"
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

SEED=42

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX \
    --seed $SEED &
done

wait
