#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"


RUN_ID="run_002"

python /home/Guanjq/NewWork/MedAlignFusion/Code/main_test.py \
    --gpu_id 2\
    --load_pth_path "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/multi_oscc/run_002+healnet/valid_Best.pth" \
    --model_task "multi_oscc" \
    --dataset "multi_oscc" \
    --fusion_type "healnet" \
    --batch_size 4 \
    --modalities "all" 