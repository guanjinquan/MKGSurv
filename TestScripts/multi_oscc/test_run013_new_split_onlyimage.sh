#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
# --- Script Configuration ---
# This script is designed to run single-GPU training.
# TODO: Adjust the variables below to match your experiment settings.

# Select the GPU to use (e.g., 0, 1, 2, ...)
GPU_ID=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

RUN_ID="run_013"
MODEL_TASK="multi_oscc"
FUSION_TYPE="msa"
MODALITIES="image" # Should match the modalities used during training.
BATCH_SIZE=8 # You can often use a larger batch size for inference.

# --- Checkpoint Path ---
# Automatically construct the path to the best model checkpoint.
# Ensure your base checkpoint path is correct.
CHECKPOINT_DIR="../Checkpoints"
MODEL_PATH="${CHECKPOINT_DIR}/${MODEL_TASK}/${RUN_ID}+${FUSION_TYPE}/valid_Best.pth"


python /home/Guanjq/NewWork/MedAlignFusion/Code/main_test.py \
    --gpu_id ${GPU_ID} \
    --load_pth_path ${MODEL_PATH} \
    --model_task ${MODEL_TASK} \
    --dataset "multi_oscc" \
    --fusion_type ${FUSION_TYPE} \
    --batch_size ${BATCH_SIZE} \
    --modalities ${MODALITIES}



echo "Training run ${RUN_ID} finished."

