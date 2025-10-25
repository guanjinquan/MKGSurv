#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
# --- Script Configuration ---
# This script is designed to run single-GPU training.
# TODO: Adjust the variables below to match your experiment settings.

# Select the GPU to use (e.g., 0, 1, 2, ...)
GPU_ID=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}


RUN_ID="hancock_run_004_align"

# --- Training Hyperparameters ---
BATCH_SIZE=8          # Number of samples per batch.
ACC_STEP=2           # Gradient accumulation steps. Effective batch size = BATCH_SIZE * ACC_STEP.
LR=1e-6               # Learning rate for the model head.
BACKBONE_LR=5e-7        # Learning rate for the model backbone.
NUM_EPOCHS=200        # Total number of training epochs.


# --- Execution ---
# The command below executes the main training script with the configured parameters.
echo "Starting training run: ${RUN_ID} on GPU: ${GPU_ID}"

python /home/Guanjq/NewWork/MedAlignFusion/Code/main_train.py \
    --gpu_id ${GPU_ID} \
    --runs_id ${RUN_ID} \
    --model_task "hancock" \
    --dataset "hancock" \
    --fusion_type "healnet" \
    --batch_size ${BATCH_SIZE} \
    --acc_step ${ACC_STEP} \
    --learning_rate ${LR} \
    --backbone_lr ${BACKBONE_LR} \
    --num_epochs ${NUM_EPOCHS} \
    --optimizer "AdamW" \
    --weight_decay 5e-6 \
    --scheduler "CosineAnnealingLR"  \
    --modalities "all" \
    --with_multimodal_align 


echo "Training run ${RUN_ID} finished."

