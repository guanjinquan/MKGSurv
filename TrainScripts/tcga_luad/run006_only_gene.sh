#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
# --- Script Configuration ---
# This script is designed to run single-GPU training.
# TODO: Adjust the variables below to match your experiment settings.

# Select the GPU to use (e.g., 0, 1, 2, ...)
GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}


RUN_ID="tcga_luad_run006"

# --- Training Hyperparameters ---
BATCH_SIZE=16          # Number of samples per batch.
ACC_STEP=1           # Gradient accumulation steps. Effective batch size = BATCH_SIZE * ACC_STEP.
LR=1e-5               # Learning rate for the model head.
BACKBONE_LR=5e-7        # Learning rate for the model backbone.
NUM_EPOCHS=50        # Total number of training epochs.


# --- Execution ---
# The command below executes the main training script with the configured parameters.
echo "Starting training run: ${RUN_ID} on GPU: ${GPU_ID}"

python /home/Guanjq/NewWork/MedAlignFusion/Code/main_train.py \
    --gpu_id ${GPU_ID} \
    --runs_id ${RUN_ID} \
    --model_task "tcga_luad" \
    --dataset "tcga_luad" \
    --fusion_type "msa" \
    --batch_size ${BATCH_SIZE} \
    --acc_step ${ACC_STEP} \
    --learning_rate ${LR} \
    --backbone_lr ${BACKBONE_LR} \
    --num_epochs ${NUM_EPOCHS} \
    --optimizer "AdamW" \
    --weight_decay 0.1 \
    --scheduler "CosineAnnealingLR"  \
    --modalities "text-treatment" \
    --fold 0


echo "Training run ${RUN_ID} finished."

