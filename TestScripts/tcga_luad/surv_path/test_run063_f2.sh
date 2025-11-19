#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
# --- Script Configuration ---
# This script is designed to run single-GPU training.
# TODO: Adjust the variables below to match your experiment settings.

# Select the GPU to use (e.g., 0, 1, 2, ...)
GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}


RUN_ID="tcga_luad_run063_seed=0"

# --- Training Hyperparameters ---
BATCH_SIZE=64          # Number of samples per batch.
ACC_STEP=1           # Gradient accumulation steps. Effective batch size = BATCH_SIZE * ACC_STEP.
LR=3e-5               # Learning rate for the model head.
NUM_EPOCHS=50        # Total number of training epochs.



# --- Execution ---
# The command below executes the main training script with the configured parameters.
echo "Starting training run: ${RUN_ID} on GPU: ${GPU_ID}"

python /home/Guanjq/NewWork/MedAlignFusion/Code/main_test.py \
    --gpu_id ${GPU_ID} \
    --runs_id ${RUN_ID} \
    --model_task "tcga_luad" \
    --dataset "tcga_luad" \
    --image_aggregater "panther" \
    --fusion_type "surv_path" \
    --batch_size ${BATCH_SIZE} \
    --acc_step ${ACC_STEP} \
    --learning_rate ${LR} \
    --num_epochs ${NUM_EPOCHS} \
    --optimizer "AdamW" \
    --weight_decay 1e-4 \
    --scheduler "CosineAnnealingLR"  \
    --modalities "image-pathology,genomics-genomics" \
    --fold 2


echo "Training run ${RUN_ID} finished."

