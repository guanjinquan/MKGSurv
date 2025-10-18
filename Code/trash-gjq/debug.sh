#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
# --- Script Configuration ---
# This script is designed to run single-GPU training.
# TODO: Adjust the variables below to match your experiment settings.

# Select the GPU to use (e.g., 0, 1, 2, ...)
GPU_ID=2
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# --- Model & Run Identifiers ---
# Specify the name of the model architecture you are using.
MODEL_NAME="dual_align_net"
# Provide a unique identifier for this specific training run.
# This helps in organizing logs and checkpoints.
RUN_ID="run_001_test"

# --- Training Hyperparameters ---
BATCH_SIZE=32          # Number of samples per batch.
ACC_STEP=1           # Gradient accumulation steps. Effective batch size = BATCH_SIZE * ACC_STEP.
LR=2e-5               # Learning rate for the model head.
BACKBONE_LR=1e-6        # Learning rate for the model backbone.
NUM_EPOCHS=200        # Total number of training epochs.


# --- Execution ---
# The command below executes the main training script with the configured parameters.
echo "Starting training run: ${RUN_ID} on GPU: ${GPU_ID}"

python /home/Guanjq/NewWork/MedAlignFusion/Code/main_train.py \
    --gpu_id ${GPU_ID} \
    --model_name ${MODEL_NAME} \
    --runs_id ${RUN_ID} \
    --batch_size ${BATCH_SIZE} \
    --acc_step ${ACC_STEP} \
    --learning_rate ${LR} \
    --backbone_lr ${BACKBONE_LR} \
    --num_epochs ${NUM_EPOCHS} \
    --dataset "multi_oscc" \
    --fusion_type "concat" \
    --optimizer "AdamW" \
    --scheduler "CosineAnnealingLR" 


echo "Training run ${RUN_ID} finished."
