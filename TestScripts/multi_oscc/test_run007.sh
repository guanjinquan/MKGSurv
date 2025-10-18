#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"

# --- Script Configuration ---
# This script is for running single-GPU inference (testing).

# Select the GPU to use (e.g., 0, 1, 2, ...)
GPU_ID=2

# --- Experiment Identification ---
# These should match the training run you want to test.
RUN_ID="run_007"
MODEL_TASK="multi_oscc"
FUSION_TYPE="msa"
MODALITIES="all" # Should match the modalities used during training.
BATCH_SIZE=8 # You can often use a larger batch size for inference.

# --- Checkpoint Path ---
# Automatically construct the path to the best model checkpoint.
# Ensure your base checkpoint path is correct.
CHECKPOINT_DIR="../Checkpoints"
MODEL_PATH="${CHECKPOINT_DIR}/${MODEL_TASK}/${RUN_ID}+${FUSION_TYPE}/valid_Best.pth"

# --- Execution ---
# The command below executes the main testing script.
echo "Starting test for run: ${RUN_ID} on GPU: ${GPU_ID}"
echo "Loading model from: ${MODEL_PATH}"

python /home/Guanjq/NewWork/MedAlignFusion/Code/main_test.py \
    --gpu_id ${GPU_ID} \
    --load_pth_path ${MODEL_PATH} \
    --model_task ${MODEL_TASK} \
    --dataset "multi_oscc" \
    --fusion_type ${FUSION_TYPE} \
    --batch_size ${BATCH_SIZE} \
    --modalities ${MODALITIES}




echo "Testing for run ${RUN_ID} finished."
