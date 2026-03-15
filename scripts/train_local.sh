#!/bin/bash

# Define paths
OUTPUT_ROOT="/home/huiwei/jiaxuanzou/linear_attn/GatedDeltaNet"
TRAIN_DATA="${OUTPUT_ROOT}/mydata/slimpajama/train"
VALIDATION_DATA="${OUTPUT_ROOT}/mydata/slimpajama/validation"
SAVE_DIR="/home/huiwei/jiaxuanzou/linear_attn/save_dir"

# Experiment settings
FULL_TRAIN_TOKENS=100000000
NAME="${NAME:-512x4k_100M_RelaxedKaczmarzQNorm_0.4B}"
MODEL="${MODEL:-RelaxedKaczmarzQNorm_0.4B}"
CONFIG="${CONFIG:-tsz512x4k}"
EVAL_ITERS="${EVAL_ITERS:-15}"
TOTAL_EVALS="${TOTAL_EVALS:-100}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
MAX_TOKENS="${MAX_TOKENS:-$FULL_TRAIN_TOKENS}"
EVAL_STEP_INTERVAL="${EVAL_STEP_INTERVAL:-0}"

# Directories
LOGS_DIR="${SAVE_DIR}/logs/${NAME}/"
WANDB_DIR="${SAVE_DIR}/wandb/${NAME}/"
TRI_CACHE_DIR="${SAVE_DIR}/triton/${NAME}/"

export PYTHONPATH="${OUTPUT_ROOT}":$PYTHONPATH
export TRITON_CACHE_DIR="${TRI_CACHE_DIR}"
export LD_LIBRARY_PATH=/home/huiwei/miniconda3/envs/jiaxuanzou/lib:$LD_LIBRARY_PATH
unset WANDB_DISABLED
export WANDB_MODE=online
export WANDB_DIR="${WANDB_DIR}"

# Create directories
mkdir -p ${LOGS_DIR}
mkdir -p ${WANDB_DIR}
mkdir -p ${TRI_CACHE_DIR}

echo "Starting training..."
echo "Model: ${MODEL}"
echo "Config: ${CONFIG}"
echo "Output Root: ${OUTPUT_ROOT}"
echo "Train Data: ${TRAIN_DATA}"
echo "Validation Data: ${VALIDATION_DATA}"
echo "Max Tokens: ${MAX_TOKENS}"
echo "Total Evals: ${TOTAL_EVALS}"
echo "W&B Mode: ${WANDB_MODE}"
echo "W&B Dir: ${WANDB_DIR}"

PYTHON_BIN="/home/huiwei/miniconda3/envs/jiaxuanzou/bin/python"

${PYTHON_BIN} -u ${OUTPUT_ROOT}/pretrain.py \
    --train_data_dir ${TRAIN_DATA} \
    --val_data_dir ${VALIDATION_DATA} \
    --output_root ${SAVE_DIR} \
    --exp_name ${NAME} \
    --model_name ${MODEL} \
    --train_config ${CONFIG} \
    --eval_iters ${EVAL_ITERS} \
    --total_evals ${TOTAL_EVALS} \
    --eval_step_interval ${EVAL_STEP_INTERVAL} \
    --learning_rate ${LR} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --max_tokens ${MAX_TOKENS}
