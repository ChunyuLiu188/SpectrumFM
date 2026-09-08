#!/usr/bin/env bash
# Multi-GPU pretraining launcher.
#   bash run_pretrain.sh                      # fresh run
#   bash run_pretrain.sh --resume ./ckpts
#   bash run_pretrain.sh --dummy --steps 20   # smoke test
set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
# GeForce cards have no NVLink; disable IB. Uncomment if NCCL hangs at init:
# export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
PORT=${MASTER_PORT:-29500}
mkdir -p logs
LOG=${LOG:-./logs/pretrain.log}

echo "launching on $NGPU GPUs, log -> $LOG"
deepspeed --num_gpus="$NGPU" --master_port="$PORT" train.py --config config.yaml "$@" 2>&1 | tee -a "$LOG"
