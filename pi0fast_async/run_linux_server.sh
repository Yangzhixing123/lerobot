#!/usr/bin/env bash
set -euo pipefail

cd /data/ywb-2/yzx/lerobot

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/ywb-2/yzx/lerobot/.cache/uv}"
export HF_HOME="${HF_HOME:-/data/ywb-2/yzx/lerobot/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-0.033}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-1.0}"
CACHE_DIR="${CACHE_DIR:-$HF_HUB_CACHE}"
LOG_DIR="${LOG_DIR:-/data/ywb-2/yzx/lerobot/pi0fast_async/logs/server}"

uv run python pi0fast_async/pi0fast_policy_server.py \
  --host="$HOST" \
  --port="$PORT" \
  --fps="$FPS" \
  --inference_latency="$INFERENCE_LATENCY" \
  --obs_queue_timeout="$OBS_QUEUE_TIMEOUT" \
  --cache_dir="$CACHE_DIR" \
  --log_dir="$LOG_DIR"
