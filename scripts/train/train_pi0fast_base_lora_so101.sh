#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/../so101-box-to-plate}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is not available. Install Conda or activate the '${CONDA_ENV}' environment first." >&2
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

if ! command -v lerobot-train >/dev/null 2>&1; then
    echo "lerobot-train is not available in the '${CONDA_ENV}' environment." >&2
    exit 1
fi

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" lerobot-train \
    --dataset.repo_id=maxlium/so101-box-to-plate \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.eval_split=0.2 \
    --policy.path=lerobot/pi0fast-base \
    --policy.input_features=null \
    --policy.output_features=null \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.push_to_hub=false \
    --policy.gradient_checkpointing=true \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.max_action_tokens=256 \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_decay_steps=20000 \
    --policy.scheduler_decay_lr=1e-5 \
    --peft.method_type=LORA \
    --peft.r=16 \
    --peft.lora_alpha=16 \
    --peft.target_modules='.*language_model\.layers\.[0-9]+\.self_attn\.(q_proj|v_proj)' \
    --output_dir=outputs/train/pi0fast_base_lora_so101 \
    --job_name=pi0fast_base_lora_so101 \
    --batch_size=1 \
    --steps=20000 \
    --log_freq=100 \
    --eval_steps=1000 \
    --max_eval_samples=128 \
    --save_freq=2000 \
    --env_eval_freq=0 \
    --wandb.enable=true \
    --wandb.project=so101-box-to-plate
