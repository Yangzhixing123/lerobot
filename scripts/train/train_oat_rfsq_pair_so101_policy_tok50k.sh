#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/../so101-box-to-plate}"
ACTION_TOKENIZER_PATH="${ACTION_TOKENIZER_PATH:-${REPO_ROOT}/outputs/train/oat_rfsq_pair_so101_tokenizer/checkpoint-00050000}"

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

if [[ ! -d "${ACTION_TOKENIZER_PATH}" ]]; then
    echo "Missing action tokenizer checkpoint: ${ACTION_TOKENIZER_PATH}" >&2
    echo "Run train_oat_rfsq_pair_so101_tokenizer.sh first." >&2
    exit 1
fi

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" lerobot-train \
    --dataset.repo_id=maxlium/so101-box-to-plate \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.eval_split=0.2 \
    --policy.type=oat_rfsq_pair \
    --policy.action_tokenizer_path="${ACTION_TOKENIZER_PATH}" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.n_obs_steps=2 \
    --policy.horizon=32 \
    --policy.n_action_steps=16 \
    --policy.optimizer_lr=5e-5 \
    --policy.optimizer_lr_observation_encoder=1e-5 \
    --policy.scheduler_warmup_steps=100 \
    --output_dir=outputs/train/oat_rfsq_pair_so101_policy_tok50k \
    --job_name=oat_rfsq_pair_so101_policy_tok50k \
    --batch_size=8 \
    --steps=20000 \
    --log_freq=100 \
    --eval_steps=1000 \
    --max_eval_samples=512 \
    --save_freq=2000 \
    --env_eval_freq=0 \
    --wandb.enable=true \
    --wandb.project=so101-box-to-plate
