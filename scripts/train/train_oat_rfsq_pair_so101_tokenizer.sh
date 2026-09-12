#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/../so101-box-to-plate}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is not available. Install Conda or activate the '${CONDA_ENV}' environment first." >&2
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

if ! command -v lerobot-train-oat-tokenizer >/dev/null 2>&1; then
    echo "lerobot-train-oat-tokenizer is not available in the '${CONDA_ENV}' environment." >&2
    exit 1
fi

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" lerobot-train-oat-tokenizer \
    --repo_id=maxlium/so101-box-to-plate \
    --root="${DATASET_ROOT}" \
    --policy_type=oat_rfsq_pair \
    --output_dir=outputs/train/oat_rfsq_pair_so101_tokenizer \
    --batch_size=256 \
    --steps=100000 \
    --save_freq=10000 \
    --device=cuda
