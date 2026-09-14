#!/usr/bin/env bash

set -euo pipefail

# Reproducible offline evaluation for PI0Fast checkpoint 040000.
#
# Usage:
#   ./scripts/eval_pi0fast_ckpt040000_all_episodes.sh
#
# Prerequisites:
#   1. Download checkpoint to: outputs/train/pi0fast_base_lora_so101/checkpoints/040000/pretrained_model
#   2. The dataset will be automatically downloaded from HuggingFace Hub on first run
#   3. Ensure you have activated the lerobot environment (conda or venv)
#
# Environment variables (optional):
#   DATASET_ROOT - Override dataset cache location (default: ~/.cache/huggingface/datasets/maxlium___so101-box-to-plate)
#   CONDA_ENV    - Override conda environment name (default: lerobot)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Configuration
CONDA_ENV="${CONDA_ENV:-lerobot}"
POLICY_DIR="${REPO_ROOT}/outputs/train/pi0fast_base_lora_so101/checkpoints/040000/pretrained_model"
DATASET_REPO_ID="maxlium/so101-box-to-plate"
# Default to HuggingFace cache location if not specified
DATASET_ROOT="${DATASET_ROOT:-${HOME}/.cache/huggingface/datasets/maxlium___so101-box-to-plate/default/0.0.0/*}"
ACTION_TOKENIZER_DIR="${REPO_ROOT}/outputs/cache/fast-action-tokenizer"
TEXT_TOKENIZER_DIR="${REPO_ROOT}/outputs/cache/paligemma-tokenizer"
FRAMES_PER_EPISODE=5
OUTPUT_DIR="${REPO_ROOT}/outputs/eval"
RESULT_JSON="${OUTPUT_DIR}/pi0fast_ckpt040000_all_episodes_safe.json"
RUN_LOG="${OUTPUT_DIR}/pi0fast_ckpt040000_all_episodes_safe.log"

# Check required files (but allow dataset to be auto-downloaded)
required_files=(
    "${POLICY_DIR}/adapter_model.safetensors"
    "${POLICY_DIR}/config.json"
    "${ACTION_TOKENIZER_DIR}/processor_config.json"
    "${TEXT_TOKENIZER_DIR}/tokenizer.json"
)
for required_file in "${required_files[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "❌ Missing required file: ${required_file}" >&2
        echo "" >&2
        echo "Please ensure you have:" >&2
        echo "  1. Downloaded the checkpoint to: ${POLICY_DIR}" >&2
        echo "  2. The tokenizer files in: ${ACTION_TOKENIZER_DIR} and ${TEXT_TOKENIZER_DIR}" >&2
        exit 1
    fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "⚠️  nvidia-smi is unavailable; this evaluation requires an NVIDIA GPU." >&2
    exit 1
fi

# Pick the GPU with the most free VRAM at launch time.
GPU_INDEX="$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
        awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 > max) {max=$2; gpu=$1}} END {print gpu}'
)"
if [[ -z "${GPU_INDEX}" ]]; then
    echo "❌ Could not select a CUDA GPU." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

# Detect Python environment
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]] || command -v conda >/dev/null 2>&1; then
    # Running in conda environment
    PYTHON_CMD="python"
    if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
        echo "⚠️  Warning: Current conda environment is '${CONDA_DEFAULT_ENV}', but script expects '${CONDA_ENV}'" >&2
        echo "   Attempting to use conda run..." >&2
        PYTHON_CMD="conda run --no-capture-output -n ${CONDA_ENV} python"
    fi
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    # Running in venv
    PYTHON_CMD="python"
else
    echo "⚠️  Warning: No conda or virtualenv detected. Using system Python." >&2
    PYTHON_CMD="python"
fi

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
# Remove HF_HUB_OFFLINE to allow dataset download
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PI0Fast Offline Evaluation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Checkpoint:  ${POLICY_DIR}"
echo "  Dataset:     ${DATASET_REPO_ID}"
echo "  Sampling:    ${FRAMES_PER_EPISODE} frames × 40 episodes = 200 frames"
echo "  GPU:         ${GPU_INDEX}"
echo "  Result JSON: ${RESULT_JSON}"
echo "  Log file:    ${RUN_LOG}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

${PYTHON_CMD} -u scripts/eval_pi0fast_dataset_actions.py \
    --policy-path "${POLICY_DIR}" \
    --action-tokenizer-path "${ACTION_TOKENIZER_DIR}" \
    --text-tokenizer-path "${TEXT_TOKENIZER_DIR}" \
    --dataset-repo-id "${DATASET_REPO_ID}" \
    --dataset-root "${DATASET_ROOT}" \
    --frames-per-episode "${FRAMES_PER_EPISODE}" \
    --device cuda \
    --video-backend torchcodec \
    --summary-only \
    --seed 0 \
    --save-json "${RESULT_JSON}" 2>&1 | tee "${RUN_LOG}"

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Evaluation complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results: ${RESULT_JSON}"
echo "  Log:     ${RUN_LOG}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
