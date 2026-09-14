#!/usr/bin/env bash

set -euo pipefail

# ============================================================================
# OAT-RFSQ Offline Action-vs-GT Evaluation Script
# ============================================================================
# This script evaluates a trained OAT-RFSQ policy by comparing its predicted
# actions against ground truth actions from a dataset. It never connects to
# a physical robot.
#
# Usage:
#   ./scripts/eval/oat_rfsq_pair_so101_policy_tok50k.sh
#
# Environment variables (all optional):
#   CHECKPOINT           - Checkpoint number (default: latest)
#   FRAMES_PER_EPISODE   - Number of frames to sample per episode (default: 5)
#   TEMPERATURE          - Sampling temperature, 0=greedy (default: 0)
#   DEVICE               - cuda or cpu (default: cuda)
#   CONDA_ENV            - Conda environment name (default: lerobot)
#   DATASET_ROOT         - Local dataset path (default: HF cache)
#
# Examples:
#   CHECKPOINT=018000 FRAMES_PER_EPISODE=10 ./scripts/eval/oat_rfsq_pair_so101_policy_tok50k.sh
#   DEVICE=cpu ./scripts/eval/oat_rfsq_pair_so101_policy_tok50k.sh
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Auto-detect conda/python environment
if command -v conda >/dev/null 2>&1 && [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    CONDA_BIN="$(which conda)"
    CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV}}"
elif command -v python >/dev/null 2>&1; then
    CONDA_BIN=""
    CONDA_ENV=""
else
    echo "❌ Error: No Python environment found" >&2
    echo "Please activate a conda environment or virtual environment" >&2
    exit 1
fi

TRAIN_DIR="${REPO_ROOT}/outputs/train/oat_rfsq_pair_so101_policy_tok50k"
DATASET_ROOT="${DATASET_ROOT:-${HOME}/.cache/huggingface/datasets/maxlium___so101-box-to-plate}"
DATASET_REPO_ID="maxlium/so101-box-to-plate"

CHECKPOINT="${CHECKPOINT:-latest}"
FRAMES_PER_EPISODE="${FRAMES_PER_EPISODE:-5}"
TEMPERATURE="${TEMPERATURE:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

if [[ "${CHECKPOINT}" == "latest" ]]; then
    shopt -s nullglob
    checkpoint_dirs=("${TRAIN_DIR}"/checkpoints/[0-9]*)
    shopt -u nullglob
    if ((${#checkpoint_dirs[@]} == 0)); then
        echo "❌ No numeric checkpoints found under ${TRAIN_DIR}/checkpoints" >&2
        exit 1
    fi
    checkpoint_dir="${checkpoint_dirs[${#checkpoint_dirs[@]} - 1]}"
else
    checkpoint_dir="${TRAIN_DIR}/checkpoints/${CHECKPOINT}"
fi

POLICY_DIR="${checkpoint_dir}/pretrained_model"
CHECKPOINT_NAME="$(basename -- "${checkpoint_dir}")"
OUTPUT_DIR="${REPO_ROOT}/outputs/eval/oat_rfsq_pair_so101_policy_tok50k"
RESULT_JSON="${OUTPUT_DIR}/checkpoint_${CHECKPOINT_NAME}_action_vs_gt.json"
RUN_LOG="${OUTPUT_DIR}/checkpoint_${CHECKPOINT_NAME}_action_vs_gt.log"

# Check required policy files
required_policy_files=(
    "${POLICY_DIR}/config.json"
    "${POLICY_DIR}/model.safetensors"
    "${POLICY_DIR}/policy_preprocessor.json"
    "${POLICY_DIR}/policy_postprocessor.json"
)
for required_file in "${required_policy_files[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "❌ Missing required policy file: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! "${FRAMES_PER_EPISODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "❌ FRAMES_PER_EPISODE must be a positive integer, got: ${FRAMES_PER_EPISODE}" >&2
    exit 1
fi

if [[ "${DEVICE}" == cuda* ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "❌ nvidia-smi is unavailable; set DEVICE=cpu for a slow CPU run." >&2
        exit 1
    fi
    if [[ -z "${GPU_INDEX:-}" ]]; then
        GPU_INDEX="$(
            nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
                awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 > max) {max=$2; gpu=$1}} END {print gpu}'
        )"
    fi
    if [[ -z "${GPU_INDEX}" ]]; then
        echo "❌ Could not select a CUDA GPU." >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
    GPU_DESCRIPTION="physical GPU ${GPU_INDEX}"
else
    GPU_DESCRIPTION="not used (${DEVICE})"
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

# Allow HuggingFace downloads for dataset
# export HF_HUB_OFFLINE=1  # Commented out to allow dataset auto-download
# export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "============================================================================"
echo "OAT-RFSQ Offline Action-vs-GT Evaluation"
echo "============================================================================"
echo "  Checkpoint:  ${POLICY_DIR}"
echo "  Dataset:     ${DATASET_ROOT}"
echo "  Sampling:    ${FRAMES_PER_EPISODE} evenly spaced frames per episode"
echo "  Temperature: ${TEMPERATURE} (0 = deterministic greedy decoding)"
echo "  Device:      ${DEVICE}; ${GPU_DESCRIPTION}"
echo "  Result:      ${RESULT_JSON}"
echo "  Log:         ${RUN_LOG}"
echo "============================================================================"
echo

if [[ -n "${CONDA_BIN}" ]]; then
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
        python -u scripts/eval/eval_pi0fast_dataset_actions.py \
        --policy-path "${POLICY_DIR}" \
        --dataset-repo-id "${DATASET_REPO_ID}" \
        --dataset-root "${DATASET_ROOT}" \
        --frames-per-episode "${FRAMES_PER_EPISODE}" \
        --temperature "${TEMPERATURE}" \
        --device "${DEVICE}" \
        --video-backend "${VIDEO_BACKEND}" \
        --summary-only \
        --fail-fast \
        --seed "${SEED}" \
        --save-json "${RESULT_JSON}" 2>&1 | tee "${RUN_LOG}"
else
    python -u scripts/eval/eval_pi0fast_dataset_actions.py \
        --policy-path "${POLICY_DIR}" \
        --dataset-repo-id "${DATASET_REPO_ID}" \
        --dataset-root "${DATASET_ROOT}" \
        --frames-per-episode "${FRAMES_PER_EPISODE}" \
        --temperature "${TEMPERATURE}" \
        --device "${DEVICE}" \
        --video-backend "${VIDEO_BACKEND}" \
        --summary-only \
        --fail-fast \
        --seed "${SEED}" \
        --save-json "${RESULT_JSON}" 2>&1 | tee "${RUN_LOG}"
fi

echo
echo "============================================================================"
echo "✅ Evaluation Complete"
echo "============================================================================"
echo "  Detailed JSON: ${RESULT_JSON}"
echo "  Console log:   ${RUN_LOG}"
echo "============================================================================"
