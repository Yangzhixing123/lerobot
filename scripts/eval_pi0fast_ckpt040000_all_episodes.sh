#!/usr/bin/env bash

set -euo pipefail

# Reproducible offline evaluation for PI0Fast checkpoint 040000.
# No command-line arguments are required: all paths and sampling settings live here.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONDA_BIN="/home/zhixingyang/miniconda3/bin/conda"
CONDA_ENV="lerobot"
POLICY_DIR="${REPO_ROOT}/outputs/train/pi0fast_base_lora_so101/checkpoints/040000/pretrained_model"
DATASET_ROOT="/home/zhixingyang/projects/so101-box-to-plate"
DATASET_REPO_ID="maxlium/so101-box-to-plate"
ACTION_TOKENIZER_DIR="${REPO_ROOT}/outputs/cache/fast-action-tokenizer"
TEXT_TOKENIZER_DIR="${REPO_ROOT}/outputs/cache/paligemma-tokenizer"
FRAMES_PER_EPISODE=5
OUTPUT_DIR="${REPO_ROOT}/outputs/eval"
RESULT_JSON="${OUTPUT_DIR}/pi0fast_ckpt040000_all_episodes_safe.json"
RUN_LOG="${OUTPUT_DIR}/pi0fast_ckpt040000_all_episodes_safe.log"

required_files=(
    "${CONDA_BIN}"
    "${POLICY_DIR}/adapter_model.safetensors"
    "${POLICY_DIR}/config.json"
    "${DATASET_ROOT}/meta/info.json"
    "${ACTION_TOKENIZER_DIR}/processor_config.json"
    "${TEXT_TOKENIZER_DIR}/tokenizer.json"
)
for required_file in "${required_files[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; this test requires an NVIDIA GPU." >&2
    exit 1
fi

# Pick the GPU with the most free VRAM at launch time.
GPU_INDEX="$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
        awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 > max) {max=$2; gpu=$1}} END {print gpu}'
)"
if [[ -z "${GPU_INDEX}" ]]; then
    echo "Could not select a CUDA GPU." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "PI0Fast offline evaluation"
echo "  checkpoint: ${POLICY_DIR}"
echo "  sampling:   ${FRAMES_PER_EPISODE} frames x 40 episodes = 200 frames"
echo "  GPU:        physical GPU ${GPU_INDEX}"
echo "  result:     ${RESULT_JSON}"
echo "  log:        ${RUN_LOG}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python -u scripts/eval_pi0fast_dataset_actions.py \
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
echo "Evaluation complete."
echo "Detailed JSON: ${RESULT_JSON}"
echo "Console log:   ${RUN_LOG}"
