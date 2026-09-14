#!/usr/bin/env bash

set -euo pipefail

# Download PI0Fast checkpoint and tokenizers from HuggingFace Hub
# Usage: ./scripts/download_checkpoint.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# TODO: Replace with your actual HuggingFace repo
HF_CHECKPOINT_REPO="YOUR_USERNAME/pi0fast-so101-checkpoint"
HF_ACTION_TOKENIZER_REPO="YOUR_USERNAME/fast-action-tokenizer"
HF_TEXT_TOKENIZER_REPO="YOUR_USERNAME/paligemma-tokenizer"

CHECKPOINT_DIR="${REPO_ROOT}/outputs/train/pi0fast_base_lora_so101/checkpoints/040000/pretrained_model"
ACTION_TOKENIZER_DIR="${REPO_ROOT}/outputs/cache/fast-action-tokenizer"
TEXT_TOKENIZER_DIR="${REPO_ROOT}/outputs/cache/paligemma-tokenizer"

echo "📥 Downloading PI0Fast checkpoint and tokenizers..."
echo

# Check if huggingface-cli is available
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "❌ huggingface-cli not found. Install with:"
    echo "   pip install huggingface_hub[cli]"
    exit 1
fi

# Download checkpoint
echo "1️⃣  Downloading checkpoint to ${CHECKPOINT_DIR}..."
mkdir -p "${CHECKPOINT_DIR}"
huggingface-cli download "${HF_CHECKPOINT_REPO}" \
    --local-dir "${CHECKPOINT_DIR}" \
    --local-dir-use-symlinks False

# Download action tokenizer
echo "2️⃣  Downloading action tokenizer to ${ACTION_TOKENIZER_DIR}..."
mkdir -p "${ACTION_TOKENIZER_DIR}"
huggingface-cli download "${HF_ACTION_TOKENIZER_REPO}" \
    --local-dir "${ACTION_TOKENIZER_DIR}" \
    --local-dir-use-symlinks False

# Download text tokenizer
echo "3️⃣  Downloading text tokenizer to ${TEXT_TOKENIZER_DIR}..."
mkdir -p "${TEXT_TOKENIZER_DIR}"
huggingface-cli download "${HF_TEXT_TOKENIZER_REPO}" \
    --local-dir "${TEXT_TOKENIZER_DIR}" \
    --local-dir-use-symlinks False

echo
echo "✅ All files downloaded successfully!"
echo
echo "You can now run:"
echo "  ./scripts/eval_pi0fast_ckpt040000_all_episodes.sh"
