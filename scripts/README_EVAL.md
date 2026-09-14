# Evaluation Scripts

This directory contains scripts for evaluating trained policies without requiring physical robot hardware.

## Available Evaluations

- **PI0Fast**: `eval/pi0fast_ckpt040000_all_episodes.sh`
- **OAT-RFSQ**: `eval/oat_rfsq_pair_so101_policy_tok50k.sh`

## Quick Start

```bash
# 1. Ensure you're in the lerobot environment
conda activate lerobot
# or: source venv/bin/activate

# 2. Download checkpoint and tokenizers from HuggingFace Hub (for PI0Fast)
./scripts/download_checkpoint.sh

# 3. Run the evaluation
./scripts/eval/pi0fast_ckpt040000_all_episodes.sh     # PI0Fast
./scripts/eval/oat_rfsq_pair_so101_policy_tok50k.sh  # OAT-RFSQ
```

## Prerequisites

### 1. Download Required Files

**Option A: Automatic Download (Recommended)**

Run the download script to fetch checkpoint and tokenizers from HuggingFace Hub:

```bash
pip install huggingface_hub[cli]  # If not already installed
./scripts/download_checkpoint.sh
```

This will download:
- PI0Fast checkpoint (7.1MB)
- Action tokenizer
- Text tokenizer (PaliGemma)

**Option B: Manual Download**

Download the files manually and place them at:

```
outputs/train/pi0fast_base_lora_so101/checkpoints/040000/pretrained_model/
├── adapter_model.safetensors (7.1MB)
├── adapter_config.json
├── config.json
├── policy_postprocessor.json
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
├── policy_preprocessor.json
├── policy_preprocessor_step_3_normalizer_processor.safetensors
└── train_config.json

outputs/cache/fast-action-tokenizer/
├── processor_config.json
├── tokenizer.json
└── ...

outputs/cache/paligemma-tokenizer/
├── tokenizer.json
└── ...
```

### 2. Dataset

The dataset (`maxlium/so101-box-to-plate`) will be automatically downloaded from HuggingFace Hub on the first run. It will be cached at:
```
~/.cache/huggingface/datasets/maxlium___so101-box-to-plate/
```

To use a custom dataset location, set the `DATASET_ROOT` environment variable:
```bash
export DATASET_ROOT=/path/to/your/dataset
./scripts/eval_pi0fast_ckpt040000_all_episodes.sh
```

## Requirements

- NVIDIA GPU with CUDA support
- Python environment with lerobot dependencies installed
- Internet connection (for downloading checkpoint and dataset)
- `huggingface_hub[cli]` for downloading checkpoint

## What You Need to Upload to GitHub

To make this reproducible for others, you need to:

### 1. Upload to GitHub
```bash
git add scripts/eval/
git add scripts/download_checkpoint.sh
git add scripts/README_EVAL.md
git commit -m "Add policy evaluation scripts"
git push
```

### 2. Upload Checkpoint & Tokenizers to HuggingFace Hub

Since checkpoint files are too large for GitHub and are in `.gitignore`, upload them to HuggingFace:

```bash
# Install HuggingFace CLI
pip install huggingface_hub[cli]

# Login to HuggingFace
huggingface-cli login

# Upload checkpoint
huggingface-cli upload YOUR_USERNAME/pi0fast-so101-checkpoint \
    outputs/train/pi0fast_base_lora_so101/checkpoints/040000/pretrained_model/

# Upload action tokenizer
huggingface-cli upload YOUR_USERNAME/fast-action-tokenizer \
    outputs/cache/fast-action-tokenizer/

# Upload text tokenizer
huggingface-cli upload YOUR_USERNAME/paligemma-tokenizer \
    outputs/cache/paligemma-tokenizer/
```

### 3. Update download_checkpoint.sh

Edit `scripts/download_checkpoint.sh` and replace:
```bash
HF_CHECKPOINT_REPO="YOUR_USERNAME/pi0fast-so101-checkpoint"
HF_ACTION_TOKENIZER_REPO="YOUR_USERNAME/fast-action-tokenizer"
HF_TEXT_TOKENIZER_REPO="YOUR_USERNAME/paligemma-tokenizer"
```

With your actual HuggingFace repository names.

## Output

The script generates:
- **JSON results**: `outputs/eval/pi0fast_ckpt040000_all_episodes_safe.json`
- **Console log**: `outputs/eval/pi0fast_ckpt040000_all_episodes_safe.log`

## Configuration

The script automatically:
- Selects the GPU with the most free VRAM
- Detects conda/venv environment
- Downloads the evaluation dataset if needed
- Samples 5 frames per episode across 40 episodes (200 frames total)

To override the conda environment name:
```bash
export CONDA_ENV=my_custom_env
./scripts/eval_pi0fast_ckpt040000_all_episodes.sh
```

## Troubleshooting

### Missing checkpoint files
```
❌ Missing required file: outputs/train/.../adapter_model.safetensors
```
→ Download the checkpoint to the correct location

### No GPU detected
```
⚠️ nvidia-smi is unavailable
```
→ Ensure NVIDIA drivers and CUDA are installed

### Dataset download issues
→ Check your internet connection and HuggingFace Hub access
