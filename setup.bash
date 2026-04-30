#!/usr/bin/env bash
# =============================================================
# setup.bash — GNR638 Project Setup
# Runs from the submission directory. Internet available here.
# inference.py runs later WITHOUT internet, uses HF cache.
# =============================================================

set -e  # stop on any error

# ------------------------------------------------------------------
# 1. Clone repo contents directly into current directory (no cd)
# ------------------------------------------------------------------
echo "=== [1/4] Cloning repository ==="
git clone --branch project https://github.com/mehtavirti/gnr638.git _tmp_gnr_clone
cp -rf _tmp_gnr_clone/. .
rm -rf _tmp_gnr_clone
echo "Repository ready."

# ------------------------------------------------------------------
# 2. Create conda environment with Python 3.11
# ------------------------------------------------------------------
echo "=== [2/4] Creating conda environment ==="
conda create -n gnr_project_env python=3.11 -y
echo "Environment created."

# ------------------------------------------------------------------
# 3. Install all dependencies
# ------------------------------------------------------------------
echo "=== [3/4] Installing dependencies ==="
conda run -n gnr_project_env pip install --upgrade pip
conda run -n gnr_project_env pip install \
    "transformers>=4.52.0" \
    accelerate \
    qwen-vl-utils \
    "bitsandbytes>=0.43.0" \
    sentencepiece \
    "opencv-python-headless>=4.8.0" \
    torchvision \
    Pillow \
    numpy \
    pandas \
    torch
echo "Dependencies installed."

# ------------------------------------------------------------------
# 4. Pre-download model into HuggingFace cache (needs internet)
#    inference.py will load from HF cache (TRANSFORMERS_OFFLINE=1)
# ------------------------------------------------------------------
echo "=== [4/4] Downloading model weights into HF cache ==="
conda run -n gnr_project_env python -c "
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, BitsAndBytesConfig

HF_MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'

print('Downloading processor...')
processor = Qwen2_5_VLProcessor.from_pretrained(
    HF_MODEL_ID,
    min_pixels=256*28*28,
    max_pixels=1280*28*28,
)
print('Processor cached.')

print('Downloading model (this takes ~10-15 mins)...')
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    HF_MODEL_ID,
    quantization_config=bnb,
    device_map='auto',
)
print('Model cached in ~/.cache/huggingface/')
print('Setup complete!')
"

echo ""
echo "=================================================="
echo "Setup finished! Now run:"
echo "  conda activate gnr_project_env"
echo "  python inference.py --test_dir <path_to_test_dir>"
echo "=================================================="
