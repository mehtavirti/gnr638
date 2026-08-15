#!/usr/bin/env bash

# Fix CRLF automatically
if file "$0" | grep -q CRLF; then
    echo "Fixing Windows line endings..."
    sed -i 's/\r$//' "$0"
fi

set -e

echo "=============================="
echo "Starting setup..."
echo "=============================="

# --------------------------------------------------
# 1. Clone repo into current directory
# --------------------------------------------------
echo "=== [1/4] Cloning repository ==="

git clone --branch project https://github.com/mehtavirti/gnr638.git _tmp_gnr_clone
cp -rf _tmp_gnr_clone/. .
rm -rf _tmp_gnr_clone

echo "Repository ready."

# --------------------------------------------------
# 2. Create conda environment
# --------------------------------------------------
echo "=== [2/4] Creating conda environment ==="
echo "Accepting conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

conda create -n gnr_project_env python=3.11 -y

echo "Environment created."

# --------------------------------------------------
# 3. Install dependencies
# --------------------------------------------------
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

# --------------------------------------------------
# 4. Download model into ./model (CRITICAL)
# --------------------------------------------------
echo "=== [4/4] Downloading model into ./model ==="

cat << 'EOF' > download_model.py
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, BitsAndBytesConfig

HF_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

print("Downloading processor...")
processor = Qwen2_5_VLProcessor.from_pretrained(HF_MODEL_ID)
processor.save_pretrained("./model")

print("Downloading model...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    HF_MODEL_ID,
    quantization_config=bnb,
    device_map="auto",
)

model.save_pretrained("./model")

print("Model saved successfully in ./model")
EOF

# run download script
conda run -n gnr_project_env python download_model.py

# cleanup
rm download_model.py

echo ""
echo "=============================="
echo "Setup complete!"
echo "=============================="
echo "Next steps:"
echo "conda activate gnr_project_env"
echo "python inference.py --test_dir <path>"