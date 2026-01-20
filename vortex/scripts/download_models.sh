#!/bin/bash
# ToonGen Model Download Script
# Downloads all required models for the ComfyUI workflow

set -e

COMFY_DIR="${COMFY_DIR:-/home/matt/nsn/ComfyUI}"
MODELS_DIR="$COMFY_DIR/models"

echo "=== ToonGen Model Downloader ==="
echo "Target: $MODELS_DIR"

# Create directories
mkdir -p "$MODELS_DIR/checkpoints"
mkdir -p "$MODELS_DIR/liveportrait"
mkdir -p "$MODELS_DIR/animatediff_models"
mkdir -p "$MODELS_DIR/clip"

# Flux.1-Schnell NF4 (Quantized for 12GB VRAM)
echo "Downloading Flux.1-Schnell NF4..."
if [ ! -f "$MODELS_DIR/checkpoints/flux1-schnell-bnb-nf4.safetensors" ]; then
    wget -O "$MODELS_DIR/checkpoints/flux1-schnell-bnb-nf4.safetensors" \
        "https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-schnell-bnb-nf4.safetensors"
fi

# ToonYou SD1.5 Checkpoint (Cartoon Style)
echo "Downloading ToonYou Beta6..."
if [ ! -f "$MODELS_DIR/checkpoints/toonyou_beta6.safetensors" ]; then
    wget -O "$MODELS_DIR/checkpoints/toonyou_beta6.safetensors" \
        "https://civitai.com/api/download/models/125771"
fi

# AnimateDiff Motion Module
echo "Downloading AnimateDiff v3..."
if [ ! -f "$MODELS_DIR/animatediff_models/mm_sd_v15_v3.ckpt" ]; then
    wget -O "$MODELS_DIR/animatediff_models/mm_sd_v15_v3.ckpt" \
        "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v3.ckpt"
fi

# LivePortrait weights (downloaded by node on first use, but we can pre-fetch)
echo "LivePortrait weights will be downloaded on first use by the ComfyUI node"

echo "=== Download Complete ==="
echo "Models installed to: $MODELS_DIR"
