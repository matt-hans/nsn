#!/usr/bin/env python3
"""Download and quantize CLIP models for Vortex pipeline.

Downloads CLIP-ViT-B-32 and CLIP-ViT-L-14 from OpenAI/OpenCLIP,
applies INT8 quantization, and caches models locally.

Usage:
    python scripts/download_and_quantize_clip.py [--device cuda] [--cache-dir ~/.cache/vortex/clip]

VRAM Budget:
    - CLIP-ViT-B-32 (INT8): ~0.3 GB
    - CLIP-ViT-L-14 (INT8): ~0.6 GB
    - Total: ~0.9 GB

Requirements:
    pip install open-clip-torch==2.23.0
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_and_quantize_clip(
    device: str = "cuda",
    cache_dir: Path = None,
) -> None:
    """Download and quantize CLIP models.

    Args:
        device: Target device ("cuda" or "cpu")
        cache_dir: Model cache directory
    """
    try:
        import open_clip
    except ImportError:
        logger.error("open_clip not found. Install with: pip install open-clip-torch==2.23.0")
        sys.exit(1)

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "vortex" / "clip"

    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Target device: {device}")

    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Download and quantize ViT-B-32
    logger.info("=" * 60)
    logger.info("Downloading CLIP-ViT-B-32...")
    logger.info("=" * 60)

    clip_b, _, preprocess_b = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device=device,
        cache_dir=str(cache_dir),
    )
    clip_b.eval()

    logger.info("Applying INT8 quantization to ViT-B-32...")
    clip_b_quantized = torch.quantization.quantize_dynamic(
        clip_b, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Save quantized model
    b32_path = cache_dir / "clip_b32_int8.pt"
    torch.save(clip_b_quantized.state_dict(), b32_path)
    logger.info(f"Saved quantized ViT-B-32 to {b32_path}")

    # Check VRAM usage
    if device == "cuda":
        torch.cuda.synchronize()
        vram_b32_mb = torch.cuda.max_memory_allocated() / (1024**2)
        logger.info(f"ViT-B-32 peak VRAM: {vram_b32_mb:.1f} MB")
        torch.cuda.reset_peak_memory_stats()

    # Clean up
    del clip_b, clip_b_quantized
    if device == "cuda":
        torch.cuda.empty_cache()

    # Download and quantize ViT-L-14
    logger.info("=" * 60)
    logger.info("Downloading CLIP-ViT-L-14...")
    logger.info("=" * 60)

    clip_l, _, preprocess_l = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        device=device,
        cache_dir=str(cache_dir),
    )
    clip_l.eval()

    logger.info("Applying INT8 quantization to ViT-L-14...")
    clip_l_quantized = torch.quantization.quantize_dynamic(
        clip_l, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Save quantized model
    l14_path = cache_dir / "clip_l14_int8.pt"
    torch.save(clip_l_quantized.state_dict(), l14_path)
    logger.info(f"Saved quantized ViT-L-14 to {l14_path}")

    # Check VRAM usage
    if device == "cuda":
        torch.cuda.synchronize()
        vram_l14_mb = torch.cuda.max_memory_allocated() / (1024**2)
        logger.info(f"ViT-L-14 peak VRAM: {vram_l14_mb:.1f} MB")

    # Clean up
    del clip_l, clip_l_quantized
    if device == "cuda":
        torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("CLIP models downloaded and quantized successfully!")
    logger.info("=" * 60)
    logger.info(f"ViT-B-32: {b32_path}")
    logger.info(f"ViT-L-14: {l14_path}")

    if device == "cuda":
        total_vram_gb = (vram_b32_mb + vram_l14_mb) / 1024
        logger.info(f"Total VRAM budget: ~{total_vram_gb:.2f} GB")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download and quantize CLIP models for Vortex"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Target device (default: cuda if available)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Model cache directory (default: ~/.cache/vortex/clip)"
    )

    args = parser.parse_args()

    try:
        download_and_quantize_clip(
            device=args.device,
            cache_dir=args.cache_dir,
        )
    except Exception as e:
        logger.error(f"Failed to download/quantize models: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
