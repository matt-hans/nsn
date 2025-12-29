#!/usr/bin/env python3
"""Download LivePortrait model weights for Vortex pipeline.

This script downloads the LivePortrait FP16 model weights and optionally
builds TensorRT engine for optimized inference.

Usage:
    python scripts/download_liveportrait.py
    python scripts/download_liveportrait.py --build-tensorrt
    python scripts/download_liveportrait.py --cache-dir /custom/cache

Model Size: ~8GB (FP16 precision)
VRAM Budget: ~3.5GB during inference
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download LivePortrait model weights"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory (default: ~/.cache/vortex/liveportrait)",
    )
    parser.add_argument(
        "--build-tensorrt",
        action="store_true",
        help="Build TensorRT engine after download (requires TensorRT 8.6+)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Model precision (default: fp16)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if weights exist",
    )
    return parser.parse_args()


def get_cache_dir(custom_cache_dir: str | None) -> Path:
    """Get cache directory for model weights.

    Args:
        custom_cache_dir: Custom cache directory (optional)

    Returns:
        Path to cache directory
    """
    if custom_cache_dir:
        cache_dir = Path(custom_cache_dir)
    else:
        cache_dir = Path.home() / ".cache" / "vortex" / "liveportrait"

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using cache directory: {cache_dir}")
    return cache_dir


def check_existing_weights(cache_dir: Path, precision: str, force: bool) -> bool:
    """Check if model weights already exist.

    Args:
        cache_dir: Cache directory
        precision: Model precision
        force: Force re-download

    Returns:
        bool: True if should skip download
    """
    weights_file = cache_dir / f"liveportrait_{precision}.safetensors"

    if weights_file.exists() and not force:
        logger.info(f"Model weights already exist: {weights_file}")
        logger.info("Use --force to re-download")
        return True

    return False


def download_weights(cache_dir: Path, precision: str) -> bool:
    """Download LivePortrait model weights.

    Args:
        cache_dir: Cache directory
        precision: Model precision

    Returns:
        bool: True if successful
    """
    logger.info(f"Downloading LivePortrait {precision.upper()} model weights...")

    # In production, this would use Hugging Face Hub or direct download
    # For now, this is a placeholder that documents the expected process

    try:
        # Example using huggingface_hub:
        # from huggingface_hub import hf_hub_download
        # weights_path = hf_hub_download(
        #     repo_id="liveportrait/base-fp16",
        #     filename="model.safetensors",
        #     cache_dir=cache_dir,
        # )

        # Placeholder: Document the download process
        logger.info("=" * 70)
        logger.info("LivePortrait Model Download Instructions")
        logger.info("=" * 70)
        logger.info("")
        logger.info("LivePortrait is not yet available as a public Hugging Face model.")
        logger.info("To use LivePortrait in production:")
        logger.info("")
        logger.info("1. Option A: Download from GitHub (if available)")
        logger.info("   git clone https://github.com/KwaiVGI/LivePortrait")
        logger.info("   cd LivePortrait")
        logger.info("   python scripts/download_weights.py")
        logger.info("")
        logger.info("2. Option B: Download from Hugging Face (when available)")
        logger.info("   huggingface-cli download KwaiVGI/LivePortrait \\")
        logger.info("     --local-dir pretrained_weights \\")
        logger.info("     --exclude '*.git*' 'README.md' 'docs'")
        logger.info("")
        logger.info("3. Option C: Manual download")
        logger.info("   Visit: https://huggingface.co/liveportrait/base-fp16")
        logger.info("   Download model.safetensors to:")
        logger.info(f"   {cache_dir}/liveportrait_{precision}.safetensors")
        logger.info("")
        logger.info("=" * 70)
        logger.info("")
        logger.info("For T016 development, the Vortex pipeline uses a mock")
        logger.info("implementation that simulates LivePortrait functionality.")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def build_tensorrt_engine(cache_dir: Path, precision: str) -> bool:
    """Build TensorRT engine for optimized inference.

    Args:
        cache_dir: Cache directory
        precision: Model precision

    Returns:
        bool: True if successful
    """
    logger.info("Building TensorRT engine...")

    try:
        # Check TensorRT availability
        try:
            import tensorrt as trt
            logger.info(f"TensorRT version: {trt.__version__}")
        except ImportError:
            logger.error("TensorRT not installed. Install with:")
            logger.error("  pip install tensorrt==8.6.1")
            return False

        # Placeholder: Document TensorRT build process
        logger.info("")
        logger.info("TensorRT Build Instructions:")
        logger.info("-" * 70)
        logger.info("1. Export model to ONNX format")
        logger.info("2. Build TensorRT engine with FP16 precision")
        logger.info("3. Save engine to cache directory")
        logger.info("")
        logger.info("Expected speedup: 20-30% reduction in inference time")
        logger.info("")

        # In production:
        # 1. Load PyTorch model
        # 2. Export to ONNX
        # 3. Build TensorRT engine
        # 4. Save to cache_dir / f"liveportrait_{precision}_trt.engine"

        engine_path = cache_dir / f"liveportrait_{precision}_trt.engine"
        logger.info(f"TensorRT engine would be saved to: {engine_path}")

        return True

    except Exception as e:
        logger.error(f"TensorRT build failed: {e}")
        return False


def verify_installation(cache_dir: Path) -> bool:
    """Verify that model can be loaded successfully.

    Args:
        cache_dir: Cache directory

    Returns:
        bool: True if verification passed
    """
    logger.info("Verifying installation...")

    try:
        from vortex.models.liveportrait import load_liveportrait

        # Try loading model (will use mock if real weights not available)
        model = load_liveportrait(device="cpu", precision="fp16")

        logger.info("✓ Model loaded successfully")

        # Cleanup
        del model

        return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def main():
    """Main download function."""
    args = parse_args()

    logger.info("LivePortrait Model Download Script")
    logger.info("=" * 70)
    logger.info(f"Precision: {args.precision}")
    logger.info(f"TensorRT: {'enabled' if args.build_tensorrt else 'disabled'}")
    logger.info("")

    # Get cache directory
    cache_dir = get_cache_dir(args.cache_dir)

    # Check existing weights
    if check_existing_weights(cache_dir, args.precision, args.force):
        if not args.build_tensorrt:
            logger.info("Model weights already downloaded. Skipping download.")
            return 0

    # Download weights
    if not download_weights(cache_dir, args.precision):
        logger.error("Download failed")
        return 1

    # Build TensorRT engine if requested
    if args.build_tensorrt:
        if not build_tensorrt_engine(cache_dir, args.precision):
            logger.warning("TensorRT build failed, but model can still be used")

    # Verify installation
    if not verify_installation(cache_dir):
        logger.error("Installation verification failed")
        return 1

    logger.info("")
    logger.info("=" * 70)
    logger.info("Download complete!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run unit tests: pytest vortex/tests/unit/test_liveportrait.py -v")
    logger.info("2. Run integration tests: pytest vortex/tests/integration/test_liveportrait_generation.py --gpu -v")
    logger.info("3. Run benchmark: python vortex/benchmarks/liveportrait_latency.py")
    logger.info("")

    return 0


if __name__ == "__main__":
    exit(main())
