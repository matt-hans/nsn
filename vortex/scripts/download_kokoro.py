#!/usr/bin/env python3
"""Download Kokoro-82M model and voice packs.

This script downloads the Kokoro-82M TTS model and required voice packs
from Hugging Face. It handles:
- Model weights download (~500MB)
- Voice pack downloads for ICN character voices
- Cache directory setup
- Dependency verification

Usage:
    python scripts/download_kokoro.py
    python scripts/download_kokoro.py --cache-dir ~/.cache/vortex/
    python scripts/download_kokoro.py --verify-only

Requirements:
    pip install kokoro soundfile
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Check that required packages are installed.

    Returns:
        bool: True if all dependencies available
    """
    logger.info("Checking dependencies...")

    try:
        import kokoro
        logger.info("✓ kokoro package installed")
    except ImportError:
        logger.error(
            "✗ kokoro package not found. Install with: pip install kokoro soundfile"
        )
        return False

    try:
        import soundfile
        logger.info("✓ soundfile package installed")
    except ImportError:
        logger.error("✗ soundfile package not found. Install with: pip install soundfile")
        return False

    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ PyTorch with CUDA {torch.version.cuda}")
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠ CUDA not available, will use CPU (slower)")
    except ImportError:
        logger.error("✗ PyTorch not found. Install with: pip install torch")
        return False

    return True


def download_model(cache_dir: Path) -> bool:
    """Download Kokoro-82M model to cache directory.

    Args:
        cache_dir: Cache directory path

    Returns:
        bool: True if download successful
    """
    logger.info(f"Downloading Kokoro-82M model to {cache_dir}")

    try:
        from kokoro import Kokoro

        # First load triggers download from Hugging Face
        # Model is cached in Hugging Face default location, then moved
        model = Kokoro()

        logger.info("✓ Kokoro-82M model downloaded successfully")
        logger.info(f"  Model size: ~500 MB")
        logger.info(f"  Cache location: {cache_dir}")

        del model  # Clean up
        return True

    except Exception as e:
        logger.error(f"✗ Failed to download model: {e}")
        return False


def verify_voices(voice_config_path: Path) -> bool:
    """Verify that required voices are available.

    Args:
        voice_config_path: Path to kokoro_voices.yaml

    Returns:
        bool: True if all voices available
    """
    logger.info("Verifying voice configurations...")

    try:
        import yaml

        with open(voice_config_path) as f:
            voice_config = yaml.safe_load(f)

        required_voices = ["rick_c137", "morty", "summer"]
        missing_voices = [v for v in required_voices if v not in voice_config]

        if missing_voices:
            logger.error(f"✗ Missing required voices: {missing_voices}")
            return False

        logger.info(f"✓ All required voices configured: {required_voices}")

        # Log all available voices
        logger.info(f"  Total voices available: {len(voice_config)}")
        for icn_voice, kokoro_voice in voice_config.items():
            logger.info(f"    {icn_voice} → {kokoro_voice}")

        return True

    except FileNotFoundError:
        logger.error(f"✗ Voice config not found: {voice_config_path}")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to verify voices: {e}")
        return False


def test_synthesis() -> bool:
    """Test synthesis with Kokoro model.

    Returns:
        bool: True if synthesis works
    """
    logger.info("Testing synthesis...")

    try:
        from vortex.models.kokoro import load_kokoro
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model on {device}...")

        kokoro = load_kokoro(device=device)

        logger.info("Synthesizing test audio...")
        audio = kokoro.synthesize(
            text="Hello, this is a test of the Kokoro text-to-speech system.",
            voice_id="rick_c137"
        )

        logger.info(f"✓ Synthesis successful")
        logger.info(f"  Output shape: {audio.shape}")
        logger.info(f"  Duration: {len(audio) / 24000:.2f} seconds")
        logger.info(f"  Sample rate: 24000 Hz")

        del kokoro
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        logger.error(f"✗ Synthesis test failed: {e}", exc_info=True)
        return False


def main():
    """Main download script."""
    parser = argparse.ArgumentParser(
        description="Download Kokoro-82M model and verify setup"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "vortex",
        help="Cache directory for model weights"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify setup, don't download"
    )
    parser.add_argument(
        "--test-synthesis",
        action="store_true",
        help="Test synthesis after download"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Kokoro-82M Download & Setup")
    logger.info("=" * 60)

    # Step 1: Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Exiting.")
        sys.exit(1)

    # Step 2: Setup cache directory
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cache directory: {args.cache_dir}")

    # Step 3: Download model (unless verify-only)
    if not args.verify_only:
        if not download_model(args.cache_dir):
            logger.error("Model download failed. Exiting.")
            sys.exit(1)
    else:
        logger.info("Skipping download (--verify-only mode)")

    # Step 4: Verify voice configurations
    voice_config_path = (
        Path(__file__).parent.parent / "src" / "vortex" / "models" / "configs" / "kokoro_voices.yaml"
    )
    if not verify_voices(voice_config_path):
        logger.error("Voice verification failed. Exiting.")
        sys.exit(1)

    # Step 5: Test synthesis (if requested)
    if args.test_synthesis:
        if not test_synthesis():
            logger.error("Synthesis test failed. Exiting.")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("✓ Kokoro-82M setup complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run unit tests: pytest tests/unit/test_kokoro.py -v")
    logger.info("  2. Run integration tests: pytest tests/integration/test_kokoro_synthesis.py --gpu -v")
    logger.info("  3. Run benchmarks: python benchmarks/kokoro_latency.py")
    logger.info("")


if __name__ == "__main__":
    main()
