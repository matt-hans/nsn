#!/usr/bin/env python3
"""
Model downloader for Vortex AI engine with retry logic and checksum verification.

Downloads:
- Flux-Schnell (NF4 quantized) - ~6GB
- LivePortrait (FP16) - ~3.5GB
- Kokoro-82M (FP32) - ~400MB
- CLIP-ViT-B-32 (INT8) - ~300MB
- CLIP-ViT-L-14 (INT8) - ~600MB

Total: ~15GB
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model definitions with URLs and expected checksums
MODELS = {
    "flux-schnell-nf4": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors",
        "path": "flux-schnell/model.safetensors",
        "checksum": None,  # Add actual checksum
        "size_gb": 6.0
    },
    "liveportrait": {
        "url": "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/liveportrait.pth",
        "path": "liveportrait/model.pth",
        "checksum": None,
        "size_gb": 3.5
    },
    "kokoro-82m": {
        "url": "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.pth",
        "path": "kokoro/model.pth",
        "checksum": None,
        "size_gb": 0.4
    },
    "clip-vit-b32": {
        "url": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin",
        "path": "clip-vit-b32/model.bin",
        "checksum": None,
        "size_gb": 0.3
    },
    "clip-vit-l14": {
        "url": "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin",
        "path": "clip-vit-l14/model.bin",
        "checksum": None,
        "size_gb": 0.6
    }
}


def verify_checksum(file_path: Path, expected_checksum: Optional[str]) -> bool:
    """Verify SHA256 checksum of downloaded file."""
    if expected_checksum is None:
        logger.warning(f"No checksum provided for {file_path}, skipping verification")
        return True

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    actual_checksum = sha256_hash.hexdigest()
    if actual_checksum != expected_checksum:
        logger.error(f"Checksum mismatch for {file_path}")
        logger.error(f"Expected: {expected_checksum}")
        logger.error(f"Got: {actual_checksum}")
        return False

    logger.info(f"Checksum verified for {file_path}")
    return True


def download_with_retry(url: str, output_path: Path, max_retries: int = 3) -> bool:
    """Download file with retry logic and progress bar."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")

            # Create parent directory
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Stream download with progress bar
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f, tqdm(
                desc=output_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    size = f.write(data)
                    pbar.update(size)

            logger.info(f"Downloaded {output_path} successfully")
            return True

        except Exception as e:
            logger.error(f"Download failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return False

    return False


def download_all_models(output_dir: Path, force: bool = False) -> bool:
    """Download all models to output directory."""
    logger.info(f"Model output directory: {output_dir}")
    logger.info(f"Total models to download: {len(MODELS)}")

    total_size = sum(model['size_gb'] for model in MODELS.values())
    logger.info(f"Total download size: ~{total_size:.1f} GB")

    success_count = 0

    for model_name, model_info in MODELS.items():
        output_path = output_dir / model_info['path']

        # Skip if already exists and not forcing
        if output_path.exists() and not force:
            logger.info(f"Model {model_name} already exists, skipping")
            success_count += 1
            continue

        logger.info(f"Downloading {model_name} (~{model_info['size_gb']} GB)")

        if download_with_retry(model_info['url'], output_path):
            # Verify checksum if provided
            if verify_checksum(output_path, model_info['checksum']):
                success_count += 1
            else:
                logger.error(f"Checksum verification failed for {model_name}")
                # Delete corrupted file
                output_path.unlink()
        else:
            logger.error(f"Failed to download {model_name}")

    logger.info(f"Successfully downloaded {success_count}/{len(MODELS)} models")
    return success_count == len(MODELS)


def main():
    parser = argparse.ArgumentParser(description="Download Vortex AI models")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/models"),
        help="Output directory for models (default: /models)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Download all models
    success = download_all_models(args.output, args.force)

    if success:
        logger.info("All models downloaded successfully!")
        return 0
    else:
        logger.error("Some models failed to download")
        return 1


if __name__ == "__main__":
    sys.exit(main())
