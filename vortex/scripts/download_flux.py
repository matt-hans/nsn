#!/usr/bin/env python
"""Pre-download Flux-Schnell model weights.

This script downloads the Flux.1-Schnell model weights (~12GB) from Hugging Face
and caches them locally. This is a one-time operation that avoids delays during
first model load.

Usage:
    python vortex/scripts/download_flux.py

Cache Location:
    ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell
"""

import sys
from pathlib import Path

from diffusers import FluxPipeline

# Add vortex to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    """Download Flux-Schnell model weights."""
    print("=" * 80)
    print("Flux-Schnell Model Download")
    print("=" * 80)
    print("Model: black-forest-labs/FLUX.1-schnell")
    print("Size: ~12 GB")
    print("Destination: ~/.cache/huggingface/hub/")
    print("=" * 80)
    print("\n⏳ Downloading model weights (this may take 10-30 minutes)...\n")

    try:
        # Download model (no quantization, just weights)
        _ = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            use_safetensors=True,
        )

        print("\n✅ Download complete!")
        print(f"Cache location: {Path.home() / '.cache' / 'huggingface' / 'hub'}")

        # Verify model files
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        flux_dirs = list(cache_dir.glob("models--black-forest-labs--*"))

        if flux_dirs:
            print(f"\nModel cached at: {flux_dirs[0]}")
            # Calculate size
            total_size = sum(f.stat().st_size for f in flux_dirs[0].rglob("*") if f.is_file())
            print(f"Total size: {total_size / 1e9:.2f} GB")

        print("\n" + "=" * 80)
        print("You can now run Flux-Schnell without download delays:")
        print("  python vortex/benchmarks/flux_vram_profile.py")
        print("  python vortex/benchmarks/flux_latency.py --iterations 50")
        print("=" * 80)

        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify Hugging Face access token (if required)")
        print("  3. Ensure sufficient disk space (~15 GB)")
        sys.exit(1)


if __name__ == "__main__":
    main()
