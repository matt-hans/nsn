#!/usr/bin/env python
"""Visual quality check for Flux-Schnell generation.

Generates a test image from a prompt and saves it for manual inspection.
Useful for verifying visual quality after model updates or configuration changes.

Usage:
    python vortex/scripts/visual_check_flux.py --prompt "scientist" --output /tmp/flux_test.png
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

# Add vortex to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vortex.models.flux import load_flux_schnell


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image.

    Args:
        tensor: Tensor of shape (3, H, W) in range [0, 1]

    Returns:
        PIL Image in RGB format
    """
    # Convert to [0, 255] and HWC format
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def main():
    """Generate and save test image."""
    parser = argparse.ArgumentParser(description="Visual quality check for Flux-Schnell")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a scientist in a laboratory",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, watermark",
        help="Negative prompt for quality control",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/flux_visual_check.png",
        help="Output path for generated image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic generation",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for visual check")
        sys.exit(1)

    device = "cuda:0"
    print("=" * 80)
    print("Flux-Schnell Visual Quality Check")
    print("=" * 80)
    print(f"Prompt: {args.prompt}")
    print(f"Negative: {args.negative_prompt}")
    print(f"Output: {args.output}")
    if args.seed:
        print(f"Seed: {args.seed}")
    print("=" * 80)

    # Load model
    print("\n⏳ Loading Flux-Schnell model...")
    flux_model = load_flux_schnell(device=device, quantization="nf4")
    print("✅ Model loaded\n")

    # Generate image
    print("🎨 Generating image...")
    result = flux_model.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        seed=args.seed,
    )

    # Convert to PIL and save
    image = tensor_to_pil(result)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print(f"✅ Image saved to: {output_path}")
    print(f"   Size: {image.size}")
    print("\nOpen image for manual inspection:")
    print(f"  open {output_path}")


if __name__ == "__main__":
    main()
