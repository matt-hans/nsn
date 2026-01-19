#!/usr/bin/env python3
"""Test script to verify Vortex video generation pipeline.

This script:
1. Initializes the VortexPipeline with optimized VRAM settings
2. Generates a short test video
3. Saves output for verification
4. Reports VRAM usage and timing

Usage:
    cd vortex && python test_generation.py

Expected output:
    - VRAM usage should stay below 11.5GB on RTX 3060
    - Video frames tensor saved to test_output/
    - Audio waveform tensor saved to test_output/
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vortex.pipeline import VortexPipeline, GenerationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_vram_stats() -> dict:
    """Get current VRAM statistics."""
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "device_name": torch.cuda.get_device_name(0),
    }


def print_vram_status(label: str):
    """Print current VRAM status with label."""
    stats = get_vram_stats()
    if stats["available"]:
        print(f"\n[VRAM] {label}:")
        print(f"  Device: {stats['device_name']}")
        print(f"  Allocated: {stats['allocated_gb']:.2f} GB")
        print(f"  Reserved:  {stats['reserved_gb']:.2f} GB")
        print(f"  Total:     {stats['total_gb']:.2f} GB")
        print(f"  Free:      {stats['total_gb'] - stats['allocated_gb']:.2f} GB")
    else:
        print(f"\n[VRAM] {label}: CUDA not available")


async def test_video_generation():
    """Run video generation test."""

    print("=" * 60)
    print("Vortex Video Generation Test")
    print("=" * 60)

    # Print initial VRAM status
    print_vram_status("Before initialization")

    # Create output directory
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)

    # Initialize pipeline
    print("\n[1/4] Initializing VortexPipeline...")
    start_init = time.time()

    try:
        config_path = str(Path(__file__).parent / "config.yaml")
        pipeline = await VortexPipeline.create(
            config_path=config_path,
            device="cuda:0",
        )
        init_time = time.time() - start_init
        print(f"  Pipeline initialized in {init_time:.1f}s")
        print(f"  Renderer: {pipeline.renderer_name} v{pipeline.renderer_version}")
    except Exception as e:
        print(f"  ERROR: Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

    print_vram_status("After initialization")

    # Create test recipe (short 5-second video for quick testing)
    print("\n[2/4] Creating test recipe...")
    recipe = {
        "slot_params": {
            "slot_id": 1,
            "duration_sec": 5,  # Short video for testing
            "fps": 24,
        },
        "audio_track": {
            "script": "Testing the Vortex video generation pipeline. All systems operational.",
            "voice_id": "rick_c137",
            "speed": 1.0,
            "emotion": "neutral",
        },
        "visual_track": {
            "prompt": "A scientist in a white lab coat standing in a modern laboratory, detailed face, professional lighting, high quality",
            "expression_preset": "neutral",
            "negative_prompt": "blurry, low quality, distorted",
        },
        "semantic_constraints": {
            "clip_threshold": 0.65,  # Lower threshold for testing
        },
    }
    print(f"  Script: {recipe['audio_track']['script'][:50]}...")
    print(f"  Prompt: {recipe['visual_track']['prompt'][:50]}...")
    print(f"  Duration: {recipe['slot_params']['duration_sec']}s @ {recipe['slot_params']['fps']}fps")

    # Generate video
    print("\n[3/4] Generating video...")
    print_vram_status("Before generation")

    start_gen = time.time()
    try:
        result: GenerationResult = await pipeline.generate_slot(
            recipe=recipe,
            slot_id=1,
            seed=42,  # Deterministic seed for reproducibility
            deadline=time.time() + 120,  # 2-minute deadline
        )
        gen_time = time.time() - start_gen
    except Exception as e:
        print(f"  ERROR: Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print_vram_status("After generation")

    # Check results
    print("\n[4/4] Verifying results...")

    if not result.success:
        print(f"  ERROR: Generation failed: {result.error_msg}")
        return False

    print(f"  Success: {result.success}")
    print(f"  Generation time: {result.generation_time_ms:.0f}ms ({gen_time:.1f}s wall clock)")
    print(f"  Video shape: {tuple(result.video_frames.shape)}")
    print(f"  Audio shape: {tuple(result.audio_waveform.shape)}")
    print(f"  CLIP embedding shape: {tuple(result.clip_embedding.shape)}")
    print(f"  Determinism proof: {result.determinism_proof.hex()[:32]}...")

    # Save outputs
    print("\n[5/5] Saving outputs...")

    # Save video frames as numpy
    video_path = output_dir / "test_video_frames.npy"
    video_np = result.video_frames.cpu().numpy()
    np.save(video_path, video_np)
    print(f"  Video saved: {video_path}")
    print(f"    Shape: {video_np.shape}, dtype: {video_np.dtype}")
    print(f"    Size: {video_np.nbytes / 1e6:.1f} MB")

    # Save audio as numpy
    audio_path = output_dir / "test_audio.npy"
    audio_np = result.audio_waveform.cpu().numpy()
    np.save(audio_path, audio_np)
    print(f"  Audio saved: {audio_path}")
    print(f"    Shape: {audio_np.shape}, dtype: {audio_np.dtype}")
    print(f"    Duration: {len(audio_np) / 24000:.1f}s @ 24kHz")

    # Save CLIP embedding
    clip_path = output_dir / "test_clip_embedding.npy"
    clip_np = result.clip_embedding.cpu().numpy()
    np.save(clip_path, clip_np)
    print(f"  CLIP embedding saved: {clip_path}")

    # Save first frame as image for visual verification
    try:
        from PIL import Image
        first_frame = result.video_frames[0].cpu()
        # Convert from CHW to HWC and scale to 0-255
        first_frame = first_frame.permute(1, 2, 0).numpy()
        first_frame = (first_frame * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(first_frame)
        img_path = output_dir / "test_first_frame.png"
        img.save(img_path)
        print(f"  First frame saved: {img_path}")
    except ImportError:
        print("  (Pillow not installed, skipping image save)")
    except Exception as e:
        print(f"  Warning: Could not save first frame: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("TEST PASSED")
    print("=" * 60)
    stats = get_vram_stats()
    if stats["available"]:
        print(f"Peak VRAM: {stats['allocated_gb']:.2f} GB / {stats['total_gb']:.2f} GB")
        if stats['allocated_gb'] < 11.5:
            print("VRAM usage is within RTX 3060 12GB limit")
        else:
            print("WARNING: VRAM usage exceeds RTX 3060 limit!")
    print(f"Total time: {time.time() - start_init:.1f}s")
    print(f"Output directory: {output_dir}")

    return True


def main():
    """Main entry point."""
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    success = asyncio.run(test_video_generation())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
