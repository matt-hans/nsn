#!/usr/bin/env python3
"""End-to-end smoke test for ToonGen pipeline.

Usage:
    python scripts/test_e2e.py

Requires:
    - ComfyUI running on localhost:8188
    - Audio models available
    - Workflow template configured
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vortex.orchestrator import VideoOrchestrator


async def main():
    """Run E2E smoke test."""
    print("=" * 60)
    print("ToonGen E2E Smoke Test")
    print("=" * 60)

    orchestrator = VideoOrchestrator(
        template_path="templates/cartoon_workflow.json",
        assets_dir="assets",
        output_dir="outputs/test",
    )

    # Health check
    print("\n[1/3] Checking component health...")
    health = await orchestrator.health_check()
    print(f"  Orchestrator: {'OK' if health['orchestrator'] else 'FAIL'}")
    print(f"  ComfyUI: {'OK' if health['comfyui'] else 'FAIL'}")

    if not health["comfyui"]:
        print("\nERROR: ComfyUI is not running. Start it with:")
        print("  docker-compose up comfyui")
        return 1

    # Generate test video
    print("\n[2/3] Generating test video...")
    print("  Prompt: 'A cartoon cat in a lab coat'")
    print("  Script: 'This is a test of the ToonGen system.'")

    try:
        result = await orchestrator.generate(
            prompt="Medium shot, cartoon style, a friendly cat in a lab coat waving at camera, bright laboratory background",
            script="This is a test of the ToonGen system. If you can hear this, everything is working correctly.",
            engine="kokoro",  # Use Kokoro for quick test
            seed=12345,
        )
    except Exception as e:
        print(f"  ERROR: Generation failed: {e}")
        return 1

    print(f"  Video: {result['video_path']}")
    print(f"  Frames: {result['frame_count']}")

    # Verify output
    print("\n[3/3] Verifying output...")
    video_path = Path(result["video_path"])

    if not video_path.exists():
        print(f"  ERROR: Video file not found: {video_path}")
        return 1

    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")

    if size_mb < 0.1:
        print("  WARNING: Video file seems too small")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
