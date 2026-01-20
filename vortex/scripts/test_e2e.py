#!/usr/bin/env python
"""End-to-end test of Vortex video generation pipeline.

Usage:
    python vortex/scripts/test_e2e.py --recipe vortex/test.json
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add vortex to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="E2E test of Vortex pipeline")
    parser.add_argument(
        "--recipe",
        type=str,
        default=str(Path(__file__).parent.parent / "test.json"),
        help="Path to recipe JSON file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., 'cuda:0', 'cpu')",
    )
    args = parser.parse_args()

    # Load recipe
    recipe_path = Path(args.recipe)
    if not recipe_path.exists():
        logger.error(f"Recipe file not found: {recipe_path}")
        sys.exit(1)

    with open(recipe_path) as f:
        data = json.load(f)

    recipe = data.get("recipe", data)
    slot_id = data.get("slot_id", 1001)
    seed = data.get("seed", 2024)

    logger.info(f"Loaded recipe from: {recipe_path}")
    logger.info(f"  Slot ID: {slot_id}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Duration: {recipe.get('slot_params', {}).get('duration_sec', 45)}s")

    # Initialize pipeline
    from vortex.pipeline import VortexPipeline

    config_path = str(Path(__file__).parent.parent / "config.yaml")
    logger.info(f"Initializing pipeline with config: {config_path}")

    try:
        pipeline = await VortexPipeline.create(
            config_path=config_path,
            device=args.device,
        )
        logger.info(f"Pipeline initialized: {pipeline.renderer_name} v{pipeline.renderer_version}")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate slot
    logger.info("=" * 60)
    logger.info("Starting video generation...")
    logger.info("=" * 60)

    try:
        result = await pipeline.generate_slot(
            recipe=recipe,
            slot_id=slot_id,
            seed=seed,
        )
    except Exception as e:
        logger.error(f"Generation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Report results
    logger.info("=" * 60)
    logger.info("GENERATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Generation time: {result.generation_time_ms:.1f}ms")

    if result.success:
        logger.info(f"  Video frames: {result.video_frames.shape}")
        logger.info(f"  Audio samples: {result.audio_waveform.shape}")
        logger.info(f"  CLIP embedding: {result.clip_embedding.shape}")
        logger.info(f"  Determinism proof: {result.determinism_proof.hex()[:32]}...")

        # Check output files
        output_dir = Path(__file__).parent.parent / "outputs"
        if output_dir.exists():
            mp4_files = list(output_dir.glob("*.mp4"))
            wav_files = list(output_dir.glob("*.wav"))
            if mp4_files:
                latest_mp4 = max(mp4_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"  Output video: {latest_mp4}")
                logger.info(f"    Size: {latest_mp4.stat().st_size / 1e6:.2f} MB")
            if wav_files:
                latest_wav = max(wav_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"  Output audio: {latest_wav}")

        logger.info("=" * 60)
        logger.info("E2E TEST PASSED")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error(f"  Error: {result.error_msg}")
        logger.info("=" * 60)
        logger.error("E2E TEST FAILED")
        logger.info("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
