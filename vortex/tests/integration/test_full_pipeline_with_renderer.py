"""End-to-end integration test for VortexPipeline with new renderer system.

This test verifies:
1. Pipeline initialization with RendererRegistry
2. DefaultRenderer (Narrative Chain) is loaded correctly
3. Full video generation works end-to-end
4. Output meets Lane 0 constraints
"""

import asyncio
import pytest
import torch

from vortex.pipeline import VortexPipeline


@pytest.mark.asyncio
async def test_pipeline_with_default_renderer():
    """Test full pipeline with DefaultRenderer (Narrative Chain)."""
    # Create pipeline with default renderer
    pipeline = await VortexPipeline.create(device="cuda:0")

    # Verify renderer loaded
    assert pipeline.renderer is not None
    assert pipeline.renderer.manifest.name == "default-narrative-chain"
    assert pipeline.renderer.manifest.deterministic is True

    # Check Lane 0 constraints
    manifest = pipeline.renderer.manifest
    assert manifest.resources.vram_gb <= 11.5, f"VRAM {manifest.resources.vram_gb}GB exceeds Lane 0 limit"
    assert manifest.resources.max_latency_ms <= 15000, f"Latency {manifest.resources.max_latency_ms}ms exceeds Lane 0 limit"

    print(f"✓ Renderer loaded: {manifest.name} v{manifest.version}")
    print(f"✓ VRAM: {manifest.resources.vram_gb}GB / 11.5GB")
    print(f"✓ Max latency: {manifest.resources.max_latency_ms}ms / 15000ms")


@pytest.mark.asyncio
async def test_minimal_video_generation():
    """Test minimal video generation (45 seconds) through new renderer system."""
    pipeline = await VortexPipeline.create(device="cuda:0")

    # Standard 45-second recipe (matches buffer configuration)
    recipe = {
        "slot_params": {"slot_id": 1, "duration_sec": 45, "fps": 24},
        "audio_track": {
            "script": "Test.",
            "voice_id": "rick_c137",
            "speed": 1.0,
            "emotion": "neutral",
        },
        "visual_track": {
            "prompt": "scientist",
            "expression_preset": "neutral",
        },
        "semantic_constraints": {"clip_threshold": 0.70},
    }

    # Generate with seed for reproducibility
    result = await pipeline.generate_slot(recipe, slot_id=1, seed=42)

    # Verify output
    assert result.success, f"Generation failed: {result.error_msg}"
    assert result.video_frames.shape[0] == 1080, f"Expected 1080 frames (45s), got {result.video_frames.shape[0]}"
    assert result.video_frames.shape[1:] == (3, 512, 512), "Invalid frame dimensions"
    assert result.audio_waveform.numel() > 0, "Empty audio waveform"
    assert result.clip_embedding.numel() > 0, "Empty CLIP embedding"
    assert len(result.determinism_proof) == 32, "Invalid determinism proof"

    print(f"✓ Generated {result.video_frames.shape[0]} frames in {result.generation_time_ms:.0f}ms")
    print(f"✓ Video shape: {tuple(result.video_frames.shape)}")
    print(f"✓ Audio samples: {result.audio_waveform.numel()}")
    print(f"✓ CLIP embedding: {result.clip_embedding.shape}")
    print(f"✓ Determinism proof: {result.determinism_proof.hex()[:16]}...")


@pytest.mark.asyncio
async def test_deterministic_generation_through_pipeline():
    """Test deterministic generation through full pipeline."""
    pipeline = await VortexPipeline.create(device="cuda:0")

    recipe = {
        "slot_params": {"slot_id": 1, "duration_sec": 45, "fps": 24},
        "audio_track": {"script": "Test determinism.", "voice_id": "rick_c137"},
        "visual_track": {"prompt": "scientist"},
    }

    seed = 42

    # Generate twice with same seed
    result1 = await pipeline.generate_slot(recipe.copy(), slot_id=1, seed=seed)
    result2 = await pipeline.generate_slot(recipe.copy(), slot_id=2, seed=seed)

    assert result1.success and result2.success

    # Verify determinism proofs match
    assert result1.determinism_proof == result2.determinism_proof, (
        f"Determinism proofs differ: {result1.determinism_proof.hex()[:16]} != {result2.determinism_proof.hex()[:16]}"
    )

    print(f"✓ Deterministic generation verified")
    print(f"✓ Both proofs match: {result1.determinism_proof.hex()[:16]}...")


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_pipeline_with_default_renderer())
    asyncio.run(test_minimal_video_generation())
    asyncio.run(test_deterministic_generation_through_pipeline())
    print("\n✓ All end-to-end tests passed!")
