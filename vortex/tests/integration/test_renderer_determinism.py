"""Integration tests for renderer determinism verification.

These tests verify that:
1. Same recipe + seed produces identical output (byte-for-byte)
2. Determinism proof is computed correctly
3. Different seeds produce different outputs
"""

import hashlib

import pytest
import torch

from vortex.renderers import RenderResult
from vortex.renderers.default import DefaultRenderer


def compute_tensor_hash(tensor: torch.Tensor) -> bytes:
    """Compute SHA256 hash of tensor bytes."""
    if tensor.numel() == 0:
        return b""
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).digest()


@pytest.fixture
def mock_config():
    """Configuration for testing (minimal but valid sizes)."""
    return {
        "device": {"name": "cpu"},
        "vram": {"soft_limit_gb": 11.0, "hard_limit_gb": 11.5},
        "models": {
            "precision": {
                "flux": "nf4",
                "cogvideox": "int8",
                "kokoro": "fp32",
                "clip": "fp16",
            }
        },
        "buffers": {
            # Must be 512x512 for CogVideoX compatibility
            "actor": {"height": 512, "width": 512, "channels": 3},
            "video": {"frames": 24, "height": 512, "width": 512, "channels": 3},
            "audio": {"samples": 24000, "sample_rate": 24000, "duration_sec": 1},
        },
    }


@pytest.fixture
def sample_recipe():
    """Sample recipe for testing (ToonGen schema)."""
    return {
        "slot_params": {"slot_id": 1, "fps": 24},
        "audio_track": {
            "script": "Test script for determinism verification.",
            "engine": "auto",
            "voice_id": "af_heart",
        },
        "visual_track": {
            "prompt": "scientist in lab coat explaining",
            "negative_prompt": "blurry, low quality, distorted face",
        },
    }


class TestDeterminismVerification:
    """Tests for verifying deterministic output."""

    @pytest.mark.asyncio
    async def test_same_seed_produces_identical_output(self, mock_config, sample_recipe):
        """Test that same recipe + seed produces identical output."""
        renderer = DefaultRenderer()
        await renderer.initialize("cpu", mock_config)

        seed = 42
        deadline = float("inf")  # No deadline for testing

        # First render
        result1 = await renderer.render(
            recipe=sample_recipe.copy(),
            slot_id=1,
            seed=seed,
            deadline=deadline,
        )

        # Second render with same seed
        result2 = await renderer.render(
            recipe=sample_recipe.copy(),
            slot_id=2,
            seed=seed,
            deadline=deadline,
        )

        # Verify both succeeded
        assert result1.success, f"First render failed: {result1.error_msg}"
        assert result2.success, f"Second render failed: {result2.error_msg}"

        # Verify determinism proofs match
        assert result1.determinism_proof == result2.determinism_proof, (
            f"Determinism proofs don't match: "
            f"{result1.determinism_proof.hex()[:16]} vs {result2.determinism_proof.hex()[:16]}"
        )

    @pytest.mark.asyncio
    async def test_different_seeds_produce_different_proofs(
        self, mock_config, sample_recipe
    ):
        """Test that different seeds produce different determinism proofs."""
        renderer = DefaultRenderer()
        await renderer.initialize("cpu", mock_config)

        deadline = float("inf")

        # Render with seed 42
        result1 = await renderer.render(
            recipe=sample_recipe.copy(),
            slot_id=1,
            seed=42,
            deadline=deadline,
        )

        # Render with seed 123
        result2 = await renderer.render(
            recipe=sample_recipe.copy(),
            slot_id=2,
            seed=123,
            deadline=deadline,
        )

        assert result1.success and result2.success

        # Proofs should be different (different seeds)
        assert result1.determinism_proof != result2.determinism_proof, (
            "Different seeds should produce different proofs"
        )

    @pytest.mark.asyncio
    async def test_determinism_proof_contains_recipe_and_seed(
        self, mock_config, sample_recipe
    ):
        """Test that determinism proof includes recipe and seed."""
        from vortex.renderers import merge_with_defaults

        renderer = DefaultRenderer()
        await renderer.initialize("cpu", mock_config)

        seed = 42
        # Use the merged recipe (same as renderer uses internally)
        merged_recipe = merge_with_defaults(sample_recipe.copy())

        result = await renderer.render(
            recipe=sample_recipe.copy(),
            slot_id=1,
            seed=seed,
            deadline=float("inf"),
        )

        assert result.success

        # Verify proof is non-empty
        assert len(result.determinism_proof) == 32, (
            f"Proof should be 32 bytes (SHA256), got {len(result.determinism_proof)}"
        )

        # Recompute proof using merged recipe (same format renderer uses)
        recomputed = renderer.compute_determinism_proof(merged_recipe, seed, result)
        assert result.determinism_proof == recomputed, (
            "Recomputed proof should match stored proof"
        )


class TestDeterminismProofComputation:
    """Tests for compute_determinism_proof method."""

    def test_proof_is_sha256(self):
        """Test that proof is a valid SHA256 hash (32 bytes)."""
        renderer = DefaultRenderer()

        recipe = {"slot_params": {"slot_id": 1, "fps": 24}}
        seed = 42
        result = RenderResult(
            video_frames=torch.randn(10, 3, 64, 64),
            audio_waveform=torch.randn(24000),
            clip_embedding=torch.randn(512),
            generation_time_ms=1000.0,
            determinism_proof=b"",
            success=True,
        )

        proof = renderer.compute_determinism_proof(recipe, seed, result)
        assert len(proof) == 32, "SHA256 produces 32 bytes"

    def test_proof_changes_with_recipe(self):
        """Test that proof changes when recipe changes."""
        renderer = DefaultRenderer()

        recipe1 = {"slot_params": {"slot_id": 1, "fps": 24}}
        recipe2 = {"slot_params": {"slot_id": 1, "fps": 30}}  # Different
        seed = 42
        result = RenderResult(
            video_frames=torch.randn(10, 3, 64, 64),
            audio_waveform=torch.randn(24000),
            clip_embedding=torch.randn(512),
            generation_time_ms=1000.0,
            determinism_proof=b"",
            success=True,
        )

        proof1 = renderer.compute_determinism_proof(recipe1, seed, result)
        proof2 = renderer.compute_determinism_proof(recipe2, seed, result)

        assert proof1 != proof2, "Different recipes should produce different proofs"

    def test_proof_changes_with_seed(self):
        """Test that proof changes when seed changes."""
        renderer = DefaultRenderer()

        recipe = {"slot_params": {"slot_id": 1, "fps": 24}}
        result = RenderResult(
            video_frames=torch.randn(10, 3, 64, 64),
            audio_waveform=torch.randn(24000),
            clip_embedding=torch.randn(512),
            generation_time_ms=1000.0,
            determinism_proof=b"",
            success=True,
        )

        proof1 = renderer.compute_determinism_proof(recipe, seed=42, result=result)
        proof2 = renderer.compute_determinism_proof(recipe, seed=123, result=result)

        assert proof1 != proof2, "Different seeds should produce different proofs"

    def test_proof_changes_with_output(self):
        """Test that proof changes when output changes."""
        renderer = DefaultRenderer()

        recipe = {"slot_params": {"slot_id": 1, "fps": 24}}
        seed = 42

        result1 = RenderResult(
            video_frames=torch.zeros(10, 3, 64, 64),
            audio_waveform=torch.zeros(24000),
            clip_embedding=torch.randn(512),
            generation_time_ms=1000.0,
            determinism_proof=b"",
            success=True,
        )

        result2 = RenderResult(
            video_frames=torch.ones(10, 3, 64, 64),  # Different
            audio_waveform=torch.zeros(24000),
            clip_embedding=torch.randn(512),
            generation_time_ms=1000.0,
            determinism_proof=b"",
            success=True,
        )

        proof1 = renderer.compute_determinism_proof(recipe, seed, result1)
        proof2 = renderer.compute_determinism_proof(recipe, seed, result2)

        assert proof1 != proof2, "Different outputs should produce different proofs"
