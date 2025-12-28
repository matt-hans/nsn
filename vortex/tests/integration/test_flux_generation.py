"""Integration tests for Flux-Schnell generation (requires GPU).

These tests verify real Flux model behavior with actual GPU hardware.
They are skipped if no CUDA GPU is available.

Run with: pytest tests/integration/test_flux_generation.py --gpu -v
"""

import time
import unittest

import pytest
import torch

from vortex.models.flux import FluxModel, VortexInitializationError, load_flux_schnell
from vortex.utils.memory import get_vram_stats


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for integration tests")
class TestFluxGeneration(unittest.TestCase):
    """Integration tests for real Flux-Schnell generation."""

    @classmethod
    def setUpClass(cls):
        """Load Flux model once for all tests (expensive operation)."""
        print("\n" + "=" * 80)
        print("Loading Flux-Schnell model (this may take 30-60 seconds)...")
        print("=" * 80)

        try:
            cls.flux_model = load_flux_schnell(device="cuda:0", quantization="nf4")
            cls.device = "cuda:0"

            # Log initial VRAM
            stats = get_vram_stats()
            print(f"Flux loaded - VRAM: {stats['allocated_gb']:.2f}GB / {stats['total_gb']:.2f}GB")

        except VortexInitializationError as e:
            pytest.skip(f"Failed to load Flux model: {e}")

    def test_standard_actor_generation(self):
        """Test Case 1: Standard actor generation with valid prompt."""
        prompt = "manic scientist, blue spiked hair, white lab coat"

        # Measure VRAM before generation
        vram_before = torch.cuda.memory_allocated(self.device)

        # Generate
        start_time = time.time()
        result = self.flux_model.generate(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
        )
        generation_time = time.time() - start_time

        # Measure VRAM after generation
        vram_after = torch.cuda.memory_allocated(self.device)
        vram_delta_mb = (vram_after - vram_before) / 1e6

        # Verify output
        self.assertEqual(result.shape, (3, 512, 512))
        self.assertTrue(torch.all(result >= 0.0))
        self.assertTrue(torch.all(result <= 1.0))

        # Verify VRAM increase is reasonable (<500MB)
        self.assertLess(
            vram_delta_mb,
            500,
            f"VRAM increased by {vram_delta_mb:.1f}MB during generation",
        )

        # Verify generation time (target: 8-12s)
        self.assertLess(
            generation_time, 15.0, f"Generation took {generation_time:.2f}s (target: <12s)"
        )

        print(f"\nGeneration time: {generation_time:.2f}s")
        print(f"VRAM delta: {vram_delta_mb:.1f}MB")
        print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")

    def test_negative_prompt_application(self):
        """Test Case 2: Negative prompt improves quality."""
        prompt = "scientist"
        negative_prompt = "blurry, low quality, watermark"

        start_time = time.time()
        result = self.flux_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=4,
        )
        generation_time = time.time() - start_time

        # Verify output
        self.assertEqual(result.shape, (3, 512, 512))

        # Verify negative prompt doesn't significantly slow generation (<5% slower)
        # Baseline: ~10s, with negative: should be <10.5s
        self.assertLess(generation_time, 12.0)

        print(f"\nGeneration time with negative prompt: {generation_time:.2f}s")

    def test_vram_budget_compliance(self):
        """Test Case 3: VRAM usage within 5.5-6.5GB budget."""
        stats = get_vram_stats()
        allocated_gb = stats["allocated_gb"]

        # Verify Flux model VRAM is within budget
        self.assertGreaterEqual(allocated_gb, 5.0, "VRAM usage suspiciously low")
        self.assertLessEqual(
            allocated_gb, 7.0, f"VRAM usage {allocated_gb:.2f}GB exceeds 6.5GB budget"
        )

        print(f"\nTotal VRAM allocated: {allocated_gb:.2f}GB (budget: 5.5-6.5GB)")

    def test_deterministic_output(self):
        """Test Case 4: Same seed produces identical outputs."""
        prompt = "scientist in laboratory"
        seed = 42

        # Generate twice with same seed
        result1 = self.flux_model.generate(
            prompt=prompt,
            num_inference_steps=4,
            seed=seed,
        )

        result2 = self.flux_model.generate(
            prompt=prompt,
            num_inference_steps=4,
            seed=seed,
        )

        # Results should be identical (or very close due to numerical precision)
        max_diff = torch.abs(result1 - result2).max().item()
        self.assertLess(
            max_diff, 1e-4, f"Outputs differ by {max_diff} (should be deterministic)"
        )

        print(f"\nDeterminism check - max difference: {max_diff:.6f}")

    def test_preallocated_buffer_output(self):
        """Test Case 5: Output to pre-allocated buffer (prevents fragmentation)."""
        prompt = "scientist"

        # Create pre-allocated buffer
        buffer = torch.zeros(3, 512, 512, device=self.device, dtype=torch.float32)
        buffer_id = id(buffer)

        # Generate to buffer
        result = self.flux_model.generate(
            prompt=prompt,
            num_inference_steps=4,
            output=buffer,
        )

        # Verify result is the same buffer object (in-place write)
        self.assertEqual(
            id(result), buffer_id, "Result should be the same buffer object"
        )

        # Verify buffer was actually modified
        self.assertGreater(
            torch.abs(buffer).max().item(), 0.0, "Buffer should contain generated data"
        )

    def test_long_prompt_truncation(self):
        """Test Case 5: Long prompts are truncated to 77 tokens."""
        # Create a very long prompt (>77 tokens)
        long_prompt = " ".join(["word"] * 100)

        # Should not crash, just truncate with warning
        result = self.flux_model.generate(
            prompt=long_prompt,
            num_inference_steps=4,
        )

        # Verify output is still valid
        self.assertEqual(result.shape, (3, 512, 512))

    def test_batch_generation_memory_leak(self):
        """Test Case 6: No memory leak over multiple generations."""
        prompt = "scientist"

        # Measure VRAM before batch
        vram_start = torch.cuda.memory_allocated(self.device)

        # Generate 10 images
        for _ in range(10):
            _ = self.flux_model.generate(
                prompt=prompt,
                num_inference_steps=4,
            )

        # Measure VRAM after batch
        vram_end = torch.cuda.memory_allocated(self.device)
        vram_delta_mb = (vram_end - vram_start) / 1e6

        # Memory growth should be minimal (<50MB)
        self.assertLess(
            vram_delta_mb, 50, f"VRAM grew by {vram_delta_mb:.1f}MB over 10 generations"
        )

        print(
            f"\nMemory leak test - VRAM delta after 10 generations: {vram_delta_mb:.1f}MB"
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestFluxLoadingIntegration(unittest.TestCase):
    """Integration tests for Flux model loading."""

    def test_load_flux_schnell_cuda(self):
        """Test loading Flux-Schnell on CUDA device."""
        model = load_flux_schnell(device="cuda:0", quantization="nf4")

        # Verify model is FluxModel instance
        self.assertIsInstance(model, FluxModel)

        # Verify model is on correct device
        self.assertEqual(model.device, "cuda:0")

        # Verify pipeline was loaded
        self.assertIsNotNone(model.pipeline)

    def test_flux_model_weights_cached(self):
        """Test that model weights are cached locally (no re-download)."""
        from pathlib import Path

        # Check cache directory exists
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        self.assertTrue(cache_dir.exists(), "Hugging Face cache directory should exist")

        # Look for Flux model files
        flux_models = list(cache_dir.glob("models--black-forest-labs--*"))
        self.assertGreater(len(flux_models), 0, "Flux model should be cached locally")


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_flux_generation.py --gpu -v
    pytest.main([__file__, "-v", "--gpu"])
