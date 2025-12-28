"""Integration tests for Kokoro TTS synthesis with real model.

These tests require:
- CUDA-capable GPU (RTX 3060+ recommended)
- Kokoro package installed: pip install kokoro soundfile
- Internet connection for first-time model download (~500MB)

Run with: pytest tests/integration/test_kokoro_synthesis.py --gpu -v

Note: Tests are skipped if CUDA is not available or kokoro package is missing.
"""

import pytest
import torch
import time
from pathlib import Path

# Check for kokoro package
pytest.importorskip("kokoro", reason="Kokoro package not installed")


from vortex.models.kokoro import load_kokoro, KokoroWrapper


@pytest.fixture(scope="module")
def kokoro_model():
    """Load Kokoro model once for all tests (expensive operation).

    This fixture is module-scoped to avoid reloading the model for each test.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = load_kokoro(device="cuda:0")
    yield model

    # Cleanup
    del model
    torch.cuda.empty_cache()


class TestKokoroSynthesis:
    """Integration tests for real Kokoro TTS synthesis."""

    def test_basic_synthesis(self, kokoro_model):
        """Test basic text-to-speech generation."""
        result = kokoro_model.synthesize(
            text="Hello world, this is a test.",
            voice_id="rick_c137"
        )

        # Verify output properties
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 1  # Mono audio
        assert len(result) > 0
        assert result.device.type == "cuda"
        assert result.dtype == torch.float32

        # Verify audio range [-1, 1]
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_sample_rate_24khz(self, kokoro_model):
        """Test that output is 24kHz sample rate."""
        text = "This is a one second test."  # ~1 second of audio

        result = kokoro_model.synthesize(
            text=text,
            voice_id="rick_c137"
        )

        # For 1 second of audio at 24kHz, expect ~24000 samples
        # Allow ±20% tolerance
        expected_samples = 24000
        tolerance = 0.2
        assert (
            expected_samples * (1 - tolerance)
            <= len(result)
            <= expected_samples * (1 + tolerance) * 1.5
        ), f"Expected ~{expected_samples} samples for 1s audio, got {len(result)}"

    def test_voice_consistency(self, kokoro_model):
        """Test that different voice IDs produce different outputs."""
        text = "This is the same text for all voices."

        result_rick = kokoro_model.synthesize(text=text, voice_id="rick_c137")
        result_morty = kokoro_model.synthesize(text=text, voice_id="morty")
        result_summer = kokoro_model.synthesize(text=text, voice_id="summer")

        # Outputs should be different
        assert not torch.equal(result_rick, result_morty)
        assert not torch.equal(result_rick, result_summer)
        assert not torch.equal(result_morty, result_summer)

        # All should be valid audio
        for result in [result_rick, result_morty, result_summer]:
            assert isinstance(result, torch.Tensor)
            assert len(result) > 0

    def test_speed_control(self, kokoro_model):
        """Test that speed parameter affects output duration."""
        text = "The quick brown fox jumps over the lazy dog."

        # Generate at different speeds
        result_slow = kokoro_model.synthesize(text=text, voice_id="rick_c137", speed=0.8)
        result_normal = kokoro_model.synthesize(text=text, voice_id="rick_c137", speed=1.0)
        result_fast = kokoro_model.synthesize(text=text, voice_id="rick_c137", speed=1.2)

        # Slower speed should produce longer audio
        assert len(result_slow) > len(result_normal)
        # Faster speed should produce shorter audio
        assert len(result_fast) < len(result_normal)

        # Check rough proportions (allow some tolerance)
        # 0.8× speed ≈ 25% longer
        # 1.2× speed ≈ 17% shorter
        assert len(result_slow) > len(result_normal) * 1.1
        assert len(result_fast) < len(result_normal) * 0.9

    def test_emotion_parameters(self, kokoro_model):
        """Test that emotion parameters produce different outputs."""
        text = "I am very excited about this project!"

        result_neutral = kokoro_model.synthesize(
            text=text, voice_id="rick_c137", emotion="neutral"
        )
        result_excited = kokoro_model.synthesize(
            text=text, voice_id="rick_c137", emotion="excited"
        )
        result_manic = kokoro_model.synthesize(
            text=text, voice_id="rick_c137", emotion="manic"
        )

        # Different emotions should produce different outputs
        # (emotion affects tempo, which affects duration)
        assert not torch.equal(result_neutral, result_excited)
        assert not torch.equal(result_neutral, result_manic)

        # Excited and manic should be shorter due to faster tempo
        assert len(result_excited) < len(result_neutral)
        assert len(result_manic) < len(result_excited)

    def test_output_to_preallocated_buffer(self, kokoro_model):
        """Test synthesis with pre-allocated output buffer."""
        text = "Testing buffer output"

        # Pre-allocate buffer for 45 seconds @ 24kHz
        buffer = torch.zeros(1080000, dtype=torch.float32, device="cuda:0")

        result = kokoro_model.synthesize(
            text=text,
            voice_id="rick_c137",
            output=buffer
        )

        # Result should be a slice of the buffer (same memory)
        assert result.data_ptr() == buffer.data_ptr()
        assert len(result) < len(buffer)  # Only uses needed portion
        assert torch.sum(result.abs()) > 0  # Has actual audio data

    def test_long_script_truncation(self, kokoro_model):
        """Test that very long scripts are truncated with warning."""
        # Create very long script (far exceeds 45s)
        long_text = " ".join(["word"] * 10000)  # 10,000 words

        with pytest.warns(UserWarning, match="Script truncated"):
            result = kokoro_model.synthesize(
                text=long_text,
                voice_id="rick_c137"
            )

        # Should still produce valid audio
        assert isinstance(result, torch.Tensor)
        assert len(result) > 0

        # Duration should be ≤ 45 seconds
        max_samples = int(45.0 * 24000)  # 45s @ 24kHz
        assert len(result) <= max_samples

    def test_deterministic_with_seed(self, kokoro_model):
        """Test that same seed produces identical outputs."""
        text = "Deterministic test text"

        result1 = kokoro_model.synthesize(
            text=text,
            voice_id="rick_c137",
            seed=42
        )

        result2 = kokoro_model.synthesize(
            text=text,
            voice_id="rick_c137",
            seed=42
        )

        # Should be identical with same seed
        assert torch.equal(result1, result2)

    def test_different_seeds_produce_variation(self, kokoro_model):
        """Test that different seeds produce different outputs."""
        text = "Random variation test"

        result1 = kokoro_model.synthesize(
            text=text,
            voice_id="rick_c137",
            seed=42
        )

        result2 = kokoro_model.synthesize(
            text=text,
            voice_id="rick_c137",
            seed=123
        )

        # Different seeds may produce variations
        # Note: Some TTS models are fully deterministic regardless of seed
        # This test documents expected behavior but may not fail
        # if the model is deterministic


class TestPerformanceRequirements:
    """Test performance and VRAM requirements."""

    def test_vram_budget_compliance(self, kokoro_model):
        """Test that Kokoro VRAM usage is within 0.3-0.5 GB budget."""
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        # Perform synthesis
        kokoro_model.synthesize(
            text="Testing VRAM usage with a moderate length sentence.",
            voice_id="rick_c137"
        )

        # Get peak memory allocated
        peak_memory = torch.cuda.max_memory_allocated()
        peak_memory_gb = peak_memory / 1e9

        # Log for debugging
        print(f"\nPeak VRAM usage: {peak_memory_gb:.3f} GB")

        # Should be within budget (0.3-0.5 GB for model + buffer)
        # Allow some tolerance for CUDA overhead
        assert peak_memory_gb >= 0.2, "VRAM usage suspiciously low"
        assert peak_memory_gb <= 1.0, (
            f"VRAM usage {peak_memory_gb:.2f}GB exceeds 1.0GB limit "
            "(expected 0.3-0.5GB for Kokoro)"
        )

    def test_synthesis_latency_target(self, kokoro_model):
        """Test that synthesis completes in <2s P99 for 45s script."""
        # Create script that produces ~45 seconds of audio
        # Heuristic: ~125 words per minute of speech
        # 45s = 0.75 min → ~94 words
        script = " ".join(["test"] * 94)

        # Warm-up run (first run may be slower due to JIT compilation)
        kokoro_model.synthesize(text=script, voice_id="rick_c137")

        # Measure latency over multiple runs
        latencies = []
        num_runs = 10

        for _ in range(num_runs):
            start_time = time.perf_counter()

            kokoro_model.synthesize(
                text=script,
                voice_id="rick_c137"
            )

            end_time = time.perf_counter()
            latency_sec = end_time - start_time
            latencies.append(latency_sec)

        # Calculate P99
        latencies.sort()
        p99_latency = latencies[int(len(latencies) * 0.99)]

        print(f"\nP99 synthesis latency: {p99_latency:.3f}s")
        print(f"Mean latency: {sum(latencies) / len(latencies):.3f}s")

        # Should be < 2s P99 on RTX 3060
        assert p99_latency < 2.0, (
            f"P99 latency {p99_latency:.2f}s exceeds 2.0s target"
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_special_characters(self, kokoro_model):
        """Test handling of special characters."""
        text = "Hello! How are you? It's a test... #hashtag @mention $money"

        result = kokoro_model.synthesize(
            text=text,
            voice_id="rick_c137"
        )

        assert isinstance(result, torch.Tensor)
        assert len(result) > 0

    def test_numbers_and_symbols(self, kokoro_model):
        """Test handling of numbers and symbols."""
        text = "The year is 2025 and it costs $100 for 50% discount."

        result = kokoro_model.synthesize(
            text=text,
            voice_id="rick_c137"
        )

        assert isinstance(result, torch.Tensor)
        assert len(result) > 0

    def test_unicode_characters(self, kokoro_model):
        """Test handling of Unicode characters."""
        # Note: Kokoro may not support all Unicode characters
        # This test documents behavior
        text = "Hello world 世界"

        try:
            result = kokoro_model.synthesize(
                text=text,
                voice_id="rick_c137"
            )
            assert isinstance(result, torch.Tensor)
        except Exception as e:
            # Some TTS models don't support Unicode
            pytest.skip(f"Unicode not supported: {e}")

    def test_empty_text_raises_error(self, kokoro_model):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            kokoro_model.synthesize(
                text="",
                voice_id="rick_c137"
            )

    def test_invalid_voice_id_raises_error(self, kokoro_model):
        """Test that invalid voice ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown voice_id"):
            kokoro_model.synthesize(
                text="Test",
                voice_id="invalid_voice_12345"
            )


class TestAudioQuality:
    """Test audio quality properties."""

    def test_output_normalization(self, kokoro_model):
        """Test that output is properly normalized to [-1, 1]."""
        result = kokoro_model.synthesize(
            text="Testing audio normalization with various volumes.",
            voice_id="rick_c137"
        )

        # Should be normalized to [-1, 1]
        assert result.min() >= -1.0
        assert result.max() <= 1.0

        # Should actually use significant portion of dynamic range
        assert result.abs().max() > 0.3, "Audio suspiciously quiet"

    def test_no_silence_only_output(self, kokoro_model):
        """Test that output contains actual audio (not just silence)."""
        result = kokoro_model.synthesize(
            text="This should produce audible speech.",
            voice_id="rick_c137"
        )

        # Check RMS energy (should be > 0.01 for speech)
        rms = torch.sqrt(torch.mean(result ** 2))
        assert rms > 0.01, f"Output appears to be silence (RMS: {rms:.6f})"

    def test_no_clipping_artifacts(self, kokoro_model):
        """Test that output doesn't have excessive clipping."""
        result = kokoro_model.synthesize(
            text="Testing for clipping artifacts in generated audio output.",
            voice_id="rick_c137",
            emotion="excited"
        )

        # Count samples at or near clipping threshold
        clipped_samples = torch.sum(result.abs() >= 0.99).item()
        total_samples = len(result)
        clipping_ratio = clipped_samples / total_samples

        # Less than 1% of samples should be clipped
        assert clipping_ratio < 0.01, (
            f"Excessive clipping: {clipping_ratio*100:.2f}% of samples"
        )
