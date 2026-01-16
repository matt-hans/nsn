"""Integration tests for LivePortrait video generation (requires GPU).

These tests verify end-to-end video generation with real LivePortrait model.
Tests are skipped if CUDA is not available.

Run with: pytest tests/integration/test_liveportrait_generation.py --gpu -v
"""

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA GPU"
)


class TestLivePortraitGeneration:
    """Integration tests for LivePortrait video generation."""

    @pytest.fixture(scope="class")
    def liveportrait_model(self):
        """Load LivePortrait model (cached for all tests)."""
        from vortex.models.liveportrait import load_liveportrait

        model = load_liveportrait(device="cuda:0", precision="fp16")
        if getattr(model, "backend_name", "") != "cli":
            pytest.skip("LivePortrait CLI backend not available in this environment")
        yield model

        # Cleanup
        del model
        torch.cuda.empty_cache()

    @pytest.fixture
    def sample_actor_image(self):
        """Generate sample 512×512 actor image."""
        return torch.rand(3, 512, 512, device="cuda:0")

    @pytest.fixture
    def sample_audio(self):
        """Generate sample 45-second audio @ 24kHz."""
        return torch.randn(int(45 * 24000), device="cuda:0")

    def test_generate_45_second_video(
        self, liveportrait_model, sample_actor_image, sample_audio
    ):
        """Test generation of standard 45-second video."""
        result = liveportrait_model.animate(
            source_image=sample_actor_image,
            driving_audio=sample_audio,
            expression_preset="neutral",
            fps=24,
            duration=45,
        )

        # Verify output shape
        assert result.shape == (1080, 3, 512, 512), f"Unexpected shape: {result.shape}"
        assert result.dtype == torch.float32
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_vram_usage_compliance(
        self, liveportrait_model, sample_actor_image, sample_audio
    ):
        """Test that VRAM usage is within 3.0-4.0GB budget."""
        torch.cuda.reset_peak_memory_stats()

        # Generate video
        liveportrait_model.animate(
            source_image=sample_actor_image,
            driving_audio=sample_audio,
            expression_preset="excited",
        )

        # Check VRAM usage
        vram_bytes = torch.cuda.max_memory_allocated()
        vram_gb = vram_bytes / 1e9

        assert (
            3.0 <= vram_gb <= 4.0
        ), f"VRAM usage {vram_gb:.2f}GB exceeds budget (3.0-4.0GB)"

    def test_generation_latency(
        self, liveportrait_model, sample_actor_image, sample_audio
    ):
        """Test that generation completes within 8s target (P99)."""
        import time

        # Warmup
        for _ in range(3):
            liveportrait_model.animate(
                source_image=sample_actor_image,
                driving_audio=sample_audio,
                expression_preset="neutral",
            )

        # Measure latency
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            liveportrait_model.animate(
                source_image=sample_actor_image,
                driving_audio=sample_audio,
                expression_preset="excited",
            )
            latency = time.perf_counter() - start
            latencies.append(latency)

        # Check P99 latency
        latencies.sort()
        p99_latency = latencies[int(0.99 * len(latencies))]

        assert (
            p99_latency < 8.0
        ), f"P99 latency {p99_latency:.2f}s exceeds 8s target"

    def test_expression_presets_produce_different_outputs(
        self, liveportrait_model, sample_actor_image, sample_audio
    ):
        """Test that different expression presets produce different outputs."""
        neutral_video = liveportrait_model.animate(
            source_image=sample_actor_image,
            driving_audio=sample_audio,
            expression_preset="neutral",
            seed=42,
        )

        excited_video = liveportrait_model.animate(
            source_image=sample_actor_image,
            driving_audio=sample_audio,
            expression_preset="excited",
            seed=42,
        )

        # Outputs should be different (not identical)
        assert not torch.equal(
            neutral_video, excited_video
        ), "Expression presets produced identical outputs"

        # But should have same structure
        assert neutral_video.shape == excited_video.shape

    def test_expression_sequence_transitions(
        self, liveportrait_model, sample_actor_image, sample_audio
    ):
        """Test smooth transitions in expression sequences."""
        result = liveportrait_model.animate(
            source_image=sample_actor_image,
            driving_audio=sample_audio,
            expression_sequence=["neutral", "excited", "manic", "calm"],
            seed=42,
        )

        # Verify output shape
        assert result.shape == (1080, 3, 512, 512)

        # Check for smooth transitions (no abrupt frame-to-frame changes)
        # Compute frame-to-frame difference
        frame_diffs = []
        for i in range(len(result) - 1):
            diff = (result[i + 1] - result[i]).abs().mean().item()
            frame_diffs.append(diff)

        # No frame should have extreme difference (threshold: 0.1)
        max_diff = max(frame_diffs)
        assert (
            max_diff < 0.1
        ), f"Abrupt transition detected: max frame diff {max_diff:.4f}"

    def test_deterministic_with_seed(
        self, liveportrait_model, sample_actor_image, sample_audio
    ):
        """Test that same seed produces identical outputs."""
        video1 = liveportrait_model.animate(
            source_image=sample_actor_image,
            driving_audio=sample_audio,
            expression_preset="neutral",
            seed=42,
        )

        video2 = liveportrait_model.animate(
            source_image=sample_actor_image,
            driving_audio=sample_audio,
            expression_preset="neutral",
            seed=42,
        )

        # Outputs should be identical
        assert torch.equal(video1, video2), "Same seed produced different outputs"

    def test_preallocated_buffer_output(
        self, liveportrait_model, sample_actor_image, sample_audio
    ):
        """Test writing to pre-allocated video buffer."""
        video_buffer = torch.zeros(1080, 3, 512, 512, device="cuda:0")

        result = liveportrait_model.animate(
            source_image=sample_actor_image,
            driving_audio=sample_audio,
            expression_preset="neutral",
            output=video_buffer,
        )

        # Should return the same buffer
        assert result.data_ptr() == video_buffer.data_ptr()
        assert (video_buffer != 0).any()  # Buffer was written to


class TestLipsyncAccuracy:
    """Integration tests for lip-sync accuracy."""

    @pytest.fixture(scope="class")
    def liveportrait_model(self):
        """Load LivePortrait model."""
        from vortex.models.liveportrait import load_liveportrait

        model = load_liveportrait(device="cuda:0", precision="fp16")
        if getattr(model, "backend_name", "") != "cli":
            pytest.skip("LivePortrait CLI backend not available in this environment")
        yield model
        del model
        torch.cuda.empty_cache()

    def test_audio_to_viseme_conversion(self):
        """Test audio-to-viseme conversion produces valid visemes."""
        from vortex.models.liveportrait import audio_to_visemes

        # Generate test audio
        test_audio = torch.randn(24000)  # 1 second @ 24kHz
        fps = 24

        visemes = audio_to_visemes(test_audio, fps, sample_rate=24000)

        # Should have 24 visemes (one per frame)
        assert len(visemes) == 24

        # Each viseme should have shape [3] and values in [0, 1]
        for viseme in visemes:
            assert viseme.shape == (3,)
            assert (viseme >= 0.0).all() and (viseme <= 1.0).all()


class TestErrorHandling:
    """Integration tests for error handling."""

    @pytest.fixture(scope="class")
    def liveportrait_model(self):
        """Load LivePortrait model."""
        from vortex.models.liveportrait import load_liveportrait

        model = load_liveportrait(device="cuda:0", precision="fp16")
        if getattr(model, "backend_name", "") != "cli":
            pytest.skip("LivePortrait CLI backend not available in this environment")
        yield model
        del model
        torch.cuda.empty_cache()

    def test_invalid_image_dimensions(self, liveportrait_model):
        """Test that invalid image dimensions raise ValueError."""
        invalid_image = torch.rand(3, 256, 256, device="cuda:0")  # Wrong size
        audio = torch.randn(int(45 * 24000), device="cuda:0")

        with pytest.raises(ValueError, match="Invalid source_image shape"):
            liveportrait_model.animate(
                source_image=invalid_image,
                driving_audio=audio,
                expression_preset="neutral",
            )

    def test_audio_truncation_warning(self, liveportrait_model, caplog):
        """Test that long audio is truncated with warning."""
        image = torch.rand(3, 512, 512, device="cuda:0")
        long_audio = torch.randn(int(60 * 24000), device="cuda:0")  # 60s

        with caplog.at_level("WARNING"):
            result = liveportrait_model.animate(
                source_image=image,
                driving_audio=long_audio,
                expression_preset="neutral",
                duration=45,
            )

        # Should still generate 1080 frames (45s @ 24fps)
        assert result.shape[0] == 1080

        # Should log truncation warning
        assert any("truncated" in record.message.lower() for record in caplog.records)


class TestVRAMProfiling:
    """VRAM profiling tests."""

    def test_liveportrait_vram_profile(self):
        """Profile VRAM usage during model loading and generation."""
        from vortex.models.liveportrait import load_liveportrait

        torch.cuda.reset_peak_memory_stats()
        initial_vram = torch.cuda.memory_allocated() / 1e9

        # Load model
        model = load_liveportrait(device="cuda:0", precision="fp16")
        if getattr(model, "backend_name", "") != "cli":
            pytest.skip("LivePortrait CLI backend not available in this environment")
        after_load_vram = torch.cuda.memory_allocated() / 1e9
        load_vram = after_load_vram - initial_vram

        print(f"\nVRAM after load: {after_load_vram:.2f}GB (delta: +{load_vram:.2f}GB)")

        # Generate video
        image = torch.rand(3, 512, 512, device="cuda:0")
        audio = torch.randn(int(45 * 24000), device="cuda:0")

        model.animate(
            source_image=image, driving_audio=audio, expression_preset="neutral"
        )

        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        generation_vram = peak_vram - after_load_vram

        print(f"VRAM during generation: {peak_vram:.2f}GB (delta: +{generation_vram:.2f}GB)")

        # Cleanup
        del model
        torch.cuda.empty_cache()

        # Verify budget compliance
        assert (
            3.0 <= peak_vram <= 4.0
        ), f"Peak VRAM {peak_vram:.2f}GB exceeds budget"
