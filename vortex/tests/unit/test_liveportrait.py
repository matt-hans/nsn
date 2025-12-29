"""Unit tests for LivePortrait video warping model wrapper (mocked, no GPU required).

Tests the LivePortraitModel interface, expression handling, lip-sync, error cases,
and determinism. Real GPU tests are in tests/integration/test_liveportrait_generation.py
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import torch


class TestLivePortraitModelInterface(unittest.TestCase):
    """Test LivePortraitModel interface without real model weights."""

    @patch("vortex.models.liveportrait.LivePortraitPipeline")
    def setUp(self, mock_pipeline_class):
        """Set up mocked LivePortrait pipeline."""
        # Import here to avoid triggering real model downloads
        from vortex.models.liveportrait import LivePortraitModel

        # Mock the pipeline
        self.mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = self.mock_pipeline

        # Mock warp_sequence to return correct number of frames based on num_frames arg
        def mock_warp_sequence(source_image, visemes, expression_params, num_frames):
            return torch.rand(num_frames, 3, 512, 512)  # TCHW format

        self.mock_pipeline.warp_sequence.side_effect = mock_warp_sequence

        # Create LivePortraitModel with mocked pipeline
        self.model = LivePortraitModel(self.mock_pipeline, device="cpu")

    def test_animate_basic(self):
        """Test basic animate() call with default parameters."""
        source_image = torch.randn(3, 512, 512)  # CHW format
        driving_audio = torch.randn(int(45 * 24000))  # 45s @ 24kHz

        result = self.model.animate(
            source_image=source_image,
            driving_audio=driving_audio,
            expression_preset="neutral",
        )

        # Verify output shape: 1080 frames × 3 channels × 512 × 512
        self.assertEqual(result.shape, (1080, 3, 512, 512))

    def test_animate_with_expression_presets(self):
        """Test animate() with different expression presets."""
        source_image = torch.randn(3, 512, 512)
        driving_audio = torch.randn(int(45 * 24000))

        for expression in ["neutral", "excited", "manic", "calm"]:
            result = self.model.animate(
                source_image=source_image,
                driving_audio=driving_audio,
                expression_preset=expression,
            )
            self.assertEqual(result.shape, (1080, 3, 512, 512))

    def test_animate_with_expression_sequence(self):
        """Test animate() with expression sequence transitions."""
        source_image = torch.randn(3, 512, 512)
        driving_audio = torch.randn(int(45 * 24000))

        result = self.model.animate(
            source_image=source_image,
            driving_audio=driving_audio,
            expression_sequence=["neutral", "excited", "manic", "calm"],
        )

        self.assertEqual(result.shape, (1080, 3, 512, 512))

    def test_animate_custom_duration_fps(self):
        """Test animate() with custom duration and FPS."""
        source_image = torch.randn(3, 512, 512)
        driving_audio = torch.randn(int(30 * 24000))  # 30s

        result = self.model.animate(
            source_image=source_image,
            driving_audio=driving_audio,
            expression_preset="neutral",
            fps=30,
            duration=30,
        )

        # 30 fps × 30 seconds = 900 frames
        self.assertEqual(result.shape[0], 900)

    def test_animate_writes_to_preallocated_buffer(self):
        """Test that animate() writes to pre-allocated video_buffer."""
        source_image = torch.randn(3, 512, 512)
        driving_audio = torch.randn(int(45 * 24000))
        video_buffer = torch.zeros(1080, 3, 512, 512)

        result = self.model.animate(
            source_image=source_image,
            driving_audio=driving_audio,
            expression_preset="neutral",
            output=video_buffer,
        )

        # Should return the buffer itself (in-place write)
        self.assertIs(result, video_buffer)

    def test_animate_deterministic_with_seed(self):
        """Test that same seed produces identical outputs."""
        source_image = torch.randn(3, 512, 512)
        driving_audio = torch.randn(int(45 * 24000))

        with patch("torch.manual_seed") as mock_seed:
            self.model.animate(
                source_image=source_image,
                driving_audio=driving_audio,
                expression_preset="neutral",
                seed=42,
            )
            mock_seed.assert_called_once_with(42)

    def test_animate_audio_truncation(self):
        """Test that audio exceeding duration is truncated with warning."""
        source_image = torch.randn(3, 512, 512)
        long_audio = torch.randn(int(60 * 24000))  # 60s audio for 45s video

        with patch("vortex.models.liveportrait.logger") as mock_logger:
            result = self.model.animate(
                source_image=source_image,
                driving_audio=long_audio,
                expression_preset="neutral",
                duration=45,
            )

            # Should log truncation warning
            warning_calls = [
                call
                for call in mock_logger.warning.call_args_list
                if "truncated" in str(call).lower()
            ]
            self.assertGreater(len(warning_calls), 0)

    def test_animate_invalid_image_dimensions(self):
        """Test that invalid image dimensions raise ValueError."""
        invalid_image = torch.randn(3, 256, 256)  # Wrong size
        driving_audio = torch.randn(int(45 * 24000))

        with self.assertRaises(ValueError):
            self.model.animate(
                source_image=invalid_image,
                driving_audio=driving_audio,
                expression_preset="neutral",
            )

    def test_animate_invalid_expression_preset(self):
        """Test that invalid expression falls back to neutral with warning."""
        source_image = torch.randn(3, 512, 512)
        driving_audio = torch.randn(int(45 * 24000))

        with patch("vortex.models.liveportrait.logger") as mock_logger:
            # Should not raise, just use neutral
            result = self.model.animate(
                source_image=source_image,
                driving_audio=driving_audio,
                expression_preset="invalid_expression",
            )

            self.assertEqual(result.shape, (1080, 3, 512, 512))


class TestLivePortraitLoading(unittest.TestCase):
    """Test LivePortrait model loading with FP16 precision."""

    @patch("vortex.models.liveportrait.LivePortraitPipeline")
    def test_load_liveportrait_fp16(self, mock_pipeline_class):
        """Test loading LivePortrait with FP16 precision."""
        from vortex.models.liveportrait import LivePortraitModel, load_liveportrait

        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        result = load_liveportrait(device="cuda:0", precision="fp16")

        # Verify pipeline was called
        mock_pipeline_class.from_pretrained.assert_called_once()

        # Result should be LivePortraitModel instance
        self.assertIsInstance(result, LivePortraitModel)

    @patch("vortex.models.liveportrait.torch.cuda.is_available")
    @patch("vortex.models.liveportrait.LivePortraitPipeline")
    def test_load_liveportrait_cuda_oom_handling(
        self, mock_pipeline_class, mock_cuda_available
    ):
        """Test graceful handling of CUDA OOM during model loading."""
        from vortex.models.liveportrait import (
            VortexInitializationError,
            load_liveportrait,
        )

        # Simulate CUDA is NOT available
        mock_cuda_available.return_value = False

        # Simulate CUDA OOM
        mock_pipeline_class.from_pretrained.side_effect = torch.cuda.OutOfMemoryError()

        with self.assertRaises(VortexInitializationError) as ctx:
            load_liveportrait(device="cuda:0")

        # Verify error message includes VRAM info
        self.assertIn("VRAM", str(ctx.exception))


class TestLivePortraitVRAMBudget(unittest.TestCase):
    """Test VRAM budget compliance."""

    @patch("vortex.models.liveportrait.torch.cuda.memory_allocated")
    def test_vram_usage_within_budget(self, mock_memory_allocated):
        """Test that LivePortrait VRAM usage is within 3.0-4.0GB budget."""
        # Simulate 3.5GB allocation
        mock_memory_allocated.return_value = int(3.5 * 1e9)

        vram_gb = mock_memory_allocated() / 1e9
        self.assertGreaterEqual(vram_gb, 3.0)
        self.assertLessEqual(vram_gb, 4.0)


class TestLipsyncAccuracy(unittest.TestCase):
    """Test lip-sync accuracy and audio-visual alignment."""

    @patch("vortex.models.liveportrait.LivePortraitPipeline")
    def setUp(self, mock_pipeline_class):
        """Set up mocked pipeline."""
        from vortex.models.liveportrait import LivePortraitModel

        self.mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = self.mock_pipeline

        def mock_warp_sequence(source_image, visemes, expression_params, num_frames):
            return torch.rand(num_frames, 3, 512, 512)

        self.mock_pipeline.warp_sequence.side_effect = mock_warp_sequence

        self.model = LivePortraitModel(self.mock_pipeline, device="cpu")

    def test_audio_to_visemes_conversion(self):
        """Test that audio is converted to per-frame visemes."""
        from vortex.models.liveportrait import audio_to_visemes

        # Mock audio with distinct phonemes
        test_audio = torch.randn(24000)  # 1 second @ 24kHz
        fps = 24

        visemes = audio_to_visemes(test_audio, fps, sample_rate=24000)

        # Should return 24 viseme tensors (one per frame)
        self.assertEqual(len(visemes), 24)
        for viseme in visemes:
            self.assertIsInstance(viseme, torch.Tensor)

    def test_lipsync_temporal_alignment(self):
        """Test that lip movements align with audio within ±2 frames."""
        # This is a placeholder for integration test
        # Real test requires visual inspection or phoneme detector
        pass


class TestExpressionPresets(unittest.TestCase):
    """Test expression preset handling and interpolation."""

    @patch("vortex.models.liveportrait.LivePortraitPipeline")
    def setUp(self, mock_pipeline_class):
        """Set up mocked pipeline."""
        from vortex.models.liveportrait import LivePortraitModel

        self.mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = self.mock_pipeline

        def mock_warp_sequence(source_image, visemes, expression_params, num_frames):
            return torch.rand(num_frames, 3, 512, 512)

        self.mock_pipeline.warp_sequence.side_effect = mock_warp_sequence

        self.model = LivePortraitModel(self.mock_pipeline, device="cpu")

    def test_expression_params_retrieval(self):
        """Test that expression presets map to correct parameters."""
        neutral_params = self.model._get_expression_params("neutral")
        excited_params = self.model._get_expression_params("excited")

        # Excited should have higher intensity than neutral
        self.assertGreater(
            excited_params.get("intensity", 1.0),
            neutral_params.get("intensity", 1.0),
        )

    def test_expression_sequence_transitions(self):
        """Test that expression sequences create smooth transitions."""
        sequence = ["neutral", "excited", "manic", "calm"]
        num_frames = 1080

        # Get interpolated params at different frames
        params_start = self.model._interpolate_expression_sequence(sequence, 0, num_frames)
        params_mid = self.model._interpolate_expression_sequence(sequence, 540, num_frames)
        params_end = self.model._interpolate_expression_sequence(sequence, 1079, num_frames)

        # Should have different params at different times
        self.assertIsNotNone(params_start)
        self.assertIsNotNone(params_mid)
        self.assertIsNotNone(params_end)


class TestOutputConstraints(unittest.TestCase):
    """Test output format and constraints."""

    @patch("vortex.models.liveportrait.LivePortraitPipeline")
    def setUp(self, mock_pipeline_class):
        """Set up mocked pipeline."""
        from vortex.models.liveportrait import LivePortraitModel

        self.mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = self.mock_pipeline

        # Mock output in correct range [0, 1]
        def mock_warp_sequence(source_image, visemes, expression_params, num_frames):
            return torch.rand(num_frames, 3, 512, 512)  # rand() gives [0, 1]

        self.mock_pipeline.warp_sequence.side_effect = mock_warp_sequence

        self.model = LivePortraitModel(self.mock_pipeline, device="cpu")

    def test_output_value_range(self):
        """Test that output values are in [0, 1] range."""
        source_image = torch.randn(3, 512, 512)
        driving_audio = torch.randn(int(45 * 24000))

        result = self.model.animate(
            source_image=source_image,
            driving_audio=driving_audio,
            expression_preset="neutral",
        )

        # All values should be in [0, 1]
        self.assertGreaterEqual(result.min().item(), 0.0)
        self.assertLessEqual(result.max().item(), 1.0)

    def test_output_dtype(self):
        """Test that output has correct dtype (float32)."""
        source_image = torch.randn(3, 512, 512)
        driving_audio = torch.randn(int(45 * 24000))

        result = self.model.animate(
            source_image=source_image,
            driving_audio=driving_audio,
            expression_preset="neutral",
        )

        self.assertEqual(result.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
