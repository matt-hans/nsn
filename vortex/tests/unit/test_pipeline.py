"""Unit tests for vortex.pipeline core orchestration.

Tests the VortexPipeline class which orchestrates video generation through
the DeterministicVideoRenderer interface.

Note: VRAM monitoring and model registry are now internal to renderers.
Those components are tested via the renderer-specific tests.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from vortex.pipeline import (
    GenerationResult,
    VortexInitializationError,
    VortexPipeline,
)
from vortex.renderers.default.renderer import (
    MemoryPressureError,
    _ModelRegistry as ModelRegistry,
    _VRAMMonitor as VRAMMonitor,
)


class TestVRAMMonitor:
    """Test VRAM pressure monitoring with soft/hard limits."""

    def test_init(self):
        """Test VRAMMonitor initialization."""
        monitor = VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.5)
        assert monitor.soft_limit_bytes == 11_000_000_000
        assert monitor.hard_limit_bytes == 11_500_000_000
        assert monitor._warning_emitted is False

    @patch("vortex.renderers.default.renderer.get_current_vram_usage", return_value=10_000_000_000)
    def test_check_below_soft_limit(self, mock_usage):
        """Test VRAM check when usage is below soft limit."""
        monitor = VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.5)
        monitor.check()  # Should not raise or warn
        assert not monitor._warning_emitted

    @patch("vortex.renderers.default.renderer.get_current_vram_usage", return_value=11_200_000_000)
    @patch("vortex.renderers.default.renderer.get_vram_stats")
    @patch("vortex.renderers.default.renderer.logger")
    def test_check_soft_limit_exceeded(self, mock_logger, mock_stats, mock_usage):
        """Test VRAM check when soft limit exceeded (warning)."""
        mock_stats.return_value = {"allocated_gb": 11.2}
        monitor = VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.5)

        monitor.check()  # First call - should warn
        assert monitor._warning_emitted
        mock_logger.warning.assert_called_once()

        mock_logger.reset_mock()
        monitor.check()  # Second call - should NOT warn again
        mock_logger.warning.assert_not_called()

    @patch("vortex.renderers.default.renderer.get_current_vram_usage", return_value=11_600_000_000)
    @patch("vortex.renderers.default.renderer.get_vram_stats")
    def test_check_hard_limit_exceeded(self, mock_stats, mock_usage):
        """Test VRAM check when hard limit exceeded (error)."""
        mock_stats.return_value = {"allocated_gb": 11.6}
        monitor = VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.5)

        with pytest.raises(MemoryPressureError, match="hard limit exceeded"):
            monitor.check()


class TestModelRegistry:
    """Test model registry with get_model() interface."""

    @patch("vortex.renderers.default.renderer.load_model")
    @patch("vortex.renderers.default.renderer.load_clip_ensemble")
    @patch("vortex.renderers.default.renderer.log_vram_snapshot")
    @patch("vortex.renderers.default.renderer.get_vram_stats")
    def test_load_all_models(self, mock_stats, mock_log, mock_clip_load, mock_load):
        """Test successful loading of all models."""
        # Mock model returns
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_clip_load.return_value = MagicMock()
        mock_stats.return_value = {"allocated_gb": 10.0}

        registry = ModelRegistry(device="cpu")
        registry.load_all_models()

        # Verify 4 models loaded (flux, liveportrait, kokoro, clip_ensemble)
        assert len(registry._models) == 4
        assert "flux" in registry
        assert "liveportrait" in registry
        assert "kokoro" in registry
        assert "clip_ensemble" in registry

        # Verify load_model called 3 times (individual models)
        assert mock_load.call_count == 3
        # Verify load_clip_ensemble called once
        assert mock_clip_load.call_count == 1

    @patch("vortex.renderers.default.renderer.load_model")
    @patch("vortex.renderers.default.renderer.load_clip_ensemble")
    @patch("vortex.renderers.default.renderer.log_vram_snapshot")
    @patch("vortex.renderers.default.renderer.get_vram_stats")
    def test_get_model_success(self, mock_stats, mock_log, mock_clip_load, mock_load):
        """Test retrieving loaded model by name."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_clip_load.return_value = MagicMock()
        mock_stats.return_value = {"allocated_gb": 10.0}

        registry = ModelRegistry(device="cpu")
        registry.load_all_models()
        flux = registry.get_model("flux")

        assert flux is mock_model

    @patch("vortex.renderers.default.renderer.load_model")
    @patch("vortex.renderers.default.renderer.load_clip_ensemble")
    @patch("vortex.renderers.default.renderer.log_vram_snapshot")
    @patch("vortex.renderers.default.renderer.get_vram_stats")
    def test_get_model_invalid_name(self, mock_stats, mock_log, mock_clip_load, mock_load):
        """Test retrieving model with invalid name raises KeyError."""
        mock_load.return_value = MagicMock()
        mock_clip_load.return_value = MagicMock()
        mock_stats.return_value = {"allocated_gb": 10.0}

        registry = ModelRegistry(device="cpu")
        registry.load_all_models()

        with pytest.raises(KeyError, match="not found"):
            registry.get_model("invalid_model")

    @patch("vortex.renderers.default.renderer.load_model")
    @patch("vortex.renderers.default.renderer.log_vram_snapshot")
    @patch("vortex.renderers.default.renderer.get_vram_stats")
    def test_cuda_oom_during_loading(self, mock_stats, mock_log, mock_load):
        """Test graceful handling of CUDA OOM during model loading."""
        from vortex.renderers.default.renderer import VortexInitializationError as RendererInitError

        # First two models succeed, third fails with OOM
        mock_load.side_effect = [
            MagicMock(),  # flux
            MagicMock(),  # liveportrait
            torch.cuda.OutOfMemoryError("CUDA OOM"),  # kokoro fails
        ]
        mock_stats.return_value = {"allocated_gb": 10.0, "total_gb": 12.0}

        registry = ModelRegistry(device="cuda:0")
        with pytest.raises(RendererInitError, match="CUDA OOM"):
            registry.load_all_models()


@pytest.mark.asyncio
class TestVortexPipelineCreation:
    """Test VortexPipeline creation and initialization."""

    async def test_create_with_mock_renderer(self):
        """Test pipeline creation with mocked renderer."""
        # Create mock renderer
        mock_renderer = MagicMock()
        mock_renderer.manifest.name = "mock-renderer"
        mock_renderer.manifest.version = "1.0.0"
        mock_renderer.manifest.resources.vram_gb = 11.0
        mock_renderer.initialize = AsyncMock()
        mock_renderer.health_check = AsyncMock(return_value=True)

        # Create pipeline with mock
        config = {"device": {"name": "cpu"}}
        pipeline = VortexPipeline(config=config, renderer=mock_renderer, device="cpu")

        assert pipeline.renderer is mock_renderer
        assert pipeline.device == "cpu"
        assert pipeline.renderer_name == "mock-renderer"


@pytest.mark.asyncio
class TestVortexPipelineGeneration:
    """Test async slot generation orchestration."""

    async def test_generate_slot_success(self):
        """Test successful slot generation."""
        # Create mock renderer with render method
        mock_renderer = MagicMock()
        mock_renderer.manifest.name = "mock-renderer"
        mock_renderer.manifest.version = "1.0.0"

        # Mock successful render result
        from vortex.renderers.types import RenderResult

        mock_result = RenderResult(
            video_frames=torch.zeros(1080, 3, 512, 512),
            audio_waveform=torch.zeros(1080000),
            clip_embedding=torch.zeros(512),
            generation_time_ms=12500.0,
            determinism_proof=b"test_proof",
            success=True,
        )
        mock_renderer.render = AsyncMock(return_value=mock_result)

        config = {"device": {"name": "cpu"}}
        pipeline = VortexPipeline(config=config, renderer=mock_renderer, device="cpu")

        recipe = {
            "audio_track": {"script": "Test audio"},
            "visual_track": {"prompt": "Test actor"},
        }

        result = await pipeline.generate_slot(recipe=recipe, slot_id=12345)

        assert isinstance(result, GenerationResult)
        assert result.success is True
        assert result.slot_id == 12345
        assert result.generation_time_ms > 0
        assert result.video_frames.shape == (1080, 3, 512, 512)
        assert result.audio_waveform.shape == (1080000,)
        assert result.clip_embedding.shape == (512,)

    async def test_generate_slot_with_seed(self):
        """Test generation with explicit seed."""
        mock_renderer = MagicMock()
        mock_renderer.manifest.name = "mock-renderer"

        from vortex.renderers.types import RenderResult

        mock_result = RenderResult(
            video_frames=torch.zeros(10, 3, 512, 512),
            audio_waveform=torch.zeros(10000),
            clip_embedding=torch.zeros(512),
            generation_time_ms=100.0,
            determinism_proof=b"proof",
            success=True,
        )
        mock_renderer.render = AsyncMock(return_value=mock_result)

        config = {}
        pipeline = VortexPipeline(config=config, renderer=mock_renderer, device="cpu")

        recipe = {"audio_track": {}, "visual_track": {}}
        result = await pipeline.generate_slot(recipe=recipe, slot_id=1, seed=42)

        # Verify render was called with the seed
        mock_renderer.render.assert_called_once()
        call_kwargs = mock_renderer.render.call_args[1]
        assert call_kwargs["seed"] == 42

    async def test_generate_slot_failure(self):
        """Test handling of failed generation."""
        mock_renderer = MagicMock()
        mock_renderer.manifest.name = "mock-renderer"

        from vortex.renderers.types import RenderResult

        mock_result = RenderResult(
            video_frames=torch.empty(0),
            audio_waveform=torch.empty(0),
            clip_embedding=torch.empty(0),
            generation_time_ms=500.0,
            determinism_proof=b"",
            success=False,
            error_msg="CUDA OOM",
        )
        mock_renderer.render = AsyncMock(return_value=mock_result)

        config = {}
        pipeline = VortexPipeline(config=config, renderer=mock_renderer, device="cpu")

        recipe = {"audio_track": {}, "visual_track": {}}
        result = await pipeline.generate_slot(recipe=recipe, slot_id=99999)

        assert result.success is False
        assert result.error_msg == "CUDA OOM"

    async def test_health_check_delegates_to_renderer(self):
        """Test health check is delegated to renderer."""
        mock_renderer = MagicMock()
        mock_renderer.health_check = AsyncMock(return_value=True)

        config = {}
        pipeline = VortexPipeline(config=config, renderer=mock_renderer, device="cpu")

        result = await pipeline.health_check()

        assert result is True
        mock_renderer.health_check.assert_called_once()


class TestGenerationResult:
    """Test GenerationResult dataclass."""

    def test_successful_result(self):
        """Test creation of successful generation result."""
        result = GenerationResult(
            video_frames=torch.zeros(1080, 3, 512, 512),
            audio_waveform=torch.zeros(1080000),
            clip_embedding=torch.zeros(512),
            generation_time_ms=12500.0,
            slot_id=12345,
            success=True,
        )

        assert result.success is True
        assert result.slot_id == 12345
        assert result.generation_time_ms == 12500.0
        assert result.error_msg is None

    def test_failed_result(self):
        """Test creation of failed generation result."""
        result = GenerationResult(
            video_frames=torch.empty(0),
            audio_waveform=torch.empty(0),
            clip_embedding=torch.empty(0),
            generation_time_ms=500.0,
            slot_id=99999,
            success=False,
            error_msg="CUDA OOM",
        )

        assert result.success is False
        assert result.error_msg == "CUDA OOM"
        assert result.video_frames.numel() == 0
