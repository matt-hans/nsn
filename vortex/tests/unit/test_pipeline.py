"""Unit tests for vortex.pipeline core orchestration."""

import asyncio
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from vortex.pipeline import (
    VortexPipeline,
    ModelRegistry,
    VRAMMonitor,
    GenerationResult,
    MemoryPressureWarning,
    MemoryPressureError,
    VortexInitializationError,
)


class TestVRAMMonitor:
    """Test VRAM pressure monitoring with soft/hard limits."""

    def test_init(self):
        """Test VRAMMonitor initialization."""
        monitor = VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.5)
        assert monitor.soft_limit_bytes == 11_000_000_000
        assert monitor.hard_limit_bytes == 11_500_000_000
        assert monitor._warning_emitted is False

    @patch("vortex.pipeline.get_current_vram_usage", return_value=10_000_000_000)  # 10GB
    def test_check_below_soft_limit(self, mock_usage):
        """Test VRAM check when usage is below soft limit."""
        monitor = VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.5)
        monitor.check()  # Should not raise or warn
        assert not monitor._warning_emitted

    @patch("vortex.pipeline.get_current_vram_usage", return_value=11_200_000_000)  # 11.2GB
    @patch("vortex.pipeline.get_vram_stats")
    @patch("vortex.pipeline.logger")
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

    @patch("vortex.pipeline.get_current_vram_usage", return_value=11_600_000_000)  # 11.6GB
    @patch("vortex.pipeline.get_vram_stats")
    def test_check_hard_limit_exceeded(self, mock_stats, mock_usage):
        """Test VRAM check when hard limit exceeded (error)."""
        mock_stats.return_value = {"allocated_gb": 11.6}
        monitor = VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.5)

        with pytest.raises(MemoryPressureError, match="hard limit exceeded"):
            monitor.check()

    def test_reset_warning(self):
        """Test warning flag reset."""
        monitor = VRAMMonitor()
        monitor._warning_emitted = True
        monitor.reset_warning()
        assert not monitor._warning_emitted


class TestModelRegistry:
    """Test model registry with get_model() interface."""

    @patch("vortex.models.load_model")
    @patch("vortex.pipeline.log_vram_snapshot")
    def test_load_all_models(self, mock_log, mock_load):
        """Test successful loading of all 5 models."""
        # Mock model returns
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        registry = ModelRegistry(device="cpu")

        # Verify all 5 models loaded
        assert len(registry._models) == 5
        assert "flux" in registry
        assert "liveportrait" in registry
        assert "kokoro" in registry
        assert "clip_b" in registry
        assert "clip_l" in registry

        # Verify load_model called 5 times
        assert mock_load.call_count == 5

    @patch("vortex.models.load_model")
    def test_get_model_success(self, mock_load):
        """Test retrieving loaded model by name."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        registry = ModelRegistry(device="cpu")
        flux = registry.get_model("flux")

        assert flux is mock_model

    @patch("vortex.models.load_model")
    def test_get_model_invalid_name(self, mock_load):
        """Test retrieving model with invalid name raises KeyError."""
        mock_load.return_value = MagicMock()
        registry = ModelRegistry(device="cpu")

        with pytest.raises(KeyError, match="not found in registry"):
            registry.get_model("invalid_model")

    @patch("vortex.models.load_model")
    def test_cuda_oom_during_loading(self, mock_load):
        """Test graceful handling of CUDA OOM during model loading."""
        # First two models succeed, third fails with OOM
        mock_load.side_effect = [
            MagicMock(),  # flux
            MagicMock(),  # liveportrait
            torch.cuda.OutOfMemoryError("CUDA OOM"),  # kokoro fails
        ]

        with pytest.raises(VortexInitializationError, match="CUDA OOM"):
            ModelRegistry(device="cuda:0")

    @patch("vortex.models.load_model")
    def test_precision_overrides(self, mock_load):
        """Test precision overrides are passed to load_model."""
        mock_load.return_value = MagicMock()
        overrides = {"flux": "fp16", "clip_b": "fp32"}

        registry = ModelRegistry(device="cpu", precision_overrides=overrides)

        # Check that load_model was called with precision for flux
        flux_call = [c for c in mock_load.call_args_list if c[0][0] == "flux"][0]
        assert flux_call[1]["precision"] == "fp16"


class TestVortexPipeline:
    """Test VortexPipeline initialization and buffer allocation."""

    @patch("vortex.pipeline.ModelRegistry")
    @patch("vortex.pipeline.VRAMMonitor")
    def test_init_success(self, mock_monitor_cls, mock_registry_cls):
        """Test successful pipeline initialization."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"

        pipeline = VortexPipeline(config_path=str(config_path), device="cpu")

        # Verify registry and monitor created
        mock_registry_cls.assert_called_once()
        mock_monitor_cls.assert_called_once()

        # Verify buffers allocated
        assert pipeline.actor_buffer is not None
        assert pipeline.video_buffer is not None
        assert pipeline.audio_buffer is not None

        # Check buffer shapes
        assert pipeline.actor_buffer.shape == (1, 3, 512, 512)
        assert pipeline.video_buffer.shape == (1080, 3, 512, 512)
        assert pipeline.audio_buffer.shape == (1080000,)

    @patch("vortex.pipeline.ModelRegistry")
    @patch("vortex.pipeline.VRAMMonitor")
    def test_buffer_device_placement(self, mock_monitor_cls, mock_registry_cls):
        """Test buffers are allocated on correct device."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"

        pipeline = VortexPipeline(config_path=str(config_path), device="cpu")

        assert pipeline.actor_buffer.device.type == "cpu"
        assert pipeline.video_buffer.device.type == "cpu"
        assert pipeline.audio_buffer.device.type == "cpu"


@pytest.mark.asyncio
class TestVortexPipelineGeneration:
    """Test async slot generation orchestration."""

    @patch("vortex.pipeline.ModelRegistry")
    @patch("vortex.pipeline.VRAMMonitor")
    async def test_generate_slot_success(self, mock_monitor_cls, mock_registry_cls):
        """Test successful slot generation with parallel orchestration."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        pipeline = VortexPipeline(config_path=str(config_path), device="cpu")

        # Mock VRAM monitor check
        mock_monitor = mock_monitor_cls.return_value
        mock_monitor.check = MagicMock()

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

    @patch("vortex.pipeline.ModelRegistry")
    @patch("vortex.pipeline.VRAMMonitor")
    async def test_generate_slot_memory_pressure(self, mock_monitor_cls, mock_registry_cls):
        """Test generation aborts on memory pressure error."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        pipeline = VortexPipeline(config_path=str(config_path), device="cpu")

        # Mock VRAM monitor to raise error
        mock_monitor = mock_monitor_cls.return_value
        mock_monitor.check.side_effect = MemoryPressureError("Hard limit exceeded")

        recipe = {"audio_track": {}, "visual_track": {}}

        result = await pipeline.generate_slot(recipe=recipe, slot_id=99999)

        # Should return failed result, not raise
        assert result.success is False
        assert "Hard limit exceeded" in result.error_msg

    @patch("vortex.pipeline.ModelRegistry")
    @patch("vortex.pipeline.VRAMMonitor")
    async def test_generate_slot_cancellation(self, mock_monitor_cls, mock_registry_cls):
        """Test graceful handling of async cancellation."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        pipeline = VortexPipeline(config_path=str(config_path), device="cpu")

        recipe = {"audio_track": {}, "visual_track": {}}

        # Create task and cancel it
        task = asyncio.create_task(pipeline.generate_slot(recipe=recipe, slot_id=77777))
        await asyncio.sleep(0.05)  # Let it start
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @patch("vortex.pipeline.ModelRegistry")
    @patch("vortex.pipeline.VRAMMonitor")
    async def test_parallel_audio_actor_generation(self, mock_monitor_cls, mock_registry_cls):
        """Test audio and actor generation run in parallel."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        pipeline = VortexPipeline(config_path=str(config_path), device="cpu")

        # Monkey-patch timing tracking
        audio_start = None
        actor_start = None
        audio_end = None
        actor_end = None

        original_gen_audio = pipeline._generate_audio
        original_gen_actor = pipeline._generate_actor

        async def tracked_audio(recipe):
            nonlocal audio_start, audio_end
            audio_start = asyncio.get_event_loop().time()
            result = await original_gen_audio(recipe)
            audio_end = asyncio.get_event_loop().time()
            return result

        async def tracked_actor(recipe):
            nonlocal actor_start, actor_end
            actor_start = asyncio.get_event_loop().time()
            result = await original_gen_actor(recipe)
            actor_end = asyncio.get_event_loop().time()
            return result

        pipeline._generate_audio = tracked_audio
        pipeline._generate_actor = tracked_actor

        recipe = {"audio_track": {}, "visual_track": {}}
        await pipeline.generate_slot(recipe=recipe, slot_id=11111)

        # Verify they started within 10ms of each other (parallel)
        assert audio_start is not None
        assert actor_start is not None
        time_diff = abs(audio_start - actor_start)
        assert time_diff < 0.01, f"Tasks not parallel (diff: {time_diff}s)"


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
