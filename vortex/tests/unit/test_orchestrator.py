"""Unit tests for VideoOrchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest
import torch


class TestVideoOrchestrator:
    """Test main orchestration pipeline."""

    @pytest.mark.asyncio
    async def test_generate_calls_renderer(self, tmp_path):
        """Should initialize renderer and call render()."""
        from vortex.orchestrator import VideoOrchestrator
        from vortex.renderers.types import RenderResult

        # Create mock config
        mock_config = """
models:
  flux:
    path: "mock/flux"
  cogvideox:
    path: "mock/cogvideox"
showrunner:
  url: "http://localhost:8000"
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(mock_config)

        # Create mock render result
        mock_result = RenderResult(
            success=True,
            video_frames=torch.zeros(24, 480, 720, 3, dtype=torch.uint8),
            audio_waveform=torch.zeros(24000 * 3),
            clip_embedding=torch.zeros(512),
            generation_time_ms=5000.0,
            determinism_proof=b"test",
        )

        with patch('vortex.orchestrator.DefaultRenderer') as mock_renderer_cls:
            with patch('vortex.orchestrator.save_render_result') as mock_save:
                # Setup mocks
                mock_renderer = mock_renderer_cls.return_value
                mock_renderer.initialize = AsyncMock()
                mock_renderer.render = AsyncMock(return_value=mock_result)

                mock_save.return_value = {
                    "video_path": tmp_path / "output.mp4",
                    "audio_path": tmp_path / "output.wav",
                }

                # Run orchestrator
                orchestrator = VideoOrchestrator(
                    config_path=str(config_path),
                    output_dir=str(tmp_path / "outputs"),
                )

                result = await orchestrator.generate(
                    slot_id=1,
                    seed=42,
                    theme="test theme",
                )

                # Verify renderer was initialized and called
                mock_renderer.initialize.assert_called_once()
                mock_renderer.render.assert_called_once()

                # Verify result
                assert result.success is True
                assert result.seed == 42

    @pytest.mark.asyncio
    async def test_returns_generation_result(self, tmp_path):
        """Should return GenerationResult with paths and metadata."""
        from vortex.orchestrator import VideoOrchestrator, GenerationResult
        from vortex.renderers.types import RenderResult

        # Create mock config
        mock_config = """
models:
  flux:
    path: "mock/flux"
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(mock_config)

        # Create mock render result with video
        mock_result = RenderResult(
            success=True,
            video_frames=torch.zeros(48, 480, 720, 3, dtype=torch.uint8),  # 48 frames
            audio_waveform=torch.zeros(24000 * 6),  # 6 seconds
            clip_embedding=torch.zeros(512),
            generation_time_ms=10000.0,
            determinism_proof=b"test",
        )

        with patch('vortex.orchestrator.DefaultRenderer') as mock_renderer_cls:
            with patch('vortex.orchestrator.save_render_result') as mock_save:
                mock_renderer = mock_renderer_cls.return_value
                mock_renderer.initialize = AsyncMock()
                mock_renderer.render = AsyncMock(return_value=mock_result)

                mock_save.return_value = {
                    "video_path": tmp_path / "video.mp4",
                    "audio_path": tmp_path / "audio.wav",
                }

                orchestrator = VideoOrchestrator(
                    config_path=str(config_path),
                    output_dir=str(tmp_path),
                )

                result = await orchestrator.generate(
                    slot_id=42,
                    seed=12345,
                )

                # Verify result type and fields
                assert isinstance(result, GenerationResult)
                assert result.success is True
                assert result.seed == 12345
                assert "video.mp4" in result.video_path
                assert "audio.wav" in result.audio_path
                assert result.generation_time_ms == 10000.0
                assert result.duration_sec == 6.0  # 48 frames / 8 fps

    @pytest.mark.asyncio
    async def test_handles_render_failure(self, tmp_path):
        """Should return failed GenerationResult on render error."""
        from vortex.orchestrator import VideoOrchestrator
        from vortex.renderers.types import RenderResult

        # Create mock config
        mock_config = "models: {}"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(mock_config)

        # Create failed render result
        mock_result = RenderResult(
            success=False,
            video_frames=torch.zeros(0),
            audio_waveform=torch.zeros(0),
            clip_embedding=torch.zeros(0),
            generation_time_ms=1000.0,
            determinism_proof=b"",
            error_msg="TTS synthesis failed",
        )

        with patch('vortex.orchestrator.DefaultRenderer') as mock_renderer_cls:
            mock_renderer = mock_renderer_cls.return_value
            mock_renderer.initialize = AsyncMock()
            mock_renderer.render = AsyncMock(return_value=mock_result)

            orchestrator = VideoOrchestrator(
                config_path=str(config_path),
                output_dir=str(tmp_path),
            )

            result = await orchestrator.generate(slot_id=1)

            assert result.success is False
            assert result.error_msg == "TTS synthesis failed"
            assert result.video_path == ""
            assert result.audio_path == ""

    @pytest.mark.asyncio
    async def test_health_check_delegates_to_renderer(self, tmp_path):
        """Should delegate health check to renderer."""
        from vortex.orchestrator import VideoOrchestrator

        mock_config = "models: {}"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(mock_config)

        with patch('vortex.orchestrator.DefaultRenderer') as mock_renderer_cls:
            mock_renderer = mock_renderer_cls.return_value
            mock_renderer.initialize = AsyncMock()
            mock_renderer.health_check = AsyncMock(return_value=True)

            orchestrator = VideoOrchestrator(
                config_path=str(config_path),
                output_dir=str(tmp_path),
            )

            # Before initialization
            health = await orchestrator.health_check()
            assert health["orchestrator"] is False
            assert health["renderer"] is False

            # After initialization
            await orchestrator.initialize()
            health = await orchestrator.health_check()
            assert health["orchestrator"] is True
            assert health["renderer"] is True

    @pytest.mark.asyncio
    async def test_shutdown_clears_renderer(self, tmp_path):
        """Should shutdown renderer and clear state."""
        from vortex.orchestrator import VideoOrchestrator

        mock_config = "models: {}"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(mock_config)

        with patch('vortex.orchestrator.DefaultRenderer') as mock_renderer_cls:
            mock_renderer = mock_renderer_cls.return_value
            mock_renderer.initialize = AsyncMock()
            mock_renderer.shutdown = AsyncMock()

            orchestrator = VideoOrchestrator(
                config_path=str(config_path),
                output_dir=str(tmp_path),
            )

            await orchestrator.initialize()
            assert orchestrator._initialized is True

            await orchestrator.shutdown()
            assert orchestrator._initialized is False
            assert orchestrator._renderer is None
            mock_renderer.shutdown.assert_called_once()


class TestGenerationResult:
    """Test GenerationResult dataclass."""

    def test_generation_result_fields(self):
        """Should have all required fields."""
        from vortex.orchestrator import GenerationResult

        result = GenerationResult(
            video_path="/path/to/video.mp4",
            audio_path="/path/to/audio.wav",
            seed=42,
            duration_sec=10.5,
            generation_time_ms=5000.0,
            success=True,
        )

        assert result.video_path == "/path/to/video.mp4"
        assert result.audio_path == "/path/to/audio.wav"
        assert result.seed == 42
        assert result.duration_sec == 10.5
        assert result.generation_time_ms == 5000.0
        assert result.success is True
        assert result.error_msg is None

    def test_generation_result_with_error(self):
        """Should support error message field."""
        from vortex.orchestrator import GenerationResult

        result = GenerationResult(
            video_path="",
            audio_path="",
            seed=0,
            duration_sec=0.0,
            generation_time_ms=100.0,
            success=False,
            error_msg="Generation failed",
        )

        assert result.success is False
        assert result.error_msg == "Generation failed"
