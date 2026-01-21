"""Unit tests for VideoOrchestrator."""

from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestVideoOrchestrator:
    """Test main orchestration pipeline."""

    @pytest.mark.asyncio
    async def test_generate_calls_audio_then_visual(self, tmp_path):
        """Should generate audio first, then dispatch to ComfyUI."""
        from vortex.orchestrator import VideoOrchestrator

        with patch('vortex.orchestrator.AudioEngine') as mock_audio_cls:
            with patch('vortex.orchestrator.AudioCompositor') as mock_mixer_cls:
                with patch('vortex.orchestrator.ComfyClient') as mock_comfy_cls:
                    with patch('vortex.orchestrator.WorkflowBuilder') as mock_builder_cls:
                        with patch('vortex.orchestrator.calculate_frame_count') as mock_frame_count:
                            # Setup mocks
                            mock_audio = mock_audio_cls.return_value
                            mock_audio.generate.return_value = str(tmp_path / "voice.wav")
                            mock_audio.unload = Mock()

                            mock_mixer = mock_mixer_cls.return_value
                            mock_mixer.mix.return_value = str(tmp_path / "mixed.wav")

                            mock_builder = mock_builder_cls.return_value
                            mock_builder.build.return_value = {"workflow": "data"}

                            mock_comfy = mock_comfy_cls.return_value
                            mock_comfy.generate = AsyncMock(
                                return_value=str(tmp_path / "output.mp4")
                            )

                            mock_frame_count.return_value = 48

                            # Run orchestrator
                            orchestrator = VideoOrchestrator(
                                template_path=str(tmp_path / "workflow.json"),
                                assets_dir=str(tmp_path / "assets"),
                            )

                            await orchestrator.generate(
                                prompt="A cool cat",
                                script="Hello world",
                            )

                            # Verify call order
                            mock_audio.generate.assert_called_once()
                            mock_audio.unload.assert_called_once()  # VRAM freed before ComfyUI
                            mock_comfy.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_both_audio_paths(self, tmp_path):
        """Should return clean audio (for lip-sync) and mixed audio (for broadcast)."""
        from vortex.orchestrator import VideoOrchestrator

        with patch('vortex.orchestrator.AudioEngine') as mock_audio_cls:
            with patch('vortex.orchestrator.AudioCompositor') as mock_mixer_cls:
                with patch('vortex.orchestrator.ComfyClient') as mock_comfy_cls:
                    with patch('vortex.orchestrator.WorkflowBuilder') as mock_builder_cls:
                        with patch('vortex.orchestrator.calculate_frame_count') as mock_frame_count:
                            mock_audio = mock_audio_cls.return_value
                            mock_audio.generate.return_value = str(tmp_path / "voice.wav")
                            mock_audio.unload = Mock()

                            mock_mixer = mock_mixer_cls.return_value
                            mock_mixer.mix.return_value = str(tmp_path / "mixed.wav")

                            mock_builder = mock_builder_cls.return_value
                            mock_builder.build.return_value = {}

                            mock_comfy = mock_comfy_cls.return_value
                            mock_comfy.generate = AsyncMock(
                                return_value=str(tmp_path / "output.mp4")
                            )

                            mock_frame_count.return_value = 48

                            orchestrator = VideoOrchestrator(
                                template_path=str(tmp_path / "workflow.json"),
                            )

                            result = await orchestrator.generate(
                                prompt="Test",
                                script="Test script",
                                bgm_name="elevator",
                            )

                            assert "video_path" in result
                            assert "clean_audio_path" in result
                            assert "mixed_audio_path" in result
