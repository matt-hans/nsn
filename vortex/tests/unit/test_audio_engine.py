"""Unit tests for AudioEngine with Bark TTS."""

from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
import numpy as np


class TestAudioEngineWithBark:
    """Test AudioEngine using Bark backend."""

    def test_audio_engine_imports(self):
        """Test that AudioEngine can be imported."""
        from vortex.core.audio import AudioEngine
        assert AudioEngine is not None

    def test_generate_calls_bark(self, tmp_path):
        """Test that generate() uses Bark engine."""
        from vortex.core.audio import AudioEngine

        # Mock BarkVoiceEngine
        mock_bark = MagicMock()
        mock_bark.synthesize.return_value = torch.randn(24000)

        with patch('vortex.core.audio.BarkVoiceEngine', return_value=mock_bark):
            engine = AudioEngine(device="cpu", output_dir=str(tmp_path))
            path = engine.generate(
                script="Hello world",
                voice_id="rick_c137",
                emotion="manic"
            )

            assert path.endswith(".wav")
            mock_bark.synthesize.assert_called_once()

    def test_unload_clears_bark_model(self, tmp_path):
        """Test that unload() clears Bark model."""
        from vortex.core.audio import AudioEngine

        mock_bark = MagicMock()

        with patch('vortex.core.audio.BarkVoiceEngine', return_value=mock_bark):
            engine = AudioEngine(device="cpu", output_dir=str(tmp_path))
            engine._bark_engine = mock_bark

            with patch('gc.collect') as mock_gc:
                engine.unload()

                mock_bark.unload.assert_called_once()
                mock_gc.assert_called()
