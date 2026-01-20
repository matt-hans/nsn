"""Unit tests for AudioEngine with graceful degradation."""

from unittest.mock import Mock, patch

import pytest


class TestAudioEngineSelection:
    """Test engine selection logic."""

    def test_auto_uses_f5_when_reference_exists(self, tmp_path):
        """Auto mode should use F5-TTS when voice reference file exists."""
        from vortex.core.audio import AudioEngine

        # Create mock reference file
        refs_dir = tmp_path / "voices"
        refs_dir.mkdir()
        ref_file = refs_dir / "manic.wav"
        ref_file.write_bytes(b"fake audio data")

        with patch.object(AudioEngine, '_load_f5'):
            with patch.object(AudioEngine, '_generate_f5') as mock_gen:
                mock_gen.return_value = str(tmp_path / "output.wav")

                engine = AudioEngine(device="cpu", assets_dir=str(refs_dir))
                engine.generate(
                    script="Hello world",
                    engine="auto",
                    voice_style="manic"
                )

                mock_gen.assert_called_once()

    def test_auto_falls_back_to_kokoro_when_reference_missing(self, tmp_path):
        """Auto mode should fall back to Kokoro when reference file is missing."""
        from vortex.core.audio import AudioEngine

        refs_dir = tmp_path / "voices"
        refs_dir.mkdir()
        # No reference file created

        with patch.object(AudioEngine, '_load_kokoro'):
            with patch.object(AudioEngine, '_generate_kokoro') as mock_gen:
                mock_gen.return_value = str(tmp_path / "output.wav")

                engine = AudioEngine(device="cpu", assets_dir=str(refs_dir))
                engine.generate(
                    script="Hello world",
                    engine="auto",
                    voice_style="nonexistent"
                )

                mock_gen.assert_called_once()

    def test_explicit_f5_raises_when_reference_missing(self, tmp_path):
        """Explicit f5_tts engine should raise FileNotFoundError when reference missing."""
        from vortex.core.audio import AudioEngine

        refs_dir = tmp_path / "voices"
        refs_dir.mkdir()

        engine = AudioEngine(device="cpu", assets_dir=str(refs_dir))

        with pytest.raises(FileNotFoundError):
            engine.generate(
                script="Hello world",
                engine="f5_tts",
                voice_style="nonexistent"
            )

    def test_explicit_kokoro_skips_f5(self, tmp_path):
        """Explicit kokoro engine should skip F5 entirely."""
        from vortex.core.audio import AudioEngine

        refs_dir = tmp_path / "voices"
        refs_dir.mkdir()
        # Create reference that would be used by F5
        ref_file = refs_dir / "manic.wav"
        ref_file.write_bytes(b"fake audio data")

        with patch.object(AudioEngine, '_generate_f5') as mock_f5:
            with patch.object(AudioEngine, '_generate_kokoro') as mock_kokoro:
                mock_kokoro.return_value = str(tmp_path / "output.wav")

                engine = AudioEngine(device="cpu", assets_dir=str(refs_dir))
                engine.generate(
                    script="Hello world",
                    engine="kokoro",
                    voice_style="manic",  # Should be ignored
                    voice_id="af_heart"
                )

                mock_f5.assert_not_called()
                mock_kokoro.assert_called_once()


class TestAudioEngineVRAM:
    """Test VRAM management."""

    def test_unload_clears_models(self, tmp_path):
        """Unload should clear model references and call empty_cache."""
        from vortex.core.audio import AudioEngine

        refs_dir = tmp_path / "voices"
        refs_dir.mkdir()

        engine = AudioEngine(device="cpu", assets_dir=str(refs_dir))
        engine._f5_model = Mock()
        engine._kokoro_model = Mock()

        with patch('torch.cuda.empty_cache'):
            with patch('gc.collect') as mock_gc:
                engine.unload()

                assert engine._f5_model is None
                assert engine._kokoro_model is None
                mock_gc.assert_called()
