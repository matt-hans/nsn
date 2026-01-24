"""Unit tests for Bark TTS model wrapper."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch


class TestBarkVoiceEngine:
    """Test suite for BarkVoiceEngine wrapper."""

    def test_import_bark_voice_engine(self):
        """Test that BarkVoiceEngine can be imported."""
        from vortex.models.bark import BarkVoiceEngine
        assert BarkVoiceEngine is not None

    def test_init_loads_configs(self):
        """Test that engine initializes with voice and emotion configs."""
        from vortex.models.bark import BarkVoiceEngine

        with patch('vortex.models.bark.preload_models'):
            engine = BarkVoiceEngine(device="cpu")

        assert "rick_c137" in engine.voice_profiles
        assert "neutral" in engine.emotion_config
        assert engine.sample_rate == 24000

    def test_get_voice_preset_returns_bark_speaker(self):
        """Test voice ID mapping to Bark speaker preset."""
        from vortex.models.bark import BarkVoiceEngine

        with patch('vortex.models.bark.preload_models'):
            engine = BarkVoiceEngine(device="cpu")

        speaker = engine._get_bark_speaker("rick_c137")
        assert speaker == "v2/en_speaker_6"

    def test_get_voice_preset_unknown_returns_default(self):
        """Test unknown voice ID returns default speaker."""
        from vortex.models.bark import BarkVoiceEngine

        with patch('vortex.models.bark.preload_models'):
            engine = BarkVoiceEngine(device="cpu")

        speaker = engine._get_bark_speaker("unknown_voice")
        assert speaker == "v2/en_speaker_0"

    def test_get_emotion_params_returns_temperatures(self):
        """Test emotion mapping returns temperature settings."""
        from vortex.models.bark import BarkVoiceEngine

        with patch('vortex.models.bark.preload_models'):
            engine = BarkVoiceEngine(device="cpu")

        params = engine._get_emotion_params("manic")
        assert params["coarse_temp"] == 0.8  # Lowered for clarity
        assert params["fine_temp"] == 0.5  # Lowered to reduce artifacts

    def test_get_emotion_params_unknown_returns_neutral(self):
        """Test unknown emotion returns neutral settings."""
        from vortex.models.bark import BarkVoiceEngine

        with patch('vortex.models.bark.preload_models'):
            engine = BarkVoiceEngine(device="cpu")

        params = engine._get_emotion_params("unknown_emotion")
        assert params["coarse_temp"] == 0.7
        assert params["fine_temp"] == 0.5


class TestBarkSynthesis:
    """Test suite for Bark synthesis functionality."""

    @pytest.fixture
    def engine_with_mocks(self):
        """Create BarkVoiceEngine with mocked bark models."""
        with patch('vortex.models.bark.preload_models'):
            with patch('vortex.models.bark.generate_audio') as mock_gen:
                from vortex.models.bark import BarkVoiceEngine
                # Mock returns 1 second of audio at 24kHz
                mock_gen.return_value = np.random.randn(24000).astype(np.float32)
                engine = BarkVoiceEngine(device="cpu")
                engine._generate_bark = Mock(return_value=np.random.randn(24000).astype(np.float32))
                yield engine

    def test_synthesize_returns_tensor(self, engine_with_mocks):
        """Test that synthesize returns a torch.Tensor."""
        result = engine_with_mocks.synthesize(
            text="Hello world",
            voice_id="rick_c137"
        )
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 1  # Mono audio

    def test_synthesize_with_output_buffer(self, engine_with_mocks):
        """Test synthesize writes to pre-allocated buffer."""
        output_buffer = torch.zeros(48000, dtype=torch.float32)

        result = engine_with_mocks.synthesize(
            text="Test",
            voice_id="rick_c137",
            output=output_buffer
        )

        # Result should be a slice of the buffer
        assert result.shape[0] <= output_buffer.shape[0]

    def test_synthesize_empty_text_raises(self, engine_with_mocks):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            engine_with_mocks.synthesize(text="", voice_id="rick_c137")

    def test_synthesize_normalizes_audio(self, engine_with_mocks):
        """Test that output audio is normalized to [-1, 1]."""
        # Mock returns unnormalized audio
        engine_with_mocks._generate_bark = Mock(
            return_value=np.array([5.0, -3.0, 2.0], dtype=np.float32)
        )

        result = engine_with_mocks.synthesize(
            text="Test",
            voice_id="rick_c137"
        )

        assert result.abs().max() <= 1.0


class TestBarkRetryFallback:
    """Test suite for Bark retry and fallback logic."""

    def test_synthesize_retries_on_failure(self):
        """Test that synthesize retries up to 3 times before fallback."""
        with patch('vortex.models.bark.preload_models'):
            with patch('vortex.models.bark.generate_audio') as mock_gen:
                from vortex.models.bark import BarkVoiceEngine

                # First two calls fail, third succeeds
                mock_gen.side_effect = [
                    RuntimeError("Bark failed"),
                    RuntimeError("Bark failed"),
                    np.random.randn(24000).astype(np.float32)
                ]

                engine = BarkVoiceEngine(device="cpu")
                result = engine.synthesize(text="Test", voice_id="rick_c137")

                assert isinstance(result, torch.Tensor)
                assert mock_gen.call_count == 3

    def test_synthesize_uses_fallback_after_max_retries(self):
        """Test that fallback audio is used after 3 failed attempts."""
        with patch('vortex.models.bark.preload_models'):
            with patch('vortex.models.bark.generate_audio') as mock_gen:
                from vortex.models.bark import BarkVoiceEngine

                # All calls fail
                mock_gen.side_effect = RuntimeError("Bark failed")

                engine = BarkVoiceEngine(device="cpu")
                # Should not raise - returns fallback audio
                result = engine.synthesize(text="Test", voice_id="rick_c137")

                assert isinstance(result, torch.Tensor)
                assert mock_gen.call_count == 3


class TestLoadBark:
    """Test suite for load_bark factory function."""

    def test_load_bark_returns_engine(self):
        """Test that load_bark returns BarkVoiceEngine instance."""
        with patch('vortex.models.bark.preload_models'):
            from vortex.models.bark import BarkVoiceEngine, load_bark

            engine = load_bark(device="cpu")

            assert isinstance(engine, BarkVoiceEngine)

    def test_load_bark_preloads_models(self):
        """Test that load_bark calls preload_models."""
        with patch('vortex.models.bark.preload_models') as mock_preload:
            from vortex.models.bark import load_bark

            load_bark(device="cpu")

            mock_preload.assert_called_once()


class TestTextNormalization:
    """Test suite for text normalization function."""

    def test_clean_text_for_bark_removes_file_extensions(self):
        """Verify text cleaning removes file extensions that cause stuttering."""
        from vortex.models.bark import _clean_text_for_bark

        # File extensions cause "dot S-S-D" stuttering
        text = "Check out desc.ssd and data.json files"
        cleaned = _clean_text_for_bark(text)
        assert ".ssd" not in cleaned
        assert ".json" not in cleaned
        assert "desc" in cleaned  # Keep the word, just not the extension

    def test_clean_text_for_bark_removes_special_chars(self):
        """Verify special characters are removed."""
        from vortex.models.bark import _clean_text_for_bark

        text = "This *weird* product costs $99.99!"
        cleaned = _clean_text_for_bark(text)
        assert "*" not in cleaned
        assert "$" not in cleaned
        assert "99" in cleaned  # Numbers are fine

    def test_clean_text_for_bark_removes_urls(self):
        """Verify URLs are removed."""
        from vortex.models.bark import _clean_text_for_bark

        text = "Visit https://example.com/path for more info"
        cleaned = _clean_text_for_bark(text)
        assert "https://" not in cleaned
        assert "example.com" not in cleaned
        assert "Visit" in cleaned
        assert "for more info" in cleaned

    def test_clean_text_for_bark_removes_paths(self):
        """Verify file paths are removed."""
        from vortex.models.bark import _clean_text_for_bark

        text = "Open the file at /home/user/data.txt"
        cleaned = _clean_text_for_bark(text)
        assert "/home/user" not in cleaned
        assert "Open the file at" in cleaned

    def test_clean_text_for_bark_preserves_ellipsis(self):
        """Verify ellipsis is preserved."""
        from vortex.models.bark import _clean_text_for_bark

        text = "Wait... what happened?"
        cleaned = _clean_text_for_bark(text)
        assert "..." in cleaned

    def test_clean_text_for_bark_normalizes_whitespace(self):
        """Verify multiple spaces are normalized."""
        from vortex.models.bark import _clean_text_for_bark

        text = "Too   many    spaces   here"
        cleaned = _clean_text_for_bark(text)
        assert "  " not in cleaned
        assert "Too many spaces here" == cleaned

    def test_clean_text_for_bark_handles_empty_string(self):
        """Verify empty string returns empty string."""
        from vortex.models.bark import _clean_text_for_bark

        assert _clean_text_for_bark("") == ""

    def test_clean_text_for_bark_handles_plain_text(self):
        """Verify plain text passes through unchanged."""
        from vortex.models.bark import _clean_text_for_bark

        text = "Hello world, how are you today?"
        cleaned = _clean_text_for_bark(text)
        assert cleaned == text


class TestBarkUnload:
    """Test suite for VRAM cleanup."""

    def test_unload_clears_cuda_cache(self):
        """Test that unload clears CUDA cache."""
        with patch('vortex.models.bark.preload_models'):
            with patch('torch.cuda.empty_cache') as mock_cache:
                from vortex.models.bark import BarkVoiceEngine

                engine = BarkVoiceEngine(device="cpu")
                engine.unload()

                mock_cache.assert_called_once()
