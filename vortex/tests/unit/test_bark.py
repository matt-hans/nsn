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
        assert params["coarse_temp"] == 0.75  # Lowered for stability
        assert params["fine_temp"] == 0.4  # Lowered to reduce artifacts

    def test_get_emotion_params_unknown_returns_neutral(self):
        """Test unknown emotion returns neutral settings."""
        from vortex.models.bark import BarkVoiceEngine

        with patch('vortex.models.bark.preload_models'):
            engine = BarkVoiceEngine(device="cpu")

        params = engine._get_emotion_params("unknown_emotion")
        assert params["coarse_temp"] == 0.8  # UPDATED: was 0.6, now 0.8 for natural pitch variation
        assert params["fine_temp"] == 0.4  # Keep low for clarity


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

    def test_clean_text_that_normalizes_to_empty(self):
        """Verify text that normalizes to empty triggers error.

        Text that is only special chars/paths/URLs should raise
        after normalization.
        """
        from vortex.models.bark import _clean_text_for_bark

        # File path only -> normalizes to empty
        assert _clean_text_for_bark("/path/to/file.txt") == ""

        # URL only -> normalizes to empty
        assert _clean_text_for_bark("https://example.com/page") == ""

        # Special characters only -> normalizes to empty
        assert _clean_text_for_bark("***") == ""

        # Mixed paths and special chars -> normalizes to empty
        assert _clean_text_for_bark("/foo/bar.json ***") == ""


class TestSynthesizeEmptyAfterNormalization:
    """Test suite for empty text after normalization handling."""

    def test_synthesize_raises_when_text_normalizes_to_empty(self):
        """Test that synthesize raises ValueError when text normalizes to empty."""
        with patch('vortex.models.bark.preload_models'):
            from vortex.models.bark import BarkVoiceEngine

            engine = BarkVoiceEngine(device="cpu")

            # File path only should raise after normalization
            with pytest.raises(ValueError, match="empty after normalization"):
                engine.synthesize(text="/path/to/file.txt", voice_id="rick_c137")

            # URL only should raise after normalization
            with pytest.raises(ValueError, match="empty after normalization"):
                engine.synthesize(text="https://example.com/page", voice_id="rick_c137")

            # Special characters only should raise after normalization
            with pytest.raises(ValueError, match="empty after normalization"):
                engine.synthesize(text="***", voice_id="rick_c137")


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


class TestBarkMinEosP:
    """Test suite for min_eos_p parameter that prevents semantic drift."""

    def test_bark_uses_min_eos_p(self):
        """Bark should use min_eos_p=0.05 to prevent gibberish/semantic drift.

        Without min_eos_p, Bark can continue generating past logical sentence
        boundaries, producing semantic gibberish like "laves are allegations".
        The min_eos_p=0.05 parameter helps Bark recognize when to stop.
        """
        with patch('vortex.models.bark.preload_models'):
            with patch('vortex.models.bark.generate_audio') as mock_gen:
                from vortex.models.bark import BarkVoiceEngine

                # Mock returns valid audio
                mock_gen.return_value = np.random.randn(24000).astype(np.float32)

                engine = BarkVoiceEngine(device="cpu")
                engine.synthesize(text="Hello world", voice_id="rick_c137")

                # Verify generate_audio was called with min_eos_p=0.05
                mock_gen.assert_called_once()
                call_kwargs = mock_gen.call_args.kwargs
                assert "min_eos_p" in call_kwargs, (
                    "min_eos_p parameter not passed to generate_audio"
                )
                assert call_kwargs["min_eos_p"] == 0.05, (
                    f"Expected min_eos_p=0.05, got {call_kwargs['min_eos_p']}"
                )

    def test_bark_min_eos_p_value_is_correct(self):
        """Verify the specific min_eos_p value of 0.05 is used consistently."""
        with patch('vortex.models.bark.preload_models'):
            with patch('vortex.models.bark.generate_audio') as mock_gen:
                from vortex.models.bark import BarkVoiceEngine

                mock_gen.return_value = np.random.randn(24000).astype(np.float32)

                engine = BarkVoiceEngine(device="cpu")

                # Test with different emotions to ensure min_eos_p is always 0.05
                for emotion in ["neutral", "manic", "excited", "sad", "angry"]:
                    mock_gen.reset_mock()
                    engine.synthesize(
                        text="Test sentence",
                        voice_id="rick_c137",
                        emotion=emotion
                    )
                    call_kwargs = mock_gen.call_args.kwargs
                    assert call_kwargs.get("min_eos_p") == 0.05, (
                        f"min_eos_p should be 0.05 for emotion '{emotion}', "
                        f"got {call_kwargs.get('min_eos_p')}"
                    )


class TestBarkTokenWhitelist:
    """Test suite for Bark token whitelist sanitization."""

    def test_preserves_valid_laughs_token(self):
        """Valid [laughs] token should be preserved."""
        from vortex.models.bark import _clean_text_for_bark
        text = "That's hilarious [laughs] indeed"
        cleaned = _clean_text_for_bark(text)
        assert "[laughs]" in cleaned

    def test_preserves_valid_gasps_token(self):
        """Valid [gasps] token should be preserved."""
        from vortex.models.bark import _clean_text_for_bark
        text = "Oh no [gasps] what happened"
        cleaned = _clean_text_for_bark(text)
        assert "[gasps]" in cleaned

    def test_preserves_valid_sighs_token(self):
        """Valid [sighs] token should be preserved."""
        from vortex.models.bark import _clean_text_for_bark
        text = "I'm so tired [sighs] of this"
        cleaned = _clean_text_for_bark(text)
        assert "[sighs]" in cleaned

    def test_preserves_valid_laughter_token(self):
        """Valid [laughter] token should be preserved."""
        from vortex.models.bark import _clean_text_for_bark
        text = "That was funny [laughter]"
        cleaned = _clean_text_for_bark(text)
        assert "[laughter]" in cleaned

    def test_preserves_valid_music_token(self):
        """Valid [music] token should be preserved."""
        from vortex.models.bark import _clean_text_for_bark
        text = "[music] Here comes the song"
        cleaned = _clean_text_for_bark(text)
        assert "[music]" in cleaned

    def test_preserves_valid_clears_throat_token(self):
        """Valid [clears throat] token should be preserved."""
        from vortex.models.bark import _clean_text_for_bark
        text = "Ahem [clears throat] attention please"
        cleaned = _clean_text_for_bark(text)
        assert "[clears throat]" in cleaned

    def test_strips_invalid_excited_token(self):
        """Invalid [excited] token should be removed."""
        from vortex.models.bark import _clean_text_for_bark
        text = "I'm so [excited] about this"
        cleaned = _clean_text_for_bark(text)
        assert "[excited]" not in cleaned
        assert "excited" not in cleaned  # Entire token removed

    def test_strips_invalid_fast_token(self):
        """Invalid [fast] token should be removed."""
        from vortex.models.bark import _clean_text_for_bark
        text = "We need to go [fast] right now"
        cleaned = _clean_text_for_bark(text)
        assert "[fast]" not in cleaned
        assert "fast" not in cleaned

    def test_strips_asterisk_stage_directions(self):
        """Asterisk stage directions like *looks around* should be removed."""
        from vortex.models.bark import _clean_text_for_bark
        text = "Hello *looks around nervously* how are you"
        cleaned = _clean_text_for_bark(text)
        assert "*" not in cleaned
        assert "looks around" not in cleaned
        assert "Hello" in cleaned
        assert "how are you" in cleaned

    def test_strips_asterisk_gasps_direction(self):
        """Asterisk *gasps* should be removed (use [gasps] instead)."""
        from vortex.models.bark import _clean_text_for_bark
        text = "Oh my *gasps* what is that"
        cleaned = _clean_text_for_bark(text)
        assert "*gasps*" not in cleaned
        assert "gasps" not in cleaned

    def test_multiple_valid_tokens_preserved(self):
        """Multiple valid tokens in one string should all be preserved."""
        from vortex.models.bark import _clean_text_for_bark
        text = "[sighs] I can't believe [laughs] this is happening [gasps]"
        cleaned = _clean_text_for_bark(text)
        assert "[sighs]" in cleaned
        assert "[laughs]" in cleaned
        assert "[gasps]" in cleaned

    def test_mixed_valid_and_invalid_tokens(self):
        """Valid tokens preserved while invalid tokens removed."""
        from vortex.models.bark import _clean_text_for_bark
        text = "I'm [excited] to see you [laughs] let's go [fast]"
        cleaned = _clean_text_for_bark(text)
        assert "[laughs]" in cleaned
        assert "[excited]" not in cleaned
        assert "[fast]" not in cleaned
        assert "excited" not in cleaned
        assert "fast" not in cleaned

    def test_preserves_ellipsis(self):
        """Ellipsis should be preserved as valid token."""
        from vortex.models.bark import _clean_text_for_bark
        text = "Wait... what?"
        cleaned = _clean_text_for_bark(text)
        assert "..." in cleaned

    def test_preserves_em_dash(self):
        """Em dash should be preserved as valid token."""
        from vortex.models.bark import _clean_text_for_bark
        text = "I think—no wait—yes that's right"
        cleaned = _clean_text_for_bark(text)
        assert "—" in cleaned


class TestSentenceSplitting:
    """Test suite for sentence splitting functionality."""

    def test_sentence_splitting(self):
        """Should split long text into sentences."""
        from vortex.models.bark import split_into_sentences

        text = "Hello world. This is a test. It works!"
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "This is a test."
        assert sentences[2] == "It works!"

    def test_sentence_splitting_preserves_bark_tokens(self):
        """Should preserve Bark tokens when splitting."""
        from vortex.models.bark import split_into_sentences

        text = "Hello [laughs]. This is fun! Really [sighs]."
        sentences = split_into_sentences(text)
        assert "[laughs]" in sentences[0]
        assert "[sighs]" in sentences[2]

    def test_sentence_splitting_single_sentence(self):
        """Should return single sentence in list."""
        from vortex.models.bark import split_into_sentences

        text = "Hello world"
        sentences = split_into_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == "Hello world"

    def test_sentence_splitting_empty_string(self):
        """Should return empty list for empty string."""
        from vortex.models.bark import split_into_sentences

        text = ""
        sentences = split_into_sentences(text)
        assert len(sentences) == 0

    def test_sentence_splitting_multiple_punctuation(self):
        """Should handle multiple sentence-ending punctuation."""
        from vortex.models.bark import split_into_sentences

        text = "What? Really! Yes."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "What?"
        assert sentences[1] == "Really!"
        assert sentences[2] == "Yes."


class TestSynthesizeWithSentenceSplitting:
    """Test suite for synthesize with sentence splitting."""

    def test_synthesize_concatenates_long_text(self):
        """Test that long text is split and audio segments are concatenated."""
        with patch('vortex.models.bark.preload_models'):
            with patch('vortex.models.bark.generate_audio') as mock_gen:
                from vortex.models.bark import BarkVoiceEngine

                # Mock returns different length audio for each call
                mock_gen.side_effect = [
                    np.random.randn(12000).astype(np.float32),  # First sentence
                    np.random.randn(10000).astype(np.float32),  # Second sentence
                    np.random.randn(8000).astype(np.float32),   # Third sentence
                ]

                engine = BarkVoiceEngine(device="cpu")
                result = engine.synthesize(
                    text="Hello world. This is a test. It works!",
                    voice_id="rick_c137"
                )

                # Should have called generate_audio 3 times (once per sentence)
                assert mock_gen.call_count == 3
                assert isinstance(result, torch.Tensor)

    def test_synthesize_short_text_no_split(self):
        """Test that short text (single sentence) is not split."""
        with patch('vortex.models.bark.preload_models'):
            with patch('vortex.models.bark.generate_audio') as mock_gen:
                from vortex.models.bark import BarkVoiceEngine

                mock_gen.return_value = np.random.randn(24000).astype(np.float32)

                engine = BarkVoiceEngine(device="cpu")
                result = engine.synthesize(
                    text="Hello world",
                    voice_id="rick_c137"
                )

                # Should have called generate_audio only once
                assert mock_gen.call_count == 1
                assert isinstance(result, torch.Tensor)


class TestUnbracketedStageDirectionConversion:
    """Test suite for converting unbracketed stage directions to Bark tokens."""

    def test_converts_sigh_to_sighs_token(self):
        """Unbracketed 'Sigh' should become [sighs]."""
        from vortex.models.bark import _clean_text_for_bark
        text = "Sigh. I'm so tired of this."
        cleaned = _clean_text_for_bark(text)
        assert "[sighs]" in cleaned

    def test_converts_lowercase_sighs(self):
        """Lowercase 'sighs' should become [sighs]."""
        from vortex.models.bark import _clean_text_for_bark
        text = "He sighs and walks away."
        cleaned = _clean_text_for_bark(text)
        assert "[sighs]" in cleaned

    def test_converts_laughs_to_laughs_token(self):
        """Unbracketed 'Laughs' should become [laughs]."""
        from vortex.models.bark import _clean_text_for_bark
        text = "Laughs uncontrollably at the joke."
        cleaned = _clean_text_for_bark(text)
        assert "[laughs]" in cleaned

    def test_converts_laughter_to_laughs_token(self):
        """Unbracketed 'Laughter' should become [laughs]."""
        from vortex.models.bark import _clean_text_for_bark
        text = "Laughter echoes through the room."
        cleaned = _clean_text_for_bark(text)
        assert "[laughs]" in cleaned

    def test_converts_gasps_to_gasps_token(self):
        """Unbracketed 'Gasps' should become [gasps]."""
        from vortex.models.bark import _clean_text_for_bark
        text = "Gasps in surprise at the reveal."
        cleaned = _clean_text_for_bark(text)
        assert "[gasps]" in cleaned

    def test_word_boundary_prevents_partial_match(self):
        """Should not match 'sigh' within words like 'sight'."""
        from vortex.models.bark import _clean_text_for_bark
        text = "The sight was laughable at first glance."
        cleaned = _clean_text_for_bark(text)
        assert "[sighs]" not in cleaned  # 'sight' should NOT match
        assert "[laughs]" not in cleaned  # 'laughable' should NOT match
        assert "sight" in cleaned
        assert "laughable" in cleaned

    def test_preserves_already_bracketed_tokens(self):
        """Already correct [sighs] tokens should remain unchanged."""
        from vortex.models.bark import _clean_text_for_bark
        text = "[sighs] I'm so tired [laughs] but happy."
        cleaned = _clean_text_for_bark(text)
        assert "[sighs]" in cleaned
        assert "[laughs]" in cleaned
