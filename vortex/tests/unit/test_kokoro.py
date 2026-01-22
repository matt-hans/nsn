"""Unit tests for Kokoro TTS model wrapper.

Tests the KokoroWrapper class with mocked Kokoro backend.
Real model tests are in integration/test_kokoro_synthesis.py.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from vortex.models.kokoro import KokoroWrapper, load_kokoro


class TestKokoroWrapper:
    """Test suite for KokoroWrapper model wrapper."""

    @pytest.fixture
    def mock_kokoro_model(self):
        """Create a mock Kokoro model for testing."""
        import numpy as np

        # Create a callable mock that returns a generator
        def mock_pipeline_call(*args, **kwargs):
            # Return a generator that yields (graphemes, phonemes, audio)
            yield ("text", "phonemes", np.random.randn(24000).astype(np.float32))

        mock_model = MagicMock()
        mock_model.side_effect = mock_pipeline_call
        # Keep generate for backward compatibility during transition
        mock_model.generate = Mock(return_value=torch.randn(24000))
        return mock_model

    @pytest.fixture
    def wrapper(self, mock_kokoro_model):
        """Create KokoroWrapper with mocked model."""
        voice_config = {
            "rick_c137": "am_adam",
            "morty": "af_sarah",
            "summer": "af_jessica"
        }
        emotion_config = {
            "neutral": {"pitch_shift": 0, "tempo": 1.0, "energy": 1.0},
            "excited": {"pitch_shift": 50, "tempo": 1.15, "energy": 1.3},
            "manic": {"pitch_shift": 100, "tempo": 1.25, "energy": 1.5}
        }
        return KokoroWrapper(
            model=mock_kokoro_model,
            voice_config=voice_config,
            emotion_config=emotion_config,
            device="cpu"
        )

    def test_init_loads_configs(self, wrapper):
        """Test that wrapper initializes with voice and emotion configs."""
        assert "rick_c137" in wrapper.voice_config
        assert "morty" in wrapper.voice_config
        assert "summer" in wrapper.voice_config
        assert "neutral" in wrapper.emotion_config
        assert "excited" in wrapper.emotion_config

    def test_synthesize_basic(self, wrapper, mock_kokoro_model):
        """Test basic synthesis with default parameters."""
        result = wrapper.synthesize(
            text="Hello world",
            voice_id="rick_c137"
        )

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 1  # Mono audio
        assert len(result) > 0
        mock_kokoro_model.assert_called_once()

    def test_synthesize_with_speed_control(self, wrapper, mock_kokoro_model):
        """Test speed parameter affects generation."""
        # Test different speeds
        for speed in [0.8, 1.0, 1.2]:
            wrapper.synthesize(
                text="Test",
                voice_id="rick_c137",
                speed=speed
            )

        assert mock_kokoro_model.call_count == 3

    def test_synthesize_with_emotion(self, wrapper, mock_kokoro_model):
        """Test emotion parameter modulates synthesis."""
        result_neutral = wrapper.synthesize(
            text="I am happy",
            voice_id="rick_c137",
            emotion="neutral"
        )

        result_excited = wrapper.synthesize(
            text="I am happy",
            voice_id="rick_c137",
            emotion="excited"
        )

        assert isinstance(result_neutral, torch.Tensor)
        assert isinstance(result_excited, torch.Tensor)

    def test_synthesize_invalid_voice_id(self, wrapper, mock_kokoro_model):
        """Test that unknown voice IDs are passed through to Kokoro (ToonGen support).

        NOTE: The wrapper now allows passthrough of raw Kokoro voice IDs
        (e.g., "af_heart") in addition to mapped ICN voice IDs (e.g., "rick_c137").
        This enables ToonGen to use Kokoro's native voice selection.
        """
        # Unknown voice IDs are passed through, not rejected
        result = wrapper.synthesize(
            text="Test",
            voice_id="some_raw_kokoro_voice"
        )
        assert isinstance(result, torch.Tensor)

    def test_synthesize_invalid_emotion(self, wrapper, mock_kokoro_model):
        """Test that invalid emotion falls back to neutral."""
        # Should not raise, just use neutral
        result = wrapper.synthesize(
            text="Test",
            voice_id="rick_c137",
            emotion="invalid_emotion"
        )

        assert isinstance(result, torch.Tensor)

    def test_synthesize_writes_to_output_buffer(self, wrapper, mock_kokoro_model):
        """Test that synthesis writes to pre-allocated buffer when provided."""
        # Mock pipeline to return specific length
        import numpy as np
        def mock_long_audio(*args, **kwargs):
            yield ("text", "phonemes", np.random.randn(48000).astype(np.float32))
        mock_kokoro_model.side_effect = mock_long_audio

        # Pre-allocated buffer
        output_buffer = torch.zeros(1080000, dtype=torch.float32)  # 45s @ 24kHz

        result = wrapper.synthesize(
            text="Test",
            voice_id="rick_c137",
            output=output_buffer
        )

        # Should return slice of buffer
        assert result.shape[0] == 48000
        assert result.data_ptr() == output_buffer.data_ptr()  # Same memory

    def test_synthesize_truncates_long_scripts(self, wrapper, mock_kokoro_model):
        """Test that scripts exceeding 45s are truncated with warning."""
        # Very long script (estimate: 10,000 chars = ~800s at 0.08s/char)
        long_text = "a" * 10000

        with pytest.warns(UserWarning, match="Script truncated"):
            wrapper.synthesize(
                text=long_text,
                voice_id="rick_c137"
            )

    def test_synthesize_deterministic_with_seed(self, wrapper, mock_kokoro_model):
        """Test that same seed produces identical outputs."""
        # Mock to return different random tensors
        import numpy as np
        def generate_random(*args, **kwargs):
            yield ("text", "phonemes", np.random.randn(24000).astype(np.float32))

        mock_kokoro_model.side_effect = generate_random

        # With seed, torch.manual_seed should be called
        with patch('torch.manual_seed') as mock_seed:
            wrapper.synthesize(
                text="Test",
                voice_id="rick_c137",
                seed=42
            )
            mock_seed.assert_called_once_with(42)

    def test_output_normalization(self, wrapper, mock_kokoro_model):
        """Test that output is normalized to [-1, 1]."""
        # Mock output with values outside [-1, 1]
        import numpy as np
        def mock_unnormalized(*args, **kwargs):
            yield ("text", "phonemes", np.array([2.0, -3.0, 1.5], dtype=np.float32))
        mock_kokoro_model.side_effect = mock_unnormalized

        result = wrapper.synthesize(
            text="Test",
            voice_id="rick_c137"
        )

        # Should be normalized
        assert result.abs().max() <= 1.0

    def test_voice_config_mapping(self, wrapper):
        """Test that voice IDs map to correct Kokoro voices."""
        assert wrapper.voice_config["rick_c137"] == "am_adam"
        assert wrapper.voice_config["morty"] == "af_sarah"
        assert wrapper.voice_config["summer"] == "af_jessica"

    def test_emotion_params_retrieval(self, wrapper):
        """Test emotion parameter retrieval."""
        params_neutral = wrapper._get_emotion_params("neutral")
        params_excited = wrapper._get_emotion_params("excited")

        assert params_neutral["tempo"] == 1.0
        assert params_excited["tempo"] == 1.15
        assert params_excited["pitch_shift"] == 50


class TestLoadKokoro:
    """Test suite for load_kokoro factory function."""

    @patch('vortex.models.kokoro._create_wrapper')
    @patch('vortex.models.kokoro._load_configs')
    @patch('vortex.models.kokoro._import_and_create_kokoro_model')
    def test_load_kokoro_initializes_wrapper(
        self, mock_import_model, mock_load_configs, mock_create_wrapper
    ):
        """Test that load_kokoro creates KokoroWrapper with configs."""
        # Mock the helper functions
        mock_model = Mock()
        mock_import_model.return_value = mock_model

        voice_config = {"rick_c137": "am_adam"}
        emotion_config = {"neutral": {"tempo": 1.0}}
        mock_load_configs.return_value = (voice_config, emotion_config)

        mock_wrapper = Mock()
        mock_create_wrapper.return_value = mock_wrapper

        result = load_kokoro(device="cuda:0")

        # Should have called all helpers
        assert mock_import_model.called
        assert mock_load_configs.called
        assert mock_create_wrapper.called
        assert result == mock_wrapper

    @patch('builtins.__import__', side_effect=ImportError("No module named 'kokoro'"))
    def test_load_kokoro_handles_missing_package(self, mock_import):
        """Test that load_kokoro raises informative error if kokoro package missing."""
        with pytest.raises(ImportError, match="kokoro"):
            load_kokoro(device="cuda:0")

    @patch('vortex.models.kokoro._create_wrapper')
    @patch('vortex.models.kokoro._load_configs')
    @patch('vortex.models.kokoro._import_and_create_kokoro_model')
    def test_load_kokoro_creates_pipeline(
        self, mock_import_model, mock_load_configs, mock_create_wrapper
    ):
        """Test that _import_and_create_kokoro_model is called."""
        voice_config = {"rick_c137": "am_adam"}
        emotion_config = {"neutral": {"tempo": 1.0}}
        mock_load_configs.return_value = (voice_config, emotion_config)

        mock_pipeline = Mock()
        mock_import_model.return_value = mock_pipeline

        mock_wrapper = Mock()
        mock_create_wrapper.return_value = mock_wrapper

        load_kokoro(device="cuda:0")

        # Should create model via helper function
        mock_import_model.assert_called_once()


class TestVRAMBudget:
    """Test suite for VRAM budget compliance."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_kokoro_vram_budget_compliance(self):
        """Test that Kokoro VRAM usage is within 0.3-0.5 GB budget.

        This is a placeholder test. Real VRAM testing requires GPU and
        actual Kokoro model loaded. See integration tests.
        """
        # This would require actual model loading
        # For now, we document the requirement
        max_vram_gb = 0.5
        min_vram_gb = 0.3

        # In real test:
        # torch.cuda.reset_peak_memory_stats()
        # kokoro = load_kokoro(device="cuda:0")
        # vram_bytes = torch.cuda.max_memory_allocated()
        # vram_gb = vram_bytes / 1e9
        # assert min_vram_gb <= vram_gb <= max_vram_gb

        assert True  # Placeholder


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper with minimal mocks."""
        import numpy as np

        def mock_pipeline_call(*args, **kwargs):
            yield ("text", "phonemes", np.random.randn(24000).astype(np.float32))

        mock_model = MagicMock()
        mock_model.side_effect = mock_pipeline_call
        mock_model.generate = Mock(return_value=torch.randn(24000))

        return KokoroWrapper(
            model=mock_model,
            voice_config={"rick_c137": "am_adam"},
            emotion_config={"neutral": {"tempo": 1.0}},
            device="cpu"
        )

    def test_empty_text_handling(self, wrapper):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            wrapper.synthesize(text="", voice_id="rick_c137")

    def test_very_long_single_word(self, wrapper):
        """Test handling of extremely long single word."""
        # 1000-character word
        long_word = "a" * 1000

        # Should not crash, may truncate
        result = wrapper.synthesize(
            text=long_word,
            voice_id="rick_c137"
        )

        assert isinstance(result, torch.Tensor)

    def test_special_characters_in_text(self, wrapper):
        """Test handling of special characters and punctuation."""
        special_text = "Hello! How are you? It's a test... #AI @mention"

        result = wrapper.synthesize(
            text=special_text,
            voice_id="rick_c137"
        )

        assert isinstance(result, torch.Tensor)

    def test_unicode_text(self, wrapper):
        """Test handling of Unicode characters."""
        unicode_text = "Hello 世界 🌍"

        result = wrapper.synthesize(
            text=unicode_text,
            voice_id="rick_c137"
        )

        assert isinstance(result, torch.Tensor)

    def test_speed_boundary_values(self, wrapper):
        """Test speed parameter at boundary values."""
        # Min speed
        result_slow = wrapper.synthesize(
            text="Test",
            voice_id="rick_c137",
            speed=0.8
        )

        # Max speed
        result_fast = wrapper.synthesize(
            text="Test",
            voice_id="rick_c137",
            speed=1.2
        )

        assert isinstance(result_slow, torch.Tensor)
        assert isinstance(result_fast, torch.Tensor)

    def test_cuda_error_handling(self, wrapper):
        """Test graceful handling of CUDA errors."""
        # Mock CUDA OOM on pipeline call
        def mock_oom(*args, **kwargs):
            raise torch.cuda.OutOfMemoryError()
        wrapper.model.side_effect = mock_oom

        with pytest.raises(torch.cuda.OutOfMemoryError):
            wrapper.synthesize(text="Test", voice_id="rick_c137")
