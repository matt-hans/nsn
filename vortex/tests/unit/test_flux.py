"""Unit tests for Flux-Schnell model wrapper (mocked, no GPU required).

Tests the FluxModel interface, config validation, prompt handling, and error cases.
Real GPU tests would be in tests/integration/test_flux_generation.py
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


class TestFluxConfig:
    """Test FluxConfig dataclass validation."""

    def test_default_config_values(self):
        """Test FluxConfig has correct 720x480 defaults for CogVideoX I2V."""
        from vortex.models.flux import FluxConfig

        config = FluxConfig()
        assert config.model_id == "black-forest-labs/FLUX.1-schnell"
        assert config.height == 480  # CogVideoX I2V native height
        assert config.width == 720  # CogVideoX I2V native width
        assert config.num_inference_steps == 4
        assert config.guidance_scale == 0.0
        assert config.max_sequence_length == 256

    def test_custom_config_values(self):
        """Test FluxConfig accepts custom resolution values."""
        from vortex.models.flux import FluxConfig

        config = FluxConfig(height=512, width=512, num_inference_steps=8)
        assert config.height == 512
        assert config.width == 512
        assert config.num_inference_steps == 8

    def test_config_height_must_be_divisible_by_16(self):
        """Test that height must be divisible by 16."""
        from vortex.models.flux import FluxConfig

        with pytest.raises(ValueError, match="height must be divisible by 16"):
            FluxConfig(height=473)

    def test_config_width_must_be_divisible_by_16(self):
        """Test that width must be divisible by 16."""
        from vortex.models.flux import FluxConfig

        with pytest.raises(ValueError, match="width must be divisible by 16"):
            FluxConfig(width=713)

    def test_config_steps_must_be_positive(self):
        """Test that num_inference_steps must be >= 1."""
        from vortex.models.flux import FluxConfig

        with pytest.raises(ValueError, match="num_inference_steps must be >= 1"):
            FluxConfig(num_inference_steps=0)

    def test_720x480_divisible_by_16(self):
        """Verify 720x480 is valid (divisible by 16)."""
        from vortex.models.flux import FluxConfig

        # Should not raise
        config = FluxConfig(height=480, width=720)
        assert config.height % 16 == 0
        assert config.width % 16 == 0


class TestFluxModelInit:
    """Test FluxModel initialization and validation."""

    def test_default_model_values(self):
        """Test FluxModel has correct defaults."""
        from vortex.models.flux import FluxModel

        model = FluxModel()
        assert model.device == "cuda"
        assert model.cache_dir is None
        assert model.quantization == "nf4"
        assert model.config.height == 480
        assert model.config.width == 720

    def test_model_cpu_device(self):
        """Test FluxModel accepts cpu device."""
        from vortex.models.flux import FluxModel

        model = FluxModel(device="cpu")
        assert model.device == "cpu"

    def test_model_invalid_device_raises(self):
        """Test FluxModel rejects invalid device."""
        from vortex.models.flux import FluxModel

        with pytest.raises(ValueError, match="device must be 'cuda' or 'cpu'"):
            FluxModel(device="invalid")

    def test_model_invalid_quantization_raises(self):
        """Test FluxModel rejects invalid quantization."""
        from vortex.models.flux import FluxModel

        with pytest.raises(ValueError, match="quantization must be 'nf4' or 'none'"):
            FluxModel(quantization="fp16")

    def test_model_is_not_loaded_initially(self):
        """Test FluxModel is not loaded after init."""
        from vortex.models.flux import FluxModel

        model = FluxModel()
        assert not model.is_loaded


class TestTextureAnchorSuffix:
    """Test the TEXTURE_ANCHOR_SUFFIX constant."""

    def test_texture_anchor_suffix_exists(self):
        """Test TEXTURE_ANCHOR_SUFFIX constant is defined."""
        from vortex.models.flux import TEXTURE_ANCHOR_SUFFIX

        assert isinstance(TEXTURE_ANCHOR_SUFFIX, str)
        assert len(TEXTURE_ANCHOR_SUFFIX) > 0

    def test_texture_anchor_suffix_content(self):
        """Test TEXTURE_ANCHOR_SUFFIX has texture keywords."""
        from vortex.models.flux import TEXTURE_ANCHOR_SUFFIX

        # Should contain texture-related terms for domain gap bridging
        assert "texture" in TEXTURE_ANCHOR_SUFFIX.lower()
        assert "grain" in TEXTURE_ANCHOR_SUFFIX.lower()


class TestFluxModelGenerate:
    """Test FluxModel generate() method with mocked pipeline."""

    @patch("diffusers.FluxPipeline")
    @patch("diffusers.quantizers.PipelineQuantizationConfig")
    def test_generate_empty_prompt_raises(self, mock_quant_config, mock_pipeline_class):
        """Test generate() raises on empty prompt."""
        from vortex.models.flux import FluxModel

        # Set up mock
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        model = FluxModel()
        model.load()

        with pytest.raises(ValueError, match="prompt cannot be empty"):
            model.generate(prompt="")

        with pytest.raises(ValueError, match="prompt cannot be empty"):
            model.generate(prompt="   ")

    @patch("diffusers.FluxPipeline")
    @patch("diffusers.quantizers.PipelineQuantizationConfig")
    def test_generate_returns_correct_shape(self, mock_quant_config, mock_pipeline_class):
        """Test generate() returns tensor with correct 720x480 shape."""
        from vortex.models.flux import FluxModel

        # Set up mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Mock the generate result - PIL image at 720x480
        mock_pil_image = MagicMock()
        mock_pil_image.__array__ = MagicMock(
            return_value=np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)
        )
        mock_result = MagicMock()
        mock_result.images = [mock_pil_image]
        mock_pipeline.return_value = mock_result

        model = FluxModel(device="cpu")
        model.load()
        result = model.generate(prompt="a test image")

        # Verify output shape is [3, 480, 720] (CHW format)
        assert result.shape == (3, 480, 720)
        assert result.dtype == torch.float32

    @patch("diffusers.FluxPipeline")
    @patch("diffusers.quantizers.PipelineQuantizationConfig")
    def test_generate_uses_config_resolution(self, mock_quant_config, mock_pipeline_class):
        """Test generate() uses config height/width."""
        from vortex.models.flux import FluxConfig, FluxModel

        # Set up mock
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Mock PIL image output
        mock_pil_image = MagicMock()
        mock_pil_image.__array__ = MagicMock(
            return_value=np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)
        )
        mock_result = MagicMock()
        mock_result.images = [mock_pil_image]
        mock_pipeline.return_value = mock_result

        config = FluxConfig(height=480, width=720)
        model = FluxModel(config=config, device="cpu")
        model.load()
        model.generate(prompt="test")

        # Verify pipeline was called with correct dimensions
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["height"] == 480
        assert call_kwargs["width"] == 720


class TestFluxModelUnload:
    """Test FluxModel unload() method."""

    @patch("diffusers.FluxPipeline")
    @patch("diffusers.quantizers.PipelineQuantizationConfig")
    @patch("torch.cuda.empty_cache")
    def test_unload_clears_pipeline(self, mock_empty_cache, mock_quant_config, mock_pipeline_class):
        """Test unload() releases pipeline and clears CUDA cache."""
        from vortex.models.flux import FluxModel

        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        model = FluxModel()
        model.load()
        assert model.is_loaded

        model.unload()
        assert not model.is_loaded
        mock_empty_cache.assert_called()

    def test_unload_safe_when_not_loaded(self):
        """Test unload() is safe to call when not loaded."""
        from vortex.models.flux import FluxModel

        model = FluxModel()
        assert not model.is_loaded

        # Should not raise
        model.unload()
        assert not model.is_loaded


class TestLoadFluxFactory:
    """Test load_flux() factory function."""

    @patch("diffusers.FluxPipeline")
    @patch("diffusers.quantizers.PipelineQuantizationConfig")
    def test_load_flux_returns_loaded_model(self, mock_quant_config, mock_pipeline_class):
        """Test load_flux() returns a loaded FluxModel."""
        from vortex.models.flux import FluxModel, load_flux

        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        model = load_flux(device="cpu", quantization="none")

        assert isinstance(model, FluxModel)
        assert model.is_loaded
        assert model.device == "cpu"
        assert model.quantization == "none"

    def test_load_flux_schnell_alias_exists(self):
        """Test load_flux_schnell alias is defined for backward compat."""
        from vortex.models.flux import load_flux, load_flux_schnell

        assert load_flux_schnell is load_flux
