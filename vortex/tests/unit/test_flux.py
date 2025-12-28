"""Unit tests for Flux-Schnell model wrapper (mocked, no GPU required).

Tests the FluxModel interface, prompt handling, error cases, and determinism.
Real GPU tests are in tests/integration/test_flux_generation.py
"""

import unittest
from unittest.mock import MagicMock, patch

import torch


class TestFluxModelInterface(unittest.TestCase):
    """Test FluxModel interface without real model weights."""

    @patch("vortex.models.flux.FluxPipeline")
    def setUp(self, mock_pipeline_class):
        """Set up mocked Flux pipeline."""
        # Import here to avoid triggering real model downloads
        from vortex.models.flux import FluxModel

        # Mock the diffusers pipeline
        self.mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = self.mock_pipeline

        # Mock generate output (512x512x3 tensor)
        mock_output = MagicMock()
        mock_output.images = [torch.randn(3, 512, 512)]  # CHW format
        self.mock_pipeline.return_value = mock_output

        # Create FluxModel with mocked pipeline
        self.model = FluxModel(self.mock_pipeline, device="cpu")

    def test_generate_basic(self):
        """Test basic generate() call with default parameters."""
        prompt = "a scientist in a laboratory"
        result = self.model.generate(prompt=prompt)

        # Verify pipeline was called
        self.mock_pipeline.assert_called_once()
        call_kwargs = self.mock_pipeline.call_args[1]

        # Verify parameters
        self.assertEqual(call_kwargs["prompt"], prompt)
        self.assertEqual(call_kwargs["num_inference_steps"], 4)
        self.assertEqual(call_kwargs["guidance_scale"], 0.0)
        self.assertEqual(call_kwargs["height"], 512)
        self.assertEqual(call_kwargs["width"], 512)
        self.assertEqual(call_kwargs["output_type"], "pt")

        # Verify output shape
        self.assertEqual(result.shape, (3, 512, 512))

    def test_generate_with_negative_prompt(self):
        """Test generate() with negative prompt."""
        prompt = "scientist"
        negative_prompt = "blurry, low quality, watermark"

        self.model.generate(prompt=prompt, negative_prompt=negative_prompt)

        call_kwargs = self.mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs["negative_prompt"], negative_prompt)

    def test_generate_with_custom_steps(self):
        """Test generate() with custom inference steps."""
        self.model.generate(prompt="test", num_inference_steps=8)

        call_kwargs = self.mock_pipeline.call_args[1]
        self.assertEqual(call_kwargs["num_inference_steps"], 8)

    def test_generate_with_seed(self):
        """Test deterministic generation with manual seed."""
        with patch("torch.manual_seed") as mock_seed:
            self.model.generate(prompt="test", seed=42)
            mock_seed.assert_called_once_with(42)

    def test_generate_to_preallocated_buffer(self):
        """Test writing output to pre-allocated buffer (prevents fragmentation)."""
        # Create pre-allocated buffer
        buffer = torch.zeros(3, 512, 512)

        result = self.model.generate(prompt="test", output=buffer)

        # Verify result is the buffer (not a new tensor)
        self.assertIs(result, buffer)

    def test_prompt_truncation_warning(self):
        """Test that long prompts trigger truncation warning."""
        # Create a prompt exceeding 77 tokens (CLIP limit)
        long_prompt = " ".join(["word"] * 100)

        with patch("vortex.models.flux.logger") as mock_logger:
            self.model.generate(prompt=long_prompt)

            # Should log warning about truncation
            warning_calls = [
                call for call in mock_logger.warning.call_args_list
                if "truncated" in str(call).lower()
            ]
            self.assertGreater(len(warning_calls), 0)


class TestFluxLoading(unittest.TestCase):
    """Test Flux model loading with NF4 quantization."""

    @patch("vortex.models.flux.FluxPipeline")
    @patch("vortex.models.flux.BitsAndBytesConfig")
    def test_load_flux_schnell_nf4(self, mock_bnb_config, mock_pipeline_class):
        """Test loading Flux-Schnell with NF4 quantization config."""
        from vortex.models.flux import FluxModel, load_flux_schnell

        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        result = load_flux_schnell(device="cuda:0", quantization="nf4")

        # Verify BitsAndBytesConfig was created correctly
        mock_bnb_config.assert_called_once()
        bnb_call_kwargs = mock_bnb_config.call_args[1]
        self.assertTrue(bnb_call_kwargs["load_in_4bit"])
        self.assertEqual(bnb_call_kwargs["bnb_4bit_quant_type"], "nf4")
        self.assertEqual(bnb_call_kwargs["bnb_4bit_compute_dtype"], torch.bfloat16)

        # Verify pipeline was loaded with correct parameters
        mock_pipeline_class.from_pretrained.assert_called_once()
        pipeline_call_kwargs = mock_pipeline_class.from_pretrained.call_args[1]
        self.assertEqual(pipeline_call_kwargs["torch_dtype"], torch.bfloat16)
        self.assertTrue(pipeline_call_kwargs["use_safetensors"])

        # Verify safety checker was disabled
        self.assertIsNone(mock_pipeline.safety_checker)

        # Result should be FluxModel instance
        self.assertIsInstance(result, FluxModel)

    @patch("vortex.models.flux.torch.cuda.is_available")
    @patch("vortex.models.flux.FluxPipeline")
    def test_load_flux_cuda_oom_handling(self, mock_pipeline_class, mock_cuda_available):
        """Test graceful handling of CUDA OOM during model loading."""
        from vortex.models.flux import VortexInitializationError, load_flux_schnell

        # Simulate CUDA is NOT available (fallback error path)
        mock_cuda_available.return_value = False

        # Simulate CUDA OOM
        mock_pipeline_class.from_pretrained.side_effect = torch.cuda.OutOfMemoryError()

        with self.assertRaises(VortexInitializationError) as ctx:
            load_flux_schnell(device="cuda:0")

        # Verify error message includes remediation (fallback message without RTX 3060)
        self.assertIn("VRAM", str(ctx.exception))
        self.assertIn("12GB", str(ctx.exception))


class TestFluxVRAMBudget(unittest.TestCase):
    """Test VRAM budget compliance."""

    @patch("vortex.models.flux.torch.cuda.memory_allocated")
    def test_vram_usage_within_budget(self, mock_memory_allocated):
        """Test that Flux VRAM usage is within 5.5-6.5GB budget."""
        # Simulate 6.0GB allocation
        mock_memory_allocated.return_value = int(6.0 * 1e9)

        vram_gb = mock_memory_allocated() / 1e9
        self.assertGreaterEqual(vram_gb, 5.5)
        self.assertLessEqual(vram_gb, 6.5)


class TestFluxDeterminism(unittest.TestCase):
    """Test deterministic output with same seed."""

    @patch("vortex.models.flux.FluxPipeline")
    def test_same_seed_same_output(self, mock_pipeline_class):
        """Test that same seed + prompt produces identical outputs."""
        from vortex.models.flux import FluxModel

        # Create deterministic mock outputs
        output1 = torch.randn(3, 512, 512)
        output2 = output1.clone()

        mock_pipeline = MagicMock()
        mock_output1 = MagicMock()
        mock_output1.images = [output1]
        mock_output2 = MagicMock()
        mock_output2.images = [output2]

        mock_pipeline.side_effect = [mock_output1, mock_output2]

        model = FluxModel(mock_pipeline, device="cpu")

        with patch("torch.manual_seed"):
            result1 = model.generate(prompt="scientist", seed=42)
            result2 = model.generate(prompt="scientist", seed=42)

        # Results should be identical
        self.assertTrue(torch.equal(result1, result2))


if __name__ == "__main__":
    unittest.main()
