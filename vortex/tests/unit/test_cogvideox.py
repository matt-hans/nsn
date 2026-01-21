"""Unit tests for CogVideoX model wrapper.

Tests the CogVideoXModel class including the generate_chain method
for autoregressive video chaining.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest
import torch

from vortex.models.cogvideox import (
    CogVideoXModel,
    VideoGenerationConfig,
    load_cogvideox,
)


class TestVideoGenerationConfig:
    """Tests for VideoGenerationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VideoGenerationConfig()
        assert config.num_frames == 49
        assert config.guidance_scale == 6.0
        assert config.num_inference_steps == 50
        assert config.fps == 8
        assert config.height == 480
        assert config.width == 720

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = VideoGenerationConfig(
            num_frames=33,
            guidance_scale=7.5,
            num_inference_steps=30,
            fps=16,
            height=512,
            width=768,
        )
        assert config.num_frames == 33
        assert config.guidance_scale == 7.5
        assert config.num_inference_steps == 30
        assert config.fps == 16
        assert config.height == 512
        assert config.width == 768

    def test_invalid_height(self) -> None:
        """Test that height must be divisible by 16."""
        with pytest.raises(ValueError, match="height must be divisible by 16"):
            VideoGenerationConfig(height=481)

    def test_invalid_width(self) -> None:
        """Test that width must be divisible by 16."""
        with pytest.raises(ValueError, match="width must be divisible by 16"):
            VideoGenerationConfig(width=721)

    def test_invalid_num_frames(self) -> None:
        """Test that num_frames must be >= 1."""
        with pytest.raises(ValueError, match="num_frames must be >= 1"):
            VideoGenerationConfig(num_frames=0)

    def test_invalid_guidance_scale(self) -> None:
        """Test that guidance_scale must be >= 1.0."""
        with pytest.raises(ValueError, match="guidance_scale must be >= 1.0"):
            VideoGenerationConfig(guidance_scale=0.5)

    def test_invalid_num_inference_steps(self) -> None:
        """Test that num_inference_steps must be >= 1."""
        with pytest.raises(ValueError, match="num_inference_steps must be >= 1"):
            VideoGenerationConfig(num_inference_steps=0)


class TestCogVideoXModel:
    """Tests for CogVideoXModel class."""

    def test_default_initialization(self) -> None:
        """Test default model initialization."""
        model = CogVideoXModel()
        assert model.model_id == "THUDM/CogVideoX-5b-I2V"
        assert model.device == "cuda"
        assert model.enable_cpu_offload is True
        assert model.cache_dir is None
        assert model.is_loaded is False

    def test_custom_initialization(self) -> None:
        """Test custom model initialization."""
        model = CogVideoXModel(
            device="cpu",
            enable_cpu_offload=False,
            cache_dir="/tmp/cache",
        )
        assert model.device == "cpu"
        assert model.enable_cpu_offload is False
        assert model.cache_dir == "/tmp/cache"

    def test_invalid_device(self) -> None:
        """Test that device must be 'cuda' or 'cpu'."""
        with pytest.raises(ValueError, match="device must be 'cuda' or 'cpu'"):
            CogVideoXModel(device="tpu")


class TestGenerateChainSignature:
    """Tests for the generate_chain method signature."""

    def test_generate_chain_exists(self) -> None:
        """Test that generate_chain method exists."""
        assert hasattr(CogVideoXModel, "generate_chain")

    def test_generate_chain_is_async(self) -> None:
        """Test that generate_chain is an async method."""
        assert inspect.iscoroutinefunction(CogVideoXModel.generate_chain)

    def test_generate_chain_parameters(self) -> None:
        """Test generate_chain has correct parameters."""
        sig = inspect.signature(CogVideoXModel.generate_chain)
        params = list(sig.parameters.keys())
        expected_params = [
            "self",
            "keyframe",
            "prompt",
            "target_duration",
            "config",
            "seed",
            "progress_callback",
        ]
        assert params == expected_params

    def test_generate_chain_parameter_defaults(self) -> None:
        """Test generate_chain parameter defaults."""
        sig = inspect.signature(CogVideoXModel.generate_chain)

        # Required params have no default
        assert sig.parameters["keyframe"].default is inspect.Parameter.empty
        assert sig.parameters["prompt"].default is inspect.Parameter.empty
        assert sig.parameters["target_duration"].default is inspect.Parameter.empty

        # Optional params default to None
        assert sig.parameters["config"].default is None
        assert sig.parameters["seed"].default is None
        assert sig.parameters["progress_callback"].default is None

    def test_generate_chain_annotations(self) -> None:
        """Test generate_chain has correct type annotations."""
        ann = CogVideoXModel.generate_chain.__annotations__

        assert "torch.Tensor" in str(ann.get("keyframe"))
        assert "str" in str(ann.get("prompt"))
        assert "float" in str(ann.get("target_duration"))
        assert "VideoGenerationConfig" in str(ann.get("config"))
        assert "int" in str(ann.get("seed"))
        assert "Callable" in str(ann.get("progress_callback"))
        assert "torch.Tensor" in str(ann.get("return"))


class TestGenerateChainLogic:
    """Tests for the generate_chain method logic (with mocking)."""

    @pytest.fixture
    def mock_model(self) -> CogVideoXModel:
        """Create a CogVideoXModel instance for testing."""
        return CogVideoXModel()

    @pytest.mark.asyncio
    async def test_generate_chain_single_chunk(self, mock_model: CogVideoXModel) -> None:
        """Test generate_chain with a short duration (single chunk)."""
        # Create mock chunk output (49 frames)
        mock_frames = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframe = torch.rand(3, 480, 720)
            result = await mock_model.generate_chain(
                keyframe=keyframe,
                prompt="test prompt",
                target_duration=6.0,  # 6s = 1 chunk at 49 frames, 8fps
                seed=42,
            )

            # Should have called generate_chunk once
            assert mock_gen.call_count == 1

            # Result should be trimmed to 48 frames (6.0 * 8 fps)
            assert result.shape[0] == 48

    @pytest.mark.asyncio
    async def test_generate_chain_multiple_chunks(self, mock_model: CogVideoXModel) -> None:
        """Test generate_chain with multiple chunks."""
        # Create mock chunk outputs
        chunk1 = torch.rand(49, 3, 480, 720)
        chunk2 = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = [chunk1, chunk2]

            keyframe = torch.rand(3, 480, 720)
            await mock_model.generate_chain(
                keyframe=keyframe,
                prompt="test prompt",
                target_duration=12.0,  # 12s = 2 chunks
                seed=42,
            )

            # Should have called generate_chunk twice
            assert mock_gen.call_count == 2

            # Second call should use different seed
            call_args_1 = mock_gen.call_args_list[0]
            call_args_2 = mock_gen.call_args_list[1]
            assert call_args_1.kwargs.get("seed") == 42
            assert call_args_2.kwargs.get("seed") == 43

    @pytest.mark.asyncio
    async def test_generate_chain_frame_overlap(self, mock_model: CogVideoXModel) -> None:
        """Test that subsequent chunks skip their first frame."""
        # First chunk: 49 frames, second chunk: 49 frames (but skip 1st)
        # So total frames = 49 + 48 = 97 frames
        # At 8fps, this gives us 97/8 = 12.125 seconds
        chunk1 = torch.rand(49, 3, 480, 720)
        chunk2 = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = [chunk1, chunk2]

            keyframe = torch.rand(3, 480, 720)
            # chunk_duration = 49/8 = 6.125s
            # target 10.0s needs ceil(10.0/6.125) = 2 chunks
            result = await mock_model.generate_chain(
                keyframe=keyframe,
                prompt="test prompt",
                target_duration=10.0,  # 2 chunks: 10/6.125 = 1.63, ceil to 2
                seed=42,
            )

            # Total before trimming: 49 + 48 = 97 frames
            # Target: 10 * 8 = 80 frames
            # Result should be trimmed to 80 frames
            assert result.shape[0] == 80

    @pytest.mark.asyncio
    async def test_generate_chain_progress_callback(self, mock_model: CogVideoXModel) -> None:
        """Test that progress callback is called correctly."""
        mock_frames = torch.rand(49, 3, 480, 720)
        callback_calls: list[tuple[int, int]] = []

        def progress_callback(chunk_num: int, total_chunks: int) -> None:
            callback_calls.append((chunk_num, total_chunks))

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframe = torch.rand(3, 480, 720)
            await mock_model.generate_chain(
                keyframe=keyframe,
                prompt="test prompt",
                target_duration=6.0,
                progress_callback=progress_callback,
            )

            # Should have called callback at start and end
            assert (0, 1) in callback_calls  # Start of chunk 0
            assert (1, 1) in callback_calls  # Completion

    @pytest.mark.asyncio
    async def test_generate_chain_uses_last_frame_as_keyframe(
        self, mock_model: CogVideoXModel
    ) -> None:
        """Test that each chunk uses the last frame of the previous chunk."""
        chunk1 = torch.rand(49, 3, 480, 720)
        chunk2 = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = [chunk1, chunk2]

            keyframe = torch.rand(3, 480, 720)
            await mock_model.generate_chain(
                keyframe=keyframe,
                prompt="test prompt",
                target_duration=12.0,
                seed=42,
            )

            # Second call should use last frame of first chunk as keyframe
            second_call = mock_gen.call_args_list[1]
            second_keyframe = second_call.kwargs.get("image")

            # The last frame of chunk1 (after first frame skip) is chunk1[-1]
            # But since frame overlap happens after append, we check chunk1[-1]
            expected_keyframe = chunk1[-1]
            assert torch.equal(second_keyframe, expected_keyframe)

    @pytest.mark.asyncio
    async def test_generate_chain_no_seed(self, mock_model: CogVideoXModel) -> None:
        """Test generate_chain with no seed (non-deterministic)."""
        mock_frames = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframe = torch.rand(3, 480, 720)
            await mock_model.generate_chain(
                keyframe=keyframe,
                prompt="test prompt",
                target_duration=6.0,
                seed=None,  # No seed
            )

            # generate_chunk should be called with seed=None
            call_args = mock_gen.call_args
            assert call_args.kwargs.get("seed") is None


class TestLoadCogVideoX:
    """Tests for the load_cogvideox factory function."""

    def test_load_cogvideox_creates_model(self) -> None:
        """Test that load_cogvideox creates and loads a model."""
        with patch.object(CogVideoXModel, "load") as mock_load:
            model = load_cogvideox(
                device="cpu",
                enable_cpu_offload=False,
            )

            assert isinstance(model, CogVideoXModel)
            assert model.device == "cpu"
            assert model.enable_cpu_offload is False
            mock_load.assert_called_once()
