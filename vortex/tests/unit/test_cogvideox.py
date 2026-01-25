"""Unit tests for CogVideoX model wrapper.

Tests the CogVideoXModel class including the generate_chunk method with I2V
(Image-to-Video) architecture where a keyframe image is required for generation.
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
        assert config.guidance_scale == 4.5  # Increased for temporal stability
        assert config.use_dynamic_cfg is True  # Dynamic CFG for better motion
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
        assert model.model_id == "THUDM/CogVideoX-5b-I2V"  # I2V model
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


class TestGenerateChunkSignature:
    """Tests for the generate_chunk method signature (I2V architecture)."""

    def test_generate_chunk_exists(self) -> None:
        """Test that generate_chunk method exists."""
        assert hasattr(CogVideoXModel, "generate_chunk")

    def test_generate_chunk_is_async(self) -> None:
        """Test that generate_chunk is an async method."""
        assert inspect.iscoroutinefunction(CogVideoXModel.generate_chunk)

    def test_generate_chunk_parameters(self) -> None:
        """Test generate_chunk has correct parameters for I2V."""
        sig = inspect.signature(CogVideoXModel.generate_chunk)
        params = list(sig.parameters.keys())
        expected_params = [
            "self",
            "image",  # Required: keyframe image
            "prompt",
            "config",
            "seed",
        ]
        assert params == expected_params

    def test_generate_chunk_parameter_defaults(self) -> None:
        """Test generate_chunk parameter defaults."""
        sig = inspect.signature(CogVideoXModel.generate_chunk)

        # Required params have no default
        assert sig.parameters["image"].default is inspect.Parameter.empty
        assert sig.parameters["prompt"].default is inspect.Parameter.empty

        # Optional params default to None
        assert sig.parameters["config"].default is None
        assert sig.parameters["seed"].default is None

    def test_generate_chunk_annotations(self) -> None:
        """Test generate_chunk has correct type annotations."""
        ann = CogVideoXModel.generate_chunk.__annotations__

        # Image can be tensor or PIL Image
        assert "torch.Tensor" in str(ann.get("image")) or "Image" in str(ann.get("image"))
        assert "str" in str(ann.get("prompt"))
        assert "VideoGenerationConfig" in str(ann.get("config"))
        assert "int" in str(ann.get("seed"))
        assert "torch.Tensor" in str(ann.get("return"))


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
    """Tests for the deprecated generate_chain method logic (with mocking).

    Note: generate_chain is deprecated in favor of generate_montage.
    The keyframe parameter is now used with I2V to generate video from
    the initial keyframe image.
    """

    @pytest.fixture
    def mock_model(self) -> CogVideoXModel:
        """Create a CogVideoXModel instance for testing."""
        return CogVideoXModel()

    @pytest.mark.asyncio
    async def test_generate_chain_single_chunk(self, mock_model: CogVideoXModel) -> None:
        """Test generate_chain with a short duration (single chunk)."""
        import warnings

        # Create mock chunk output (49 frames)
        mock_frames = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframe = torch.rand(3, 480, 720)
            # Expect deprecation warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = await mock_model.generate_chain(
                    keyframe=keyframe,
                    prompt="test prompt",
                    target_duration=6.0,  # 6s = 1 chunk at 49 frames, 8fps
                    seed=42,
                )
                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "deprecated" in str(w[0].message).lower()

            # Should have called generate_chunk once
            assert mock_gen.call_count == 1

            # Result should be trimmed to 48 frames (6.0 * 8 fps)
            assert result.shape[0] == 48

    @pytest.mark.asyncio
    async def test_generate_chain_multiple_chunks(self, mock_model: CogVideoXModel) -> None:
        """Test generate_chain with multiple chunks."""
        import warnings

        # Create mock chunk outputs
        chunk1 = torch.rand(49, 3, 480, 720)
        chunk2 = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = [chunk1, chunk2]

            keyframe = torch.rand(3, 480, 720)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
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

            # First call should use original keyframe, second uses last frame of chunk1
            assert call_args_1.kwargs.get("image") is not None
            assert call_args_2.kwargs.get("image") is not None

    @pytest.mark.asyncio
    async def test_generate_chain_frame_overlap(self, mock_model: CogVideoXModel) -> None:
        """Test that subsequent chunks skip their first frame."""
        import warnings

        # First chunk: 49 frames, second chunk: 49 frames (but skip 1st)
        # So total frames = 49 + 48 = 97 frames
        # At 8fps, this gives us 97/8 = 12.125 seconds
        chunk1 = torch.rand(49, 3, 480, 720)
        chunk2 = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = [chunk1, chunk2]

            keyframe = torch.rand(3, 480, 720)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
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
        import warnings

        mock_frames = torch.rand(49, 3, 480, 720)
        callback_calls: list[tuple[int, int]] = []

        def progress_callback(chunk_num: int, total_chunks: int) -> None:
            callback_calls.append((chunk_num, total_chunks))

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframe = torch.rand(3, 480, 720)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
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
    async def test_generate_chain_no_seed(self, mock_model: CogVideoXModel) -> None:
        """Test generate_chain with no seed (non-deterministic)."""
        import warnings

        mock_frames = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframe = torch.rand(3, 480, 720)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                await mock_model.generate_chain(
                    keyframe=keyframe,
                    prompt="test prompt",
                    target_duration=6.0,
                    seed=None,  # No seed
                )

            # generate_chunk should be called with seed=None
            call_args = mock_gen.call_args
            assert call_args.kwargs.get("seed") is None


class TestGenerateMontageSignature:
    """Tests for the generate_montage method signature."""

    def test_generate_montage_exists(self) -> None:
        """Test that generate_montage method exists."""
        assert hasattr(CogVideoXModel, "generate_montage")

    def test_generate_montage_is_async(self) -> None:
        """Test that generate_montage is an async method."""
        assert inspect.iscoroutinefunction(CogVideoXModel.generate_montage)

    def test_generate_montage_parameters(self) -> None:
        """Test generate_montage has correct parameters for I2V (with keyframes)."""
        sig = inspect.signature(CogVideoXModel.generate_montage)
        params = list(sig.parameters.keys())
        expected_params = [
            "self",
            "keyframes",
            "prompts",
            "config",
            "seed",
            "trim_frames",
            "progress_callback",
        ]
        assert params == expected_params

    def test_generate_montage_parameter_defaults(self) -> None:
        """Test generate_montage parameter defaults."""
        sig = inspect.signature(CogVideoXModel.generate_montage)

        # Required params have no default
        assert sig.parameters["keyframes"].default is inspect.Parameter.empty
        assert sig.parameters["prompts"].default is inspect.Parameter.empty

        # Optional params have defaults
        assert sig.parameters["config"].default is None
        assert sig.parameters["seed"].default is None
        assert sig.parameters["trim_frames"].default == 40
        assert sig.parameters["progress_callback"].default is None

    def test_generate_montage_annotations(self) -> None:
        """Test generate_montage has correct type annotations."""
        ann = CogVideoXModel.generate_montage.__annotations__

        assert "list" in str(ann.get("keyframes"))
        assert "list" in str(ann.get("prompts"))
        assert "VideoGenerationConfig" in str(ann.get("config"))
        assert "int" in str(ann.get("seed"))
        assert "int" in str(ann.get("trim_frames"))
        assert "Callable" in str(ann.get("progress_callback"))
        assert "torch.Tensor" in str(ann.get("return"))


class TestGenerateMontageLogic:
    """Tests for the generate_montage method logic (with mocking).

    Updated for I2V architecture where generate_montage takes keyframes and prompts.
    Each scene is generated from its corresponding keyframe image and text prompt.
    """

    @pytest.fixture
    def mock_model(self) -> CogVideoXModel:
        """Create a CogVideoXModel instance for testing."""
        return CogVideoXModel()

    @pytest.mark.asyncio
    async def test_generate_montage_single_scene(self, mock_model: CogVideoXModel) -> None:
        """Test generate_montage with a single scene."""
        # Create mock chunk output (49 frames)
        mock_frames = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframes = [torch.rand(3, 480, 720)]
            prompts = ["Scene 1 description"]
            result = await mock_model.generate_montage(
                keyframes=keyframes,
                prompts=prompts,
                seed=42,
            )

            # Should have called generate_chunk once
            assert mock_gen.call_count == 1

            # Result should be trimmed to 40 frames (default trim_frames)
            assert result.shape[0] == 40

    @pytest.mark.asyncio
    async def test_generate_montage_multiple_scenes(self, mock_model: CogVideoXModel) -> None:
        """Test generate_montage with multiple scenes."""
        # Create mock chunk outputs
        mock_frames = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframes = [torch.rand(3, 480, 720) for _ in range(3)]
            prompts = ["Scene 1", "Scene 2", "Scene 3"]
            result = await mock_model.generate_montage(
                keyframes=keyframes,
                prompts=prompts,
                seed=42,
            )

            # Should have called generate_chunk three times
            assert mock_gen.call_count == 3

            # Result should be 3 * 40 = 120 frames
            assert result.shape[0] == 120

    @pytest.mark.asyncio
    async def test_generate_montage_derived_seeds(self, mock_model: CogVideoXModel) -> None:
        """Test that each scene gets a derived seed."""
        mock_frames = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframes = [torch.rand(3, 480, 720) for _ in range(3)]
            prompts = ["Scene 1", "Scene 2", "Scene 3"]
            await mock_model.generate_montage(
                keyframes=keyframes,
                prompts=prompts,
                seed=100,
            )

            # Check seeds: 100, 101, 102
            for i, call in enumerate(mock_gen.call_args_list):
                assert call.kwargs.get("seed") == 100 + i

    @pytest.mark.asyncio
    async def test_generate_montage_no_seed(self, mock_model: CogVideoXModel) -> None:
        """Test generate_montage without seed (non-deterministic)."""
        mock_frames = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframes = [torch.rand(3, 480, 720)]
            prompts = ["Scene 1"]
            await mock_model.generate_montage(
                keyframes=keyframes,
                prompts=prompts,
                seed=None,
            )

            # Should have called with seed=None
            call_args = mock_gen.call_args
            assert call_args.kwargs.get("seed") is None

    @pytest.mark.asyncio
    async def test_generate_montage_custom_trim_frames(self, mock_model: CogVideoXModel) -> None:
        """Test generate_montage with custom trim_frames."""
        mock_frames = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframes = [torch.rand(3, 480, 720) for _ in range(2)]
            prompts = ["Scene 1", "Scene 2"]
            result = await mock_model.generate_montage(
                keyframes=keyframes,
                prompts=prompts,
                trim_frames=30,  # Custom trim
            )

            # Result should be 2 * 30 = 60 frames
            assert result.shape[0] == 60

    @pytest.mark.asyncio
    async def test_generate_montage_no_trim(self, mock_model: CogVideoXModel) -> None:
        """Test generate_montage without trimming (trim_frames=0)."""
        mock_frames = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframes = [torch.rand(3, 480, 720)]
            prompts = ["Scene 1"]
            result = await mock_model.generate_montage(
                keyframes=keyframes,
                prompts=prompts,
                trim_frames=0,  # No trimming
            )

            # Result should keep all 49 frames
            assert result.shape[0] == 49

    @pytest.mark.asyncio
    async def test_generate_montage_empty_prompts(
        self, mock_model: CogVideoXModel
    ) -> None:
        """Test that empty prompts list raises ValueError."""
        with pytest.raises(ValueError, match="prompts list cannot be empty"):
            await mock_model.generate_montage(
                keyframes=[],
                prompts=[],
            )

    @pytest.mark.asyncio
    async def test_generate_montage_mismatched_lengths(
        self, mock_model: CogVideoXModel
    ) -> None:
        """Test that mismatched keyframes/prompts lengths raises ValueError."""
        with pytest.raises(ValueError, match="keyframes and prompts must have same length"):
            await mock_model.generate_montage(
                keyframes=[torch.rand(3, 480, 720)],
                prompts=["Scene 1", "Scene 2"],  # 2 prompts but only 1 keyframe
            )

    @pytest.mark.asyncio
    async def test_generate_montage_uses_keyframe_and_prompt_per_scene(
        self, mock_model: CogVideoXModel
    ) -> None:
        """Test that each scene uses its own keyframe and prompt (I2V generation)."""
        mock_frames = torch.rand(49, 3, 480, 720)

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframes = [torch.rand(3, 480, 720) for _ in range(3)]
            await mock_model.generate_montage(
                keyframes=keyframes,
                prompts=["Scene 1 prompt", "Scene 2 prompt", "Scene 3 prompt"],
            )

            # Verify each call uses the correct prompt
            assert mock_gen.call_args_list[0].kwargs["prompt"] == "Scene 1 prompt"
            assert mock_gen.call_args_list[1].kwargs["prompt"] == "Scene 2 prompt"
            assert mock_gen.call_args_list[2].kwargs["prompt"] == "Scene 3 prompt"

            # Verify each call uses a keyframe
            for call in mock_gen.call_args_list:
                assert call.kwargs.get("image") is not None

    @pytest.mark.asyncio
    async def test_generate_montage_progress_callback(
        self, mock_model: CogVideoXModel
    ) -> None:
        """Test that progress callback is called correctly for montage."""
        mock_frames = torch.rand(49, 3, 480, 720)
        callback_calls: list[tuple[int, int]] = []

        def progress_callback(scene_num: int, total_scenes: int) -> None:
            callback_calls.append((scene_num, total_scenes))

        with patch.object(mock_model, "generate_chunk", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_frames

            keyframes = [torch.rand(3, 480, 720) for _ in range(3)]
            prompts = ["Scene 1", "Scene 2", "Scene 3"]
            await mock_model.generate_montage(
                keyframes=keyframes,
                prompts=prompts,
                progress_callback=progress_callback,
            )

            # Should have called callback before each scene and after completion
            assert (0, 3) in callback_calls  # Before scene 0
            assert (1, 3) in callback_calls  # Before scene 1
            assert (2, 3) in callback_calls  # Before scene 2
            assert (3, 3) in callback_calls  # Completion
            assert len(callback_calls) == 4


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
