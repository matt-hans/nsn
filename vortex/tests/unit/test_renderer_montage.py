"""Unit tests for DefaultRenderer I2V Montage architecture.

Tests for:
- _generate_video() method with Script object, keyframes, and motion prompts
- _generate_keyframes() method with Flux model
- render() method integration with audio-first I2V flow
- Proper model offloading during keyframe and video generation

These tests verify the I2V (Image-to-Video) Montage architecture where:
1. Script contains 3-scene storyboard
2. Flux generates 3 keyframe images at 720x480
3. CogVideoX I2V animates each keyframe into video clips
4. Clips are concatenated with hard cuts
"""

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from vortex.models.showrunner import Script
from vortex.renderers.default.renderer import MOTION_STYLE_SUFFIX, DefaultRenderer


class TestGenerateVideoMontageSignature:
    """Tests for the _generate_video method signature with I2V architecture."""

    def test_generate_video_exists(self) -> None:
        """Test that _generate_video method exists."""
        assert hasattr(DefaultRenderer, "_generate_video")

    def test_generate_video_is_async(self) -> None:
        """Test that _generate_video is an async method."""
        assert inspect.iscoroutinefunction(DefaultRenderer._generate_video)

    def test_generate_video_parameters(self) -> None:
        """Test _generate_video has correct parameters for I2V."""
        sig = inspect.signature(DefaultRenderer._generate_video)
        params = list(sig.parameters.keys())
        # I2V: script, keyframes, frames_per_scene, seed
        expected_params = ["self", "script", "keyframes", "frames_per_scene", "seed"]
        assert params == expected_params

    def test_generate_video_script_annotation(self) -> None:
        """Test _generate_video has Script type annotation."""
        ann = DefaultRenderer._generate_video.__annotations__
        assert "Script" in str(ann.get("script"))


class TestGenerateKeyframesSignature:
    """Tests for the _generate_keyframes method signature."""

    def test_generate_keyframes_exists(self) -> None:
        """Test that _generate_keyframes method exists."""
        assert hasattr(DefaultRenderer, "_generate_keyframes")

    def test_generate_keyframes_is_async(self) -> None:
        """Test that _generate_keyframes is an async method."""
        assert inspect.iscoroutinefunction(DefaultRenderer._generate_keyframes)

    def test_generate_keyframes_parameters(self) -> None:
        """Test _generate_keyframes has correct parameters."""
        sig = inspect.signature(DefaultRenderer._generate_keyframes)
        params = list(sig.parameters.keys())
        expected_params = ["self", "script", "seed"]
        assert params == expected_params


class TestUnloadFluxHelper:
    """Tests for the _unload_flux helper method."""

    def test_unload_flux_exists(self) -> None:
        """Test that _unload_flux method exists."""
        assert hasattr(DefaultRenderer, "_unload_flux")

    def test_unload_flux_is_sync(self) -> None:
        """Test that _unload_flux is a synchronous method (not async)."""
        assert not inspect.iscoroutinefunction(DefaultRenderer._unload_flux)


class TestGenerateVideoMontageLogic:
    """Tests for the _generate_video method logic with I2V mocking."""

    @pytest.fixture
    def mock_renderer(self) -> DefaultRenderer:
        """Create a DefaultRenderer with mocked dependencies for I2V."""
        renderer = DefaultRenderer.__new__(DefaultRenderer)

        # Mock model registry
        renderer._model_registry = MagicMock()
        renderer._model_registry.offloading_enabled = True
        renderer._model_registry.prepare_for_stage = MagicMock()

        # Mock CogVideoX model (I2V)
        mock_cogvideox = MagicMock()
        renderer._model_registry.get_cogvideox.return_value = mock_cogvideox

        return renderer

    @pytest.fixture
    def valid_script(self) -> Script:
        """Create a valid Script with 3-scene storyboard."""
        return Script(
            setup="Welcome to the interdimensional cable network!",
            punchline="Where everything is made up and the points don't matter!",
            subject_visual="a charismatic TV host with slicked-back hair in a shiny suit",
            storyboard=[
                "Scene 1: Host standing in futuristic studio, neon lights",
                "Scene 2: Host gesturing wildly at floating screens",
                "Scene 3: Host disappearing into a portal, waving goodbye",
            ],
            video_prompts=[
                "A charismatic TV host standing in a futuristic studio",
                "A charismatic TV host gesturing wildly",
                "A charismatic TV host disappearing into a portal",
            ],
        )

    @pytest.fixture
    def valid_keyframes(self) -> list[torch.Tensor]:
        """Create 3 valid keyframe tensors at 720x480."""
        return [torch.rand(3, 480, 720) for _ in range(3)]

    @pytest.mark.asyncio
    async def test_generate_video_calls_generate_montage_with_keyframes(
        self,
        mock_renderer: DefaultRenderer,
        valid_script: Script,
        valid_keyframes: list[torch.Tensor],
    ) -> None:
        """Test that _generate_video calls CogVideoX.generate_montage() with keyframes."""
        # Mock generate_montage to return video tensor
        mock_video = torch.rand(120, 3, 480, 720)  # 3 scenes x 40 frames
        mock_cogvideox = mock_renderer._model_registry.get_cogvideox()
        mock_cogvideox.generate_montage = AsyncMock(return_value=mock_video)

        await mock_renderer._generate_video(
            valid_script, valid_keyframes, frames_per_scene=40, seed=42
        )

        # Verify generate_montage was called
        mock_cogvideox.generate_montage.assert_called_once()

        # Verify it was called with keyframes (I2V)
        call_kwargs = mock_cogvideox.generate_montage.call_args[1]
        assert "keyframes" in call_kwargs
        assert len(call_kwargs["keyframes"]) == 3
        assert call_kwargs["keyframes"] == valid_keyframes

        # Verify motion prompts include MOTION_STYLE_SUFFIX
        assert "prompts" in call_kwargs
        assert len(call_kwargs["prompts"]) == 3
        for prompt in call_kwargs["prompts"]:
            assert MOTION_STYLE_SUFFIX in prompt

        # Verify trim_frames
        assert call_kwargs["trim_frames"] == 40

    @pytest.mark.asyncio
    async def test_generate_video_raises_for_keyframes_storyboard_mismatch(
        self, mock_renderer: DefaultRenderer, valid_script: Script
    ) -> None:
        """Test that _generate_video raises ValueError for keyframes/storyboard mismatch."""
        # Only 2 keyframes for 3 scenes
        short_keyframes = [torch.rand(3, 480, 720) for _ in range(2)]

        with pytest.raises(ValueError, match="mismatch"):
            await mock_renderer._generate_video(
                valid_script, short_keyframes, frames_per_scene=40, seed=42
            )

    @pytest.mark.asyncio
    async def test_generate_video_returns_video_tensor(
        self,
        mock_renderer: DefaultRenderer,
        valid_script: Script,
        valid_keyframes: list[torch.Tensor],
    ) -> None:
        """Test that _generate_video returns the video tensor from generate_montage."""
        expected_video = torch.rand(120, 3, 480, 720)
        mock_renderer._model_registry.get_cogvideox().generate_montage = AsyncMock(
            return_value=expected_video
        )

        result = await mock_renderer._generate_video(
            valid_script, valid_keyframes, frames_per_scene=40, seed=42
        )

        assert torch.equal(result, expected_video)

    @pytest.mark.asyncio
    async def test_generate_video_prepares_for_video_stage(
        self,
        mock_renderer: DefaultRenderer,
        valid_script: Script,
        valid_keyframes: list[torch.Tensor],
    ) -> None:
        """Test that _generate_video prepares model registry for video stage."""
        mock_video = torch.rand(120, 3, 480, 720)
        mock_renderer._model_registry.get_cogvideox().generate_montage = AsyncMock(
            return_value=mock_video
        )

        await mock_renderer._generate_video(
            valid_script, valid_keyframes, frames_per_scene=40, seed=42
        )

        # Verify prepare_for_stage was called with "video"
        mock_renderer._model_registry.prepare_for_stage.assert_called_with("video")

    @pytest.mark.asyncio
    async def test_generate_video_respects_frames_per_scene(
        self,
        mock_renderer: DefaultRenderer,
        valid_script: Script,
        valid_keyframes: list[torch.Tensor],
    ) -> None:
        """Test that _generate_video passes correct trim_frames based on frames_per_scene."""
        mock_video = torch.rand(90, 3, 480, 720)  # 3 scenes x 30 frames
        mock_renderer._model_registry.get_cogvideox().generate_montage = AsyncMock(
            return_value=mock_video
        )

        # Request 30 frames per scene
        await mock_renderer._generate_video(
            valid_script, valid_keyframes, frames_per_scene=30, seed=42
        )

        call_kwargs = mock_renderer._model_registry.get_cogvideox().generate_montage.call_args[1]
        assert call_kwargs["trim_frames"] == 30

    @pytest.mark.asyncio
    async def test_generate_video_caps_trim_frames_at_40(
        self,
        mock_renderer: DefaultRenderer,
        valid_script: Script,
        valid_keyframes: list[torch.Tensor],
    ) -> None:
        """Test that trim_frames is capped at 40 to avoid tail degradation."""
        mock_video = torch.rand(120, 3, 480, 720)
        mock_renderer._model_registry.get_cogvideox().generate_montage = AsyncMock(
            return_value=mock_video
        )

        # Request 50 frames per scene (exceeds cap)
        await mock_renderer._generate_video(
            valid_script, valid_keyframes, frames_per_scene=50, seed=42
        )

        call_kwargs = mock_renderer._model_registry.get_cogvideox().generate_montage.call_args[1]
        # Should be capped at 40
        assert call_kwargs["trim_frames"] == 40


class TestGenerateKeyframesLogic:
    """Tests for the _generate_keyframes method logic."""

    @pytest.fixture
    def mock_renderer(self) -> DefaultRenderer:
        """Create a DefaultRenderer with mocked dependencies for keyframe generation."""
        renderer = DefaultRenderer.__new__(DefaultRenderer)

        # Mock model registry
        renderer._model_registry = MagicMock()
        renderer._model_registry.prepare_for_stage = MagicMock()

        # Mock Flux model
        mock_flux = MagicMock()
        renderer._model_registry.get_flux.return_value = mock_flux

        # Mock actor buffer
        renderer._actor_buffer = torch.zeros(1, 3, 480, 720)

        return renderer

    @pytest.fixture
    def valid_script(self) -> Script:
        """Create a valid Script with 3-scene storyboard."""
        return Script(
            setup="Welcome!",
            punchline="Goodbye!",
            subject_visual="a cartoon TV host",
            storyboard=[
                "Scene 1: Host waves",
                "Scene 2: Host talks",
                "Scene 3: Host exits",
            ],
            video_prompts=[],
        )

    @pytest.mark.asyncio
    async def test_generate_keyframes_calls_flux_for_each_scene(
        self, mock_renderer: DefaultRenderer, valid_script: Script
    ) -> None:
        """Test that _generate_keyframes calls Flux.generate() for each storyboard scene."""
        mock_flux = mock_renderer._model_registry.get_flux()
        mock_flux.generate.return_value = torch.rand(3, 480, 720)

        await mock_renderer._generate_keyframes(valid_script, seed=42)

        # Should be called once per storyboard scene
        assert mock_flux.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_keyframes_uses_fixed_seed(
        self, mock_renderer: DefaultRenderer, valid_script: Script
    ) -> None:
        """Test that all keyframes use the same seed for subject consistency."""
        mock_flux = mock_renderer._model_registry.get_flux()
        mock_flux.generate.return_value = torch.rand(3, 480, 720)

        await mock_renderer._generate_keyframes(valid_script, seed=42)

        # All calls should use the same seed (not seed + i)
        for call in mock_flux.generate.call_args_list:
            assert call[1]["seed"] == 42

    @pytest.mark.asyncio
    async def test_generate_keyframes_returns_list_of_tensors(
        self, mock_renderer: DefaultRenderer, valid_script: Script
    ) -> None:
        """Test that _generate_keyframes returns a list of keyframe tensors."""
        mock_flux = mock_renderer._model_registry.get_flux()
        mock_flux.generate.return_value = torch.rand(3, 480, 720)

        keyframes = await mock_renderer._generate_keyframes(valid_script, seed=42)

        assert isinstance(keyframes, list)
        assert len(keyframes) == 3
        for kf in keyframes:
            assert isinstance(kf, torch.Tensor)

    @pytest.mark.asyncio
    async def test_generate_keyframes_prepares_for_keyframe_stage(
        self, mock_renderer: DefaultRenderer, valid_script: Script
    ) -> None:
        """Test that _generate_keyframes prepares model registry for keyframe stage."""
        mock_flux = mock_renderer._model_registry.get_flux()
        mock_flux.generate.return_value = torch.rand(3, 480, 720)

        await mock_renderer._generate_keyframes(valid_script, seed=42)

        mock_renderer._model_registry.prepare_for_stage.assert_called_with("keyframe")


class TestScriptVisualPromptProperty:
    """Tests for Script.visual_prompt backward compatibility."""

    def test_visual_prompt_returns_first_video_prompt(self) -> None:
        """Test that Script.visual_prompt returns first video_prompt."""
        script = Script(
            setup="Setup text",
            punchline="Punchline text",
            subject_visual="test subject",
            storyboard=["First scene", "Second scene", "Third scene"],
            video_prompts=["First prompt", "Second prompt", "Third prompt"],
        )

        assert script.visual_prompt == "First prompt"

    def test_visual_prompt_empty_for_empty_video_prompts(self) -> None:
        """Test that Script.visual_prompt returns empty string for empty video_prompts."""
        script = Script(
            setup="Setup text",
            punchline="Punchline text",
            subject_visual="test subject",
            storyboard=["First scene", "Second scene", "Third scene"],
            video_prompts=[],
        )

        assert script.visual_prompt == ""
