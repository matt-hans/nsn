"""Unit tests for DefaultRenderer montage architecture.

Tests for:
- _generate_video() method with Script object and storyboard
- render() method integration with montage flow
- Proper Flux/CogVideoX model offloading during montage

These tests verify the montage architecture where:
1. Script contains 3-scene storyboard
2. Flux generates 3 independent keyframes
3. CogVideoX generates 3 independent clips via generate_montage()
4. Clips are concatenated with hard cuts
"""

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from vortex.models.showrunner import Script
from vortex.renderers.default.renderer import DefaultRenderer


class TestGenerateVideoMontageSignature:
    """Tests for the _generate_video method signature with montage architecture."""

    def test_generate_video_exists(self) -> None:
        """Test that _generate_video method exists."""
        assert hasattr(DefaultRenderer, "_generate_video")

    def test_generate_video_is_async(self) -> None:
        """Test that _generate_video is an async method."""
        assert inspect.iscoroutinefunction(DefaultRenderer._generate_video)

    def test_generate_video_parameters(self) -> None:
        """Test _generate_video has correct parameters for montage."""
        sig = inspect.signature(DefaultRenderer._generate_video)
        params = list(sig.parameters.keys())
        expected_params = ["self", "script", "recipe", "seed"]
        assert params == expected_params

    def test_generate_video_script_annotation(self) -> None:
        """Test _generate_video has Script type annotation."""
        ann = DefaultRenderer._generate_video.__annotations__
        assert "Script" in str(ann.get("script"))


class TestGenerateVideoMontageLogic:
    """Tests for the _generate_video method logic with mocking."""

    @pytest.fixture
    def mock_renderer(self) -> DefaultRenderer:
        """Create a DefaultRenderer with mocked dependencies."""
        renderer = DefaultRenderer.__new__(DefaultRenderer)

        # Mock model registry
        renderer._model_registry = MagicMock()
        renderer._model_registry.offloading_enabled = True
        renderer._model_registry.prepare_for_stage = MagicMock()

        # Mock Flux model
        mock_flux = MagicMock()
        mock_flux.unload = MagicMock()
        renderer._model_registry.get_model.return_value = mock_flux

        # Mock CogVideoX model
        mock_cogvideox = MagicMock()
        renderer._model_registry.get_cogvideox.return_value = mock_cogvideox

        # Mock actor buffer for keyframe generation
        renderer._actor_buffer = torch.zeros(1, 3, 512, 512)

        return renderer

    @pytest.fixture
    def valid_script(self) -> Script:
        """Create a valid Script with 3-scene storyboard."""
        return Script(
            setup="Welcome to the interdimensional cable network!",
            punchline="Where everything is made up and the points don't matter!",
            storyboard=[
                "Scene 1: Host standing in futuristic studio, neon lights",
                "Scene 2: Host gesturing wildly at floating screens",
                "Scene 3: Host disappearing into a portal, waving goodbye",
            ],
        )

    @pytest.fixture
    def valid_recipe(self) -> dict:
        """Create a valid recipe dict."""
        return {
            "video": {
                "style_prompt": "cartoon style, vibrant colors",
                "guidance_scale": 5.0,
            },
            "slot_params": {
                "fps": 8,
            },
        }

    @pytest.mark.asyncio
    async def test_generate_video_calls_keyframe_for_each_scene(
        self, mock_renderer: DefaultRenderer, valid_script: Script, valid_recipe: dict
    ) -> None:
        """Test that _generate_video generates a keyframe for each storyboard scene."""
        # Mock _generate_keyframe to return tensor
        mock_keyframe = torch.rand(3, 512, 512)
        mock_renderer._generate_keyframe = AsyncMock(return_value=mock_keyframe)

        # Mock generate_montage to return video tensor
        mock_video = torch.rand(120, 3, 480, 720)  # 3 scenes × 40 frames
        mock_renderer._model_registry.get_cogvideox().generate_montage = AsyncMock(
            return_value=mock_video
        )

        await mock_renderer._generate_video(valid_script, valid_recipe, seed=42)

        # Should call _generate_keyframe 3 times (once per scene)
        assert mock_renderer._generate_keyframe.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_video_uses_derived_seeds(
        self, mock_renderer: DefaultRenderer, valid_script: Script, valid_recipe: dict
    ) -> None:
        """Test that each keyframe uses a derived seed for determinism."""
        mock_keyframe = torch.rand(3, 512, 512)
        mock_renderer._generate_keyframe = AsyncMock(return_value=mock_keyframe)

        mock_video = torch.rand(120, 3, 480, 720)
        mock_renderer._model_registry.get_cogvideox().generate_montage = AsyncMock(
            return_value=mock_video
        )

        base_seed = 100
        await mock_renderer._generate_video(valid_script, valid_recipe, seed=base_seed)

        # Check seeds: base_seed + 0, base_seed + 1, base_seed + 2
        # _generate_keyframe(full_prompt, recipe, scene_seed) - seed is 3rd positional arg
        calls = mock_renderer._generate_keyframe.call_args_list
        assert calls[0][0][2] == 100  # Scene 0: seed = base_seed + 0
        assert calls[1][0][2] == 101  # Scene 1: seed = base_seed + 1
        assert calls[2][0][2] == 102  # Scene 2: seed = base_seed + 2

    @pytest.mark.asyncio
    async def test_generate_video_calls_generate_montage(
        self, mock_renderer: DefaultRenderer, valid_script: Script, valid_recipe: dict
    ) -> None:
        """Test that _generate_video calls CogVideoX.generate_montage()."""
        mock_keyframe = torch.rand(3, 512, 512)
        mock_renderer._generate_keyframe = AsyncMock(return_value=mock_keyframe)

        mock_video = torch.rand(120, 3, 480, 720)
        mock_cogvideox = mock_renderer._model_registry.get_cogvideox()
        mock_cogvideox.generate_montage = AsyncMock(return_value=mock_video)

        await mock_renderer._generate_video(valid_script, valid_recipe, seed=42)

        # Verify generate_montage was called
        mock_cogvideox.generate_montage.assert_called_once()

        # Verify it was called with 3 keyframes and 3 prompts
        call_kwargs = mock_cogvideox.generate_montage.call_args[1]
        assert len(call_kwargs["keyframes"]) == 3
        assert len(call_kwargs["prompts"]) == 3
        assert call_kwargs["prompts"] == valid_script.storyboard
        assert call_kwargs["trim_frames"] == 40  # 5s per scene @ 8fps

    @pytest.mark.asyncio
    async def test_generate_video_unloads_flux_before_cogvideox(
        self, mock_renderer: DefaultRenderer, valid_script: Script, valid_recipe: dict
    ) -> None:
        """Test that Flux is unloaded before CogVideoX generation."""
        mock_keyframe = torch.rand(3, 512, 512)
        mock_renderer._generate_keyframe = AsyncMock(return_value=mock_keyframe)

        mock_video = torch.rand(120, 3, 480, 720)
        mock_renderer._model_registry.get_cogvideox().generate_montage = AsyncMock(
            return_value=mock_video
        )

        mock_flux = MagicMock()
        mock_flux.unload = MagicMock()
        mock_renderer._model_registry.get_model.return_value = mock_flux

        await mock_renderer._generate_video(valid_script, valid_recipe, seed=42)

        # Verify Flux.unload() was called
        mock_flux.unload.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_video_raises_for_insufficient_storyboard(
        self, mock_renderer: DefaultRenderer, valid_recipe: dict
    ) -> None:
        """Test that _generate_video raises ValueError for < 3 scenes."""
        # Script with only 2 scenes
        short_script = Script(
            setup="Test setup",
            punchline="Test punchline",
            storyboard=["Scene 1", "Scene 2"],  # Only 2 scenes
        )

        with pytest.raises(ValueError, match="must have 3-scene storyboard"):
            await mock_renderer._generate_video(short_script, valid_recipe, seed=42)

    @pytest.mark.asyncio
    async def test_generate_video_raises_for_empty_storyboard(
        self, mock_renderer: DefaultRenderer, valid_recipe: dict
    ) -> None:
        """Test that _generate_video raises ValueError for empty storyboard."""
        empty_script = Script(
            setup="Test setup",
            punchline="Test punchline",
            storyboard=[],  # Empty storyboard
        )

        with pytest.raises(ValueError, match="must have 3-scene storyboard"):
            await mock_renderer._generate_video(empty_script, valid_recipe, seed=42)

    @pytest.mark.asyncio
    async def test_generate_video_combines_style_prompt(
        self, mock_renderer: DefaultRenderer, valid_script: Script
    ) -> None:
        """Test that style_prompt from recipe is combined with scene prompts."""
        mock_keyframe = torch.rand(3, 512, 512)
        mock_renderer._generate_keyframe = AsyncMock(return_value=mock_keyframe)

        mock_video = torch.rand(120, 3, 480, 720)
        mock_renderer._model_registry.get_cogvideox().generate_montage = AsyncMock(
            return_value=mock_video
        )

        recipe_with_style = {
            "video": {"style_prompt": "cyberpunk aesthetic, neon glow"},
        }

        await mock_renderer._generate_video(valid_script, recipe_with_style, seed=42)

        # Check that first keyframe call includes style_prompt
        first_call = mock_renderer._generate_keyframe.call_args_list[0]
        prompt_arg = first_call[0][0]  # First positional arg is prompt
        assert "cyberpunk aesthetic, neon glow" in prompt_arg
        assert "Scene 1" in prompt_arg  # Original scene prompt

    @pytest.mark.asyncio
    async def test_generate_video_returns_video_tensor(
        self, mock_renderer: DefaultRenderer, valid_script: Script, valid_recipe: dict
    ) -> None:
        """Test that _generate_video returns the video tensor from generate_montage."""
        mock_keyframe = torch.rand(3, 512, 512)
        mock_renderer._generate_keyframe = AsyncMock(return_value=mock_keyframe)

        expected_video = torch.rand(120, 3, 480, 720)
        mock_renderer._model_registry.get_cogvideox().generate_montage = AsyncMock(
            return_value=expected_video
        )

        result = await mock_renderer._generate_video(valid_script, valid_recipe, seed=42)

        assert torch.equal(result, expected_video)


class TestScriptStoryboardProperty:
    """Tests for Script.visual_prompt backward compatibility."""

    def test_visual_prompt_returns_first_scene(self) -> None:
        """Test that Script.visual_prompt returns first storyboard scene."""
        script = Script(
            setup="Setup text",
            punchline="Punchline text",
            storyboard=["First scene", "Second scene", "Third scene"],
        )

        assert script.visual_prompt == "First scene"

    def test_visual_prompt_empty_for_empty_storyboard(self) -> None:
        """Test that Script.visual_prompt returns empty string for empty storyboard."""
        script = Script(
            setup="Setup text",
            punchline="Punchline text",
            storyboard=[],
        )

        assert script.visual_prompt == ""
