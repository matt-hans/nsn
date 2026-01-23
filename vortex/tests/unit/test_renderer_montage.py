"""Unit tests for DefaultRenderer T2V Montage architecture.

Tests for:
- _generate_video() method with Script object and video_prompts
- render() method integration with T2V montage flow
- Proper CogVideoX model offloading during T2V generation

These tests verify the T2V Montage architecture where:
1. Script contains 3-scene storyboard and 3 video_prompts
2. CogVideoX generates 3 independent clips directly from video_prompts (T2V)
3. Clips are concatenated with hard cuts (no keyframes, no Flux)
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
        """Create a DefaultRenderer with mocked dependencies for T2V."""
        renderer = DefaultRenderer.__new__(DefaultRenderer)

        # Mock model registry
        renderer._model_registry = MagicMock()
        renderer._model_registry.offloading_enabled = True
        renderer._model_registry.prepare_for_stage = MagicMock()

        # Mock CogVideoX model (no Flux needed for T2V)
        mock_cogvideox = MagicMock()
        renderer._model_registry.get_cogvideox.return_value = mock_cogvideox

        return renderer

    @pytest.fixture
    def valid_script(self) -> Script:
        """Create a valid Script with 3-scene storyboard and video_prompts."""
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
                "A charismatic TV host with slicked-back hair in a shiny suit standing in a futuristic studio with neon lights, camera slowly zooms in, 2D cel-shaded cartoon, adult swim aesthetic",
                "A charismatic TV host with slicked-back hair in a shiny suit gesturing wildly at floating holographic screens, dynamic camera pan, 2D cel-shaded cartoon, adult swim aesthetic",
                "A charismatic TV host with slicked-back hair in a shiny suit disappearing into a swirling portal, waving goodbye, camera pulls back, 2D cel-shaded cartoon, adult swim aesthetic",
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
    async def test_generate_video_calls_generate_montage_with_video_prompts(
        self, mock_renderer: DefaultRenderer, valid_script: Script, valid_recipe: dict
    ) -> None:
        """Test that _generate_video calls CogVideoX.generate_montage() with video_prompts."""
        # Mock generate_montage to return video tensor
        mock_video = torch.rand(120, 3, 480, 720)  # 3 scenes x 40 frames
        mock_cogvideox = mock_renderer._model_registry.get_cogvideox()
        mock_cogvideox.generate_montage = AsyncMock(return_value=mock_video)

        await mock_renderer._generate_video(valid_script, valid_recipe, seed=42)

        # Verify generate_montage was called
        mock_cogvideox.generate_montage.assert_called_once()

        # Verify it was called with video_prompts (T2V - no keyframes)
        call_kwargs = mock_cogvideox.generate_montage.call_args[1]
        assert "prompts" in call_kwargs
        assert len(call_kwargs["prompts"]) == 3
        assert call_kwargs["prompts"] == valid_script.video_prompts
        assert call_kwargs["trim_frames"] == 40  # 5s per scene @ 8fps
        # T2V: no keyframes parameter
        assert "keyframes" not in call_kwargs

    @pytest.mark.asyncio
    async def test_generate_video_raises_for_insufficient_video_prompts(
        self, mock_renderer: DefaultRenderer, valid_recipe: dict
    ) -> None:
        """Test that _generate_video raises ValueError for < 3 video_prompts."""
        # Script with only 2 video_prompts
        short_script = Script(
            setup="Test setup",
            punchline="Test punchline",
            subject_visual="test subject",
            storyboard=["Scene 1", "Scene 2", "Scene 3"],
            video_prompts=["Prompt 1", "Prompt 2"],  # Only 2 prompts
        )

        with pytest.raises(ValueError, match="must have 3 video_prompts"):
            await mock_renderer._generate_video(short_script, valid_recipe, seed=42)

    @pytest.mark.asyncio
    async def test_generate_video_raises_for_empty_video_prompts(
        self, mock_renderer: DefaultRenderer, valid_recipe: dict
    ) -> None:
        """Test that _generate_video raises ValueError for empty video_prompts."""
        empty_script = Script(
            setup="Test setup",
            punchline="Test punchline",
            subject_visual="test subject",
            storyboard=["Scene 1", "Scene 2", "Scene 3"],
            video_prompts=[],  # Empty video_prompts
        )

        with pytest.raises(ValueError, match="must have 3 video_prompts"):
            await mock_renderer._generate_video(empty_script, valid_recipe, seed=42)

    @pytest.mark.asyncio
    async def test_generate_video_returns_video_tensor(
        self, mock_renderer: DefaultRenderer, valid_script: Script, valid_recipe: dict
    ) -> None:
        """Test that _generate_video returns the video tensor from generate_montage."""
        expected_video = torch.rand(120, 3, 480, 720)
        mock_renderer._model_registry.get_cogvideox().generate_montage = AsyncMock(
            return_value=expected_video
        )

        result = await mock_renderer._generate_video(valid_script, valid_recipe, seed=42)

        assert torch.equal(result, expected_video)

    @pytest.mark.asyncio
    async def test_generate_video_prepares_for_video_stage(
        self, mock_renderer: DefaultRenderer, valid_script: Script, valid_recipe: dict
    ) -> None:
        """Test that _generate_video prepares model registry for video stage."""
        mock_video = torch.rand(120, 3, 480, 720)
        mock_renderer._model_registry.get_cogvideox().generate_montage = AsyncMock(
            return_value=mock_video
        )

        await mock_renderer._generate_video(valid_script, valid_recipe, seed=42)

        # Verify prepare_for_stage was called with "video"
        mock_renderer._model_registry.prepare_for_stage.assert_called_with("video")


class TestScriptVideoPromptsProperty:
    """Tests for Script.visual_prompt returns first video_prompt."""

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
