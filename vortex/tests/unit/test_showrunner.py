"""Unit tests for Showrunner script generator.

Tests the Showrunner class including:
- Fallback template functionality
- Showrunner initialization
- Ollama health check (is_available)
- Async script generation (generate_script)
- JSON parsing edge cases
- Error handling (ShowrunnerError)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vortex.models.showrunner import (
    ADULT_SWIM_STYLE,
    SCRIPT_PROMPT_TEMPLATE,
    Script,
    Showrunner,
    ShowrunnerError,
    _get_fallback_templates,
)


class TestScriptVideoPrompts:
    """Test suite for Script.video_prompts field."""

    def test_script_has_video_prompts_attribute(self):
        """Script should have video_prompts attribute."""
        script = Script(
            setup="Test setup",
            punchline="Test punchline",
            subject_visual="test subject",
            storyboard=["Scene 1", "Scene 2", "Scene 3"],
            video_prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
        )
        assert hasattr(script, "video_prompts")
        assert len(script.video_prompts) == 3

    def test_visual_prompt_returns_first_video_prompt(self):
        """Script.visual_prompt should return first video_prompt."""
        script = Script(
            setup="Test",
            punchline="Test",
            subject_visual="subject",
            storyboard=["S1", "S2", "S3"],
            video_prompts=["First prompt", "Second prompt", "Third prompt"],
        )
        assert script.visual_prompt == "First prompt"

    def test_visual_prompt_returns_empty_for_empty_video_prompts(self):
        """Script.visual_prompt should return empty string for empty video_prompts."""
        script = Script(
            setup="Test",
            punchline="Test",
            subject_visual="subject",
            storyboard=["S1", "S2", "S3"],
            video_prompts=[],
        )
        assert script.visual_prompt == ""


class TestFallbackTemplates:
    """Test suite for fallback template configuration."""

    def test_fallback_templates_has_minimum_count(self):
        """Ensure we have at least 10 diverse templates."""
        templates = _get_fallback_templates()
        assert len(templates) >= 10, (
            f"Expected at least 10 templates, found {len(templates)}"
        )

    def test_fallback_templates_are_script_objects(self):
        """Each template must be a Script object with required attributes."""
        templates = _get_fallback_templates()
        for i, template in enumerate(templates):
            assert isinstance(template, Script), (
                f"Template {i} must be a Script object, got {type(template).__name__}"
            )
            # Check that Script has expected attributes
            assert hasattr(template, "setup"), f"Template {i} missing 'setup'"
            assert hasattr(template, "punchline"), f"Template {i} missing 'punchline'"
            assert hasattr(template, "storyboard"), f"Template {i} missing 'storyboard'"
            assert hasattr(template, "video_prompts"), f"Template {i} missing 'video_prompts'"
            # visual_prompt is a property derived from video_prompts
            assert hasattr(template, "visual_prompt"), f"Template {i} missing 'visual_prompt'"

    def test_fallback_templates_have_video_prompts(self):
        """Each template must have exactly 3 video_prompts."""
        templates = _get_fallback_templates()
        for i, template in enumerate(templates):
            assert hasattr(template, "video_prompts"), (
                f"Template {i} missing 'video_prompts'"
            )
            assert isinstance(template.video_prompts, list), (
                f"Template {i} video_prompts must be a list"
            )
            assert len(template.video_prompts) == 3, (
                f"Template {i} must have exactly 3 video_prompts, got {len(template.video_prompts)}"
            )
            for j, prompt in enumerate(template.video_prompts):
                assert isinstance(prompt, str), (
                    f"Template {i} video_prompt {j+1} must be a string"
                )
                assert prompt.strip(), (
                    f"Template {i} video_prompt {j+1} cannot be empty"
                )

    def test_fallback_templates_have_non_empty_strings(self):
        """All template fields must be non-empty strings."""
        templates = _get_fallback_templates()
        for i, template in enumerate(templates):
            # Check setup and punchline
            assert isinstance(template.setup, str), (
                f"Template {i} field 'setup' must be a string"
            )
            assert template.setup.strip(), (
                f"Template {i} field 'setup' cannot be empty"
            )
            assert isinstance(template.punchline, str), (
                f"Template {i} field 'punchline' must be a string"
            )
            assert template.punchline.strip(), (
                f"Template {i} field 'punchline' cannot be empty"
            )
            # visual_prompt is derived from video_prompts[0]
            assert isinstance(template.visual_prompt, str), (
                f"Template {i} field 'visual_prompt' must be a string"
            )
            assert template.visual_prompt.strip(), (
                f"Template {i} field 'visual_prompt' cannot be empty"
            )

    def test_fallback_templates_have_3_scene_storyboards(self):
        """Each template must have exactly 3 scenes in storyboard."""
        templates = _get_fallback_templates()
        for i, template in enumerate(templates):
            assert isinstance(template.storyboard, list), (
                f"Template {i} storyboard must be a list"
            )
            assert len(template.storyboard) == 3, (
                f"Template {i} must have exactly 3 scenes, got {len(template.storyboard)}"
            )
            for j, scene in enumerate(template.storyboard):
                assert isinstance(scene, str), (
                    f"Template {i} scene {j+1} must be a string"
                )
                assert scene.strip(), (
                    f"Template {i} scene {j+1} cannot be empty"
                )


class TestShowrunnerGetFallbackScript:
    """Test suite for Showrunner.get_fallback_script method."""

    @pytest.fixture
    def showrunner(self):
        """Create a Showrunner instance for testing."""
        return Showrunner()

    def test_get_fallback_script_returns_script_object(self, showrunner):
        """get_fallback_script should return a Script dataclass."""
        script = showrunner.get_fallback_script("test theme")
        assert isinstance(script, Script)

    def test_get_fallback_script_has_required_attributes(self, showrunner):
        """Returned Script should have setup, punchline, and visual_prompt."""
        script = showrunner.get_fallback_script("test theme")

        assert hasattr(script, "setup")
        assert hasattr(script, "punchline")
        assert hasattr(script, "visual_prompt")

    def test_get_fallback_script_values_are_non_empty(self, showrunner):
        """All Script fields should have non-empty string values."""
        script = showrunner.get_fallback_script("test theme")

        assert isinstance(script.setup, str) and script.setup.strip()
        assert isinstance(script.punchline, str) and script.punchline.strip()
        assert isinstance(script.visual_prompt, str) and script.visual_prompt.strip()

    def test_get_fallback_script_deterministic_with_same_theme(self, showrunner):
        """Same theme should produce same script (deterministic selection)."""
        theme = "bizarre infomercial"
        script1 = showrunner.get_fallback_script(theme)
        script2 = showrunner.get_fallback_script(theme)

        assert script1.setup == script2.setup
        assert script1.punchline == script2.punchline
        assert script1.visual_prompt == script2.visual_prompt

    def test_get_fallback_script_deterministic_with_explicit_seed(self, showrunner):
        """Explicit seed should produce deterministic results."""
        seed = 12345
        script1 = showrunner.get_fallback_script("any theme", seed=seed)
        script2 = showrunner.get_fallback_script("different theme", seed=seed)

        assert script1.setup == script2.setup
        assert script1.punchline == script2.punchline
        assert script1.visual_prompt == script2.visual_prompt

    def test_get_fallback_script_different_themes_can_produce_different_scripts(
        self, showrunner
    ):
        """Different themes should (usually) produce different scripts."""
        # Note: With hash-based selection, different themes should map to
        # different templates most of the time
        themes = [
            "theme_a",
            "theme_b",
            "theme_c",
            "theme_d",
            "theme_e",
        ]
        scripts = [showrunner.get_fallback_script(t) for t in themes]
        setups = {s.setup for s in scripts}

        # At least some should be different (probabilistic but very likely)
        assert len(setups) > 1, "Expected different themes to produce varied scripts"

    def test_get_fallback_script_different_seeds_produce_different_scripts(
        self, showrunner
    ):
        """Different seeds should produce different scripts."""
        # Use seeds that we know map to different templates
        scripts = []
        for seed in range(100):  # Try up to 100 seeds
            script = showrunner.get_fallback_script("theme", seed=seed)
            if not scripts or script.setup != scripts[0].setup:
                scripts.append(script)
            if len(scripts) >= 2:
                break

        assert len(scripts) >= 2, "Expected different seeds to produce varied scripts"
        assert scripts[0].setup != scripts[1].setup

    def test_get_fallback_script_tone_parameter_accepted(self, showrunner):
        """Tone parameter should be accepted (for interface consistency)."""
        # Should not raise
        script = showrunner.get_fallback_script("theme", tone="deadpan")
        assert isinstance(script, Script)

    def test_get_fallback_script_returns_template_from_collection(self, showrunner):
        """Returned script should come from fallback templates."""
        script = showrunner.get_fallback_script("test theme")

        # Check that the script matches one of the templates
        templates = _get_fallback_templates()
        template_setups = {t.setup for t in templates}
        assert script.setup in template_setups, (
            "Script setup should match a template"
        )


class TestShowrunnerInit:
    """Test Showrunner initialization."""

    def test_default_initialization(self):
        """Showrunner should initialize with default values."""
        sr = Showrunner()
        assert sr.base_url == "http://localhost:11434"
        assert sr.model == "llama3:8b"
        assert sr.timeout == 30.0

    def test_custom_initialization(self):
        """Showrunner should accept custom configuration."""
        sr = Showrunner(
            base_url="http://custom:1234",
            model="custom-model",
            timeout=60.0
        )
        assert sr.base_url == "http://custom:1234"
        assert sr.model == "custom-model"
        assert sr.timeout == 60.0

    def test_base_url_trailing_slash_stripped(self):
        """Trailing slashes should be stripped from base_url."""
        sr = Showrunner(base_url="http://localhost:11434/")
        assert sr.base_url == "http://localhost:11434"

    def test_base_url_multiple_trailing_slashes_stripped(self):
        """Multiple trailing slashes should be stripped from base_url."""
        sr = Showrunner(base_url="http://localhost:11434///")
        # rstrip("/") removes all trailing slashes
        assert not sr.base_url.endswith("/")


class TestShowrunnerIsAvailable:
    """Test is_available() Ollama health check."""

    @patch("vortex.models.showrunner.httpx.Client")
    def test_is_available_returns_true_when_model_present(self, mock_client_class):
        """Should return True when Ollama has the model."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]
        }
        mock_client.get.return_value = mock_response

        sr = Showrunner()
        assert sr.is_available() is True

    @patch("vortex.models.showrunner.httpx.Client")
    def test_is_available_returns_true_with_prefix_match(self, mock_client_class):
        """Should return True when model matches by prefix."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3:8b-instruct-q4_0"}]
        }
        mock_client.get.return_value = mock_response

        sr = Showrunner(model="llama3:8b")
        assert sr.is_available() is True

    @patch("vortex.models.showrunner.httpx.Client")
    def test_is_available_returns_false_when_model_missing(self, mock_client_class):
        """Should return False when model not in Ollama."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "mistral:7b"}]
        }
        mock_client.get.return_value = mock_response

        sr = Showrunner(model="llama3:8b")
        assert sr.is_available() is False

    @patch("vortex.models.showrunner.httpx.Client")
    def test_is_available_returns_false_when_models_list_empty(self, mock_client_class):
        """Should return False when Ollama has no models."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_client.get.return_value = mock_response

        sr = Showrunner()
        assert sr.is_available() is False

    @patch("vortex.models.showrunner.httpx.Client")
    def test_is_available_returns_false_on_non_200_status(self, mock_client_class):
        """Should return False when Ollama returns non-200 status."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client.get.return_value = mock_response

        sr = Showrunner()
        assert sr.is_available() is False

    @patch("vortex.models.showrunner.httpx.Client")
    def test_is_available_returns_false_on_connection_error(self, mock_client_class):
        """Should return False when Ollama connection fails."""
        mock_client_class.return_value.__enter__.side_effect = httpx.ConnectError(
            "Connection refused"
        )

        sr = Showrunner()
        assert sr.is_available() is False

    @patch("vortex.models.showrunner.httpx.Client")
    def test_is_available_returns_false_on_timeout(self, mock_client_class):
        """Should return False when Ollama request times out."""
        mock_client_class.return_value.__enter__.side_effect = httpx.TimeoutException(
            "Request timed out"
        )

        sr = Showrunner()
        assert sr.is_available() is False

    @patch("vortex.models.showrunner.httpx.Client")
    def test_is_available_returns_false_on_generic_exception(self, mock_client_class):
        """Should return False when any exception occurs."""
        mock_client_class.return_value.__enter__.side_effect = Exception(
            "Unexpected error"
        )

        sr = Showrunner()
        assert sr.is_available() is False


class TestShowrunnerGenerateScript:
    """Test async generate_script() method."""

    @pytest.mark.asyncio
    @patch("vortex.models.showrunner.httpx.AsyncClient")
    async def test_generate_script_returns_script(self, mock_client_class):
        """Should parse Ollama response into Script object."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        # Include video_prompts in the response (new T2V format)
        json_response = (
            '{"setup": "Test setup", "punchline": "Test punch", '
            '"subject_visual": "a test subject", '
            '"storyboard": ["Scene 1: Visual 1", "Scene 2: Visual 2", "Scene 3: Visual 3"], '
            '"video_prompts": ["Video prompt 1", "Video prompt 2", "Video prompt 3"]}'
        )
        mock_response.json.return_value = {"response": json_response}
        mock_client.post = AsyncMock(return_value=mock_response)

        sr = Showrunner()
        script = await sr.generate_script("test theme")

        assert isinstance(script, Script)
        assert script.setup == "Test setup"
        assert script.punchline == "Test punch"
        assert script.storyboard == [
            "Scene 1: Visual 1",
            "Scene 2: Visual 2",
            "Scene 3: Visual 3",
        ]
        # visual_prompt returns first video_prompt (T2V)
        assert script.visual_prompt == "Video prompt 1"

    @pytest.mark.asyncio
    @patch("vortex.models.showrunner.httpx.AsyncClient")
    async def test_generate_script_calls_correct_endpoint(self, mock_client_class):
        """Should call the correct Ollama API endpoint."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"setup": "S", "punchline": "P", '
                        '"storyboard": ["S1", "S2", "S3"]}'
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        sr = Showrunner(base_url="http://test:1234", model="test-model")
        await sr.generate_script("test theme", tone="deadpan")

        # Verify the call was made to the correct endpoint
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://test:1234/api/generate"
        assert call_args[1]["json"]["model"] == "test-model"

    @pytest.mark.asyncio
    @patch("vortex.models.showrunner.httpx.AsyncClient")
    async def test_generate_script_handles_markdown_json(self, mock_client_class):
        """Should extract JSON from markdown code blocks."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

        markdown_response = """Here's the script:
```json
{
  "setup": "Markdown setup",
  "punchline": "Markdown punch",
  "subject_visual": "a cartoon host",
  "storyboard": ["Scene 1: MD visual", "Scene 2: MD visual", "Scene 3: MD visual"],
  "video_prompts": ["MD video 1", "MD video 2", "MD video 3"]
}
```
Hope you like it!"""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": markdown_response}
        mock_client.post = AsyncMock(return_value=mock_response)

        sr = Showrunner()
        script = await sr.generate_script("test")

        assert script.setup == "Markdown setup"
        assert script.punchline == "Markdown punch"
        assert script.storyboard == [
            "Scene 1: MD visual",
            "Scene 2: MD visual",
            "Scene 3: MD visual",
        ]
        # visual_prompt returns first video_prompt (T2V)
        assert script.visual_prompt == "MD video 1"

    @pytest.mark.asyncio
    @patch("vortex.models.showrunner.httpx.AsyncClient")
    async def test_generate_script_handles_legacy_visual_prompt_format(self, mock_client_class):
        """Should handle legacy visual_prompt format by converting to 3-scene storyboard.

        This test ensures backward compatibility with older Ollama models that may
        return the legacy format with single visual_prompt instead of storyboard.
        When video_prompts are missing, they are auto-generated from storyboard.
        """
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

        # Legacy format with visual_prompt instead of storyboard
        legacy_json_response = (
            '{"setup": "Legacy setup", "punchline": "Legacy punch", '
            '"visual_prompt": "Legacy visual description"}'
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": legacy_json_response}
        mock_client.post = AsyncMock(return_value=mock_response)

        sr = Showrunner()
        script = await sr.generate_script("test")

        # Should convert legacy visual_prompt to 3-scene storyboard
        assert script.setup == "Legacy setup"
        assert script.punchline == "Legacy punch"
        # Legacy visual_prompt gets duplicated to all 3 scenes
        assert script.storyboard == [
            "Legacy visual description",
            "Legacy visual description",
            "Legacy visual description",
        ]
        # video_prompts are auto-generated from storyboard + subject + style
        # visual_prompt returns first video_prompt which contains the scene + style
        assert "Legacy visual description" in script.visual_prompt
        assert ADULT_SWIM_STYLE in script.visual_prompt

    @pytest.mark.asyncio
    @patch("vortex.models.showrunner.httpx.AsyncClient")
    async def test_generate_script_raises_on_empty_theme(self, mock_client_class):
        """Should raise ShowrunnerError on empty theme."""
        sr = Showrunner()

        with pytest.raises(ShowrunnerError) as exc_info:
            await sr.generate_script("")

        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @patch("vortex.models.showrunner.httpx.AsyncClient")
    async def test_generate_script_raises_on_whitespace_theme(self, mock_client_class):
        """Should raise ShowrunnerError on whitespace-only theme."""
        sr = Showrunner()

        with pytest.raises(ShowrunnerError) as exc_info:
            await sr.generate_script("   ")

        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @patch("vortex.models.showrunner.httpx.AsyncClient")
    async def test_generate_script_raises_on_connection_error(self, mock_client_class):
        """Should raise ShowrunnerError on connection failure."""
        mock_client_class.return_value.__aenter__.side_effect = httpx.ConnectError(
            "Connection refused"
        )

        sr = Showrunner()
        with pytest.raises(ShowrunnerError) as exc_info:
            await sr.generate_script("test")

        assert "connect" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @patch("vortex.models.showrunner.httpx.AsyncClient")
    async def test_generate_script_raises_on_timeout(self, mock_client_class):
        """Should raise ShowrunnerError on request timeout."""
        mock_client_class.return_value.__aenter__.side_effect = httpx.TimeoutException(
            "Request timed out"
        )

        sr = Showrunner()
        with pytest.raises(ShowrunnerError) as exc_info:
            await sr.generate_script("test")

        assert "timed out" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @patch("vortex.models.showrunner.httpx.AsyncClient")
    async def test_generate_script_raises_on_non_200_status(self, mock_client_class):
        """Should raise ShowrunnerError on non-200 response."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client.post = AsyncMock(return_value=mock_response)

        sr = Showrunner()
        with pytest.raises(ShowrunnerError) as exc_info:
            await sr.generate_script("test")

        assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("vortex.models.showrunner.httpx.AsyncClient")
    async def test_generate_script_raises_on_invalid_json(self, mock_client_class):
        """Should raise ShowrunnerError on unparseable response."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "This is not valid JSON at all"}
        mock_client.post = AsyncMock(return_value=mock_response)

        sr = Showrunner()
        with pytest.raises(ShowrunnerError) as exc_info:
            await sr.generate_script("test")

        assert "json" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @patch("vortex.models.showrunner.httpx.AsyncClient")
    async def test_generate_script_raises_on_missing_fields(self, mock_client_class):
        """Should raise ShowrunnerError when response is missing required fields."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"setup": "Only setup, missing other fields"}'
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        sr = Showrunner()
        with pytest.raises(ShowrunnerError) as exc_info:
            await sr.generate_script("test")

        assert "missing" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @patch("vortex.models.showrunner.httpx.AsyncClient")
    async def test_generate_script_uses_moderate_temperature(self, mock_client_class):
        """Should use temperature 0.7 for balanced creativity/coherence."""
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"setup": "S", "punchline": "P", "storyboard": ["S1", "S2", "S3"]}'
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        sr = Showrunner()
        await sr.generate_script("test theme")

        call_args = mock_client.post.call_args
        options = call_args[1]["json"]["options"]
        assert options["temperature"] == 0.7
        assert options["top_p"] == 0.9


class TestShowrunnerJsonParsing:
    """Test JSON extraction edge cases via _extract_json and _parse_script_response."""

    @pytest.fixture
    def showrunner(self):
        """Create a Showrunner instance for testing."""
        return Showrunner()

    def test_extract_json_from_pure_json(self, showrunner):
        """Should handle pure JSON response."""
        json_str = '{"setup": "S", "punchline": "P", "visual_prompt": "V"}'
        result = showrunner._extract_json(json_str)
        assert result == json_str

    def test_extract_json_from_pure_json_with_whitespace(self, showrunner):
        """Should handle pure JSON with surrounding whitespace."""
        json_str = '{"setup": "S", "punchline": "P", "visual_prompt": "V"}'
        result = showrunner._extract_json(f"  {json_str}  ")
        assert result == json_str

    def test_extract_json_from_markdown_code_block(self, showrunner):
        """Should extract JSON from ```json ... ``` blocks."""
        text = """Here's your script:
```json
{"setup": "S", "punchline": "P", "visual_prompt": "V"}
```
"""
        result = showrunner._extract_json(text)
        # Result should be parseable JSON
        import json
        data = json.loads(result)
        assert data["setup"] == "S"

    def test_extract_json_from_markdown_code_block_no_language(self, showrunner):
        """Should extract JSON from ``` ... ``` blocks without language specifier."""
        text = """Here's your script:
```
{"setup": "S", "punchline": "P", "visual_prompt": "V"}
```
"""
        result = showrunner._extract_json(text)
        import json
        data = json.loads(result)
        assert data["setup"] == "S"

    def test_extract_json_from_embedded_braces(self, showrunner):
        """Should find JSON object in surrounding text."""
        text = (
            'Sure! Here you go: {"setup": "S", "punchline": "P", '
            '"visual_prompt": "V"} Hope this helps!'
        )
        result = showrunner._extract_json(text)
        import json
        data = json.loads(result)
        assert data["setup"] == "S"

    def test_extract_json_multiline(self, showrunner):
        """Should extract multiline JSON."""
        text = """The script is:
{
  "setup": "Multiline setup",
  "punchline": "Multiline punch",
  "visual_prompt": "Multiline visual"
}
End of script."""
        result = showrunner._extract_json(text)
        import json
        data = json.loads(result)
        assert data["setup"] == "Multiline setup"

    def test_extract_json_raises_on_no_json(self, showrunner):
        """Should raise ShowrunnerError when no JSON found."""
        text = "This text contains no JSON at all, just plain text."
        with pytest.raises(ShowrunnerError) as exc_info:
            showrunner._extract_json(text)
        assert "could not extract json" in str(exc_info.value).lower()

    def test_parse_script_response_validates_string_types(self, showrunner):
        """Should raise ShowrunnerError when fields are not strings."""
        json_str = '{"setup": 123, "punchline": "P", "storyboard": ["S1", "S2", "S3"]}'
        with pytest.raises(ShowrunnerError) as exc_info:
            showrunner._parse_script_response(json_str)
        assert "string" in str(exc_info.value).lower()

    def test_parse_script_response_validates_non_empty_fields(self, showrunner):
        """Should raise ShowrunnerError when fields are empty strings."""
        json_str = '{"setup": "", "punchline": "P", "storyboard": ["S1", "S2", "S3"]}'
        with pytest.raises(ShowrunnerError) as exc_info:
            showrunner._parse_script_response(json_str)
        assert "empty" in str(exc_info.value).lower()

    def test_parse_script_response_validates_whitespace_only_fields(self, showrunner):
        """Should raise ShowrunnerError when fields are whitespace-only."""
        json_str = '{"setup": "   ", "punchline": "P", "storyboard": ["S1", "S2", "S3"]}'
        with pytest.raises(ShowrunnerError) as exc_info:
            showrunner._parse_script_response(json_str)
        assert "empty" in str(exc_info.value).lower()

    def test_parse_script_response_strips_whitespace(self, showrunner):
        """Should strip leading/trailing whitespace from fields."""
        json_str = (
            '{"setup": "  Setup  ", "punchline": "  Punch  ", '
            '"storyboard": ["  S1  ", "  S2  ", "  S3  "]}'
        )
        script = showrunner._parse_script_response(json_str)
        assert script.setup == "Setup"
        assert script.punchline == "Punch"
        assert script.storyboard == ["S1", "S2", "S3"]

    def test_parse_script_response_handles_legacy_visual_prompt(self, showrunner):
        """Should convert legacy visual_prompt to 3-scene storyboard.

        When video_prompts are missing, they are auto-generated from storyboard + style.
        """
        json_str = '{"setup": "Setup", "punchline": "Punch", "visual_prompt": "Visual"}'
        script = showrunner._parse_script_response(json_str)
        assert script.setup == "Setup"
        assert script.punchline == "Punch"
        # Legacy visual_prompt gets duplicated to all 3 scenes
        assert script.storyboard == ["Visual", "Visual", "Visual"]
        # video_prompts auto-generated: visual_prompt returns first which includes style
        assert "Visual" in script.visual_prompt
        assert ADULT_SWIM_STYLE in script.visual_prompt


class TestShowrunnerError:
    """Test ShowrunnerError exception class."""

    def test_showrunner_error_is_exception(self):
        """ShowrunnerError should be an Exception subclass."""
        assert issubclass(ShowrunnerError, Exception)

    def test_showrunner_error_can_be_raised(self):
        """ShowrunnerError should be raisable with a message."""
        with pytest.raises(ShowrunnerError) as exc_info:
            raise ShowrunnerError("Test error message")
        assert "Test error message" in str(exc_info.value)

    def test_showrunner_error_preserves_cause(self):
        """ShowrunnerError should preserve the original exception cause."""
        original = ValueError("Original error")
        try:
            raise ShowrunnerError("Wrapped error") from original
        except ShowrunnerError as e:
            assert e.__cause__ is original


class TestShowrunnerPromptConstraints:
    """Test suite for vocabulary and transformation constraints in prompts."""

    def test_prompt_contains_vocabulary_rule(self):
        """SCRIPT_PROMPT_TEMPLATE should contain vocabulary constraints."""
        assert "VOCABULARY RULE" in SCRIPT_PROMPT_TEMPLATE
        assert "neologism" in SCRIPT_PROMPT_TEMPLATE.lower()

    def test_prompt_contains_transformation_rule(self):
        """SCRIPT_PROMPT_TEMPLATE should prohibit shape transformation."""
        assert "TRANSFORMATION RULE" in SCRIPT_PROMPT_TEMPLATE
        assert "maintain" in SCRIPT_PROMPT_TEMPLATE.lower()

    def test_prompt_has_bad_examples(self):
        """SCRIPT_PROMPT_TEMPLATE should include BAD examples to avoid."""
        # Check for neologism bad examples
        assert "Flug" in SCRIPT_PROMPT_TEMPLATE or "Zorblax" in SCRIPT_PROMPT_TEMPLATE
        # Check for transformation bad examples
        assert "transforms into" in SCRIPT_PROMPT_TEMPLATE.lower()

    def test_prompt_has_good_examples(self):
        """SCRIPT_PROMPT_TEMPLATE should include GOOD examples."""
        # Check for vocabulary good examples
        assert "blob" in SCRIPT_PROMPT_TEMPLATE.lower()
        # Check for transformation good examples
        assert "bounce" in SCRIPT_PROMPT_TEMPLATE.lower()
