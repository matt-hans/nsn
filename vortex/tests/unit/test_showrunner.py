"""Unit tests for Showrunner script generator.

Tests the Showrunner class including fallback template functionality.
"""

import pytest

from vortex.models.showrunner import (
    FALLBACK_TEMPLATES,
    Script,
    Showrunner,
)


class TestFallbackTemplates:
    """Test suite for fallback template configuration."""

    def test_fallback_templates_has_minimum_count(self):
        """Ensure we have at least 10 diverse templates."""
        assert len(FALLBACK_TEMPLATES) >= 10, (
            f"Expected at least 10 templates, found {len(FALLBACK_TEMPLATES)}"
        )

    def test_fallback_templates_have_required_fields(self):
        """Each template must have setup, punchline, and visual_prompt."""
        required_fields = {"setup", "punchline", "visual_prompt"}

        for i, template in enumerate(FALLBACK_TEMPLATES):
            missing = required_fields - set(template.keys())
            assert not missing, (
                f"Template {i} missing required fields: {missing}"
            )

    def test_fallback_templates_have_non_empty_strings(self):
        """All template fields must be non-empty strings."""
        for i, template in enumerate(FALLBACK_TEMPLATES):
            for field in ["setup", "punchline", "visual_prompt"]:
                value = template[field]
                assert isinstance(value, str), (
                    f"Template {i} field '{field}' must be a string"
                )
                assert value.strip(), (
                    f"Template {i} field '{field}' cannot be empty"
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
        """Returned script should come from FALLBACK_TEMPLATES."""
        script = showrunner.get_fallback_script("test theme")

        # Check that the script matches one of the templates
        template_setups = {t["setup"] for t in FALLBACK_TEMPLATES}
        assert script.setup in template_setups, (
            "Script setup should match a template"
        )
