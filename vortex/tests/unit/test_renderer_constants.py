"""Unit tests for renderer constants and style prompts."""


class TestVisualStylePrompt:
    """Tests for VISUAL_STYLE_PROMPT constant."""

    def test_visual_style_includes_spatial_anchoring(self):
        """VISUAL_STYLE_PROMPT should include spatial consistency terms.

        These terms help Flux generate images where subjects are properly
        anchored in the scene, preventing background dissociation and
        objects clipping through floor shadows.
        """
        from vortex.renderers.default.renderer import VISUAL_STYLE_PROMPT

        assert "grounded" in VISUAL_STYLE_PROMPT.lower()
        assert "perspective" in VISUAL_STYLE_PROMPT.lower()

    def test_visual_style_includes_cartoon_style(self):
        """VISUAL_STYLE_PROMPT should include cartoon style terms."""
        from vortex.renderers.default.renderer import VISUAL_STYLE_PROMPT

        assert "cartoon" in VISUAL_STYLE_PROMPT.lower()
        assert "cel shaded" in VISUAL_STYLE_PROMPT.lower()

    def test_visual_style_avoids_problematic_terms(self):
        """VISUAL_STYLE_PROMPT should not include terms that cause artifacts.

        Halftone texture was removed because it causes swirling during
        VAE downsampling.
        """
        from vortex.renderers.default.renderer import VISUAL_STYLE_PROMPT

        assert "halftone" not in VISUAL_STYLE_PROMPT.lower()


class TestMotionStyleSuffix:
    """Tests for MOTION_STYLE_SUFFIX constant."""

    def test_motion_style_includes_animation_terms(self):
        """MOTION_STYLE_SUFFIX should include animation terms for CogVideoX."""
        from vortex.renderers.default.renderer import MOTION_STYLE_SUFFIX

        assert "animation" in MOTION_STYLE_SUFFIX.lower()
        assert "motion" in MOTION_STYLE_SUFFIX.lower()
