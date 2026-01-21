"""Unit tests for Narrative Chain recipe schema validation."""

from vortex.renderers import (
    get_recipe_defaults,
    merge_with_defaults,
    validate_recipe,
)


class TestValidateRecipe:
    """Tests for validate_recipe function."""

    def test_valid_minimal_recipe(self):
        """Test that a minimal valid recipe passes validation."""
        recipe = {
            "slot_params": {"slot_id": 1},
        }
        errors = validate_recipe(recipe)
        assert errors == []

    def test_valid_recipe_with_auto_script(self):
        """Test recipe with LLM-driven script generation."""
        recipe = {
            "slot_params": {"slot_id": 1001, "seed": 42, "target_duration": 12.0},
            "narrative": {"theme": "alien talk show", "tone": "manic", "auto_script": True},
            "audio": {"voice_id": "am_michael", "speed": 1.1},
            "video": {"style_prompt": "neon colors, alien world"},
            "quality": {"clip_threshold": 0.70},
        }
        errors = validate_recipe(recipe)
        assert errors == []

    def test_valid_recipe_with_manual_script(self):
        """Test recipe with manually provided script."""
        recipe = {
            "slot_params": {"slot_id": 1, "fps": 8},
            "narrative": {
                "auto_script": False,
                "script": {
                    "setup": "Ever wonder why aliens never call back?",
                    "punchline": "Because they're busy with interdimensional cable!",
                    "visual_prompt": "Alien sitting on couch watching TV",
                },
            },
            "audio": {"voice_id": "af_heart", "speed": 1.0},
        }
        errors = validate_recipe(recipe)
        assert errors == []

    def test_missing_slot_params(self):
        """Test that missing slot_params is detected."""
        recipe = {
            "narrative": {"theme": "test"},
        }
        errors = validate_recipe(recipe)
        assert any("slot_params" in e for e in errors)

    def test_missing_slot_id(self):
        """Test that missing slot_id is detected."""
        recipe = {
            "slot_params": {"fps": 8},  # Missing slot_id
        }
        errors = validate_recipe(recipe)
        assert any("slot_id" in e for e in errors)

    def test_invalid_slot_params_type(self):
        """Test that invalid slot_params type is detected."""
        recipe = {
            "slot_params": "invalid",  # Should be dict
        }
        errors = validate_recipe(recipe)
        assert any("object" in e for e in errors)

    def test_invalid_slot_id_type(self):
        """Test that non-integer slot_id is detected."""
        recipe = {
            "slot_params": {"slot_id": "one"},  # Should be int
        }
        errors = validate_recipe(recipe)
        assert any("integer" in e for e in errors)

    def test_invalid_tone_enum(self):
        """Test that invalid tone value is detected."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "narrative": {"tone": "boring"},  # Invalid enum value
        }
        errors = validate_recipe(recipe)
        assert any("tone" in e and "absurd" in e for e in errors)

    def test_missing_script_when_auto_script_false(self):
        """Test that script is required when auto_script=False."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "narrative": {"auto_script": False},  # Missing script
        }
        errors = validate_recipe(recipe)
        assert any("narrative.script required" in e for e in errors)

    def test_missing_script_fields_when_auto_script_false(self):
        """Test that all script fields are required when auto_script=False."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "narrative": {
                "auto_script": False,
                "script": {"setup": "Hello"},  # Missing punchline and visual_prompt
            },
        }
        errors = validate_recipe(recipe)
        assert any("punchline" in e for e in errors)
        assert any("visual_prompt" in e for e in errors)

    def test_invalid_bgm_volume_range(self):
        """Test that bgm_volume out of range is detected."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "audio": {"bgm_volume": 1.5},  # Out of range (0.0-1.0)
        }
        errors = validate_recipe(recipe)
        assert any("bgm_volume" in e and "between" in e for e in errors)

    def test_invalid_clip_threshold_range(self):
        """Test that clip_threshold out of range is detected."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "quality": {"clip_threshold": -0.1},  # Out of range (0.0-1.0)
        }
        errors = validate_recipe(recipe)
        assert any("clip_threshold" in e and "between" in e for e in errors)

    def test_invalid_max_retries_type(self):
        """Test that non-integer max_retries is detected."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "quality": {"max_retries": 3.5},  # Should be int
        }
        errors = validate_recipe(recipe)
        assert any("max_retries" in e and "integer" in e for e in errors)

    def test_invalid_speed_type(self):
        """Test that non-number speed is detected."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "audio": {"speed": "fast"},  # Should be number
        }
        errors = validate_recipe(recipe)
        assert any("speed" in e and "number" in e for e in errors)

    def test_invalid_guidance_scale_type(self):
        """Test that non-number guidance_scale is detected."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "video": {"guidance_scale": "high"},  # Should be number
        }
        errors = validate_recipe(recipe)
        assert any("guidance_scale" in e and "number" in e for e in errors)

    def test_invalid_target_duration_type(self):
        """Test that non-number target_duration is detected."""
        recipe = {
            "slot_params": {"slot_id": 1, "target_duration": "long"},
        }
        errors = validate_recipe(recipe)
        assert any("target_duration" in e and "number" in e for e in errors)

    def test_invalid_fps_type(self):
        """Test that non-integer fps is detected."""
        recipe = {
            "slot_params": {"slot_id": 1, "fps": 8.5},  # Should be int
        }
        errors = validate_recipe(recipe)
        assert any("fps" in e and "integer" in e for e in errors)


class TestGetRecipeDefaults:
    """Tests for get_recipe_defaults function."""

    def test_returns_all_sections(self):
        """Test that defaults include all sections."""
        defaults = get_recipe_defaults()
        assert "slot_params" in defaults
        assert "narrative" in defaults
        assert "audio" in defaults
        assert "video" in defaults
        assert "quality" in defaults

    def test_default_fps(self):
        """Test default fps is 8 (CogVideoX native)."""
        defaults = get_recipe_defaults()
        assert defaults["slot_params"]["fps"] == 8

    def test_default_target_duration(self):
        """Test default target_duration is 12.0."""
        defaults = get_recipe_defaults()
        assert defaults["slot_params"]["target_duration"] == 12.0

    def test_default_auto_script(self):
        """Test default auto_script is True (LLM-driven)."""
        defaults = get_recipe_defaults()
        assert defaults["narrative"]["auto_script"] is True

    def test_default_theme(self):
        """Test default theme."""
        defaults = get_recipe_defaults()
        assert defaults["narrative"]["theme"] == "bizarre infomercial"

    def test_default_tone(self):
        """Test default tone."""
        defaults = get_recipe_defaults()
        assert defaults["narrative"]["tone"] == "absurd"

    def test_default_voice_id(self):
        """Test default voice_id."""
        defaults = get_recipe_defaults()
        assert defaults["audio"]["voice_id"] == "af_heart"

    def test_default_speed(self):
        """Test default speech speed."""
        defaults = get_recipe_defaults()
        assert defaults["audio"]["speed"] == 1.0

    def test_default_bgm_volume(self):
        """Test default BGM volume."""
        defaults = get_recipe_defaults()
        assert defaults["audio"]["bgm_volume"] == 0.3

    def test_default_style_prompt(self):
        """Test default style prompt contains key aesthetic terms."""
        defaults = get_recipe_defaults()
        style = defaults["video"]["style_prompt"]
        assert "cartoon" in style
        assert "interdimensional cable" in style

    def test_default_negative_prompt(self):
        """Test default negative prompt."""
        defaults = get_recipe_defaults()
        assert "realistic" in defaults["video"]["negative_prompt"]
        assert "blurry" in defaults["video"]["negative_prompt"]

    def test_default_guidance_scale(self):
        """Test default CogVideoX guidance scale."""
        defaults = get_recipe_defaults()
        assert defaults["video"]["guidance_scale"] == 6.0

    def test_default_clip_threshold(self):
        """Test default CLIP threshold."""
        defaults = get_recipe_defaults()
        assert defaults["quality"]["clip_threshold"] == 0.70

    def test_default_max_retries(self):
        """Test default max retries."""
        defaults = get_recipe_defaults()
        assert defaults["quality"]["max_retries"] == 3


class TestMergeWithDefaults:
    """Tests for merge_with_defaults function."""

    def test_preserves_provided_values(self):
        """Test that provided values are preserved."""
        recipe = {
            "slot_params": {"slot_id": 1001, "fps": 12},
            "narrative": {"theme": "custom theme", "tone": "deadpan"},
            "audio": {"voice_id": "am_michael", "speed": 1.2},
        }
        merged = merge_with_defaults(recipe)
        assert merged["slot_params"]["fps"] == 12
        assert merged["narrative"]["theme"] == "custom theme"
        assert merged["narrative"]["tone"] == "deadpan"
        assert merged["audio"]["voice_id"] == "am_michael"
        assert merged["audio"]["speed"] == 1.2

    def test_fills_missing_with_defaults(self):
        """Test that missing values are filled with defaults."""
        recipe = {
            "slot_params": {"slot_id": 1},
        }
        merged = merge_with_defaults(recipe)
        # Check defaults were applied
        assert merged["slot_params"]["fps"] == 8
        assert merged["slot_params"]["target_duration"] == 12.0
        assert merged["narrative"]["auto_script"] is True
        assert merged["audio"]["voice_id"] == "af_heart"
        assert merged["video"]["guidance_scale"] == 6.0
        assert merged["quality"]["clip_threshold"] == 0.70

    def test_adds_missing_sections(self):
        """Test that missing sections are added."""
        recipe = {
            "slot_params": {"slot_id": 1},
        }
        merged = merge_with_defaults(recipe)
        assert "narrative" in merged
        assert "audio" in merged
        assert "video" in merged
        assert "quality" in merged

    def test_deep_merge_nested_script(self):
        """Test that nested script section is merged correctly."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "narrative": {
                "auto_script": False,
                "script": {
                    "setup": "Custom setup",
                    "punchline": "Custom punchline",
                    "visual_prompt": "Custom visual",
                },
            },
        }
        merged = merge_with_defaults(recipe)
        # User values preserved
        assert merged["narrative"]["script"]["setup"] == "Custom setup"
        assert merged["narrative"]["script"]["punchline"] == "Custom punchline"
        assert merged["narrative"]["script"]["visual_prompt"] == "Custom visual"
        # Default theme still applied
        assert merged["narrative"]["theme"] == "bizarre infomercial"

    def test_preserves_extra_sections(self):
        """Test that extra sections not in defaults are preserved."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "custom_data": {"key": "value"},
        }
        merged = merge_with_defaults(recipe)
        assert "custom_data" in merged
        assert merged["custom_data"]["key"] == "value"

    def test_merged_recipe_is_independent(self):
        """Test that merged recipe is a deep copy, not sharing references."""
        recipe = {
            "slot_params": {"slot_id": 1},
        }
        merged1 = merge_with_defaults(recipe)
        merged2 = merge_with_defaults(recipe)

        # Modify merged1
        merged1["narrative"]["theme"] = "modified"

        # merged2 should be unaffected
        assert merged2["narrative"]["theme"] == "bizarre infomercial"
