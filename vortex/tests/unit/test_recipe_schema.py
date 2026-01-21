"""Unit tests for ToonGen recipe schema validation."""

from vortex.renderers import (
    get_recipe_defaults,
    merge_with_defaults,
    validate_recipe,
)


class TestValidateRecipe:
    """Tests for validate_recipe function."""

    def test_valid_recipe(self):
        """Test that a valid recipe passes validation."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "audio_track": {"script": "Hello world!"},
            "visual_track": {"prompt": "scientist in lab coat"},
        }
        errors = validate_recipe(recipe)
        assert errors == []

    def test_valid_recipe_with_all_fields(self):
        """Test that a recipe with all optional fields passes validation."""
        recipe = {
            "slot_params": {"slot_id": 1, "fps": 30, "seed": 42},
            "audio_track": {
                "script": "Hello world!",
                "engine": "f5_tts",
                "voice_style": "rick",
                "voice_id": "af_heart",
            },
            "audio_environment": {
                "bgm": "ambient_music",
                "sfx": "thunder",
                "mix_ratio": 0.5,
            },
            "visual_track": {
                "prompt": "scientist in lab coat",
                "negative_prompt": "blurry, distorted",
            },
        }
        errors = validate_recipe(recipe)
        assert errors == []

    def test_missing_slot_params(self):
        """Test that missing slot_params is detected."""
        recipe = {
            "audio_track": {"script": "Hello"},
            "visual_track": {"prompt": "Test"},
        }
        errors = validate_recipe(recipe)
        assert any("slot_params" in e for e in errors)

    def test_missing_audio_track(self):
        """Test that missing audio_track is detected."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "visual_track": {"prompt": "Test"},
        }
        errors = validate_recipe(recipe)
        assert any("audio_track" in e for e in errors)

    def test_missing_visual_track(self):
        """Test that missing visual_track is detected."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "audio_track": {"script": "Hello"},
        }
        errors = validate_recipe(recipe)
        assert any("visual_track" in e for e in errors)

    def test_missing_slot_id(self):
        """Test that missing slot_id is detected."""
        recipe = {
            "slot_params": {"fps": 24},  # Missing slot_id
            "audio_track": {"script": "Hello"},
            "visual_track": {"prompt": "Test"},
        }
        errors = validate_recipe(recipe)
        assert any("slot_id" in e for e in errors)

    def test_missing_script(self):
        """Test that missing script is detected."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "audio_track": {"voice_id": "af_heart"},  # Missing script
            "visual_track": {"prompt": "Test"},
        }
        errors = validate_recipe(recipe)
        assert any("script" in e for e in errors)

    def test_missing_prompt(self):
        """Test that missing prompt is detected."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "audio_track": {"script": "Hello"},
            "visual_track": {"negative_prompt": "blurry"},  # Missing prompt
        }
        errors = validate_recipe(recipe)
        assert any("prompt" in e for e in errors)

    def test_invalid_slot_params_type(self):
        """Test that invalid slot_params type is detected."""
        recipe = {
            "slot_params": "invalid",  # Should be dict
            "audio_track": {"script": "Hello"},
            "visual_track": {"prompt": "Test"},
        }
        errors = validate_recipe(recipe)
        assert any("object" in e for e in errors)

    def test_invalid_slot_id_type(self):
        """Test that non-integer slot_id is detected."""
        recipe = {
            "slot_params": {"slot_id": "one"},  # Should be int
            "audio_track": {"script": "Hello"},
            "visual_track": {"prompt": "Test"},
        }
        errors = validate_recipe(recipe)
        assert any("integer" in e for e in errors)


class TestGetRecipeDefaults:
    """Tests for get_recipe_defaults function."""

    def test_returns_all_sections(self):
        """Test that defaults include all sections."""
        defaults = get_recipe_defaults()
        assert "slot_params" in defaults
        assert "audio_track" in defaults
        assert "audio_environment" in defaults
        assert "visual_track" in defaults

    def test_default_fps(self):
        """Test default fps is 24."""
        defaults = get_recipe_defaults()
        assert defaults["slot_params"]["fps"] == 24

    def test_default_engine(self):
        """Test default TTS engine is auto."""
        defaults = get_recipe_defaults()
        assert defaults["audio_track"]["engine"] == "auto"

    def test_default_voice_id(self):
        """Test default voice_id."""
        defaults = get_recipe_defaults()
        assert defaults["audio_track"]["voice_id"] == "af_heart"

    def test_default_mix_ratio(self):
        """Test default audio mix ratio."""
        defaults = get_recipe_defaults()
        assert defaults["audio_environment"]["mix_ratio"] == 0.3

    def test_default_negative_prompt(self):
        """Test default negative prompt."""
        defaults = get_recipe_defaults()
        assert "blurry" in defaults["visual_track"]["negative_prompt"]


class TestMergeWithDefaults:
    """Tests for merge_with_defaults function."""

    def test_preserves_provided_values(self):
        """Test that provided values are preserved."""
        recipe = {
            "slot_params": {"slot_id": 1, "fps": 30},
            "audio_track": {"script": "Custom script", "voice_id": "bf_emma"},
            "visual_track": {"prompt": "Custom prompt"},
        }
        merged = merge_with_defaults(recipe)
        assert merged["slot_params"]["fps"] == 30
        assert merged["audio_track"]["voice_id"] == "bf_emma"
        assert merged["visual_track"]["prompt"] == "Custom prompt"

    def test_fills_missing_with_defaults(self):
        """Test that missing values are filled with defaults."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "audio_track": {"script": "Hello"},
            "visual_track": {"prompt": "Test"},
        }
        merged = merge_with_defaults(recipe)
        # Check defaults were applied
        assert merged["slot_params"]["fps"] == 24
        assert merged["audio_track"]["engine"] == "auto"
        assert merged["audio_track"]["voice_id"] == "af_heart"
        assert "blurry" in merged["visual_track"]["negative_prompt"]

    def test_adds_missing_sections(self):
        """Test that missing sections are added."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "audio_track": {"script": "Hello"},
            "visual_track": {"prompt": "Test"},
            # audio_environment is missing
        }
        merged = merge_with_defaults(recipe)
        assert "audio_environment" in merged
        assert merged["audio_environment"]["mix_ratio"] == 0.3

    def test_audio_environment_defaults(self):
        """Test that audio_environment gets proper defaults."""
        recipe = {
            "slot_params": {"slot_id": 1},
            "audio_track": {"script": "Hello"},
            "visual_track": {"prompt": "Test"},
        }
        merged = merge_with_defaults(recipe)
        assert merged["audio_environment"]["bgm"] is None
        assert merged["audio_environment"]["sfx"] is None
        assert merged["audio_environment"]["mix_ratio"] == 0.3
