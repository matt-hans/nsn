"""Unit tests for recipe schema validation."""

import pytest

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
            "slot_params": {"slot_id": 1, "duration_sec": 45},
            "audio_track": {"script": "Hello world!"},
            "visual_track": {"prompt": "scientist in lab coat"},
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
            "slot_params": {"slot_id": 1, "duration_sec": 45},
            "visual_track": {"prompt": "Test"},
        }
        errors = validate_recipe(recipe)
        assert any("audio_track" in e for e in errors)

    def test_missing_visual_track(self):
        """Test that missing visual_track is detected."""
        recipe = {
            "slot_params": {"slot_id": 1, "duration_sec": 45},
            "audio_track": {"script": "Hello"},
        }
        errors = validate_recipe(recipe)
        assert any("visual_track" in e for e in errors)

    def test_missing_slot_id(self):
        """Test that missing slot_id is detected."""
        recipe = {
            "slot_params": {"duration_sec": 45},  # Missing slot_id
            "audio_track": {"script": "Hello"},
            "visual_track": {"prompt": "Test"},
        }
        errors = validate_recipe(recipe)
        assert any("slot_id" in e for e in errors)

    def test_missing_script(self):
        """Test that missing script is detected."""
        recipe = {
            "slot_params": {"slot_id": 1, "duration_sec": 45},
            "audio_track": {"voice_id": "rick"},  # Missing script
            "visual_track": {"prompt": "Test"},
        }
        errors = validate_recipe(recipe)
        assert any("script" in e for e in errors)

    def test_missing_prompt(self):
        """Test that missing prompt is detected."""
        recipe = {
            "slot_params": {"slot_id": 1, "duration_sec": 45},
            "audio_track": {"script": "Hello"},
            "visual_track": {"expression_preset": "neutral"},  # Missing prompt
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


class TestGetRecipeDefaults:
    """Tests for get_recipe_defaults function."""

    def test_returns_all_sections(self):
        """Test that defaults include all sections."""
        defaults = get_recipe_defaults()
        assert "slot_params" in defaults
        assert "audio_track" in defaults
        assert "visual_track" in defaults
        assert "semantic_constraints" in defaults

    def test_default_duration(self):
        """Test default duration is 45 seconds."""
        defaults = get_recipe_defaults()
        assert defaults["slot_params"]["duration_sec"] == 45

    def test_default_fps(self):
        """Test default fps is 24."""
        defaults = get_recipe_defaults()
        assert defaults["slot_params"]["fps"] == 24

    def test_default_voice_id(self):
        """Test default voice_id."""
        defaults = get_recipe_defaults()
        assert defaults["audio_track"]["voice_id"] == "rick_c137"

    def test_default_expression_preset(self):
        """Test default expression_preset."""
        defaults = get_recipe_defaults()
        assert defaults["visual_track"]["expression_preset"] == "neutral"


class TestMergeWithDefaults:
    """Tests for merge_with_defaults function."""

    def test_preserves_provided_values(self):
        """Test that provided values are preserved."""
        recipe = {
            "slot_params": {"slot_id": 1, "duration_sec": 30},
            "audio_track": {"script": "Custom script", "voice_id": "morty"},
            "visual_track": {"prompt": "Custom prompt"},
        }
        merged = merge_with_defaults(recipe)
        assert merged["slot_params"]["duration_sec"] == 30
        assert merged["audio_track"]["voice_id"] == "morty"
        assert merged["visual_track"]["prompt"] == "Custom prompt"

    def test_fills_missing_with_defaults(self):
        """Test that missing values are filled with defaults."""
        recipe = {
            "slot_params": {"slot_id": 1, "duration_sec": 45},
            "audio_track": {"script": "Hello"},
            "visual_track": {"prompt": "Test"},
        }
        merged = merge_with_defaults(recipe)
        # Check defaults were applied
        assert merged["slot_params"]["fps"] == 24
        assert merged["audio_track"]["voice_id"] == "rick_c137"
        assert merged["audio_track"]["speed"] == 1.0
        assert merged["visual_track"]["expression_preset"] == "neutral"

    def test_adds_missing_sections(self):
        """Test that missing sections are added."""
        recipe = {
            "slot_params": {"slot_id": 1, "duration_sec": 45},
            "audio_track": {"script": "Hello"},
            "visual_track": {"prompt": "Test"},
            # semantic_constraints is missing
        }
        merged = merge_with_defaults(recipe)
        assert "semantic_constraints" in merged
        assert merged["semantic_constraints"]["clip_threshold"] == 0.70
