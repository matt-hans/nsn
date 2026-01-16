"""Standardized recipe schema for Lane 0 video renderers.

All renderers must accept recipes conforming to this schema. The schema
defines the standard fields that the network uses for task routing and
BFT verification. Renderers may ignore unknown fields but must process
all required fields.
"""

from __future__ import annotations

from typing import Any

# JSON Schema for Lane 0 recipe validation
RECIPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["slot_params", "audio_track", "visual_track"],
    "properties": {
        "slot_params": {
            "type": "object",
            "required": ["slot_id", "duration_sec"],
            "properties": {
                "slot_id": {"type": "integer", "description": "Unique slot identifier"},
                "duration_sec": {
                    "type": "integer",
                    "default": 45,
                    "description": "Target duration in seconds",
                },
                "fps": {
                    "type": "integer",
                    "default": 24,
                    "description": "Frames per second",
                },
                "seed": {
                    "type": "integer",
                    "description": "Deterministic seed (generated if not provided)",
                },
            },
        },
        "audio_track": {
            "type": "object",
            "required": ["script"],
            "properties": {
                "script": {"type": "string", "description": "Text to synthesize as speech"},
                "voice_id": {
                    "type": "string",
                    "default": "rick_c137",
                    "description": "Voice identifier",
                },
                "speed": {
                    "type": "number",
                    "default": 1.0,
                    "minimum": 0.5,
                    "maximum": 2.0,
                    "description": "Speech speed multiplier",
                },
                "emotion": {
                    "type": "string",
                    "default": "neutral",
                    "enum": ["neutral", "excited", "sad", "angry", "manic"],
                    "description": "Emotion modulation",
                },
            },
        },
        "visual_track": {
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {"type": "string", "description": "Image generation prompt"},
                "negative_prompt": {
                    "type": "string",
                    "default": "",
                    "description": "Negative prompt for image generation",
                },
                "expression_preset": {
                    "type": "string",
                    "default": "neutral",
                    "description": "Base expression preset",
                },
                "expression_sequence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sequence of expression transitions",
                },
                "driving_source": {
                    "type": "string",
                    "description": "Optional motion template path or CID",
                },
            },
        },
        "semantic_constraints": {
            "type": "object",
            "properties": {
                "clip_threshold": {
                    "type": "number",
                    "default": 0.70,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "CLIP similarity threshold for verification",
                },
            },
        },
    },
}


def validate_recipe(recipe: dict[str, Any]) -> list[str]:
    """Validate recipe against schema, returning list of errors.

    Args:
        recipe: Recipe dict to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    # Check required top-level fields
    for field in ("slot_params", "audio_track", "visual_track"):
        if field not in recipe:
            errors.append(f"Missing required field: {field}")
            continue
        if not isinstance(recipe[field], dict):
            errors.append(f"Field '{field}' must be an object")

    if errors:
        return errors

    # Validate slot_params
    slot_params = recipe["slot_params"]
    if "slot_id" not in slot_params:
        errors.append("Missing required field: slot_params.slot_id")
    elif not isinstance(slot_params.get("slot_id"), int):
        errors.append("slot_params.slot_id must be an integer")

    if "duration_sec" not in slot_params:
        errors.append("Missing required field: slot_params.duration_sec")
    elif not isinstance(slot_params.get("duration_sec"), int):
        errors.append("slot_params.duration_sec must be an integer")

    # Validate audio_track
    audio_track = recipe["audio_track"]
    if "script" not in audio_track:
        errors.append("Missing required field: audio_track.script")
    elif not isinstance(audio_track.get("script"), str):
        errors.append("audio_track.script must be a string")

    # Validate visual_track
    visual_track = recipe["visual_track"]
    if "prompt" not in visual_track:
        errors.append("Missing required field: visual_track.prompt")
    elif not isinstance(visual_track.get("prompt"), str):
        errors.append("visual_track.prompt must be a string")

    return errors


def get_recipe_defaults() -> dict[str, Any]:
    """Return a recipe dict with all defaults filled in.

    Returns:
        Recipe dict with default values for all optional fields
    """
    return {
        "slot_params": {
            "slot_id": 0,
            "duration_sec": 45,
            "fps": 24,
        },
        "audio_track": {
            "script": "",
            "voice_id": "rick_c137",
            "speed": 1.0,
            "emotion": "neutral",
        },
        "visual_track": {
            "prompt": "",
            "negative_prompt": "",
            "expression_preset": "neutral",
            "expression_sequence": [],
        },
        "semantic_constraints": {
            "clip_threshold": 0.70,
        },
    }


def merge_with_defaults(recipe: dict[str, Any]) -> dict[str, Any]:
    """Merge recipe with defaults for missing optional fields.

    Args:
        recipe: Partial recipe dict

    Returns:
        Complete recipe dict with defaults applied
    """
    defaults = get_recipe_defaults()
    result: dict[str, Any] = {}

    for section in defaults:
        if section not in recipe:
            result[section] = defaults[section].copy()
        else:
            result[section] = defaults[section].copy()
            result[section].update(recipe[section])

    return result
