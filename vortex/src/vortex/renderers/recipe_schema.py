"""Standardized recipe schema for ToonGen video generation.

This schema defines the API contract between clients and the ToonGen
orchestrator. It replaces the legacy Vortex schema with fields
appropriate for ComfyUI-based workflow orchestration.
"""

from __future__ import annotations

from typing import Any

# JSON Schema for ToonGen recipe validation
RECIPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["slot_params", "audio_track", "visual_track"],
    "properties": {
        "slot_params": {
            "type": "object",
            "required": ["slot_id"],
            "properties": {
                "slot_id": {
                    "type": "integer",
                    "description": "Unique slot identifier",
                },
                "fps": {
                    "type": "integer",
                    "default": 24,
                    "description": "Frames per second",
                },
                "seed": {
                    "type": "integer",
                    "description": "Deterministic seed (random if not provided)",
                },
            },
        },
        "audio_track": {
            "type": "object",
            "required": ["script"],
            "properties": {
                "script": {
                    "type": "string",
                    "description": "Text to synthesize as speech",
                },
                "engine": {
                    "type": "string",
                    "enum": ["auto", "f5_tts", "kokoro"],
                    "default": "auto",
                    "description": "TTS engine selection (auto tries F5, falls back to Kokoro)",
                },
                "voice_style": {
                    "type": "string",
                    "description": "F5-TTS voice reference filename (without .wav)",
                },
                "voice_id": {
                    "type": "string",
                    "default": "af_heart",
                    "description": "Kokoro voice ID (used if engine=kokoro or as fallback)",
                },
            },
        },
        "audio_environment": {
            "type": "object",
            "properties": {
                "bgm": {
                    "type": "string",
                    "description": "Background music filename (without .wav)",
                },
                "sfx": {
                    "type": "string",
                    "description": "Sound effect filename (without .wav)",
                },
                "mix_ratio": {
                    "type": "number",
                    "default": 0.3,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "BGM volume relative to voice",
                },
            },
        },
        "visual_track": {
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Full scene prompt for Flux image generation",
                },
                "negative_prompt": {
                    "type": "string",
                    "default": "blurry, low quality, distorted face, extra limbs",
                    "description": "Negative prompt for image generation",
                },
            },
        },
    },
}


def validate_recipe(recipe: dict[str, Any]) -> list[str]:
    """Validate recipe against ToonGen schema, returning list of errors.

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
            "fps": 24,
        },
        "audio_track": {
            "script": "",
            "engine": "auto",
            "voice_id": "af_heart",
        },
        "audio_environment": {
            "bgm": None,
            "sfx": None,
            "mix_ratio": 0.3,
        },
        "visual_track": {
            "prompt": "",
            "negative_prompt": "blurry, low quality, distorted face, extra limbs",
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
