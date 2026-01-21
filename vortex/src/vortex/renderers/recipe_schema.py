"""Standardized recipe schema for Narrative Chain video generation.

This schema defines the API contract between clients and the Vortex
orchestrator. It uses an LLM-first approach where:
1. LLM generates comedic script (setup/punchline) + visual prompt
2. TTS synthesizes the script to audio (sets duration)
3. Image generation creates the scene
4. Video generation animates based on audio

The narrative section controls script generation, with auto_script=True
delegating to the LLM or auto_script=False using a provided script.
"""

from __future__ import annotations

from typing import Any

# JSON Schema for Narrative Chain recipe validation
RECIPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["slot_params"],
    "properties": {
        "slot_params": {
            "type": "object",
            "required": ["slot_id"],
            "properties": {
                "slot_id": {
                    "type": "integer",
                    "description": "Unique slot identifier",
                },
                "seed": {
                    "type": "integer",
                    "description": "Deterministic seed (random if not provided)",
                },
                "target_duration": {
                    "type": "number",
                    "default": 12.0,
                    "description": "Target video duration in seconds (10-15)",
                },
                "fps": {
                    "type": "integer",
                    "default": 8,
                    "description": "Output frame rate (CogVideoX native: 8)",
                },
            },
        },
        "narrative": {
            "type": "object",
            "properties": {
                "theme": {
                    "type": "string",
                    "default": "bizarre infomercial",
                    "description": "Topic for LLM script generation",
                },
                "tone": {
                    "type": "string",
                    "enum": ["absurd", "deadpan", "manic"],
                    "default": "absurd",
                    "description": "Comedic tone for script",
                },
                "auto_script": {
                    "type": "boolean",
                    "default": True,
                    "description": "True = LLM generates, False = use provided script",
                },
                "script": {
                    "type": "object",
                    "description": "Manual script (used if auto_script=False)",
                    "properties": {
                        "setup": {
                            "type": "string",
                            "description": "Premise sentence",
                        },
                        "punchline": {
                            "type": "string",
                            "description": "Punchline sentence",
                        },
                        "visual_prompt": {
                            "type": "string",
                            "description": "Scene description for image generation",
                        },
                    },
                },
            },
        },
        "audio": {
            "type": "object",
            "properties": {
                "voice_id": {
                    "type": "string",
                    "default": "af_heart",
                    "description": "Kokoro voice ID",
                },
                "speed": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Speech speed multiplier",
                },
                "bgm": {
                    "type": "string",
                    "description": "Background music filename",
                },
                "bgm_volume": {
                    "type": "number",
                    "default": 0.3,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "BGM volume",
                },
            },
        },
        "video": {
            "type": "object",
            "properties": {
                "style_prompt": {
                    "type": "string",
                    "default": (
                        "cartoon style, vibrant colors, surreal, "
                        "interdimensional cable aesthetic"
                    ),
                    "description": "Style additions appended to visual prompt",
                },
                "negative_prompt": {
                    "type": "string",
                    "default": "realistic, photographic, blurry, low quality",
                    "description": "Negative prompt",
                },
                "guidance_scale": {
                    "type": "number",
                    "default": 6.0,
                    "description": "CogVideoX CFG scale",
                },
            },
        },
        "quality": {
            "type": "object",
            "properties": {
                "clip_threshold": {
                    "type": "number",
                    "default": 0.70,
                    "description": "Minimum CLIP score for acceptance",
                },
                "max_retries": {
                    "type": "integer",
                    "default": 3,
                    "description": "Retry count if CLIP fails",
                },
            },
        },
    },
}


def validate_recipe(recipe: dict[str, Any]) -> list[str]:
    """Validate recipe against Narrative Chain schema, returning list of errors.

    The schema requires:
    - slot_params with slot_id (integer)

    Optional sections (narrative, audio, video, quality) are validated
    for correct types if present. When auto_script=False, the narrative.script
    section must contain setup, punchline, and visual_prompt.

    Args:
        recipe: Recipe dict to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    # Check required top-level field
    if "slot_params" not in recipe:
        errors.append("Missing required field: slot_params")
        return errors

    if not isinstance(recipe["slot_params"], dict):
        errors.append("Field 'slot_params' must be an object")
        return errors

    # Validate slot_params.slot_id (required)
    slot_params = recipe["slot_params"]
    if "slot_id" not in slot_params:
        errors.append("Missing required field: slot_params.slot_id")
    elif not isinstance(slot_params.get("slot_id"), int):
        errors.append("slot_params.slot_id must be an integer")

    # Validate slot_params.target_duration (optional)
    if "target_duration" in slot_params:
        target_duration = slot_params["target_duration"]
        if not isinstance(target_duration, (int, float)):
            errors.append("slot_params.target_duration must be a number")

    # Validate slot_params.fps (optional)
    if "fps" in slot_params:
        if not isinstance(slot_params["fps"], int):
            errors.append("slot_params.fps must be an integer")

    # Validate slot_params.seed (optional)
    if "seed" in slot_params:
        if not isinstance(slot_params["seed"], int):
            errors.append("slot_params.seed must be an integer")

    # Validate narrative section (optional)
    if "narrative" in recipe:
        narrative = recipe["narrative"]
        if not isinstance(narrative, dict):
            errors.append("Field 'narrative' must be an object")
        else:
            # Validate tone enum
            if "tone" in narrative:
                valid_tones = ["absurd", "deadpan", "manic"]
                if narrative["tone"] not in valid_tones:
                    errors.append(
                        f"narrative.tone must be one of: {', '.join(valid_tones)}"
                    )

            # If auto_script=False, script section is required
            auto_script = narrative.get("auto_script", True)
            if not auto_script:
                if "script" not in narrative:
                    errors.append(
                        "narrative.script required when auto_script=False"
                    )
                elif not isinstance(narrative["script"], dict):
                    errors.append("narrative.script must be an object")
                else:
                    script = narrative["script"]
                    for field in ("setup", "punchline", "visual_prompt"):
                        if field not in script:
                            errors.append(
                                f"Missing required field: narrative.script.{field}"
                            )
                        elif not isinstance(script.get(field), str):
                            errors.append(
                                f"narrative.script.{field} must be a string"
                            )

    # Validate audio section (optional)
    if "audio" in recipe:
        audio = recipe["audio"]
        if not isinstance(audio, dict):
            errors.append("Field 'audio' must be an object")
        else:
            if "speed" in audio:
                if not isinstance(audio["speed"], (int, float)):
                    errors.append("audio.speed must be a number")
            if "bgm_volume" in audio:
                bgm_vol = audio["bgm_volume"]
                if not isinstance(bgm_vol, (int, float)):
                    errors.append("audio.bgm_volume must be a number")
                elif not (0.0 <= bgm_vol <= 1.0):
                    errors.append("audio.bgm_volume must be between 0.0 and 1.0")

    # Validate video section (optional)
    if "video" in recipe:
        video = recipe["video"]
        if not isinstance(video, dict):
            errors.append("Field 'video' must be an object")
        else:
            if "guidance_scale" in video:
                if not isinstance(video["guidance_scale"], (int, float)):
                    errors.append("video.guidance_scale must be a number")

    # Validate quality section (optional)
    if "quality" in recipe:
        quality = recipe["quality"]
        if not isinstance(quality, dict):
            errors.append("Field 'quality' must be an object")
        else:
            if "clip_threshold" in quality:
                threshold = quality["clip_threshold"]
                if not isinstance(threshold, (int, float)):
                    errors.append("quality.clip_threshold must be a number")
                elif not (0.0 <= threshold <= 1.0):
                    errors.append(
                        "quality.clip_threshold must be between 0.0 and 1.0"
                    )
            if "max_retries" in quality:
                if not isinstance(quality["max_retries"], int):
                    errors.append("quality.max_retries must be an integer")

    return errors


def get_recipe_defaults() -> dict[str, Any]:
    """Return a recipe dict with all defaults filled in.

    Returns a complete recipe structure with default values suitable
    for LLM-driven script generation (auto_script=True).

    Returns:
        Recipe dict with default values for all optional fields
    """
    return {
        "slot_params": {
            "slot_id": 0,
            "target_duration": 12.0,
            "fps": 8,
        },
        "narrative": {
            "theme": "bizarre infomercial",
            "tone": "absurd",
            "auto_script": True,
            "script": {
                "setup": "",
                "punchline": "",
                "visual_prompt": "",
            },
        },
        "audio": {
            "voice_id": "af_heart",
            "speed": 1.0,
            "bgm": None,
            "bgm_volume": 0.3,
        },
        "video": {
            "style_prompt": (
                "cartoon style, vibrant colors, surreal, "
                "interdimensional cable aesthetic"
            ),
            "negative_prompt": "realistic, photographic, blurry, low quality",
            "guidance_scale": 6.0,
        },
        "quality": {
            "clip_threshold": 0.70,
            "max_retries": 3,
        },
    }


def merge_with_defaults(recipe: dict[str, Any]) -> dict[str, Any]:
    """Merge recipe with defaults for missing optional fields.

    Performs a two-level merge: top-level sections are merged with
    defaults, and within each section, individual fields are merged.
    This preserves user-provided values while filling in missing fields.

    Args:
        recipe: Partial recipe dict

    Returns:
        Complete recipe dict with defaults applied
    """
    defaults = get_recipe_defaults()
    result: dict[str, Any] = {}

    for section in defaults:
        if section not in recipe:
            # Section missing entirely - use defaults
            result[section] = _deep_copy(defaults[section])
        else:
            # Merge section with defaults
            result[section] = _deep_copy(defaults[section])
            if isinstance(recipe[section], dict):
                _deep_merge(result[section], recipe[section])
            else:
                # Non-dict provided for section - use as-is (will fail validation)
                result[section] = recipe[section]

    # Preserve any extra sections not in defaults
    for section in recipe:
        if section not in defaults:
            result[section] = recipe[section]

    return result


def _deep_copy(obj: Any) -> Any:
    """Create a deep copy of nested dict/list structures.

    Args:
        obj: Object to copy

    Returns:
        Deep copy of the object
    """
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_copy(item) for item in obj]
    return obj


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Recursively merge source dict into target dict.

    Nested dicts are merged recursively. Other values from source
    overwrite values in target.

    Args:
        target: Dict to merge into (modified in place)
        source: Dict to merge from
    """
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(target[key], value)
        else:
            target[key] = _deep_copy(value)
