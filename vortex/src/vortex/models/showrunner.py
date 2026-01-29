"""Showrunner - LLM-based script generator for Interdimensional Cable content.

This module provides the Showrunner class that uses Ollama to generate
surreal, absurdist comedy scripts in the style of Rick and Morty's
Interdimensional Cable segments.

The Showrunner is part of the Narrative Chain pipeline (Phase 2.1) and:
- Generates scripts BEFORE any video generation
- Runs via HTTP API (no GPU usage in Vortex process)
- Uses async operations to not block the pipeline
- Fails fast if Ollama isn't running (caller handles fallback)

Example:
    >>> showrunner = Showrunner()
    >>> if showrunner.is_available():
    ...     script = await showrunner.generate_script(
    ...         theme="bizarre infomercial",
    ...         tone="absurd"
    ...     )
    ...     print(script.setup)
    ...     print(script.punchline)
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import yaml

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z']+")


def _is_text_coherent(text: str) -> bool:
    """Lightweight sanity check to filter obvious TTS-gibberish inputs."""
    if not text or not text.strip():
        return False
    if len(text) < 8:
        return False
    words = _WORD_RE.findall(text)
    if len(words) < 4:
        return False
    long_words = [w for w in words if len(w) > 18]
    if len(long_words) / len(words) > 0.08:
        return False
    non_alpha = sum(
        1
        for ch in text
        if not (ch.isalpha() or ch.isspace() or ch in ".,!?'-[]")
    )
    if non_alpha / len(text) > 0.08:
        return False
    if re.search(r"(.)\1{3,}", text):
        return False
    return True

# Valid tone options for script generation
ToneType = Literal["absurd", "deadpan", "manic"]

# Bark TTS paralinguistic token instructions for script generation
BARK_TOKEN_INSTRUCTIONS = """
For dialogue, you may ONLY use these exact audio tokens:
- [laughs] - character laughs (NOT [laughter])
- [sighs] - character sighs
- [gasps] - character gasps in surprise
- ... (ellipsis) - hesitation or trailing off
- CAPITALIZED WORDS - shouting or emphasis

DO NOT use these tokens (they cause audio gibberish):
- [excited], [angry], [sad], [happy], [fast], [slow]
- [clears throat] (unreliable)
- *stage directions* like *looks around*
- Any bracketed text not in the valid list above

Example: "[gasps] Oh my GOD... [laughs] That's ridiculous!"
"""

# Adult Swim / Interdimensional Cable visual style for T2V prompts
ADULT_SWIM_STYLE = (
    "2D cel-shaded cartoon, flat colors, rough expressive linework, "
    "adult swim aesthetic, muted palette, messy ink lines, asterisk pupils, "
    "exaggerated proportions, squash and stretch animation"
)


# Lazy-loaded fallback templates
_FALLBACK_TEMPLATES: list | None = None


def _load_fallback_templates() -> list:
    """Load fallback script templates from YAML config file."""
    config_path = Path(__file__).parent / "configs" / "fallback_scripts.yaml"

    if not config_path.exists():
        logger.warning(f"Fallback scripts config not found at {config_path}")
        return []

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse fallback scripts YAML: {e}")
        return []

    templates: list = []

    for t in data.get("templates", []):
        try:
            templates.append(Script(
                setup=t["setup"],
                punchline=t["punchline"],
                subject_visual=t.get("subject_visual", "the main character"),
                storyboard=t.get("storyboard", []),
                video_prompts=t.get("video_prompts", []),
            ))
        except KeyError as e:
            logger.warning(f"Skipping malformed fallback template, missing key: {e}")
            continue

    return templates


def _get_fallback_templates() -> list:
    """Get fallback templates, loading from YAML if needed."""
    global _FALLBACK_TEMPLATES
    if _FALLBACK_TEMPLATES is None:
        _FALLBACK_TEMPLATES = _load_fallback_templates()
    return _FALLBACK_TEMPLATES


class ShowrunnerError(Exception):
    """Base exception for Showrunner errors.

    Raised when Ollama is unavailable, returns invalid responses,
    or any other script generation failure occurs.
    """

    pass


@dataclass
class Script:
    """Generated comedy script with 3-scene storyboard and T2V prompts.

    Attributes:
        setup: Opening premise (1 sentence, may include Bark tokens)
        punchline: Absurd conclusion (1 sentence, may include Bark tokens)
        subject_visual: Consistent visual description of the main subject
        storyboard: List of 3 brief scene descriptions (for logging)
        video_prompts: List of 3 dense T2V prompts (50-100 words each)
    """

    setup: str
    punchline: str
    subject_visual: str
    storyboard: list[str]
    video_prompts: list[str]  # Dense prompts for CogVideoX T2V

    @property
    def visual_prompt(self) -> str:
        """Return first video_prompt for backward compatibility."""
        return self.video_prompts[0] if self.video_prompts else ""


# Prompt template for generating Interdimensional Cable scripts with 3-scene storyboard
SCRIPT_PROMPT_TEMPLATE = """You are a writer for "Interdimensional Cable" - absurdist TV from infinite dimensions.
Write a SHORT surreal TV clip about: {theme}
Tone: {tone}

This could be ANY type of interdimensional TV content:
- Bizarre commercials for impossible products
- Surreal talk shows with weird hosts
- Breaking news from absurd realities
- Movie trailers for films that shouldn't exist
- Game shows with incomprehensible rules
- Public service announcements gone wrong
- Random channel-surfing moments of pure chaos

The clip has 3 QUICK CUTS (5 seconds each):
- Scene 1: Hook - Grab attention with something weird
- Scene 2: Escalation - Things get weirder
- Scene 3: Payoff - Maximum absurdity or abrupt cut

IMPORTANT: Define a MAIN SUBJECT that appears in ALL scenes.
{bark_tokens}
VOCABULARY RULE: Use ONLY simple, existing English words for product names and descriptions.
NO made-up sci-fi names, alien languages, or neologisms - the TTS system cannot pronounce them.
BAD: "Flug", "Zorblax", "Quantumify", "Glorpnax"
GOOD: "blob", "sphere", "crystal", "machine", "goo", "cube"

TRANSFORMATION RULE: The MAIN SUBJECT must maintain its physical form/shape across ALL scenes.
NO transforming, morphing, shapeshifting, or becoming other objects.
The subject CAN: move, rotate, bounce, grow/shrink slightly, change color, interact with objects.
The subject CANNOT: become a different object, melt into something else, split into pieces.
BAD: "The blob transforms into pancakes" or "The sphere becomes a cube"
GOOD: "The blob bounces higher" or "The sphere spins faster"

GEOMETRY RULE: The subject's SHAPE must be explicitly stated and IDENTICAL in all video_prompts.
Pick ONE geometric descriptor (sphere, cube, cylinder, blob, etc.) and repeat it exactly.
The shape must not change, warp, or become ambiguous between scenes.
BAD: Scene 1 "a cube", Scene 2 "a rounded shape", Scene 3 "the object"
GOOD: Scene 1 "a blue cube", Scene 2 "the blue cube", Scene 3 "the blue cube"

EMOTION INTENSITY RULE: Avoid EXTREME emotional states that cause visual distortion.
The subject should express emotions through ACTION, not dramatic facial/body transformation.
BAD: "turns bright red with rage", "face contorts in fury", "body shakes violently"
GOOD: "waves arms excitedly", "bounces with joy", "slumps disappointedly"

Format your response ONLY as JSON (no markdown, no explanation):
{{
  "setup": "Opening line/premise with optional [laughs], [sighs], [gasps] tokens",
  "punchline": "Closing line or absurd conclusion with optional tokens",
  "subject_visual": "Detailed visual description of the MAIN SUBJECT including: specific COLOR (e.g., bright blue), exact SHAPE (e.g., rectangular with rounded corners), and TYPE (e.g., anthropomorphic toaster). Under 30 words.",
  "storyboard": [
    "Scene 1: Brief description",
    "Scene 2: Brief description",
    "Scene 3: Brief description"
  ],
  "video_prompts": [
    "DENSE 50-100 word prompt for scene 1. Include: the subject, action, camera movement, lighting, environment. End with: {style}",
    "DENSE 50-100 word prompt for scene 2. Include: the subject, action, camera movement, lighting, environment. End with: {style}",
    "DENSE 50-100 word prompt for scene 3. Include: the subject, action, camera movement, lighting, environment. End with: {style}"
  ]
}}

Rules for video_prompts:
- Each prompt must be 50-100 words
- Start with the subject description from subject_visual
- Describe specific actions and movements
- Include camera direction (zoom in, pan, static shot, shaky cam)
- Include lighting and atmosphere
- ALWAYS end each prompt with: "{style}"
- Focus on VISUALS only - no dialogue, no sound descriptions
- DO NOT describe text, signs, billboards, logos, or readable elements
  (video models cannot render text)
"""


class Showrunner:
    """LLM-based script generator for Interdimensional Cable content.

    Uses Ollama to generate surreal, absurdist comedy scripts. The generator
    connects to a locally running Ollama instance via HTTP API.

    Attributes:
        base_url: Ollama API base URL
        model: Model name to use
        timeout: Request timeout in seconds

    Example:
        >>> showrunner = Showrunner(
        ...     base_url="http://localhost:11434",
        ...     model="llama3:8b",
        ...     timeout=30.0
        ... )
        >>> if showrunner.is_available():
        ...     script = await showrunner.generate_script("bizarre infomercial")
        ...     print(f"Setup: {script.setup}")
        ...     print(f"Punchline: {script.punchline}")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3:8b",
        timeout: float = 30.0,
    ):
        """Initialize the showrunner.

        Args:
            base_url: Ollama API base URL
            model: Model name to use (must be available in Ollama)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        logger.info(
            "Showrunner initialized",
            extra={
                "base_url": self.base_url,
                "model": self.model,
                "timeout": self.timeout,
            },
        )

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available.

        Makes a quick health check to the Ollama API and verifies
        the configured model is available.

        Returns:
            True if Ollama is running and model is available, False otherwise
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                # Check if Ollama is running
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    logger.warning(
                        f"Ollama health check failed: status {response.status_code}"
                    )
                    return False

                # Check if the configured model is available
                data = response.json()
                models = data.get("models", [])
                model_names = [m.get("name", "") for m in models]

                # Check for exact match or prefix match (e.g., "llama3:8b" matches "llama3:8b")
                model_available = any(
                    name == self.model or name.startswith(f"{self.model}")
                    for name in model_names
                )

                if not model_available:
                    logger.warning(
                        f"Model '{self.model}' not found in Ollama",
                        extra={"available_models": model_names},
                    )
                    return False

                logger.debug(
                    "Ollama health check passed",
                    extra={"model": self.model, "available_models": model_names},
                )
                return True

        except httpx.RequestError as e:
            logger.warning(f"Ollama connection failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Ollama health check error: {e}")
            return False

    async def generate_script(
        self,
        theme: str,
        tone: ToneType = "absurd",
    ) -> Script:
        """Generate a script for the given theme.

        Sends a request to Ollama to generate a surreal comedy script
        based on the provided theme and tone.

        Args:
            theme: Topic or concept for the script (e.g., "bizarre infomercial")
            tone: Comedic tone ("absurd", "deadpan", "manic")

        Returns:
            Script with setup, punchline, and visual_prompt

        Raises:
            ShowrunnerError: If Ollama is unavailable or returns invalid response
        """
        # Validate inputs
        if not theme or not theme.strip():
            raise ShowrunnerError("Theme cannot be empty")

        # Build the prompt with Bark TTS token instructions and style
        prompt = SCRIPT_PROMPT_TEMPLATE.format(
            theme=theme,
            tone=tone,
            bark_tokens=BARK_TOKEN_INSTRUCTIONS,
            style=ADULT_SWIM_STYLE,
        )

        logger.debug(
            "Generating script",
            extra={"theme": theme, "tone": tone, "model": self.model},
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,  # Balanced: creative but coherent (was 0.9)
                            "top_p": 0.9,        # Slightly tighter sampling (was 0.95)
                        },
                    },
                )

                if response.status_code != 200:
                    raise ShowrunnerError(
                        f"Ollama API error: status {response.status_code}, "
                        f"body: {response.text}"
                    )

                data = response.json()
                raw_response = data.get("response", "")

                logger.debug(
                    "Received Ollama response",
                    extra={"response_length": len(raw_response)},
                )

        except httpx.RequestError as e:
            raise ShowrunnerError(f"Failed to connect to Ollama: {e}") from e
        except httpx.TimeoutException as e:
            raise ShowrunnerError(
                f"Ollama request timed out after {self.timeout}s"
            ) from e

        # Parse the JSON response
        script = self._parse_script_response(raw_response)

        logger.info(
            "Script generated successfully",
            extra={
                "theme": theme,
                "tone": tone,
                "setup_length": len(script.setup),
                "punchline_length": len(script.punchline),
            },
        )

        return script

    def _parse_script_response(self, raw_response: str) -> Script:
        """Parse the raw LLM response into a Script object.

        Attempts to extract JSON from the response, handling common
        formatting issues like markdown code blocks or extra text.

        Args:
            raw_response: Raw text response from Ollama

        Returns:
            Parsed Script object with 3-scene storyboard and video_prompts

        Raises:
            ShowrunnerError: If response cannot be parsed as valid JSON
        """
        # Try to extract JSON from the response
        json_str = self._extract_json(raw_response)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse script JSON",
                extra={"raw_response": raw_response[:500], "error": str(e)},
            )
            raise ShowrunnerError(
                f"Invalid JSON response from Ollama: {e}"
            ) from e

        # Validate required fields
        required_string_fields = ["setup", "punchline"]
        missing_fields = [f for f in required_string_fields if f not in data]

        # Check for storyboard (new) or visual_prompt (legacy)
        has_storyboard = "storyboard" in data
        has_visual_prompt = "visual_prompt" in data
        has_video_prompts = "video_prompts" in data

        if not has_storyboard and not has_visual_prompt:
            missing_fields.append("storyboard")

        if missing_fields:
            raise ShowrunnerError(
                f"Missing required fields in response: {missing_fields}"
            )

        # Validate string field types and content
        for field in required_string_fields:
            value = data[field]
            if not isinstance(value, str):
                raise ShowrunnerError(
                    f"Field '{field}' must be a string, got {type(value).__name__}"
                )
            if not value.strip():
                raise ShowrunnerError(f"Field '{field}' cannot be empty")
            if not _is_text_coherent(value):
                raise ShowrunnerError(
                    f"Field '{field}' failed coherence check; falling back to templates"
                )

        # Parse storyboard (or convert from legacy visual_prompt)
        if has_storyboard:
            storyboard = data["storyboard"]
            if not isinstance(storyboard, list):
                raise ShowrunnerError(
                    f"Field 'storyboard' must be a list, got {type(storyboard).__name__}"
                )
        else:
            # Legacy format: convert visual_prompt to single-scene storyboard
            visual_prompt = data["visual_prompt"].strip()
            storyboard = [visual_prompt, visual_prompt, visual_prompt]
            logger.warning("Using legacy visual_prompt format, duplicating to 3 scenes")

        # Validate storyboard has 3 scenes
        if len(storyboard) != 3:
            logger.warning(f"Expected 3 scenes, got {len(storyboard)}. Padding/trimming.")
            if len(storyboard) < 3:
                while len(storyboard) < 3:
                    storyboard.append(f"Scene {len(storyboard)+1}: continuation of the action")
            else:
                storyboard = storyboard[:3]

        # Validate each scene is a non-empty string
        cleaned_storyboard = []
        for i, scene in enumerate(storyboard):
            if not isinstance(scene, str):
                raise ShowrunnerError(
                    f"Scene {i+1} must be a string, got {type(scene).__name__}"
                )
            cleaned_scene = scene.strip()
            if not cleaned_scene:
                cleaned_scene = f"Scene {i+1}: continuation of the action"
            cleaned_storyboard.append(cleaned_scene)

        # Parse subject_visual (new field for subject anchoring)
        subject_visual = data.get("subject_visual", "")
        if not subject_visual or not subject_visual.strip():
            subject_visual = "the main character"
            logger.warning("Missing subject_visual, using generic fallback")
        else:
            subject_visual = subject_visual.strip()

        # Parse video_prompts (new field for T2V)
        if has_video_prompts:
            video_prompts = data["video_prompts"]
            if not isinstance(video_prompts, list):
                raise ShowrunnerError(
                    f"Field 'video_prompts' must be a list, got {type(video_prompts).__name__}"
                )
        else:
            # Generate video_prompts from storyboard + subject + style
            logger.warning("Missing video_prompts, generating from storyboard")
            video_prompts = []
            for scene in cleaned_storyboard:
                prompt = f"{subject_visual}, {scene}. {ADULT_SWIM_STYLE}"
                video_prompts.append(prompt)

        # Validate video_prompts has 3 entries
        if len(video_prompts) != 3:
            logger.warning(f"Expected 3 video_prompts, got {len(video_prompts)}. Padding/trimming.")
            if len(video_prompts) < 3:
                while len(video_prompts) < 3:
                    prompt = f"{subject_visual}, Scene {len(video_prompts)+1}. {ADULT_SWIM_STYLE}"
                    video_prompts.append(prompt)
            else:
                video_prompts = video_prompts[:3]

        # Validate each video_prompt is a non-empty string
        cleaned_video_prompts = []
        for i, prompt in enumerate(video_prompts):
            if not isinstance(prompt, str):
                raise ShowrunnerError(
                    f"video_prompt {i+1} must be a string, got {type(prompt).__name__}"
                )
            cleaned_prompt = prompt.strip()
            if not cleaned_prompt:
                cleaned_prompt = f"{subject_visual}, Scene {i+1}. {ADULT_SWIM_STYLE}"
            cleaned_video_prompts.append(cleaned_prompt)

        return Script(
            setup=data["setup"].strip(),
            punchline=data["punchline"].strip(),
            subject_visual=subject_visual,
            storyboard=cleaned_storyboard,
            video_prompts=cleaned_video_prompts,
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text that may contain extra content.

        Handles common LLM response patterns:
        - Pure JSON
        - JSON wrapped in markdown code blocks
        - JSON with leading/trailing text

        Args:
            text: Raw text that should contain JSON

        Returns:
            Extracted JSON string

        Raises:
            ShowrunnerError: If no valid JSON object found
        """
        text = text.strip()

        # Try direct parse first
        if text.startswith("{") and text.endswith("}"):
            return text

        # Try to extract from markdown code block
        code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        match = re.search(code_block_pattern, text)
        if match:
            return match.group(1).strip()

        # Try to find JSON object in the text
        json_pattern = r"\{[^{}]*\}"
        match = re.search(json_pattern, text)
        if match:
            return match.group(0)

        # Try to find multi-line JSON (with nested braces)
        # This handles cases where visual_prompt contains no nested braces
        start_idx = text.find("{")
        if start_idx != -1:
            end_idx = text.rfind("}")
            if end_idx > start_idx:
                return text[start_idx : end_idx + 1]

        raise ShowrunnerError(
            f"Could not extract JSON from response: {text[:200]}..."
        )

    async def unload_model(self) -> bool:
        """Unload the model from Ollama to free VRAM.

        Sends a request to Ollama with keep_alive=0 to immediately unload
        the model from GPU memory. This is useful after script generation
        to free VRAM for other models in the pipeline.

        Returns:
            True if unload succeeded or model wasn't loaded, False on error

        Example:
            >>> showrunner = Showrunner()
            >>> script = await showrunner.generate_script("theme")
            >>> await showrunner.unload_model()  # Free VRAM
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Send a minimal request with keep_alive=0 to unload
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": "",  # Empty prompt
                        "keep_alive": 0,  # Unload immediately
                    },
                )

                if response.status_code == 200:
                    logger.info(
                        "Ollama model unloaded successfully",
                        extra={"model": self.model},
                    )
                    return True
                else:
                    logger.warning(
                        f"Ollama unload returned status {response.status_code}",
                        extra={"model": self.model},
                    )
                    return False

        except httpx.RequestError as e:
            logger.warning(f"Failed to unload Ollama model: {e}")
            return False
        except Exception as e:
            logger.warning(f"Ollama unload error: {e}")
            return False

    def get_fallback_script(
        self,
        theme: str,
        tone: ToneType = "absurd",
        seed: int | None = None,
    ) -> Script:
        """Get a random pre-written script when Ollama is unavailable.

        Args:
            theme: Used to seed selection for determinism when no seed provided.
            tone: Unused, kept for interface consistency.
            seed: Optional seed for deterministic selection.

        Returns:
            Script from the fallback templates.
        """
        templates = _get_fallback_templates()

        if not templates:
            # Emergency fallback if YAML fails to load
            return Script(
                setup="Welcome to Interdimensional Cable!",
                punchline="Where anything can happen!",
                subject_visual="a cartoon TV host",
                storyboard=[
                    "Scene 1: Host waves",
                    "Scene 2: Host talks",
                    "Scene 3: Host exits",
                ],
                video_prompts=[
                    f"A cartoon TV host waves at the camera. {ADULT_SWIM_STYLE}",
                    f"A cartoon TV host talks with exaggerated gestures. {ADULT_SWIM_STYLE}",
                    f"A cartoon TV host exits stage left. {ADULT_SWIM_STYLE}",
                ],
            )

        # Use theme hash + seed for deterministic selection
        rng = random.Random()
        if seed is not None:
            rng.seed(seed)
        else:
            rng.seed(hash(theme))

        script = rng.choice(templates)

        logger.info(
            "Using fallback script template",
            extra={
                "theme": theme,
                "tone": tone,
                "seed": seed,
                "setup_preview": script.setup[:50],
                "num_scenes": len(script.storyboard),
            },
        )

        return script
