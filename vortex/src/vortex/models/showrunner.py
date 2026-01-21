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
import re
from dataclasses import dataclass
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

# Valid tone options for script generation
ToneType = Literal["absurd", "deadpan", "manic"]


# Fallback templates for when Ollama is unavailable
# Each template follows Interdimensional Cable's absurdist style
FALLBACK_TEMPLATES: list[dict[str, str]] = [
    # Fake commercials - products that shouldn't exist
    {
        "setup": "Are you tired of your regular teeth?",
        "punchline": "Try Teeth-B-Gone! Now your mouth is just a smooth hole!",
        "visual_prompt": (
            "a cartoon man with no teeth smiling at the camera, "
            "infomercial style, bright colors, surreal, 1990s commercial aesthetic"
        ),
    },
    {
        "setup": "Introducing the new Plumbus 2.0.",
        "punchline": "It does the same thing, but now it's blue!",
        "visual_prompt": (
            "a strange alien device called plumbus, colorful, "
            "alien technology, cartoon style, product showcase lighting"
        ),
    },
    {
        "setup": "Do your hands keep falling off?",
        "punchline": "Stick-It-Back Hand Glue - because duct tape is for quitters!",
        "visual_prompt": (
            "a person reattaching their cartoon hand with glue, "
            "infomercial style, bright studio lighting, surreal body horror comedy"
        ),
    },
    {
        "setup": "Tired of sleeping horizontally like some kind of floor person?",
        "punchline": "The Vertical Sleep Pod - stand up and pass out like nature intended!",
        "visual_prompt": (
            "a person sleeping standing up in a futuristic pod, "
            "neon colors, sci-fi commercial aesthetic, absurd product design"
        ),
    },
    # Talk shows with weird hosts
    {
        "setup": (
            "Welcome back to Cooking with Regret, "
            "I'm your host, a sentient cloud of disappointment."
        ),
        "punchline": (
            "Today we're making my father's approval - "
            "just kidding, that's impossible!"
        ),
        "visual_prompt": (
            "a sad cloud creature hosting a cooking show, "
            "pastel kitchen set, surreal cartoon style, studio lighting"
        ),
    },
    {
        "setup": (
            "This is Personal Space, "
            "the show that explores the boundaries of personal space."
        ),
        "punchline": (
            "Step one: stay out of my personal space. "
            "Step two: stay out of my personal space."
        ),
        "visual_prompt": (
            "a nervous man in a spotlight talking to camera, "
            "empty black studio, uncomfortable close-up, talk show format"
        ),
    },
    # News broadcasts with absurd topics
    {
        "setup": (
            "Breaking news: local man discovers "
            "his reflection has been living a better life."
        ),
        "punchline": (
            "The reflection reportedly has a nicer apartment "
            "and remembers birthdays!"
        ),
        "visual_prompt": (
            "a news anchor at a desk with a mirror showing a happier version, "
            "news broadcast style, dramatic lighting, surreal"
        ),
    },
    {
        "setup": (
            "In other news, scientists confirm "
            "the moon is just a really committed frisbee."
        ),
        "punchline": (
            "The original thrower is expected to catch it "
            "in approximately four billion years!"
        ),
        "visual_prompt": (
            "a news graphic showing the moon as a frisbee, "
            "space background, news broadcast style, infographic overlay"
        ),
    },
    # Public service announcements gone wrong
    {
        "setup": "This is a public service announcement: your furniture has feelings too.",
        "punchline": "That chair you never sit in? It knows. It knows and it's sad.",
        "visual_prompt": (
            "a sad empty chair with cartoon eyes, living room setting, "
            "PSA style graphics, melancholy mood, surreal"
        ),
    },
    {
        "setup": (
            "Remember kids, always look both ways "
            "before crossing into a parallel dimension."
        ),
        "punchline": "You might see yourself, and honestly, that guy is a jerk!",
        "visual_prompt": (
            "a safety PSA showing a child at a dimensional portal crosswalk, "
            "cartoon style, warning colors, educational aesthetic"
        ),
    },
    # Infomercials for impossible products
    {
        "setup": "Have you ever wanted to taste colors?",
        "punchline": (
            "Introducing Synesthesia Snacks - "
            "now purple tastes exactly how you'd expect!"
        ),
        "visual_prompt": (
            "colorful snacks floating in psychedelic space, "
            "infomercial product shot, vibrant colors, trippy visuals"
        ),
    },
    {
        "setup": "Is your gravity getting old and boring?",
        "punchline": "Try New Gravity - same direction, but with a fresh pine scent!",
        "visual_prompt": (
            "a spray bottle labeled New Gravity with things floating around it, "
            "infomercial style, clean white background, absurd product"
        ),
    },
    {
        "setup": "Can't stop thinking about that embarrassing thing from eight years ago?",
        "punchline": "Memory Hole - just pour it in your ear and forget responsibly!",
        "visual_prompt": (
            "a person pouring liquid into their ear from a bottle, "
            "retro infomercial style, bright colors, unsettling smile"
        ),
    },
    {
        "setup": "Introducing the Procrastinator's Clock - it's always tomorrow!",
        "punchline": (
            "Why do today what you can do never? "
            "That's the Procrastinator's promise!"
        ),
        "visual_prompt": (
            "a clock showing 'tomorrow' instead of time, "
            "product photography, studio lighting, absurd gadget commercial"
        ),
    },
    {
        "setup": "From the makers of Nothing comes Something, but not much.",
        "punchline": "Something - because you deserve barely more than nothing!",
        "visual_prompt": (
            "an empty box with the word Something on it, "
            "minimalist product shot, ironic advertising, deadpan commercial style"
        ),
    },
]


class ShowrunnerError(Exception):
    """Base exception for Showrunner errors.

    Raised when Ollama is unavailable, returns invalid responses,
    or any other script generation failure occurs.
    """

    pass


@dataclass
class Script:
    """Generated script for a video segment.

    Attributes:
        setup: The premise of the script (1 sentence)
        punchline: The absurd conclusion (1 sentence)
        visual_prompt: Scene description for image generation
    """

    setup: str
    punchline: str
    visual_prompt: str


# Prompt template for generating Interdimensional Cable scripts
SCRIPT_PROMPT_TEMPLATE = """You are a writer for "Interdimensional Cable", \
an absurdist comedy show from Rick and Morty.

Write a SHORT surreal commercial/scene about: {theme}
Tone: {tone}

Format your response ONLY as JSON (no markdown, no explanation):
{{
  "setup": "The premise in one sentence",
  "punchline": "The absurd conclusion in one sentence",
  "visual_prompt": "Scene description for image generation, include style hints"
}}

Rules:
- Keep it weird and unexpected
- No more than 15 seconds when spoken aloud
- Visual prompt should describe ONE clear scene, not a sequence
- Include visual style words like "cartoon", "surreal", "neon colors"\
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

        # Build the prompt
        prompt = SCRIPT_PROMPT_TEMPLATE.format(theme=theme, tone=tone)

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
                            "temperature": 0.9,  # Higher creativity for absurdist content
                            "top_p": 0.95,
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
            Parsed Script object

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
        required_fields = ["setup", "punchline", "visual_prompt"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ShowrunnerError(
                f"Missing required fields in response: {missing_fields}"
            )

        # Validate field types and content
        for field in required_fields:
            value = data[field]
            if not isinstance(value, str):
                raise ShowrunnerError(
                    f"Field '{field}' must be a string, got {type(value).__name__}"
                )
            if not value.strip():
                raise ShowrunnerError(f"Field '{field}' cannot be empty")

        return Script(
            setup=data["setup"].strip(),
            punchline=data["punchline"].strip(),
            visual_prompt=data["visual_prompt"].strip(),
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

    def get_fallback_script(
        self,
        theme: str,
        tone: ToneType = "absurd",
        seed: int | None = None,
    ) -> Script:
        """Get a random pre-written script when Ollama is unavailable.

        This method provides a fallback mechanism when the Ollama LLM service
        is down or unavailable. It returns a pre-written script from the
        FALLBACK_TEMPLATES collection, ensuring the pipeline can continue
        even without LLM access.

        Args:
            theme: Used to seed selection for determinism when no seed provided.
                   The actual theme content is not used for filtering.
            tone: Unused, kept for interface consistency with generate_script().
            seed: Optional seed for deterministic selection. If None, the hash
                  of the theme string is used as the seed.

        Returns:
            Script from the fallback templates with setup, punchline,
            and visual_prompt fields populated.

        Example:
            >>> showrunner = Showrunner()
            >>> # Get deterministic script based on theme
            >>> script = showrunner.get_fallback_script("bizarre infomercial")
            >>> print(script.setup)

            >>> # Get deterministic script with explicit seed
            >>> script = showrunner.get_fallback_script("any theme", seed=42)
        """
        import random

        # Use theme hash + seed for deterministic selection
        rng = random.Random()
        if seed is not None:
            rng.seed(seed)
        else:
            rng.seed(hash(theme))

        template = rng.choice(FALLBACK_TEMPLATES)

        logger.info(
            "Using fallback script template",
            extra={
                "theme": theme,
                "tone": tone,
                "seed": seed,
                "setup_preview": template["setup"][:50],
            },
        )

        return Script(
            setup=template["setup"],
            punchline=template["punchline"],
            visual_prompt=template["visual_prompt"],
        )
