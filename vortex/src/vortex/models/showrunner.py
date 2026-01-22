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
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

# Valid tone options for script generation
ToneType = Literal["absurd", "deadpan", "manic"]

# Bark TTS paralinguistic token instructions for script generation
BARK_TOKEN_INSTRUCTIONS = """
For dialogue, you may use these audio tokens to add expressiveness:
- [laughter] or [laughs] - character laughs
- [sighs] - character sighs
- [gasps] - character gasps in surprise
- [clears throat] - character clears throat
- ... (ellipsis) - hesitation or trailing off
- CAPITALIZED WORDS - shouting or emphasis

Example: "[gasps] Oh my GOD... [laughs] That's the most ridiculous thing I've ever seen!"
"""


class ShowrunnerError(Exception):
    """Base exception for Showrunner errors.

    Raised when Ollama is unavailable, returns invalid responses,
    or any other script generation failure occurs.
    """

    pass


@dataclass
class Script:
    """Generated comedy script with 3-scene storyboard.

    Attributes:
        setup: Opening premise (1 sentence)
        punchline: Absurd conclusion (1 sentence)
        storyboard: List of 3 visual scene descriptions for montage
    """

    setup: str
    punchline: str
    storyboard: list[str]

    @property
    def visual_prompt(self) -> str:
        """Return first scene for backward compatibility."""
        return self.storyboard[0] if self.storyboard else ""


# Fallback templates for when Ollama is unavailable
# Each template follows Interdimensional Cable's absurdist style with 3-scene storyboards
FALLBACK_TEMPLATES: list[Script] = [
    # Fake commercials - products that shouldn't exist
    Script(
        setup="[clears throat] Are you tired of your... regular teeth?",
        punchline="Try Teeth-B-Gone! [laughs] Now your mouth is just a SMOOTH hole!",
        storyboard=[
            (
                "Scene 1: A frustrated cartoon man pointing at his normal teeth "
                "in a bathroom mirror, infomercial lighting, 1990s aesthetic"
            ),
            (
                "Scene 2: The man applies a glowing product to his teeth, "
                "sparkles and magical effects, bright colors, transformation sequence"
            ),
            (
                "Scene 3: The man smiles revealing a completely smooth toothless mouth, "
                "thumbs up to camera, surreal body horror, disturbingly happy"
            ),
        ],
    ),
    Script(
        setup="Introducing the new Plumbus 2.0.",
        punchline="It does the same thing, but now it's blue!",
        storyboard=[
            (
                "Scene 1: A pink alien plumbus device on a white pedestal, "
                "product showcase lighting, mysterious alien technology"
            ),
            (
                "Scene 2: Factory workers dipping the plumbus in blue dye, "
                "assembly line aesthetic, cartoon style, neon lighting"
            ),
            (
                "Scene 3: The blue plumbus spinning majestically, confetti falling, "
                "product reveal moment, absurd celebration, same weird shape"
            ),
        ],
    ),
    Script(
        setup="[sighs] Do your hands keep falling off?",
        punchline="Stick-It-Back Hand Glue - because duct tape is for QUITTERS!",
        storyboard=[
            (
                "Scene 1: A cartoon person looking sadly at their detached hand "
                "on the floor, suburban living room, surreal body horror comedy"
            ),
            (
                "Scene 2: Close-up of glue bottle being applied to wrist stump, "
                "infomercial demonstration style, bright studio lighting"
            ),
            (
                "Scene 3: Person waving both hands triumphantly at camera, "
                "one slightly crooked, big smile, product success moment"
            ),
        ],
    ),
    Script(
        setup="[gasps] Tired of sleeping horizontally like some kind of FLOOR person?",
        punchline="The Vertical Sleep Pod - stand up and pass out like NATURE intended!",
        storyboard=[
            (
                "Scene 1: A person lying in bed looking disgusted at themselves, "
                "black and white footage, infomercial problem setup"
            ),
            (
                "Scene 2: Futuristic vertical pod opening with steam and neon lights, "
                "sci-fi commercial aesthetic, dramatic reveal"
            ),
            (
                "Scene 3: Happy person sleeping standing up in the pod, "
                "peaceful expression, city skyline behind, absurd product design"
            ),
        ],
    ),
    # Talk shows with weird hosts
    Script(
        setup=(
            "[sighs] Welcome back to Cooking with Regret... "
            "I'm your host, a sentient cloud of disappointment."
        ),
        punchline=(
            "Today we're making my father's approval - "
            "[laughs] just kidding, that's IMPOSSIBLE!"
        ),
        storyboard=[
            (
                "Scene 1: A sad gray cloud with eyes floating behind a kitchen counter, "
                "pastel studio set, cooking show format, surreal cartoon style"
            ),
            (
                "Scene 2: The cloud attempts to mix ingredients but they phase through it, "
                "bowls and spoons floating, existential sadness, studio lighting"
            ),
            (
                "Scene 3: Empty plate presentation with a single tear drop from the cloud, "
                "dramatic close-up, cooking show finale lighting, melancholy"
            ),
        ],
    ),
    Script(
        setup=(
            "This is Personal Space, "
            "the show that explores the boundaries of personal space."
        ),
        punchline=(
            "Step one: stay out of my personal space. "
            "Step two: stay out of my personal space."
        ),
        storyboard=[
            (
                "Scene 1: A nervous bald man in a spotlight on empty black stage, "
                "uncomfortable close-up, sweat visible, talk show format"
            ),
            (
                "Scene 2: The man drawing a chalk circle around himself frantically, "
                "paranoid expression, dramatic shadows, surreal"
            ),
            (
                "Scene 3: Extreme close-up of the man's face filling entire screen, "
                "wild eyes, ironic violation of personal space, unsettling"
            ),
        ],
    ),
    # News broadcasts with absurd topics
    Script(
        setup=(
            "[clears throat] Breaking news: local man discovers "
            "his reflection has been living a BETTER life."
        ),
        punchline=(
            "[gasps] The reflection reportedly has a nicer apartment "
            "and remembers birthdays!"
        ),
        storyboard=[
            (
                "Scene 1: News anchor at desk with breaking news graphics, "
                "professional broadcast style, dramatic lighting"
            ),
            (
                "Scene 2: Split screen showing sad man vs happy reflection in mirror, "
                "reflection's side has nicer furniture, surreal comparison"
            ),
            (
                "Scene 3: The reflection waving smugly from inside the mirror, "
                "holding a birthday cake, man crying outside, news format"
            ),
        ],
    ),
    Script(
        setup=(
            "In other news, scientists confirm "
            "the moon is just a really committed frisbee."
        ),
        punchline=(
            "The original thrower is expected to catch it "
            "in approximately four billion years!"
        ),
        storyboard=[
            (
                "Scene 1: News anchor gesturing at space graphic behind them, "
                "professional news broadcast, serious expression"
            ),
            (
                "Scene 2: Animation of giant hand throwing frisbee-moon into space, "
                "cosmic background, trajectory lines, infographic style"
            ),
            (
                "Scene 3: Silhouette of giant figure waiting with mitt in space, "
                "timer showing billions of years, patient expression, surreal scale"
            ),
        ],
    ),
    # Public service announcements gone wrong
    Script(
        setup="This is a public service announcement: your furniture has feelings too.",
        punchline="That chair you never sit in? It knows. It knows and it's sad.",
        storyboard=[
            (
                "Scene 1: PSA title card with serious font and warning colors, "
                "government broadcast aesthetic, dramatic music implied"
            ),
            (
                "Scene 2: Living room with furniture that has cartoon eyes, "
                "the chair in corner looks lonely, melancholy lighting"
            ),
            (
                "Scene 3: Close-up of the sad chair with a single tear, "
                "cobwebs forming, family laughing on couch in background, surreal guilt"
            ),
        ],
    ),
    Script(
        setup=(
            "Remember kids, always look both ways "
            "before crossing into a parallel dimension."
        ),
        punchline="You might see yourself, and honestly, that guy is a jerk!",
        storyboard=[
            (
                "Scene 1: Cartoon child at a crosswalk but instead of street "
                "there is a swirling portal, educational PSA style"
            ),
            (
                "Scene 2: Child looking left and right seeing alternate versions of self, "
                "one is rude and sticking tongue out, warning colors"
            ),
            (
                "Scene 3: Child and alternate self in fistfight, "
                "safety mascot shrugging in corner, cartoon violence, PSA gone wrong"
            ),
        ],
    ),
    # Infomercials for impossible products
    Script(
        setup="Have you ever wanted to taste colors?",
        punchline=(
            "Introducing Synesthesia Snacks - "
            "now purple tastes exactly how you'd expect!"
        ),
        storyboard=[
            (
                "Scene 1: Person staring longingly at a rainbow, "
                "dramatic lighting, existential yearning, infomercial problem setup"
            ),
            (
                "Scene 2: Package of Synesthesia Snacks glowing with prismatic colors, "
                "product hero shot, psychedelic background, trippy visuals"
            ),
            (
                "Scene 3: Person eating snacks while colors visibly enter their mouth, "
                "ecstatic expression, synesthetic explosion, vibrant surreal"
            ),
        ],
    ),
    Script(
        setup="Is your gravity getting old and boring?",
        punchline="Try New Gravity - same direction, but with a fresh pine scent!",
        storyboard=[
            (
                "Scene 1: Bored family standing normally on ground looking disappointed, "
                "black and white, infomercial problem framing"
            ),
            (
                "Scene 2: Spray bottle labeled New Gravity being sprayed into air, "
                "green pine-scented particles visible, clean white background"
            ),
            (
                "Scene 3: Same family still standing normally but now smiling and sniffing air, "
                "pine trees appearing around them, absurd satisfaction"
            ),
        ],
    ),
    Script(
        setup="[sighs] Can't stop thinking about that embarrassing thing from eight years ago?",
        punchline="Memory Hole - just pour it in your ear and forget RESPONSIBLY! [laughs]",
        storyboard=[
            (
                "Scene 1: Person lying awake at 3am with thought bubble showing cringe moment, "
                "dark bedroom, anxious expression, relatable horror"
            ),
            (
                "Scene 2: Cheerful person pouring glowing liquid into their own ear, "
                "retro infomercial style, bright colors, unsettling smile"
            ),
            (
                "Scene 3: Person with empty eyes and peaceful smile, thought bubble now blank, "
                "maybe too peaceful, slightly disturbing satisfaction"
            ),
        ],
    ),
    Script(
        setup="Introducing the Procrastinator's Clock - it's always tomorrow!",
        punchline=(
            "Why do today what you can do never? "
            "That's the Procrastinator's promise!"
        ),
        storyboard=[
            (
                "Scene 1: Stressed person surrounded by tasks and regular clocks, "
                "overwhelming chaos, deadline panic, black and white"
            ),
            (
                "Scene 2: The Procrastinator's Clock revealed showing only TOMORROW, "
                "product spotlight, studio lighting, magical glow"
            ),
            (
                "Scene 3: Person relaxing on couch with piled up tasks burning behind them, "
                "zen expression, flames reflected in eyes, absurd peace"
            ),
        ],
    ),
    Script(
        setup="From the makers of Nothing comes Something, but not much.",
        punchline="Something - because you deserve barely more than nothing!",
        storyboard=[
            (
                "Scene 1: Empty void with the word NOTHING floating, "
                "minimalist black space, philosophical emptiness"
            ),
            (
                "Scene 2: A tiny speck appears labeled SOMETHING, epic reveal lighting, "
                "dramatic music implied, ironic grandeur"
            ),
            (
                "Scene 3: Person holding nearly empty box labeled Something, "
                "forced smile, deadpan commercial style, existential product"
            ),
        ],
    ),
]


# Prompt template for generating Interdimensional Cable scripts with 3-scene storyboard
SCRIPT_PROMPT_TEMPLATE = """You are a writer for "Interdimensional Cable" - absurdist commercials.
Write a SHORT surreal commercial about: {theme}
Tone: {tone}

The commercial has 3 QUICK CUTS (5 seconds each):
- Scene 1: Setup - Introduce the absurd product/situation
- Scene 2: Escalation - Things get weirder
- Scene 3: Punchline - Maximum chaos/absurdity
{bark_tokens}
Format your response ONLY as JSON (no markdown, no explanation):
{{
  "setup": "One sentence premise",
  "punchline": "One sentence absurd conclusion",
  "storyboard": [
    "Scene 1: [Visual description - what we SEE in first 5 seconds]",
    "Scene 2: [Visual description - what we SEE in seconds 5-10]",
    "Scene 3: [Visual description - what we SEE in seconds 10-15]"
  ]
}}

Rules:
- Keep each scene description under 50 words
- Focus on VISUALS, not dialogue
- Include style words like "cartoon", "surreal", "neon colors"
- Use Bark audio tokens in setup/punchline for expressive speech\
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

        # Build the prompt with Bark TTS token instructions
        prompt = SCRIPT_PROMPT_TEMPLATE.format(
            theme=theme, tone=tone, bark_tokens=BARK_TOKEN_INSTRUCTIONS
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
            Parsed Script object with 3-scene storyboard

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

        # Validate required fields (storyboard replaces visual_prompt)
        required_string_fields = ["setup", "punchline"]
        missing_fields = [f for f in required_string_fields if f not in data]

        # Check for storyboard (new) or visual_prompt (legacy)
        has_storyboard = "storyboard" in data
        has_visual_prompt = "visual_prompt" in data

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
                # Pad with generic scene
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

        return Script(
            setup=data["setup"].strip(),
            punchline=data["punchline"].strip(),
            storyboard=cleaned_storyboard,
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

        This method provides a fallback mechanism when the Ollama LLM service
        is down or unavailable. It returns a pre-written Script from the
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
            and 3-scene storyboard fields populated.

        Example:
            >>> showrunner = Showrunner()
            >>> # Get deterministic script based on theme
            >>> script = showrunner.get_fallback_script("bizarre infomercial")
            >>> print(script.setup)
            >>> print(script.storyboard)  # List of 3 scene descriptions

            >>> # Get deterministic script with explicit seed
            >>> script = showrunner.get_fallback_script("any theme", seed=42)
        """
        # Use theme hash + seed for deterministic selection
        rng = random.Random()
        if seed is not None:
            rng.seed(seed)
        else:
            rng.seed(hash(theme))

        script = rng.choice(FALLBACK_TEMPLATES)

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
