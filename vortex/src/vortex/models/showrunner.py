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
