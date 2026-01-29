"""Vortex Video Orchestrator - Narrative Chain pipeline.

Thin wrapper around DefaultRenderer that:
1. Loads config from config.yaml
2. Initializes the renderer
3. Provides a simple generate() interface
4. Handles output file saving
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vortex.renderers.default.renderer import DefaultRenderer
from vortex.utils.render_output import save_render_result

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of video generation."""

    video_path: str
    audio_path: str
    seed: int
    duration_sec: float
    generation_time_ms: float
    success: bool
    error_msg: str | None = None


class VideoOrchestrator:
    """Orchestrates the Narrative Chain video generation pipeline.

    This is a thin wrapper around DefaultRenderer that provides a simple
    interface for video generation with file output handling.

    Pipeline: Showrunner -> Kokoro TTS -> Flux keyframe -> CogVideoX video -> CLIP
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        output_dir: str = "outputs",
        device: str = "cuda",
    ):
        """Initialize orchestrator.

        Args:
            config_path: Path to vortex config.yaml
            output_dir: Directory for output files
            device: PyTorch device for models
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._device = device

        # Load config
        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)

        # Create renderer (lazy initialization)
        self._renderer: DefaultRenderer | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the renderer and load models."""
        if self._initialized:
            return

        logger.info("Initializing VideoOrchestrator...")
        self._renderer = DefaultRenderer()
        await self._renderer.initialize(self._device, self._config)
        self._initialized = True
        logger.info("VideoOrchestrator initialized")

    async def generate(
        self,
        slot_id: int,
        seed: int | None = None,
        theme: str = "bizarre infomercial",
        tone: str = "absurd",
        target_duration: float = 12.0,
        voice_id: str = "rick_c137",
        deadline_sec: float = 150.0,
    ) -> GenerationResult:
        """Generate a video clip using the Narrative Chain pipeline.

        Args:
            slot_id: Unique identifier for this generation
            seed: Deterministic seed (random if not provided)
            theme: Topic for LLM script generation
            tone: Comedic tone ("absurd", "deadpan", "manic")
            target_duration: Target video duration in seconds
            voice_id: Bark voice ID
            deadline_sec: Maximum time allowed for generation

        Returns:
            GenerationResult with paths to output files
        """
        if not self._initialized:
            await self.initialize()

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        logger.info(
            f"Starting generation: slot_id={slot_id}, seed={seed}, theme='{theme}'"
        )

        # Build recipe
        recipe: dict[str, Any] = {
            "slot_params": {
                "slot_id": slot_id,
                "seed": seed,
                "target_duration": target_duration,
                "fps": 8,
            },
            "narrative": {
                "theme": theme,
                "tone": tone,
                "auto_script": True,
            },
            "audio": {
                "voice_id": voice_id,
            },
        }

        # Set deadline
        deadline = time.time() + deadline_sec

        # Render
        assert self._renderer is not None
        result = await self._renderer.render(recipe, slot_id, seed, deadline)

        if not result.success:
            return GenerationResult(
                video_path="",
                audio_path="",
                seed=seed,
                duration_sec=0.0,
                generation_time_ms=result.generation_time_ms,
                success=False,
                error_msg=result.error_msg,
            )

        # Save output files
        output_paths = save_render_result(
            result=result,
            output_dir=self.output_dir,
            fps=8,
            sample_rate=24000,
            slot_id=slot_id,
            seed=seed,
        )

        # Calculate duration from video frames
        num_frames = result.video_frames.shape[0] if result.video_frames.numel() > 0 else 0
        duration_sec = num_frames / 8.0 if num_frames > 0 else 0.0

        return GenerationResult(
            video_path=str(output_paths["video_path"]),
            audio_path=str(output_paths["audio_path"]),
            seed=seed,
            duration_sec=duration_sec,
            generation_time_ms=result.generation_time_ms,
            success=True,
        )

    async def health_check(self) -> dict[str, bool]:
        """Check health of all components.

        Returns:
            Dict with component health status
        """
        if not self._initialized or self._renderer is None:
            return {
                "orchestrator": False,
                "renderer": False,
            }

        renderer_ok = await self._renderer.health_check()

        return {
            "orchestrator": True,
            "renderer": renderer_ok,
        }

    async def shutdown(self) -> None:
        """Shutdown orchestrator and release resources."""
        if self._renderer is not None:
            await self._renderer.shutdown()
            self._renderer = None
        self._initialized = False
        logger.info("VideoOrchestrator shutdown complete")
