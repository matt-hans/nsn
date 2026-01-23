"""Vortex core pipeline - Modular video generation orchestration.

This module implements the VortexPipeline class that:
- Loads pluggable video renderers via RendererRegistry
- Orchestrates async generation through the DeterministicVideoRenderer interface
- Supports modular Lane 0 video generation backends
- Returns GenerationResult with video frames, audio, CLIP embedding, metadata

The pipeline delegates all model loading, VRAM management, and generation logic
to the configured renderer. This enables network users to run alternative video
generation backends (e.g., StableVideoDiffusion, CogVideoX) that implement the
DeterministicVideoRenderer interface.

Example:
    >>> pipeline = await VortexPipeline.create(config_path="config.yaml")
    >>> result = await pipeline.generate_slot(recipe=recipe, slot_id=12345)
    >>> print(f"Generated in {result.generation_time_ms}ms")
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from vortex.renderers import (
    DeterministicVideoRenderer,
    RendererNotFoundError,
    RendererRegistry,
    RenderResult,
    policy_from_config,
)
from vortex.utils.render_output import save_render_result

logger = logging.getLogger(__name__)


class VortexInitializationError(Exception):
    """Raised when pipeline initialization fails."""

    pass


@dataclass
class GenerationResult:
    """Result of a single slot generation.

    Attributes:
        video_frames: Tensor of shape (num_frames, channels, height, width)
        audio_waveform: Tensor of shape (num_samples,)
        clip_embedding: Combined CLIP embedding from dual ensemble
        generation_time_ms: Total time from start to completion (milliseconds)
        slot_id: Unique slot identifier
        success: Whether generation completed successfully
        error_msg: Error message if success=False
        determinism_proof: Hash for BFT consensus verification (if successful)
    """

    video_frames: torch.Tensor
    audio_waveform: torch.Tensor
    clip_embedding: torch.Tensor
    generation_time_ms: float
    slot_id: int
    success: bool = True
    error_msg: str | None = None
    determinism_proof: bytes = b""


class VortexPipeline:
    """Core Vortex pipeline - Modular video generation orchestration.

    Uses the RendererRegistry to load and manage DeterministicVideoRenderer
    implementations. All model loading, VRAM management, and generation logic
    is delegated to the configured renderer.

    Example:
        >>> pipeline = await VortexPipeline.create(config_path="config.yaml")
        >>> result = await pipeline.generate_slot(recipe=recipe, slot_id=12345)
        >>> print(f"Generated in {result.generation_time_ms}ms")

    Alternative renderers can be used by changing the 'renderers.default' config:
        renderers:
          default: "stable-video-diffusion"
    """

    def __init__(
        self,
        config: dict[str, Any],
        renderer: DeterministicVideoRenderer,
        device: str,
    ):
        """Initialize pipeline with config and renderer.

        Use VortexPipeline.create() for async initialization instead.

        Args:
            config: Loaded configuration dict
            renderer: Initialized DeterministicVideoRenderer
            device: Target device string
        """
        self.config = config
        self.renderer = renderer
        self.device = device
        self._initialized = True

    @classmethod
    async def create(
        cls,
        config_path: str | None = None,
        device: str | None = None,
        renderer_name: str | None = None,
    ) -> VortexPipeline:
        """Create and initialize a VortexPipeline.

        This is the recommended way to create a pipeline. It:
        1. Loads configuration from config.yaml
        2. Creates a RendererRegistry from the renderers directory
        3. Loads and initializes the specified renderer
        4. Returns a ready-to-use pipeline

        Args:
            config_path: Path to config.yaml (default: vortex/config.yaml)
            device: Override device from config (e.g., "cpu" for testing)
            renderer_name: Override renderer from config

        Returns:
            Initialized VortexPipeline ready for generation

        Raises:
            VortexInitializationError: If initialization fails
            RendererNotFoundError: If specified renderer not found

        Example:
            >>> pipeline = await VortexPipeline.create()
            >>> result = await pipeline.generate_slot(recipe, slot_id=1)
        """
        # Load configuration
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config.yaml")

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise VortexInitializationError(f"Failed to load config: {e}") from e

        # Enforce offline model loading when configured.
        if config.get("models", {}).get("local_only", False):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info(
                "Local-only mode enabled (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)"
            )

        # Determine device
        device = device or config.get("device", {}).get("name", "cuda:0")
        logger.info(f"Initializing Vortex pipeline on device: {device}")

        # Load renderer registry
        renderers_config = config.get("renderers", {})
        base_path = Path(__file__).parent

        policy = policy_from_config(renderers_config.get("policy"))
        registry = RendererRegistry.from_config(
            renderers_config,
            base_path,
            policy=policy,
            load_renderers=True,
        )

        # Get renderer name
        renderer_name = renderer_name or renderers_config.get(
            "default", "default-narrative-chain"
        )

        # Get and initialize renderer
        try:
            renderer = registry.get(renderer_name)
        except RendererNotFoundError:
            # If no renderers in registry, try loading default directly
            available = registry.list_renderers()
            if not available:
                logger.info("No renderers in registry, loading default renderer directly")
                from vortex.renderers.default import DefaultRenderer

                renderer = DefaultRenderer()
            else:
                raise

        logger.info(f"Initializing renderer: {renderer.manifest.name}")
        await renderer.initialize(device, config)

        # Verify renderer health
        if not await renderer.health_check():
            raise VortexInitializationError(
                f"Renderer '{renderer.manifest.name}' health check failed"
            )

        logger.info(
            "Vortex pipeline initialized successfully",
            extra={
                "device": device,
                "renderer": renderer.manifest.name,
                "renderer_version": renderer.manifest.version,
                "vram_gb": renderer.manifest.resources.vram_gb,
            },
        )

        return cls(config=config, renderer=renderer, device=device)

    async def generate_slot(
        self,
        recipe: dict[str, Any],
        slot_id: int,
        seed: int | None = None,
        deadline: float | None = None,
    ) -> GenerationResult:
        """Generate a single slot (45-second video) from recipe.

        Delegates generation to the configured renderer. The renderer handles:
        - Model orchestration (Flux, CogVideoX, Kokoro, CLIP)
        - VRAM management
        - Deterministic seed propagation
        - Deadline tracking

        Args:
            recipe: Recipe dict with:
                - slot_params: {slot_id, duration_sec, fps}
                - audio_track: {script, voice_id, speed, emotion}
                - visual_track: {prompt, expression_preset, ...}
                - semantic_constraints: {clip_threshold, ...}
            slot_id: Unique slot identifier
            seed: Deterministic seed (random if not provided)
            deadline: Absolute deadline timestamp (default: now + 45s)

        Returns:
            GenerationResult with video, audio, CLIP embedding, metadata

        Example:
            >>> recipe = {
            ...     "slot_params": {"slot_id": 1, "duration_sec": 45},
            ...     "audio_track": {"script": "Hello world!", "voice_id": "rick_c137"},
            ...     "visual_track": {"prompt": "scientist in lab coat"},
            ... }
            >>> result = await pipeline.generate_slot(recipe, slot_id=1)
        """
        start_time = time.time()

        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Set deadline if not provided (default: 45 seconds from now)
        if deadline is None:
            deadline = time.time() + 45.0

        # Add slot_id to recipe if not present
        if "slot_params" not in recipe:
            recipe["slot_params"] = {}
        recipe["slot_params"]["slot_id"] = slot_id

        logger.info(
            "Starting slot generation",
            extra={
                "slot_id": slot_id,
                "seed": seed,
                "deadline_in": deadline - time.time(),
                "renderer": self.renderer.manifest.name,
            },
        )

        # Delegate to renderer
        render_result: RenderResult = await self.renderer.render(
            recipe=recipe,
            slot_id=slot_id,
            seed=seed,
            deadline=deadline,
        )

        # Convert RenderResult to GenerationResult
        generation_time_ms = (time.time() - start_time) * 1000

        result = GenerationResult(
            video_frames=render_result.video_frames,
            audio_waveform=render_result.audio_waveform,
            clip_embedding=render_result.clip_embedding,
            generation_time_ms=generation_time_ms,
            slot_id=slot_id,
            success=render_result.success,
            error_msg=render_result.error_msg,
            determinism_proof=render_result.determinism_proof,
        )

        if result.success:
            logger.info(
                "Slot generation completed",
                extra={
                    "slot_id": slot_id,
                    "generation_time_ms": generation_time_ms,
                    "proof_hash": result.determinism_proof.hex()[:16] + "...",
                },
            )
            outputs_cfg = self.config.get("outputs", {})
            if outputs_cfg.get("enabled", True):
                output_dir = outputs_cfg.get("directory", "outputs")
                include_audio = outputs_cfg.get("include_audio_in_mp4", True)
                fps = recipe.get("slot_params", {}).get("fps", 24)
                sample_rate = (
                    self.config.get("buffers", {})
                    .get("audio", {})
                    .get("sample_rate", 24000)
                )
                try:
                    paths = save_render_result(
                        render_result,
                        output_dir=output_dir,
                        fps=fps,
                        sample_rate=sample_rate,
                        slot_id=slot_id,
                        seed=seed,
                        include_audio_in_mp4=include_audio,
                    )
                    logger.info(
                        "Saved render output",
                        extra={
                            "video_path": str(paths["video_path"]),
                            "audio_path": str(paths["audio_path"]),
                        },
                    )
                except Exception as exc:
                    logger.error("Failed to save render output: %s", exc)
        else:
            logger.error(
                "Slot generation failed",
                extra={
                    "slot_id": slot_id,
                    "error": result.error_msg,
                },
            )

        return result

    async def health_check(self) -> bool:
        """Check if pipeline and renderer are healthy.

        Returns:
            True if ready for generation, False otherwise
        """
        return await self.renderer.health_check()

    @property
    def renderer_name(self) -> str:
        """Return the name of the active renderer."""
        return self.renderer.manifest.name

    @property
    def renderer_version(self) -> str:
        """Return the version of the active renderer."""
        return self.renderer.manifest.version
