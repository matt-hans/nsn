"""Vortex core pipeline - Static VRAM manager and generation orchestration.

This module implements the foundational VortexPipeline class that:
- Loads all AI models once at initialization (static VRAM residency)
- Pre-allocates output buffers to prevent fragmentation
- Monitors VRAM pressure with soft/hard limits
- Orchestrates async generation (parallel audio + actor, sequential video)
- Returns GenerationResult with video frames, audio, CLIP embedding, metadata

CRITICAL: All models remain loaded in VRAM at all times. No swapping.
Total VRAM budget: 11.8GB (fits RTX 3060 12GB with 500MB safety margin).
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import yaml

from vortex.models import ModelName, load_model
from vortex.utils.memory import get_current_vram_usage, get_vram_stats, log_vram_snapshot

logger = logging.getLogger(__name__)


class VortexInitializationError(Exception):
    """Raised when pipeline initialization fails (e.g., CUDA OOM)."""

    pass


class MemoryPressureWarning(Warning):
    """Raised when VRAM usage exceeds soft limit (11.0GB)."""

    pass


class MemoryPressureError(Exception):
    """Raised when VRAM usage exceeds hard limit (11.5GB)."""

    pass


@dataclass
class GenerationResult:
    """Result of a single slot generation.

    Attributes:
        video_frames: Tensor of shape (num_frames, height, width, channels)
        audio_waveform: Tensor of shape (num_samples,)
        clip_embedding: Combined CLIP embedding from dual ensemble
        generation_time_ms: Total time from start to completion (milliseconds)
        slot_id: Unique slot identifier
        success: Whether generation completed successfully
        error_msg: Error message if success=False
    """

    video_frames: torch.Tensor
    audio_waveform: torch.Tensor
    clip_embedding: torch.Tensor
    generation_time_ms: float
    slot_id: int
    success: bool = True
    error_msg: Optional[str] = None


class ModelRegistry:
    """Registry for loaded models with get_model() interface.

    Manages the lifecycle of all 5 models (Flux, LivePortrait, Kokoro, CLIP×2).
    Models are loaded once and never unloaded (static VRAM residency).

    Example:
        >>> registry = ModelRegistry(device="cuda:0")
        >>> flux = registry.get_model("flux")
        >>> clip_b = registry.get_model("clip_b")
    """

    def __init__(self, device: str, precision_overrides: Optional[Dict[ModelName, str]] = None):
        """Initialize model registry.

        Args:
            device: Target device (e.g., "cuda:0", "cpu")
            precision_overrides: Optional precision overrides per model
        """
        self.device = device
        self.precision_overrides = precision_overrides or {}
        self._models: Dict[ModelName, nn.Module] = {}
        self._load_all_models()

    def _load_all_models(self) -> None:
        """Load all 5 models into registry.

        Models loaded: flux, liveportrait, kokoro, clip_b, clip_l
        Total VRAM: ~10.8GB (6.0 + 3.5 + 0.4 + 0.3 + 0.6 + 1.0 overhead)

        Raises:
            VortexInitializationError: If CUDA OOM occurs during loading
        """
        model_names: list[ModelName] = ["flux", "liveportrait", "kokoro", "clip_b", "clip_l"]

        try:
            for name in model_names:
                logger.info(f"Loading model: {name}")
                precision = self.precision_overrides.get(name)
                model = load_model(name, device=self.device, precision=precision)
                self._models[name] = model
                log_vram_snapshot(f"after_{name}_load")

            logger.info(
                "All models loaded successfully",
                extra={"total_models": len(self._models), "vram_gb": get_vram_stats()["allocated_gb"]},
            )

        except torch.cuda.OutOfMemoryError as e:
            stats = get_vram_stats()
            error_msg = (
                f"CUDA OOM during model loading. "
                f"Allocated: {stats['allocated_gb']:.2f}GB, "
                f"Total: {stats['total_gb']:.2f}GB. "
                f"Remediation: Upgrade to GPU with >=12GB VRAM (RTX 3060 minimum)."
            )
            logger.error(error_msg, exc_info=True)
            # Clean up partial models
            self._models.clear()
            raise VortexInitializationError(error_msg) from e

    def get_model(self, name: ModelName) -> nn.Module:
        """Get a loaded model by name.

        Args:
            name: Model name (flux, liveportrait, kokoro, clip_b, clip_l)

        Returns:
            nn.Module: The requested model

        Raises:
            KeyError: If model name is invalid or not loaded

        Example:
            >>> flux = registry.get_model("flux")
            >>> output = flux(input_tensor)
        """
        if name not in self._models:
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available: {list(self._models.keys())}"
            )
        return self._models[name]

    def __contains__(self, name: ModelName) -> bool:
        """Check if model is loaded in registry."""
        return name in self._models


class VRAMMonitor:
    """VRAM pressure monitoring with soft/hard limits.

    Tracks VRAM usage and emits warnings/errors when limits are exceeded:
    - Soft limit (11.0GB): Log warning, continue
    - Hard limit (11.5GB): Raise MemoryPressureError

    Example:
        >>> monitor = VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.5)
        >>> monitor.check()  # May raise MemoryPressureError
    """

    def __init__(self, soft_limit_gb: float = 11.0, hard_limit_gb: float = 11.5):
        """Initialize VRAM monitor.

        Args:
            soft_limit_gb: Soft limit in GB (warning threshold)
            hard_limit_gb: Hard limit in GB (error threshold)
        """
        self.soft_limit_bytes = int(soft_limit_gb * 1e9)
        self.hard_limit_bytes = int(hard_limit_gb * 1e9)
        self._warning_emitted = False

    def check(self) -> None:
        """Check current VRAM usage against limits.

        Emits MemoryPressureWarning if soft limit exceeded (once per instance).
        Raises MemoryPressureError if hard limit exceeded.

        Raises:
            MemoryPressureError: If VRAM usage exceeds hard limit

        Example:
            >>> monitor.check()  # May log warning or raise error
        """
        current_usage = get_current_vram_usage()
        stats = get_vram_stats()

        if current_usage > self.hard_limit_bytes:
            error_msg = (
                f"VRAM hard limit exceeded: {stats['allocated_gb']:.2f}GB "
                f"> {self.hard_limit_bytes / 1e9:.2f}GB. "
                f"Generation aborted to prevent CUDA OOM."
            )
            logger.error(error_msg, extra=stats)
            raise MemoryPressureError(error_msg)

        if current_usage > self.soft_limit_bytes and not self._warning_emitted:
            logger.warning(
                "VRAM soft limit exceeded: %.2fGB > %.2fGB. "
                "Monitor for OOM. Consider reducing model size or batch size.",
                stats["allocated_gb"],
                self.soft_limit_bytes / 1e9,
                extra=stats,
            )
            self._warning_emitted = True

    def reset_warning(self) -> None:
        """Reset warning flag (for testing or after memory cleanup)."""
        self._warning_emitted = False


class VortexPipeline:
    """Core Vortex pipeline - Static VRAM manager and generation orchestration.

    Loads all models once, pre-allocates buffers, orchestrates async generation.
    This is the main interface for video slot generation.

    VRAM Budget:
        - Flux-Schnell (NF4): ~6.0 GB
        - LivePortrait (FP16): ~3.5 GB
        - Kokoro-82M (FP32): ~0.4 GB
        - CLIP-ViT-B-32 (INT8): ~0.3 GB
        - CLIP-ViT-L-14 (INT8): ~0.6 GB
        - Buffers + Overhead: ~1.0 GB
        - TOTAL: ~11.8 GB

    Example:
        >>> pipeline = VortexPipeline(config_path="config.yaml")
        >>> result = await pipeline.generate_slot(recipe=recipe, slot_id=12345)
        >>> print(f"Generated in {result.generation_time_ms}ms")
    """

    def __init__(self, config_path: Optional[str] = None, device: Optional[str] = None):
        """Initialize Vortex pipeline.

        Args:
            config_path: Path to config.yaml (default: vortex/config.yaml)
            device: Override device from config (e.g., "cpu" for testing)

        Raises:
            VortexInitializationError: If initialization fails (CUDA OOM, etc.)
        """
        # Load configuration
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Device setup
        self.device = device or self.config["device"]["name"]
        logger.info(f"Initializing Vortex pipeline on device: {self.device}")

        # Model registry (loads all models once)
        precision_overrides = self.config["models"]["precision"]
        self.model_registry = ModelRegistry(
            device=self.device, precision_overrides=precision_overrides
        )

        # VRAM monitor
        self.vram_monitor = VRAMMonitor(
            soft_limit_gb=self.config["vram"]["soft_limit_gb"],
            hard_limit_gb=self.config["vram"]["hard_limit_gb"],
        )

        # Pre-allocate output buffers (prevents fragmentation)
        self._allocate_buffers()

        # Log final VRAM state
        log_vram_snapshot("pipeline_initialized")
        stats = get_vram_stats()
        logger.info(
            "Vortex pipeline initialized successfully",
            extra={
                "device": self.device,
                "models_loaded": 5,
                "vram_allocated_gb": stats["allocated_gb"],
                "vram_total_gb": stats["total_gb"],
            },
        )

    def _allocate_buffers(self) -> None:
        """Pre-allocate output buffers to prevent fragmentation during generation.

        Buffers:
            - actor_buffer: (1, channels, height, width) for single actor image
            - video_buffer: (frames, channels, height, width) for video sequence
            - audio_buffer: (samples,) for audio waveform
        """
        buf_cfg = self.config["buffers"]

        # Actor buffer (512x512x3)
        self.actor_buffer = torch.zeros(
            1,
            buf_cfg["actor"]["channels"],
            buf_cfg["actor"]["height"],
            buf_cfg["actor"]["width"],
            device=self.device,
            dtype=torch.float32,
        )

        # Video buffer (1080 frames × 512x512x3)
        self.video_buffer = torch.zeros(
            buf_cfg["video"]["frames"],
            buf_cfg["video"]["channels"],
            buf_cfg["video"]["height"],
            buf_cfg["video"]["width"],
            device=self.device,
            dtype=torch.float32,
        )

        # Audio buffer (1080000 samples = 45s @ 24kHz)
        self.audio_buffer = torch.zeros(
            buf_cfg["audio"]["samples"],
            device=self.device,
            dtype=torch.float32,
        )

        logger.info(
            "Output buffers pre-allocated",
            extra={
                "actor_shape": tuple(self.actor_buffer.shape),
                "video_shape": tuple(self.video_buffer.shape),
                "audio_shape": tuple(self.audio_buffer.shape),
            },
        )

    async def generate_slot(self, recipe: Dict, slot_id: int) -> GenerationResult:
        """Generate a single slot (45-second video) from recipe.

        Orchestration:
            1. Parallel: Audio (Kokoro) + Actor image (Flux)
            2. Sequential: Video warping (LivePortrait)
            3. Verification: Dual CLIP embedding

        Args:
            recipe: Recipe dict with audio_track, visual_track, semantic_constraints
            slot_id: Unique slot identifier

        Returns:
            GenerationResult with video, audio, CLIP embedding, metadata

        Raises:
            MemoryPressureError: If VRAM exceeds hard limit during generation
            asyncio.TimeoutError: If generation exceeds timeout (20s default)

        Example:
            >>> recipe = {"audio_track": {...}, "visual_track": {...}}
            >>> result = await pipeline.generate_slot(recipe, slot_id=12345)
        """
        start_time = time.time()
        timeout = self.config["pipeline"]["generation_timeout_sec"]

        try:
            # Check VRAM before starting
            self.vram_monitor.check()

            # Phase 1: Parallel audio + actor generation
            if self.config["pipeline"]["parallel_audio_actor"]:
                audio_task = asyncio.create_task(self._generate_audio(recipe))
                actor_task = asyncio.create_task(self._generate_actor(recipe))

                # Wait for both with timeout
                audio_result, actor_result = await asyncio.wait_for(
                    asyncio.gather(audio_task, actor_task),
                    timeout=timeout,
                )
            else:
                # Sequential fallback (for debugging)
                audio_result = await self._generate_audio(recipe)
                actor_result = await self._generate_actor(recipe)

            # Phase 2: Sequential video warping
            video_result = await self._generate_video(actor_result, audio_result)

            # Phase 3: CLIP verification
            clip_embedding = await self._verify_semantic(video_result, recipe)

            # Compute total time
            generation_time_ms = (time.time() - start_time) * 1000

            return GenerationResult(
                video_frames=video_result,
                audio_waveform=audio_result,
                clip_embedding=clip_embedding,
                generation_time_ms=generation_time_ms,
                slot_id=slot_id,
                success=True,
            )

        except asyncio.CancelledError:
            logger.warning(f"Slot {slot_id} generation cancelled")
            raise

        except Exception as e:
            logger.error(f"Slot {slot_id} generation failed: {e}", exc_info=True)
            return GenerationResult(
                video_frames=torch.empty(0),
                audio_waveform=torch.empty(0),
                clip_embedding=torch.empty(0),
                generation_time_ms=(time.time() - start_time) * 1000,
                slot_id=slot_id,
                success=False,
                error_msg=str(e),
            )

    async def _generate_audio(self, recipe: Dict) -> torch.Tensor:
        """Generate audio waveform using Kokoro TTS.

        Args:
            recipe: Recipe with audio_track section

        Returns:
            Audio waveform tensor (reuses self.audio_buffer)
        """
        # TODO(T017): Replace with real Kokoro TTS implementation
        await asyncio.sleep(0.1)  # Simulate 100ms generation
        return self.audio_buffer.clone()

    async def _generate_actor(self, recipe: Dict) -> torch.Tensor:
        """Generate actor image using Flux-Schnell.

        Args:
            recipe: Recipe with visual_track section

        Returns:
            Actor image tensor (reuses self.actor_buffer)
        """
        # TODO(T015): Replace with real Flux-Schnell implementation
        await asyncio.sleep(0.1)  # Simulate 100ms generation
        return self.actor_buffer.clone()

    async def _generate_video(self, actor_img: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Generate video using LivePortrait warping.

        Args:
            actor_img: Base actor image
            audio: Audio waveform for lip sync

        Returns:
            Video frames tensor (reuses self.video_buffer)
        """
        # TODO(T016): Replace with real LivePortrait implementation
        await asyncio.sleep(0.1)  # Simulate 100ms generation
        return self.video_buffer.clone()

    async def _verify_semantic(self, video: torch.Tensor, recipe: Dict) -> torch.Tensor:
        """Dual CLIP semantic verification.

        Args:
            video: Generated video frames
            recipe: Recipe with semantic_constraints

        Returns:
            Combined CLIP embedding (B-32 + L-14 ensemble)
        """
        # TODO(T018): Replace with real dual CLIP ensemble
        await asyncio.sleep(0.05)  # Simulate 50ms verification
        # Return mock 512-dim embedding
        return torch.randn(512, device=self.device, dtype=torch.float32)
