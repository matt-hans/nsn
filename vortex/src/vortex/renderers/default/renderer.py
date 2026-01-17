"""Default Lane 0 renderer implementation.

This renderer wraps the existing Flux-Schnell, LivePortrait, Kokoro, and dual CLIP
pipeline as a DeterministicVideoRenderer. It serves as the reference implementation
for Lane 0 video generation and maintains backward compatibility with the existing
VortexPipeline behavior.

Architecture:
    - Flux-Schnell (NF4, 6.0GB): Generates actor images from prompts
    - LivePortrait (FP16, 3.5GB): Animates images with audio-driven lip-sync
    - Kokoro-82M (FP32, 0.4GB): Text-to-speech synthesis
    - CLIP ViT-B-32 + ViT-L-14 (INT8, 0.9GB): Dual ensemble semantic verification

Total VRAM: 11.8GB (fits RTX 3060 12GB with 500MB safety margin)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from vortex.models import ModelName, load_model
from vortex.renderers.base import DeterministicVideoRenderer
from vortex.renderers.recipe_schema import merge_with_defaults, validate_recipe
from vortex.renderers.types import RendererManifest, RenderResult
from vortex.utils.memory import get_current_vram_usage, get_vram_stats, log_vram_snapshot

logger = logging.getLogger(__name__)


class VortexInitializationError(Exception):
    """Raised when renderer initialization fails."""

    pass


class MemoryPressureError(Exception):
    """Raised when VRAM usage exceeds hard limit."""

    pass


class _ModelRegistry:
    """Registry for loaded models with get_model() interface.

    Internal class for managing model lifecycle. Models are loaded once
    and remain in VRAM (static residency).
    """

    def __init__(self, device: str, precision_overrides: dict[ModelName, str] | None = None):
        self.device = device
        self.precision_overrides = precision_overrides or {}
        self._models: dict[ModelName, nn.Module] = {}

    def load_all_models(self) -> None:
        """Load all 5 models into registry."""
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
                extra={
                    "total_models": len(self._models),
                    "vram_gb": get_vram_stats()["allocated_gb"],
                },
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
            self._models.clear()
            raise VortexInitializationError(error_msg) from e

    def get_model(self, name: ModelName) -> nn.Module:
        """Get a loaded model by name."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        return self._models[name]

    def __contains__(self, name: ModelName) -> bool:
        return name in self._models


class _VRAMMonitor:
    """VRAM pressure monitoring with soft/hard limits."""

    def __init__(self, soft_limit_gb: float = 11.0, hard_limit_gb: float = 11.5):
        self.soft_limit_bytes = int(soft_limit_gb * 1e9)
        self.hard_limit_bytes = int(hard_limit_gb * 1e9)
        self._warning_emitted = False

    def check(self) -> None:
        """Check current VRAM usage against limits."""
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
                "VRAM soft limit exceeded: %.2fGB > %.2fGB.",
                stats["allocated_gb"],
                self.soft_limit_bytes / 1e9,
                extra=stats,
            )
            self._warning_emitted = True


class DefaultRenderer(DeterministicVideoRenderer):
    """Default Lane 0 renderer using Flux-Schnell, LivePortrait, Kokoro, and dual CLIP.

    This renderer wraps the existing pipeline implementation as a DeterministicVideoRenderer.
    It maintains backward compatibility while adding:
    - Explicit seed propagation for determinism
    - Determinism proof computation for BFT consensus
    - Deadline awareness

    Example:
        >>> renderer = DefaultRenderer()
        >>> await renderer.initialize("cuda:0", config)
        >>> result = await renderer.render(recipe, slot_id=1, seed=42, deadline=time.time() + 45)
    """

    def __init__(self, manifest: RendererManifest | None = None):
        """Initialize renderer.

        Args:
            manifest: Optional pre-loaded manifest (loaded from file if None)
        """
        self._manifest = manifest or self._load_manifest()
        self._model_registry: _ModelRegistry | None = None
        self._vram_monitor: _VRAMMonitor | None = None
        self._device: str = "cpu"
        self._config: dict[str, Any] = {}
        self._initialized = False

        # Pre-allocated buffers (set during initialize)
        self._actor_buffer: torch.Tensor | None = None
        self._video_buffer: torch.Tensor | None = None
        self._audio_buffer: torch.Tensor | None = None

    def _load_manifest(self) -> RendererManifest:
        """Load manifest from default/manifest.yaml."""
        manifest_path = Path(__file__).parent / "manifest.yaml"
        with open(manifest_path) as f:
            raw = yaml.safe_load(f)
        return RendererManifest.from_dict(raw)

    @property
    def manifest(self) -> RendererManifest:
        """Return renderer manifest."""
        return self._manifest

    async def initialize(self, device: str, config: dict[str, Any]) -> None:
        """Initialize renderer and load models.

        Args:
            device: Target device (e.g., "cuda:0", "cpu")
            config: Configuration dict from config.yaml
        """
        self._device = device
        self._config = config
        logger.info(f"Initializing DefaultRenderer on device: {device}")

        # Initialize model registry and load models
        precision_overrides = config.get("models", {}).get("precision", {})
        self._model_registry = _ModelRegistry(device, precision_overrides)
        self._model_registry.load_all_models()

        # Initialize VRAM monitor
        vram_config = config.get("vram", {})
        self._vram_monitor = _VRAMMonitor(
            soft_limit_gb=vram_config.get("soft_limit_gb", 11.0),
            hard_limit_gb=vram_config.get("hard_limit_gb", 11.5),
        )

        # Pre-allocate output buffers
        self._allocate_buffers(config)

        self._initialized = True
        log_vram_snapshot("renderer_initialized")
        logger.info("DefaultRenderer initialized successfully")

    def _allocate_buffers(self, config: dict[str, Any]) -> None:
        """Pre-allocate output buffers to prevent fragmentation."""
        buf_cfg = config.get("buffers", {})

        # Actor buffer (512x512x3)
        actor_cfg = buf_cfg.get("actor", {})
        self._actor_buffer = torch.zeros(
            1,
            actor_cfg.get("channels", 3),
            actor_cfg.get("height", 512),
            actor_cfg.get("width", 512),
            device=self._device,
            dtype=torch.float32,
        )

        # Video buffer (1080 frames × 512x512x3)
        video_cfg = buf_cfg.get("video", {})
        self._video_buffer = torch.zeros(
            video_cfg.get("frames", 1080),
            video_cfg.get("channels", 3),
            video_cfg.get("height", 512),
            video_cfg.get("width", 512),
            device=self._device,
            dtype=torch.float32,
        )

        # Audio buffer (1080000 samples = 45s @ 24kHz)
        audio_cfg = buf_cfg.get("audio", {})
        self._audio_buffer = torch.zeros(
            audio_cfg.get("samples", 1080000),
            device=self._device,
            dtype=torch.float32,
        )

        logger.info(
            "Output buffers pre-allocated",
            extra={
                "actor_shape": tuple(self._actor_buffer.shape),
                "video_shape": tuple(self._video_buffer.shape),
                "audio_shape": tuple(self._audio_buffer.shape),
            },
        )

    async def render(
        self,
        recipe: dict[str, Any],
        slot_id: int,
        seed: int,
        deadline: float,
    ) -> RenderResult:
        """Render a single slot from recipe.

        Args:
            recipe: Standardized recipe dict
            slot_id: Unique slot identifier
            seed: Deterministic seed for reproducibility
            deadline: Absolute deadline timestamp

        Returns:
            RenderResult with video, audio, embedding, and determinism proof
        """
        if not self._initialized:
            raise RuntimeError("Renderer not initialized. Call initialize() first.")

        start_time = time.time()

        # Validate and merge recipe with defaults
        errors = validate_recipe(recipe)
        if errors:
            return RenderResult(
                video_frames=torch.empty(0),
                audio_waveform=torch.empty(0),
                clip_embedding=torch.empty(0),
                generation_time_ms=0.0,
                determinism_proof=b"",
                success=False,
                error_msg=f"Recipe validation failed: {'; '.join(errors)}",
            )

        recipe = merge_with_defaults(recipe)

        try:
            # Check VRAM before starting
            assert self._vram_monitor is not None
            self._vram_monitor.check()

            # Set deterministic seed for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Phase 1: Parallel audio + actor generation
            audio_task = asyncio.create_task(self._generate_audio(recipe, seed))
            actor_task = asyncio.create_task(self._generate_actor(recipe, seed))

            audio_result, actor_result = await asyncio.gather(audio_task, actor_task)

            # Check deadline after parallel phase
            time_remaining = deadline - time.time()
            if time_remaining < 10.0:  # Need at least 10s for video + CLIP
                raise TimeoutError(
                    f"Deadline would be exceeded: {time_remaining:.1f}s remaining"
                )

            # Phase 2: Sequential video warping
            video_result = await self._generate_video(actor_result, audio_result, recipe, seed)

            # Check deadline before CLIP
            time_remaining = deadline - time.time()
            if time_remaining < 2.0:  # Need at least 2s for CLIP
                raise TimeoutError(
                    f"Deadline would be exceeded: {time_remaining:.1f}s remaining"
                )

            # Phase 3: CLIP verification
            clip_embedding = await self._verify_semantic(video_result, recipe)

            # Compute generation time
            generation_time_ms = (time.time() - start_time) * 1000

            # Build result
            result = RenderResult(
                video_frames=video_result,
                audio_waveform=audio_result,
                clip_embedding=clip_embedding,
                generation_time_ms=generation_time_ms,
                determinism_proof=b"",  # Computed below
                success=True,
            )

            # Compute determinism proof
            result.determinism_proof = self.compute_determinism_proof(recipe, seed, result)

            return result

        except Exception as e:
            logger.error(f"Slot {slot_id} render failed: {e}", exc_info=True)
            return RenderResult(
                video_frames=torch.empty(0),
                audio_waveform=torch.empty(0),
                clip_embedding=torch.empty(0),
                generation_time_ms=(time.time() - start_time) * 1000,
                determinism_proof=b"",
                success=False,
                error_msg=str(e),
            )

    async def _generate_audio(self, recipe: dict[str, Any], seed: int) -> torch.Tensor:
        """Generate audio waveform using Kokoro TTS.

        Args:
            recipe: Recipe with audio_track section
            seed: Deterministic seed

        Returns:
            Audio waveform tensor
        """
        assert self._audio_buffer is not None
        # TODO(T017): Replace with real Kokoro TTS implementation
        # Kokoro should use seed for deterministic output
        # In real implementation: kokoro.synthesize(..., seed=seed)
        await asyncio.sleep(0.1)  # Simulate generation
        return self._audio_buffer

    async def _generate_actor(self, recipe: dict[str, Any], seed: int) -> torch.Tensor:
        """Generate actor image using Flux-Schnell.

        Args:
            recipe: Recipe with visual_track section
            seed: Deterministic seed

        Returns:
            Actor image tensor
        """
        assert self._actor_buffer is not None
        # TODO(T015): Replace with real Flux-Schnell implementation
        # Flux should use seed for deterministic output
        # In real implementation: flux.generate(..., seed=seed)
        await asyncio.sleep(0.1)  # Simulate generation
        return self._actor_buffer

    async def _generate_video(
        self,
        actor_img: torch.Tensor,
        audio: torch.Tensor,
        recipe: dict[str, Any],
        seed: int,
    ) -> torch.Tensor:
        """Generate video using LivePortrait warping.

        Args:
            actor_img: Base actor image
            audio: Audio waveform for lip sync
            recipe: Recipe with visual_track overrides
            seed: Deterministic seed

        Returns:
            Video frames tensor
        """
        assert self._model_registry is not None
        assert self._video_buffer is not None

        liveportrait = self._model_registry.get_model("liveportrait")

        visual_track = recipe.get("visual_track", {})
        expression_sequence = visual_track.get("expression_sequence")
        expression_preset = visual_track.get("expression_preset", "neutral")
        driving_source = visual_track.get("driving_source")

        slot_params = recipe.get("slot_params", {})
        duration = slot_params.get("duration_sec", 45)
        fps = slot_params.get("fps", 24)

        if actor_img.dim() == 4 and actor_img.shape[0] == 1:
            actor_img = actor_img[0]

        if not hasattr(liveportrait, "animate"):
            logger.warning("LivePortrait model missing animate(); using zeroed buffer")
            self._video_buffer.zero_()
            return self._video_buffer

        # Check if pre-allocated buffer matches requested frame count
        expected_frames = int(duration * fps)
        output_buffer = None
        if self._video_buffer is not None and self._video_buffer.shape[0] == expected_frames:
            output_buffer = self._video_buffer
        else:
            logger.debug(
                "Buffer size mismatch: buffer=%s, expected=%d frames. LivePortrait will allocate.",
                self._video_buffer.shape[0] if self._video_buffer is not None else None,
                expected_frames,
            )

        # LivePortrait should use seed for deterministic warping
        video = liveportrait.animate(
            source_image=actor_img,
            driving_audio=audio,
            expression_preset=expression_preset,
            expression_sequence=expression_sequence,
            fps=fps,
            duration=duration,
            output=output_buffer,
            driving_source=Path(driving_source) if driving_source else None,
            seed=seed,  # Pass seed for determinism
        )

        if not isinstance(video, torch.Tensor):
            logger.warning("LivePortrait animate() returned non-tensor; using buffer")
            return self._video_buffer

        return video

    async def _verify_semantic(
        self, video: torch.Tensor, recipe: dict[str, Any]
    ) -> torch.Tensor:
        """Dual CLIP semantic verification.

        Args:
            video: Generated video frames
            recipe: Recipe with semantic_constraints

        Returns:
            Combined CLIP embedding (B-32 + L-14 ensemble)
        """
        # TODO(T018): Replace with real dual CLIP ensemble
        await asyncio.sleep(0.05)  # Simulate verification
        return torch.randn(512, device=self._device, dtype=torch.float32)

    def compute_determinism_proof(
        self,
        recipe: dict[str, Any],
        seed: int,
        result: RenderResult,
    ) -> bytes:
        """Compute determinism proof hash for BFT consensus.

        The proof is SHA256(canonical_recipe || seed || output_hash) where:
        - canonical_recipe: JSON-serialized recipe with sorted keys
        - seed: 8-byte little-endian integer
        - output_hash: SHA256 of concatenated video/audio tensor bytes

        Args:
            recipe: The recipe dict used for generation
            seed: The seed used for generation
            result: The RenderResult from generation

        Returns:
            32-byte SHA256 hash
        """
        hasher = hashlib.sha256()

        # Canonical recipe (sorted keys, consistent formatting)
        canonical_recipe = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
        hasher.update(canonical_recipe.encode("utf-8"))

        # Seed as 8-byte little-endian
        hasher.update(seed.to_bytes(8, byteorder="little", signed=False))

        # Output hash (video frames + audio waveform)
        if result.video_frames.numel() > 0:
            # Move to CPU for hashing
            video_bytes = result.video_frames.detach().cpu().numpy().tobytes()
            hasher.update(hashlib.sha256(video_bytes).digest())

        if result.audio_waveform.numel() > 0:
            audio_bytes = result.audio_waveform.detach().cpu().numpy().tobytes()
            hasher.update(hashlib.sha256(audio_bytes).digest())

        return hasher.digest()

    async def health_check(self) -> bool:
        """Verify renderer is healthy and models are loaded.

        Returns:
            True if renderer is ready to accept render() calls
        """
        if not self._initialized:
            return False

        if self._model_registry is None:
            return False

        # Check all required models are loaded
        required_models: list[ModelName] = ["flux", "liveportrait", "kokoro", "clip_b", "clip_l"]
        for model_name in required_models:
            if model_name not in self._model_registry:
                logger.warning(f"Health check failed: model '{model_name}' not loaded")
                return False

        # Check VRAM is within limits
        try:
            assert self._vram_monitor is not None
            self._vram_monitor.check()
        except MemoryPressureError:
            logger.warning("Health check failed: VRAM pressure")
            return False

        return True
