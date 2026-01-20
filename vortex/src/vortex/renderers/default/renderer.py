"""Default Lane 0 renderer implementation.

This renderer wraps the existing Flux-Schnell, LivePortrait, Kokoro, and dual CLIP
pipeline as a DeterministicVideoRenderer. It serves as the reference implementation
for Lane 0 video generation and maintains backward compatibility with the existing
VortexPipeline behavior.

Architecture:
    - Flux-Schnell (NF4, 6.0GB): Generates actor images from prompts
    - Kokoro-82M (FP32, 0.4GB): Text-to-speech synthesis
    - LivePortrait (FP16, 3.5GB): Animates images using audio-gated motion driver
    - CLIP ViT-B-32 + ViT-L-14 (FP16, 0.6GB): Dual ensemble semantic verification

Pipeline Flow:
    Recipe -> Kokoro (TTS) -> Audio
          -> Flux (image) -> Actor
    Actor + Audio -> LivePortrait (audio-gated) -> Video
    Video -> CLIP (verify) -> Embedding

VRAM Modes:
    - Static residency (default): All models stay on GPU (~10GB peak)
    - Sequential offloading: Models moved to CPU between stages (~6.5GB peak)

The sequential offloading mode is automatically enabled for GPUs with <12GB VRAM.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from vortex.models import ModelName, load_model
from vortex.models.clip_ensemble import DualClipResult, load_clip_ensemble
from vortex.renderers.base import DeterministicVideoRenderer
from vortex.renderers.recipe_schema import merge_with_defaults, validate_recipe
from vortex.renderers.types import RendererManifest, RenderResult
from vortex.utils.memory import get_current_vram_usage, get_vram_stats, log_vram_snapshot
from vortex.utils.offloader import ModelOffloader

logger = logging.getLogger(__name__)


class VortexInitializationError(Exception):
    """Raised when renderer initialization fails."""

    pass


class MemoryPressureError(Exception):
    """Raised when VRAM usage exceeds hard limit."""

    pass


class _ModelRegistry:
    """Registry for loaded models with get_model() interface.

    Supports two VRAM modes:
    - Static residency: All models stay on GPU (default for >=12GB VRAM)
    - Sequential offloading: Models moved to CPU between stages (<12GB VRAM)
    """

    def __init__(
        self,
        device: str,
        precision_overrides: dict[ModelName, str] | None = None,
        offloading_mode: str = "auto",
        cache_dir: str | None = None,
        local_only: bool = False,
    ):
        self.device = device
        self.precision_overrides = precision_overrides or {}
        self._models: dict[ModelName, nn.Module] = {}
        self._offloading_mode = offloading_mode
        self._offloader: Optional[ModelOffloader] = None
        self._cache_dir = cache_dir
        self._local_only = local_only

        # Determine if offloading should be enabled
        self._offloading_enabled = self._should_enable_offloading()

        if self._offloading_enabled:
            self._offloader = ModelOffloader(gpu_device=device, enabled=True)
            logger.info(
                "Model offloading ENABLED (sequential loading to reduce VRAM)",
                extra={"mode": offloading_mode}
            )
        else:
            logger.info(
                "Model offloading DISABLED (static residency)",
                extra={"mode": offloading_mode}
            )

    def _should_enable_offloading(self) -> bool:
        """Determine if offloading should be enabled based on mode and VRAM."""
        if self._offloading_mode == "always":
            return True
        if self._offloading_mode == "never":
            return False
        # Auto mode: enable if VRAM < 12GB
        if torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            should_offload = total_vram_gb < 12.0
            logger.info(
                f"Auto offloading: VRAM={total_vram_gb:.1f}GB, offloading={'enabled' if should_offload else 'disabled'}"
            )
            return should_offload
        return False

    def load_all_models(self) -> None:
        """Load all models into registry."""
        # All required models for audio-gated pipeline
        single_models: list[ModelName] = ["flux", "liveportrait", "kokoro"]

        try:
            for name in single_models:
                logger.info(f"Loading model: {name}")
                precision = self.precision_overrides.get(name)

                # With offloading, load to CPU first to avoid VRAM spike
                if self._offloading_enabled:
                    model = load_model(
                        name,
                        device="cpu",
                        precision=precision,
                        cache_dir=self._cache_dir,
                        local_only=self._local_only,
                    )
                    self._models[name] = model
                    if self._offloader:
                        self._offloader.register(name, model, initial_device="cpu")
                else:
                    model = load_model(
                        name,
                        device=self.device,
                        precision=precision,
                        cache_dir=self._cache_dir,
                        local_only=self._local_only,
                    )
                    self._models[name] = model

                log_vram_snapshot(f"after_{name}_load")

            # Load ClipEnsemble
            logger.info("Loading model: clip_ensemble")
            clip_precision = self.precision_overrides.get("clip_b", "fp16")

            # Note: clip_ensemble uses its own cache at ~/.cache/vortex/clip
            # (managed by download_and_quantize_clip.py script), so we don't
            # pass the shared HuggingFace cache_dir here.
            if self._offloading_enabled:
                clip_ensemble = load_clip_ensemble(
                    device="cpu",
                    precision=clip_precision,
                    local_only=self._local_only,
                )
                self._models["clip_ensemble"] = clip_ensemble
                if self._offloader:
                    self._offloader.register("clip_ensemble", clip_ensemble, initial_device="cpu")
            else:
                clip_ensemble = load_clip_ensemble(
                    device=self.device,
                    precision=clip_precision,
                    local_only=self._local_only,
                )
                self._models["clip_ensemble"] = clip_ensemble

            log_vram_snapshot("after_clip_ensemble_load")

            logger.info(
                "All models loaded successfully",
                extra={
                    "total_models": len(self._models),
                    "vram_gb": get_vram_stats()["allocated_gb"],
                    "offloading_enabled": self._offloading_enabled,
                },
            )

        except torch.cuda.OutOfMemoryError as e:
            stats = get_vram_stats()
            error_msg = (
                f"CUDA OOM during model loading. "
                f"Allocated: {stats['allocated_gb']:.2f}GB, "
                f"Total: {stats['total_gb']:.2f}GB. "
                f"Try enabling offloading with models.offloading='always' in config."
            )
            logger.error(error_msg, exc_info=True)
            self._models.clear()
            raise VortexInitializationError(error_msg) from e

    def get_model(self, name: ModelName) -> nn.Module:
        """Get a loaded model by name."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        return self._models[name]

    def prepare_for_stage(self, stage: str) -> None:
        """Prepare models for a pipeline stage (offload others if enabled).

        Args:
            stage: Stage name ("audio", "image", "video", "clip")
        """
        if not self._offloading_enabled or not self._offloader:
            return

        # Map stages to required models
        stage_models = {
            "audio": "kokoro",
            "image": "flux",
            "video": "liveportrait",
            "clip": "clip_ensemble",
        }

        model_name = stage_models.get(stage)
        if model_name:
            logger.info(f"Preparing for stage '{stage}': loading {model_name} to GPU")
            self._offloader.offload_all_except(model_name)

    def __contains__(self, name: ModelName) -> bool:
        return name in self._models

    @property
    def offloading_enabled(self) -> bool:
        """Check if offloading is enabled."""
        return self._offloading_enabled


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
        models_config = config.get("models", {})
        precision_overrides = models_config.get("precision", {})
        offloading_mode = models_config.get("offloading", "auto")
        cache_dir = models_config.get("cache_dir")
        cache_dir_path = Path(cache_dir).expanduser() if cache_dir else None
        cache_dir_str = str(cache_dir_path) if cache_dir_path else None
        local_only = bool(models_config.get("local_only", False))

        self._model_registry = _ModelRegistry(
            device,
            precision_overrides,
            offloading_mode=offloading_mode,
            cache_dir=cache_dir_str,
            local_only=local_only,
        )
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

        # Video buffer - reduced from 1080 to 300 frames (10s @ 30fps)
        # This saves ~2.5GB VRAM for LivePortrait CLI subprocess
        video_cfg = buf_cfg.get("video", {})
        self._video_buffer = torch.zeros(
            video_cfg.get("frames", 300),  # 10 seconds @ 30fps (was 1080 = 45s @ 24fps)
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
            assert self._model_registry is not None
            self._vram_monitor.check()

            # Set deterministic seed for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Check if offloading is enabled (affects parallelization strategy)
            offloading_enabled = self._model_registry.offloading_enabled

            if offloading_enabled:
                # Sequential mode: one model on GPU at a time
                logger.info("Using sequential generation (offloading enabled)")

                # Phase 1a: Audio generation (Kokoro)
                self._model_registry.prepare_for_stage("audio")
                audio_result = await self._generate_audio(recipe, seed)

                # Phase 1b: Actor image generation (Flux)
                self._model_registry.prepare_for_stage("image")
                actor_result = await self._generate_actor(recipe, seed)
            else:
                # Parallel mode: all models resident (default for >=12GB VRAM)
                logger.info("Using parallel generation (static residency)")
                audio_task = asyncio.create_task(self._generate_audio(recipe, seed))
                actor_task = asyncio.create_task(self._generate_actor(recipe, seed))
                audio_result, actor_result = await asyncio.gather(audio_task, actor_task)

            # Phase 1c: Lip inpainting (optional, creates mouth void for animation)
            lip_inpaint_cfg = self._config.get("quality", {}).get("lip_inpainting", {})
            if lip_inpaint_cfg.get("enabled", False):
                actor_result = await self._inpaint_lips(actor_result, recipe, seed)

            # Check deadline after audio + actor phase
            time_remaining = deadline - time.time()
            if time_remaining < 15.0:  # Need at least 15s for motion + video + CLIP
                raise TimeoutError(
                    f"Deadline would be exceeded: {time_remaining:.1f}s remaining"
                )

            # Phase 2: Video rendering with audio-gated driver
            # (Combines motion generation + rendering in one step)
            if offloading_enabled:
                self._model_registry.prepare_for_stage("video")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            video_result = await self._generate_video_gated(
                actor_result, audio_result, recipe, seed
            )
            video_result = self._apply_cinematic_shake(video_result)

            # Check deadline before CLIP
            time_remaining = deadline - time.time()
            if time_remaining < 2.0:  # Need at least 2s for CLIP
                raise TimeoutError(
                    f"Deadline would be exceeded: {time_remaining:.1f}s remaining"
                )

            # Phase 4: CLIP verification
            if offloading_enabled:
                self._model_registry.prepare_for_stage("clip")
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
            Audio waveform tensor of shape (samples,) at 24kHz
        """
        assert self._model_registry is not None
        assert self._audio_buffer is not None

        # Get Kokoro model from registry
        kokoro = self._model_registry.get_model("kokoro")

        # Extract audio parameters from recipe
        audio_track = recipe.get("audio_track", {})
        script = audio_track.get("script", "")
        voice_id = audio_track.get("voice_id", "rick_c137")
        speed = audio_track.get("speed", 1.0)
        emotion = audio_track.get("emotion", "neutral")

        # Validate script is not empty
        if not script or not script.strip():
            logger.warning("Empty script provided, returning zeroed audio buffer")
            self._audio_buffer.zero_()
            return self._audio_buffer

        # Check if model has synthesize method (real vs mock)
        if not hasattr(kokoro, "synthesize"):
            logger.warning("Kokoro model missing synthesize(); using zeroed buffer")
            self._audio_buffer.zero_()
            return self._audio_buffer

        # Generate audio with deterministic seed
        try:
            audio = kokoro.synthesize(
                text=script,
                voice_id=voice_id,
                speed=speed,
                emotion=emotion,
                output=self._audio_buffer,
                seed=seed,
            )
            return audio
        except ValueError as e:
            logger.warning(f"Kokoro synthesis failed: {e}, using zeroed buffer")
            self._audio_buffer.zero_()
            return self._audio_buffer
        except Exception as e:
            logger.error(f"Unexpected Kokoro error: {e}", exc_info=True)
            self._audio_buffer.zero_()
            return self._audio_buffer

    async def _generate_actor(self, recipe: dict[str, Any], seed: int) -> torch.Tensor:
        """Generate actor image using Flux-Schnell.

        Args:
            recipe: Recipe with visual_track section
            seed: Deterministic seed

        Returns:
            Actor image tensor of shape [1, 3, 512, 512]
        """
        assert self._model_registry is not None
        assert self._actor_buffer is not None

        # Get Flux model from registry
        flux = self._model_registry.get_model("flux")

        # Extract visual parameters from recipe
        visual_track = recipe.get("visual_track", {})
        prompt = visual_track.get("prompt", "")
        negative_prompt = visual_track.get("negative_prompt", "")

        quality_cfg = self._config.get("quality", {})
        prompt_cfg = quality_cfg.get("prompt_steering", {})
        if prompt_cfg.get("enabled", True):
            system_prefix = prompt_cfg.get(
                "positive_prefix",
                "medium shot, looking at viewer, symmetrical face, centered composition, eye contact, ",
            )
            system_negative = prompt_cfg.get(
                "negative_prefix",
                "profile view, side view, looking away, skewed, distorted, back of head, asymmetrical",
            )
            prompt = f"{system_prefix}{prompt}".strip()
            negative_prompt = f"{system_negative}, {negative_prompt}".strip(", ")

        # Validate prompt is not empty
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided, returning zeroed actor buffer")
            self._actor_buffer.zero_()
            return self._actor_buffer

        # Check if model has generate method (real vs mock)
        if not hasattr(flux, "generate"):
            logger.warning("Flux model missing generate(); using zeroed buffer")
            self._actor_buffer.zero_()
            return self._actor_buffer

        # Generate actor image with deterministic seed
        try:
            # Flux expects output buffer shape [3, 512, 512]
            output_buffer = (
                self._actor_buffer.squeeze(0)
                if self._actor_buffer.dim() == 4
                else self._actor_buffer
            )

            image = flux.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=4,  # Schnell fast variant
                guidance_scale=0.0,  # Unconditional for speed
                output=output_buffer,
                seed=seed,
            )

            # Ensure consistent return shape [1, 3, 512, 512]
            if image.dim() == 3:
                return image.unsqueeze(0)
            return image

        except ValueError as e:
            logger.warning(f"Flux generation failed: {e}, using zeroed buffer")
            self._actor_buffer.zero_()
            return self._actor_buffer
        except Exception as e:
            logger.error(f"Unexpected Flux error: {e}", exc_info=True)
            self._actor_buffer.zero_()
            return self._actor_buffer

    async def _inpaint_lips(
        self,
        actor_img: torch.Tensor,
        recipe: dict[str, Any],
        seed: int,
    ) -> torch.Tensor:
        """Inpaint lip region to create mouth interior texture.

        This creates a "mouth void" in sealed-lip images before LivePortrait
        animation, preventing the "rubber mask" artifact where lip skin
        stretches instead of revealing teeth/cavity.

        Args:
            actor_img: Actor image tensor [1, 3, H, W]
            recipe: Recipe with visual_track
            seed: Deterministic seed

        Returns:
            Inpainted actor image tensor [1, 3, H, W]
        """
        assert self._model_registry is not None

        lip_cfg = self._config.get("quality", {}).get("lip_inpainting", {})

        # Import face landmarks utility
        try:
            from vortex.utils.face_landmarks import create_mouth_void_mask
        except ImportError as e:
            logger.warning(f"Face landmarks not available: {e}, skipping lip inpainting")
            return actor_img

        # Get Flux model for inpainting
        try:
            flux = self._model_registry.get_model("flux")
        except KeyError:
            logger.warning("Flux model not available for inpainting, skipping")
            return actor_img

        if not hasattr(flux, "inpaint_region"):
            logger.warning("Flux model missing inpaint_region(), skipping lip inpainting")
            return actor_img

        # Ensure correct shape for processing
        img = actor_img.squeeze(0) if actor_img.dim() == 4 else actor_img  # [3, H, W]

        # Create lip mask from landmarks
        dilation_px = int(lip_cfg.get("dilation_px", 8))
        mask = create_mouth_void_mask(img, device=self._device, dilation_px=dilation_px)

        if mask is None:
            logger.warning("Face detection failed, skipping lip inpainting")
            return actor_img

        # Inpainting prompt
        inpaint_prompt = lip_cfg.get(
            "prompt",
            "slightly parted lips, visible teeth, natural shadow"
        )
        denoising_strength = float(lip_cfg.get("denoising_strength", 0.45))

        logger.info(
            "Inpainting lip region",
            extra={
                "mask_coverage": float(mask.mean()),
                "denoising_strength": denoising_strength,
            }
        )

        try:
            inpainted = flux.inpaint_region(
                image=img,
                mask=mask,
                prompt=inpaint_prompt,
                denoising_strength=denoising_strength,
                seed=seed,
            )

            # Restore batch dimension
            if inpainted.dim() == 3:
                inpainted = inpainted.unsqueeze(0)

            return inpainted

        except Exception as e:
            logger.warning(f"Lip inpainting failed: {e}, using original image")
            return actor_img

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

        driver_audio = audio
        quality_cfg = self._config.get("quality", {})
        audio_cfg = quality_cfg.get("driver_audio", {})
        if audio_cfg.get("enabled", True):
            driver_audio = self._process_audio_for_driver(audio)

        driver_sample_rate = int(audio_cfg.get("sample_rate", 16000))

        # LivePortrait should use seed for deterministic warping
        video = liveportrait.animate(
            source_image=actor_img,
            driving_audio=driver_audio,
            expression_preset=expression_preset,
            expression_sequence=expression_sequence,
            fps=fps,
            duration=duration,
            output=output_buffer,
            driving_source=Path(driving_source) if driving_source else None,
            seed=seed,  # Pass seed for determinism
            driver_sample_rate=driver_sample_rate,
        )

        if not isinstance(video, torch.Tensor):
            logger.warning("LivePortrait animate() returned non-tensor; using buffer")
            return self._video_buffer

        return video

    async def _generate_video_gated(
        self,
        actor_img: torch.Tensor,
        audio: torch.Tensor,
        recipe: dict[str, Any],
        seed: int,
    ) -> torch.Tensor:
        """Generate video using audio-gated motion driver.

        Uses a template-based approach with audio envelope gating
        for reliable lip-sync animation.

        Args:
            actor_img: Actor image tensor [1, 3, 512, 512] or [3, 512, 512]
            audio: Audio waveform tensor at 24kHz
            recipe: Recipe with slot_params
            seed: Deterministic seed

        Returns:
            Video frames tensor [num_frames, 3, 512, 512]
        """
        assert self._model_registry is not None
        assert self._video_buffer is not None

        liveportrait = self._model_registry.get_model("liveportrait")

        slot_params = recipe.get("slot_params", {})
        fps = slot_params.get("fps", 24)

        # Ensure correct shape
        if actor_img.dim() == 4 and actor_img.shape[0] == 1:
            actor_img = actor_img[0]

        if not hasattr(liveportrait, "animate_gated"):
            logger.warning(
                "LivePortrait missing animate_gated(); falling back to viseme-based"
            )
            return await self._generate_video(
                actor_img.unsqueeze(0),
                audio,
                recipe,
                seed,
            )

        # Save audio to temp file for librosa
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name
            sf.write(audio_path, audio.cpu().numpy(), 24000)

        try:
            # Detect animation style from recipe
            visual_track = recipe.get("visual_track", {})
            style = self._detect_animation_style(visual_track)

            logger.info(
                "Generating video with audio-gated driver",
                extra={"fps": fps, "audio_samples": audio.shape[0], "style": style},
            )

            video = liveportrait.animate_gated(
                source_image=actor_img,
                audio_path=audio_path,
                fps=fps,
                template_name="d7",
                style=style,
            )

            if not isinstance(video, torch.Tensor):
                logger.warning("animate_gated returned non-tensor")
                return self._video_buffer

            return video

        finally:
            # Cleanup temp file
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def _process_audio_for_driver(self, audio_24k: torch.Tensor) -> torch.Tensor:
        """Prepare audio for 16kHz LivePortrait driving."""
        quality_cfg = self._config.get("quality", {})
        audio_cfg = quality_cfg.get("driver_audio", {})
        pad_seconds = float(audio_cfg.get("pad_silence_sec", 0.2))
        target_sample_rate = int(audio_cfg.get("sample_rate", 16000))

        pad_samples = int(pad_seconds * 24000)
        padded = F.pad(audio_24k, (pad_samples, pad_samples), mode="constant", value=0.0)

        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for driver audio resampling") from exc

        resampler = torchaudio.transforms.Resample(
            orig_freq=24000,
            new_freq=target_sample_rate,
        ).to(padded.device)
        return resampler(padded.unsqueeze(0)).squeeze(0)

    def _detect_animation_style(self, visual_track: dict[str, Any]) -> str:
        """Detect animation style from recipe or prompt keywords.

        Precedence: Explicit override > Auto-detect > Default (realistic)

        Args:
            visual_track: Visual track section of recipe

        Returns:
            Animation style: "realistic", "cartoon", or "exaggerated"
        """
        # 1. Check explicit override
        if "animation_style" in visual_track:
            return visual_track["animation_style"]

        # 2. Auto-detect from prompt keywords
        prompt = visual_track.get("prompt", "").lower()
        cartoon_keywords = {
            "cartoon", "anime", "illustration", "drawing", "sketch",
            "vector art", "pixar", "disney", "animated style", "2d",
            "cel shaded", "flat style", "caricature"
        }

        if any(k in prompt for k in cartoon_keywords):
            return "cartoon"

        return "realistic"

    def _apply_cinematic_shake(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Apply subtle handheld-style motion to reduce static background feel."""
        quality_cfg = self._config.get("quality", {})
        shake_cfg = quality_cfg.get("camera_shake", {})
        if not shake_cfg.get("enabled", True):
            return video_frames

        strength = float(shake_cfg.get("strength", 0.002))
        zoom = float(shake_cfg.get("zoom", 0.002))

        t, c, h, w = video_frames.shape
        time_axis = torch.linspace(0, t / 24.0, t, device=video_frames.device)
        tx = strength * torch.sin(time_axis * 1.5) + (strength * 0.5) * torch.sin(time_axis * 3.7)
        ty = (strength * 0.5) * torch.cos(time_axis * 1.2)
        scale = 1.0 + zoom * torch.sin(time_axis * 0.5)

        theta = torch.zeros((t, 2, 3), device=video_frames.device, dtype=video_frames.dtype)
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, video_frames.size(), align_corners=False)
        return F.grid_sample(
            video_frames,
            grid,
            align_corners=False,
            padding_mode="reflection",
        )

    async def _verify_semantic(
        self, video: torch.Tensor, recipe: dict[str, Any]
    ) -> torch.Tensor:
        """Dual CLIP semantic verification.

        Args:
            video: Generated video frames tensor [T, C, H, W]
            recipe: Recipe with visual_track (prompt) and semantic_constraints

        Returns:
            Combined CLIP embedding (512-dim, L2-normalized) for BFT consensus
        """
        assert self._model_registry is not None

        # Get ClipEnsemble from registry
        try:
            clip_ensemble = self._model_registry.get_model("clip_ensemble")
        except KeyError:
            logger.warning("ClipEnsemble not loaded; returning random embedding")
            return torch.randn(512, device=self._device, dtype=torch.float32)

        # Extract parameters from recipe
        visual_track = recipe.get("visual_track", {})
        prompt = visual_track.get("prompt", "")
        semantic_constraints = recipe.get("semantic_constraints", {})
        clip_threshold = semantic_constraints.get("clip_threshold", 0.70)

        # Validate inputs
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt for CLIP verification; returning random embedding")
            return torch.randn(512, device=self._device, dtype=torch.float32)

        if not hasattr(clip_ensemble, "verify"):
            logger.warning("ClipEnsemble missing verify(); returning random embedding")
            return torch.randn(512, device=self._device, dtype=torch.float32)

        if video.numel() == 0:
            logger.warning("Empty video frames for CLIP verification; returning random embedding")
            return torch.randn(512, device=self._device, dtype=torch.float32)

        # Perform semantic verification
        try:
            result: DualClipResult = clip_ensemble.verify(
                video_frames=video,
                prompt=prompt,
                threshold=clip_threshold,
                seed=None,  # Already seeded at render() level
            )

            # Log verification results
            logger.info(
                "CLIP verification completed",
                extra={
                    "score_b": result.score_clip_b,
                    "score_l": result.score_clip_l,
                    "ensemble_score": result.ensemble_score,
                    "self_check_passed": result.self_check_passed,
                    "outlier_detected": result.outlier_detected,
                },
            )

            if not result.self_check_passed:
                logger.warning("CLIP self-check FAILED: video may not match prompt semantically")
            if result.outlier_detected:
                logger.warning("CLIP outlier detected: potential adversarial content")

            # Return the embedding for BFT consensus
            embedding = result.embedding
            if embedding.device != self._device:
                embedding = embedding.to(self._device)

            return embedding

        except ValueError as e:
            logger.warning(f"CLIP verification failed: {e}")
            return torch.randn(512, device=self._device, dtype=torch.float32)
        except RuntimeError as e:
            logger.error(f"CLIP verification runtime error: {e}", exc_info=True)
            return torch.randn(512, device=self._device, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Unexpected CLIP error: {e}", exc_info=True)
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
        required_models = ["flux", "liveportrait", "kokoro", "clip_ensemble"]
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
