"""Default Lane 0 renderer implementation with Narrative Chain architecture.

This renderer implements the Narrative Chain pipeline for Lane 0 video generation:
1. Showrunner (LLM) generates comedic script
2. Kokoro synthesizes script to audio (sets duration)
3. Flux generates keyframe image
4. CogVideoX animates keyframe to video
5. CLIP verifies semantic consistency

Architecture:
    - Showrunner (external Ollama): Generates script with setup/punchline/visual_prompt
    - Kokoro-82M (FP32, 0.4GB): Text-to-speech synthesis at 24kHz
    - Flux-Schnell (NF4, 6.0GB): Generates keyframe images from prompts
    - CogVideoX-5B (INT8, ~10-11GB): Generates video from keyframe + prompt
    - CLIP ViT-B-32 + ViT-L-14 (FP16, 0.6GB): Dual ensemble semantic verification

Pipeline Flow:
    Recipe -> Showrunner (script) -> Kokoro (TTS) -> Audio
                                  -> Flux (keyframe) -> Image
    Image + Prompt -> CogVideoX (video) -> Video
    Video -> CLIP (verify) -> Embedding

VRAM Budget: ~11GB peak during CogVideoX phase
Target Duration: 10-15 seconds at 8fps
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from vortex.models.clip_ensemble import ClipEnsemble, DualClipResult
from vortex.models.cogvideox import CogVideoXModel, VideoGenerationConfig
from vortex.models.kokoro import load_kokoro
from vortex.models.showrunner import Script, Showrunner
from vortex.renderers.base import DeterministicVideoRenderer
from vortex.renderers.recipe_schema import merge_with_defaults, validate_recipe
from vortex.renderers.types import RendererManifest, RenderResult
from vortex.utils.memory import get_current_vram_usage, get_vram_stats, log_vram_snapshot

# Type alias for model names - used for type hints
ModelName = str

logger = logging.getLogger(__name__)


class VortexInitializationError(Exception):
    """Raised when renderer initialization fails."""

    pass


class MemoryPressureError(Exception):
    """Raised when VRAM usage exceeds hard limit."""

    pass


class _ModelRegistry:
    """Registry for loaded models with sequential loading for Narrative Chain.

    The Narrative Chain pipeline requires sequential model loading due to
    CogVideoX's large VRAM footprint (~10-11GB with INT8 quantization).
    Models are loaded to CPU and moved to GPU only when needed.

    Models:
    - flux: Keyframe image generation (6GB)
    - kokoro: TTS audio synthesis (0.4GB)
    - cogvideox: Video generation (10-11GB with INT8)
    - clip_ensemble: Semantic verification (0.6GB)
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
        self._models: dict[str, nn.Module | CogVideoXModel] = {}
        self._offloading_mode = offloading_mode
        self._cache_dir = cache_dir
        self._local_only = local_only

        # CogVideoX is handled separately (not an nn.Module)
        self._cogvideox: CogVideoXModel | None = None

        # Determine if offloading should be enabled
        self._offloading_enabled = self._should_enable_offloading()

        if self._offloading_enabled:
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
        # Auto mode: enable if VRAM < 16GB (CogVideoX needs ~10-11GB)
        if torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            should_offload = total_vram_gb < 16.0
            logger.info(
                f"Auto offloading: VRAM={total_vram_gb:.1f}GB, "
                f"offloading={'enabled' if should_offload else 'disabled'}"
            )
            return should_offload
        return False

    def load_all_models(self) -> None:
        """Load all models into registry.

        For Narrative Chain, we load:
        - flux (keyframe generation) - placeholder until implemented
        - kokoro (TTS)
        - clip_ensemble (verification) - placeholder until implemented
        - cogvideox (video generation) - handled separately
        """
        try:
            # Load Kokoro TTS
            logger.info("Loading model: kokoro")
            target_device = "cpu" if self._offloading_enabled else self.device
            kokoro = load_kokoro(device=target_device)
            self._models["kokoro"] = kokoro
            log_vram_snapshot("after_kokoro_load")

            # Load Flux-Schnell keyframe generator
            from vortex.models.flux import FluxModel
            logger.info("Loading model: flux")
            flux = FluxModel(device=self.device, cache_dir=self._cache_dir)
            # Lazy load - will load when first used (via generate() method)
            self._models["flux"] = flux
            log_vram_snapshot("after_flux_load")

            # CLIP ensemble placeholder - will be implemented in Phase 4.4
            logger.info("Initializing CLIP ensemble placeholder (not yet implemented)")
            self._models["clip_ensemble"] = ClipEnsemble(device=self.device)
            log_vram_snapshot("after_clip_ensemble_load")

            # Initialize CogVideoX (lazy-loaded, doesn't load weights yet)
            logger.info("Initializing CogVideoX model wrapper")
            self._cogvideox = CogVideoXModel(
                device=self.device,
                enable_cpu_offload=True,  # Always use CPU offload for CogVideoX
                cache_dir=self._cache_dir,
            )
            # Note: CogVideoX.load() is called lazily when needed
            self._models["cogvideox"] = self._cogvideox

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

    def get_model(self, name: str) -> nn.Module | CogVideoXModel:
        """Get a loaded model by name."""
        if name not in self._models:
            raise KeyError(
                f"Model '{name}' not found. Available: {list(self._models.keys())}"
            )
        return self._models[name]

    def get_cogvideox(self) -> CogVideoXModel:
        """Get the CogVideoX model instance."""
        if self._cogvideox is None:
            raise RuntimeError("CogVideoX not initialized")
        return self._cogvideox

    def prepare_for_stage(self, stage: str) -> None:
        """Prepare models for a pipeline stage.

        Note: Full offloading support will be implemented in a future phase.
        Currently, this method just logs the stage transition.

        Args:
            stage: Stage name ("audio", "image", "video", "clip")
        """
        if not self._offloading_enabled:
            return

        # Map stages to required models
        stage_models = {
            "audio": "kokoro",
            "image": "flux",
            "video": None,  # CogVideoX handles its own offloading
            "clip": "clip_ensemble",
        }

        model_name = stage_models.get(stage)
        if model_name:
            logger.info(f"Preparing for stage '{stage}': {model_name}")
            # Note: Full offloading implementation deferred to Phase 4.5
            # For now, CogVideoX handles its own CPU offloading internally

    def __contains__(self, name: str) -> bool:
        return name in self._models

    @property
    def offloading_enabled(self) -> bool:
        """Check if offloading is enabled."""
        return self._offloading_enabled


class _FluxPlaceholder(nn.Module):
    """Placeholder for Flux-Schnell keyframe generator.

    This placeholder will be replaced with the actual Flux implementation
    in Phase 4.3. For now, it generates random noise images to allow
    pipeline testing.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        logger.warning(
            "Using Flux placeholder - actual model not implemented yet"
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        output: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate a placeholder keyframe image.

        Args:
            prompt: Text prompt (logged but not used)
            negative_prompt: Negative prompt (ignored)
            num_inference_steps: Inference steps (ignored)
            guidance_scale: Guidance scale (ignored)
            output: Optional pre-allocated buffer
            seed: Random seed for reproducibility

        Returns:
            Random noise tensor [3, 512, 512]
        """
        logger.warning(f"Flux placeholder generating noise for prompt: {prompt[:50]}...")

        if seed is not None:
            torch.manual_seed(seed)

        # Generate random image with slight structure (colored noise)
        if output is not None:
            output.uniform_(0, 1)
            return output
        else:
            return torch.rand(3, 512, 512, device=self.device)


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
    """Default Lane 0 renderer using Narrative Chain architecture.

    This renderer implements the full Narrative Chain pipeline:
    1. Showrunner (LLM) generates script from theme
    2. Kokoro synthesizes script to speech
    3. Flux generates keyframe image
    4. CogVideoX generates video from keyframe
    5. CLIP verifies semantic consistency

    The pipeline is designed for 12GB VRAM GPUs with sequential model loading.

    Example:
        >>> renderer = DefaultRenderer()
        >>> await renderer.initialize("cuda:0", config)
        >>> result = await renderer.render(recipe, slot_id=1, seed=42, deadline=time.time() + 60)
    """

    def __init__(self, manifest: RendererManifest | None = None):
        """Initialize renderer.

        Args:
            manifest: Optional pre-loaded manifest (loaded from file if None)
        """
        self._manifest = manifest or self._load_manifest()
        self._model_registry: _ModelRegistry | None = None
        self._vram_monitor: _VRAMMonitor | None = None
        self._showrunner: Showrunner | None = None
        self._device: str = "cpu"
        self._config: dict[str, Any] = {}
        self._initialized = False

        # Pre-allocated buffers (set during initialize)
        self._actor_buffer: torch.Tensor | None = None
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

        # Initialize Showrunner (external LLM, no VRAM usage)
        llm_config = config.get("llm", {})
        self._showrunner = Showrunner(
            base_url=llm_config.get("base_url", "http://localhost:11434"),
            model=llm_config.get("model", "llama3:8b"),
            timeout=llm_config.get("timeout_s", 30.0),
        )

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

        # Actor buffer (512x512x3) for Flux output
        actor_cfg = buf_cfg.get("actor", {})
        self._actor_buffer = torch.zeros(
            1,
            actor_cfg.get("channels", 3),
            actor_cfg.get("height", 512),
            actor_cfg.get("width", 512),
            device=self._device,
            dtype=torch.float32,
        )

        # Audio buffer (24kHz * 15 seconds = 360000 samples max)
        audio_cfg = buf_cfg.get("audio", {})
        self._audio_buffer = torch.zeros(
            audio_cfg.get("samples", 360000),  # 15 seconds @ 24kHz
            device=self._device,
            dtype=torch.float32,
        )

        logger.info(
            "Output buffers pre-allocated",
            extra={
                "actor_shape": tuple(self._actor_buffer.shape),
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
        """Render a single slot from recipe using Narrative Chain pipeline.

        Pipeline Flow:
        1. Script Generation: Showrunner (LLM) generates script or use provided
        2. Audio Generation: Kokoro TTS synthesizes speech
        3. Keyframe Generation: Flux generates scene image
        4. Video Generation: CogVideoX animates keyframe
        5. CLIP Verification: Dual ensemble semantic check

        Args:
            recipe: Standardized recipe dict (see recipe_schema.py)
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
            assert self._showrunner is not None
            self._vram_monitor.check()

            # Set deterministic seed for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Phase 1: Script Generation (Showrunner - external, no VRAM)
            script = await self._generate_script(recipe, seed)
            logger.info(
                "Script generated",
                extra={
                    "setup_length": len(script.setup),
                    "punchline_length": len(script.punchline),
                    "visual_prompt_length": len(script.visual_prompt),
                },
            )

            # Unload Ollama model to free VRAM for subsequent stages
            await self._showrunner.unload_model()

            # Check deadline after script generation
            time_remaining = deadline - time.time()
            if time_remaining < 50.0:
                raise TimeoutError(
                    f"Deadline would be exceeded: {time_remaining:.1f}s remaining"
                )

            # Phase 2: Audio Generation (Kokoro ~0.4GB)
            self._model_registry.prepare_for_stage("audio")
            full_script = f"{script.setup} {script.punchline}"
            audio_result = await self._generate_audio(full_script, recipe, seed)
            logger.info(
                "Audio generated",
                extra={"audio_samples": audio_result.shape[0]},
            )

            # Phase 3: Keyframe Generation (Flux ~6GB)
            self._model_registry.prepare_for_stage("image")
            # Combine visual_prompt with video.style_prompt
            video_config = recipe.get("video", {})
            style_prompt = video_config.get("style_prompt", "")
            combined_prompt = f"{script.visual_prompt}, {style_prompt}".strip(", ")
            keyframe = await self._generate_keyframe(combined_prompt, recipe, seed)
            logger.info(
                "Keyframe generated",
                extra={"keyframe_shape": list(keyframe.shape)},
            )

            # Unload Flux before CogVideoX (which needs ~10-11GB)
            # This is critical to avoid VRAM fragmentation
            if self._model_registry.offloading_enabled:
                flux = self._model_registry.get_model("flux")
                if hasattr(flux, "unload"):
                    logger.info("Unloading Flux model before CogVideoX")
                    flux.unload()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

            # Check deadline before video generation
            time_remaining = deadline - time.time()
            if time_remaining < 30.0:
                raise TimeoutError(
                    f"Deadline would be exceeded: {time_remaining:.1f}s remaining"
                )

            # Phase 4: Video Generation (CogVideoX ~10-11GB)
            self._model_registry.prepare_for_stage("video")
            video_result = await self._generate_video(
                keyframe, script.visual_prompt, recipe, seed
            )
            logger.info(
                "Video generated",
                extra={
                    "video_frames": video_result.shape[0],
                    "video_shape": list(video_result.shape),
                },
            )

            # Unload CogVideoX before CLIP to free VRAM
            cogvideox = self._model_registry.get_cogvideox()
            cogvideox.unload()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Check deadline before CLIP
            time_remaining = deadline - time.time()
            if time_remaining < 5.0:
                raise TimeoutError(
                    f"Deadline would be exceeded: {time_remaining:.1f}s remaining"
                )

            # Phase 5: CLIP Verification (~0.6GB)
            self._model_registry.prepare_for_stage("clip")
            clip_embedding = await self._verify_semantic(
                video_result, script.visual_prompt, recipe
            )

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
            result.determinism_proof = self.compute_determinism_proof(
                recipe, seed, result
            )

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

    async def _generate_script(
        self, recipe: dict[str, Any], seed: int
    ) -> Script:
        """Generate script using Showrunner or use provided script.

        Args:
            recipe: Recipe with narrative section
            seed: Deterministic seed for fallback selection

        Returns:
            Script with setup, punchline, and visual_prompt
        """
        assert self._showrunner is not None

        narrative = recipe.get("narrative", {})
        auto_script = narrative.get("auto_script", True)
        theme = narrative.get("theme", "bizarre infomercial")
        tone = narrative.get("tone", "absurd")

        if auto_script:
            # Generate script with Showrunner (LLM)
            if self._showrunner.is_available():
                try:
                    script = await self._showrunner.generate_script(
                        theme=theme,
                        tone=tone,
                    )
                    logger.info(
                        "Script generated by Showrunner",
                        extra={"theme": theme, "tone": tone},
                    )
                    return script
                except Exception as e:
                    logger.warning(
                        f"Showrunner failed, using fallback: {e}",
                        extra={"theme": theme},
                    )
                    return self._showrunner.get_fallback_script(
                        theme=theme, tone=tone, seed=seed
                    )
            else:
                logger.info(
                    "Showrunner unavailable, using fallback script",
                    extra={"theme": theme},
                )
                return self._showrunner.get_fallback_script(
                    theme=theme, tone=tone, seed=seed
                )
        else:
            # Use provided script from recipe
            script_data = narrative.get("script", {})
            return Script(
                setup=script_data.get("setup", ""),
                punchline=script_data.get("punchline", ""),
                visual_prompt=script_data.get("visual_prompt", ""),
            )

    async def _generate_audio(
        self, script: str, recipe: dict[str, Any], seed: int
    ) -> torch.Tensor:
        """Generate audio waveform using Kokoro TTS.

        Args:
            script: Full text to synthesize (setup + punchline)
            recipe: Recipe with audio section
            seed: Deterministic seed

        Returns:
            Audio waveform tensor of shape (samples,) at 24kHz
        """
        assert self._model_registry is not None
        assert self._audio_buffer is not None

        # Get Kokoro model from registry
        kokoro = self._model_registry.get_model("kokoro")

        # Extract audio parameters from recipe
        audio_config = recipe.get("audio", {})
        voice_id = audio_config.get("voice_id", "af_heart")
        speed = audio_config.get("speed", 1.0)

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

        # Generate audio with deterministic seed (wrap in thread to avoid blocking)
        try:
            audio = await asyncio.to_thread(
                kokoro.synthesize,
                text=script,
                voice_id=voice_id,
                speed=speed,
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

    async def _generate_keyframe(
        self, prompt: str, recipe: dict[str, Any], seed: int
    ) -> torch.Tensor:
        """Generate keyframe image using Flux-Schnell.

        Args:
            prompt: Combined visual_prompt + style_prompt
            recipe: Recipe with video section
            seed: Deterministic seed

        Returns:
            Keyframe image tensor of shape [3, H, W] (CogVideoX expects this)
        """
        assert self._model_registry is not None
        assert self._actor_buffer is not None

        # Get Flux model from registry
        flux = self._model_registry.get_model("flux")

        # Extract video parameters from recipe
        video_config = recipe.get("video", {})
        negative_prompt = video_config.get("negative_prompt", "")

        # Validate prompt is not empty
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided, returning zeroed actor buffer")
            self._actor_buffer.zero_()
            return self._actor_buffer.squeeze(0)

        # Check if model has generate method (real vs mock)
        if not hasattr(flux, "generate"):
            logger.warning("Flux model missing generate(); using zeroed buffer")
            self._actor_buffer.zero_()
            return self._actor_buffer.squeeze(0)

        # Generate keyframe image with deterministic seed (wrap in thread to avoid blocking)
        try:
            # Flux expects output buffer shape [3, 512, 512]
            output_buffer = (
                self._actor_buffer.squeeze(0)
                if self._actor_buffer.dim() == 4
                else self._actor_buffer
            )

            image = await asyncio.to_thread(
                flux.generate,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=4,  # Schnell fast variant
                guidance_scale=0.0,  # Unconditional for speed
                output=output_buffer,
                seed=seed,
            )

            # Return [3, H, W] for CogVideoX
            if image.dim() == 4:
                return image.squeeze(0)
            return image

        except ValueError as e:
            logger.warning(f"Flux generation failed: {e}, using zeroed buffer")
            self._actor_buffer.zero_()
            return self._actor_buffer.squeeze(0)
        except Exception as e:
            logger.error(f"Unexpected Flux error: {e}", exc_info=True)
            self._actor_buffer.zero_()
            return self._actor_buffer.squeeze(0)

    async def _generate_video(
        self,
        keyframe: torch.Tensor,
        visual_prompt: str,
        recipe: dict[str, Any],
        seed: int,
    ) -> torch.Tensor:
        """Generate video using CogVideoX from keyframe.

        Args:
            keyframe: Keyframe image tensor [3, H, W]
            visual_prompt: Scene description for video generation
            recipe: Recipe with slot_params and video sections
            seed: Deterministic seed

        Returns:
            Video frames tensor [num_frames, 3, H, W]
        """
        assert self._model_registry is not None

        cogvideox = self._model_registry.get_cogvideox()

        # Extract parameters from recipe
        slot_params = recipe.get("slot_params", {})
        target_duration = slot_params.get("target_duration", 12.0)
        fps = slot_params.get("fps", 8)

        video_config = recipe.get("video", {})
        guidance_scale = video_config.get("guidance_scale", 6.0)

        # Build video generation config
        config = VideoGenerationConfig(
            fps=fps,
            guidance_scale=guidance_scale,
        )

        logger.info(
            "Starting CogVideoX video generation",
            extra={
                "target_duration": target_duration,
                "fps": fps,
                "guidance_scale": guidance_scale,
                "keyframe_shape": list(keyframe.shape),
            },
        )

        # Generate video chain for target duration
        video_frames = await cogvideox.generate_chain(
            keyframe=keyframe,
            prompt=visual_prompt,
            target_duration=target_duration,
            config=config,
            seed=seed,
        )

        return video_frames

    async def _verify_semantic(
        self,
        video: torch.Tensor,
        visual_prompt: str,
        recipe: dict[str, Any],
    ) -> torch.Tensor:
        """Dual CLIP semantic verification.

        Args:
            video: Generated video frames tensor [T, C, H, W]
            visual_prompt: Original visual prompt for verification
            recipe: Recipe with quality section

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
        quality_config = recipe.get("quality", {})
        clip_threshold = quality_config.get("clip_threshold", 0.70)

        # Validate inputs
        if not visual_prompt or not visual_prompt.strip():
            logger.warning(
                "Empty prompt for CLIP verification; returning random embedding"
            )
            return torch.randn(512, device=self._device, dtype=torch.float32)

        if not hasattr(clip_ensemble, "verify"):
            logger.warning("ClipEnsemble missing verify(); returning random embedding")
            return torch.randn(512, device=self._device, dtype=torch.float32)

        if video.numel() == 0:
            logger.warning(
                "Empty video frames for CLIP verification; returning random embedding"
            )
            return torch.randn(512, device=self._device, dtype=torch.float32)

        # Perform semantic verification
        try:
            result: DualClipResult = clip_ensemble.verify(
                video_frames=video,
                prompt=visual_prompt,
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
                logger.warning(
                    "CLIP self-check FAILED: video may not match prompt semantically"
                )
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
        required_models = ["flux", "kokoro", "cogvideox", "clip_ensemble"]
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

    async def shutdown(self) -> None:
        """Shutdown renderer and release all resources.

        Unloads all models, clears pre-allocated buffers, and resets state.
        Safe to call multiple times.
        """
        logger.info("Shutting down DefaultRenderer...")

        # Unload CogVideoX if loaded
        if self._model_registry is not None:
            try:
                cogvideox = self._model_registry.get_cogvideox()
                cogvideox.unload()
            except Exception as e:
                logger.warning(f"Error unloading CogVideoX: {e}")

        # Clear model registry (triggers cleanup)
        self._model_registry = None

        # Clear pre-allocated buffers
        self._actor_buffer = None
        self._video_buffer = None
        self._audio_buffer = None

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = False
        logger.info("DefaultRenderer shutdown complete")
