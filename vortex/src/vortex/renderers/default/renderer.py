"""Default Lane 0 renderer implementation with I2V (Image-to-Video) Montage architecture.

This renderer implements the I2V Montage pipeline for Lane 0 video generation:
1. Showrunner (LLM) generates comedic script with 3-scene storyboard
2. Bark synthesizes script to audio FIRST (sets exact duration)
3. Flux generates 3 keyframe images from storyboard scenes
4. CogVideoX I2V animates each keyframe into a video clip
5. Clips concatenated with hard cuts
6. CLIP verifies semantic consistency

Architecture:
    - Showrunner (external Ollama): Generates script with storyboard[3]
    - Bark (FP16, ~1.5GB): Text-to-speech synthesis at 24kHz with emotion support
    - Flux-Schnell (NF4, ~6GB): Generates 3 keyframe images at 720x480
    - CogVideoX-5B-I2V (INT8, ~10-11GB): Animates keyframes to video (I2V)
    - CLIP ViT-B-32 + ViT-L-14 (FP16, 0.6GB): Dual ensemble semantic verification

Pipeline Flow (I2V Audio-First):
    Recipe -> Showrunner (script) -> Bark (TTS) -> Audio (defines duration)

    storyboard[3] -> Flux (keyframes) -> unload Flux

    keyframes[3] + motion_prompts -> CogVideoX I2V -> concatenate -> Video

    Video -> CLIP (verify) -> Embedding

VRAM Budget: ~11GB peak during CogVideoX phase (Flux unloaded before CogVideoX)
Target Duration: Audio-driven (e.g., 14.2s audio = 227 frames @ 16fps)
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

from vortex.models.bark import load_bark
from vortex.models.clip_ensemble import ClipEnsemble, DualClipResult
from vortex.models.cogvideox import CogVideoXModel, VideoGenerationConfig
from vortex.models.flux import FluxModel
from vortex.models.showrunner import Script, Showrunner
from vortex.renderers.base import DeterministicVideoRenderer
from vortex.renderers.recipe_schema import merge_with_defaults, validate_recipe
from vortex.renderers.types import RendererManifest, RenderResult
from vortex.utils.memory import get_current_vram_usage, get_vram_stats, log_vram_snapshot

# Type alias for model names - used for type hints
ModelName = str

logger = logging.getLogger(__name__)

# VAE-safe style: solid colors and clean edges prevent aliasing artifacts
# REMOVED: "halftone texture" (causes swirling during VAE downsampling)
# ADDED: "vector art, solid colors" (mathematically stable for 8x downsampling)
VISUAL_STYLE_PROMPT = (
    "1990s cartoon style, thick clean outlines, flat solid colors, vector art, "
    "cel shaded, high definition, saturday morning cartoon, no gradients, "
    "grounded on floor, consistent perspective, stable background"
)

# Motion style suffix appended to video prompts for CogVideoX I2V
MOTION_STYLE_SUFFIX = ", smooth animation, continuous motion, fluid movement"

# Legacy clean style prompt (kept for backward compatibility)
CLEAN_STYLE_PROMPT = "cartoon style, 2d animation, flat colors, high definition, 4k"


class VortexInitializationError(Exception):
    """Raised when renderer initialization fails."""

    pass


class MemoryPressureError(Exception):
    """Raised when VRAM usage exceeds hard limit."""

    pass


class _ModelRegistry:
    """Registry for loaded models with sequential loading for I2V Montage.

    The I2V Montage pipeline requires sequential model loading due to
    CogVideoX's large VRAM footprint (~10-11GB with INT8 quantization).
    Models are loaded to CPU and moved to GPU only when needed.

    Models:
    - bark: TTS audio synthesis with emotion support (~1.5GB)
    - flux: Keyframe image generation (NF4, ~6GB) - unloaded before CogVideoX
    - cogvideox: Video generation from keyframes (I2V, 10-11GB with INT8)
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
        self._models: dict[str, nn.Module | CogVideoXModel | FluxModel] = {}
        self._offloading_mode = offloading_mode
        self._cache_dir = cache_dir
        self._local_only = local_only

        # CogVideoX and Flux are handled separately (not nn.Module)
        self._cogvideox: CogVideoXModel | None = None
        self._flux: FluxModel | None = None

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

        For I2V Montage, we load:
        - bark (TTS with emotion support)
        - flux (keyframe generation) - lazy-loaded, unloaded before CogVideoX
        - clip_ensemble (semantic verification)
        - cogvideox (I2V video generation) - handled separately
        """
        try:
            # Load Bark TTS
            logger.info("Loading model: bark")
            target_device = "cpu" if self._offloading_enabled else self.device
            bark = load_bark(device=target_device)
            self._models["bark"] = bark
            log_vram_snapshot("after_bark_load")

            # Initialize Flux (lazy-loaded, doesn't load weights yet)
            logger.info("Initializing Flux model wrapper")
            self._flux = FluxModel(
                device=self.device,
                cache_dir=self._cache_dir,
                quantization="nf4",  # NF4 quantization for memory efficiency
            )
            # Note: Flux.load() is called lazily when needed
            self._models["flux"] = self._flux
            log_vram_snapshot("after_flux_init")

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

    def get_flux(self) -> FluxModel:
        """Get the Flux model instance."""
        if self._flux is None:
            raise RuntimeError("Flux not initialized")
        return self._flux

    def prepare_for_stage(self, stage: str) -> None:
        """Prepare models for a pipeline stage.

        Note: Full offloading support will be implemented in a future phase.
        Currently, this method just logs the stage transition.

        Args:
            stage: Stage name ("audio", "keyframe", "video", "clip")
        """
        if not self._offloading_enabled:
            return

        # Map stages to required models
        stage_models = {
            "audio": "bark",
            "keyframe": "flux",  # Flux for keyframe generation
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
    """Default Lane 0 renderer using T2V Montage architecture.

    This renderer implements the T2V Montage pipeline:
    1. Showrunner (LLM) generates script with video_prompts from theme
    2. Bark synthesizes script to speech with emotion
    3. CogVideoX generates 3 video clips from video_prompts (T2V)
    4. Clips concatenated with hard cuts for 15s montage
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
        self._audio_buffer: torch.Tensor | None = None
        self._actor_buffer: torch.Tensor | None = None

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

        # Audio buffer (24kHz * 15 seconds = 360000 samples max)
        audio_cfg = buf_cfg.get("audio", {})
        self._audio_buffer = torch.zeros(
            audio_cfg.get("samples", 360000),  # 15 seconds @ 24kHz
            device=self._device,
            dtype=torch.float32,
        )

        # Actor buffer for keyframe images (720x480 at CogVideoX native resolution)
        actor_cfg = buf_cfg.get("actor", {})
        self._actor_buffer = torch.zeros(
            1,
            actor_cfg.get("channels", 3),
            actor_cfg.get("height", 480),   # CogVideoX native height
            actor_cfg.get("width", 720),    # CogVideoX native width
            device=self._device,
            dtype=torch.float32,
        )

        logger.info(
            "Output buffers pre-allocated",
            extra={
                "audio_shape": tuple(self._audio_buffer.shape),
                "actor_shape": tuple(self._actor_buffer.shape),
            },
        )

    async def render(
        self,
        recipe: dict[str, Any],
        slot_id: int,
        seed: int,
        deadline: float,
    ) -> RenderResult:
        """Render a single slot from recipe using I2V Audio-First pipeline.

        Pipeline Flow (Audio-First):
        1. Script Generation: Showrunner (LLM) generates script with storyboard
        2. Audio Generation FIRST: Bark TTS - defines exact video duration
        3. Calculate video frames: audio_duration * fps = target_frames
        4. Keyframe Generation: Flux generates 3 keyframe images
        5. Unload Flux: Free VRAM for CogVideoX
        6. Video Generation: CogVideoX I2V animates keyframes
        7. CLIP Verification: Dual ensemble semantic check

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
                    "storyboard_scenes": len(script.storyboard),
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

            # Phase 2: Audio Generation FIRST (Bark ~1.5GB) - defines video duration
            self._model_registry.prepare_for_stage("audio")
            full_script = f"{script.setup} {script.punchline}"
            audio_result = await self._generate_audio(full_script, recipe, seed)
            logger.info(
                "Audio generated",
                extra={"audio_samples": audio_result.shape[0]},
            )

            # Calculate video frames from audio duration
            # Audio is at 24kHz, video is at 16fps
            audio_sample_rate = 24000
            video_fps = 16
            audio_duration_s = audio_result.shape[0] / audio_sample_rate
            total_video_frames = int(audio_duration_s * video_fps)
            num_scenes = len(script.storyboard)
            # Ensure minimum 65 frames per scene (~4s at 16fps)
            frames_per_scene = (
                max(65, total_video_frames // num_scenes) if num_scenes > 0 else 65
            )

            logger.info(
                "Audio-first calculation",
                extra={
                    "audio_duration_s": f"{audio_duration_s:.2f}",
                    "total_video_frames": total_video_frames,
                    "frames_per_scene": frames_per_scene,
                },
            )

            # Phase 3: Keyframe Generation (Flux ~6GB)
            keyframes = await self._generate_keyframes(script, seed)
            logger.info(
                "Keyframes generated",
                extra={"num_keyframes": len(keyframes)},
            )

            # Phase 4: Unload Flux to free VRAM for CogVideoX
            self._unload_flux()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Phase 5: Video Generation from Keyframes (CogVideoX I2V ~10-11GB)
            video_result = await self._generate_video(
                script=script,
                keyframes=keyframes,
                frames_per_scene=frames_per_scene,
                seed=seed,
            )
            logger.info(
                "Video montage generated",
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

            # Phase 6: CLIP Verification (~0.6GB)
            # Uses subject_visual for semantic verification of the montage
            self._model_registry.prepare_for_stage("clip")
            clip_embedding = await self._verify_semantic(
                video_result, script.subject_visual, recipe
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
            Script with setup, punchline, storyboard, and video_prompts
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
            # Handle storyboard - convert legacy visual_prompt to 3-scene storyboard
            storyboard = script_data.get("storyboard", [])
            if not storyboard:
                visual_prompt = script_data.get("visual_prompt", "")
                storyboard = [visual_prompt, visual_prompt, visual_prompt] if visual_prompt else []
            # Handle video_prompts - use storyboard as fallback for T2V
            video_prompts = script_data.get("video_prompts", [])
            if not video_prompts:
                # If no video_prompts provided, use storyboard prompts with style
                video_prompts = [
                    f"{scene}, {CLEAN_STYLE_PROMPT}" for scene in storyboard
                ] if storyboard else []
            return Script(
                setup=script_data.get("setup", ""),
                punchline=script_data.get("punchline", ""),
                subject_visual=script_data.get("subject_visual", ""),
                storyboard=storyboard,
                video_prompts=video_prompts,
            )

    async def _generate_audio(
        self, script: str, recipe: dict[str, Any], seed: int
    ) -> torch.Tensor:
        """Generate audio waveform using Bark TTS.

        Args:
            script: Full text to synthesize (setup + punchline)
            recipe: Recipe with audio section
            seed: Deterministic seed

        Returns:
            Audio waveform tensor of shape (samples,) at 24kHz
        """
        assert self._model_registry is not None
        assert self._audio_buffer is not None

        # Get Bark model from registry
        bark = self._model_registry.get_model("bark")

        # Extract audio parameters from recipe
        audio_config = recipe.get("audio", {})
        voice_id = audio_config.get("voice_id", "v2/en_speaker_6")

        # Validate script is not empty
        if not script or not script.strip():
            logger.warning("Empty script provided, returning zeroed audio buffer")
            self._audio_buffer.zero_()
            return self._audio_buffer

        # Check if model has synthesize method (real vs mock)
        if not hasattr(bark, "synthesize"):
            logger.warning("Bark model missing synthesize(); using zeroed buffer")
            self._audio_buffer.zero_()
            return self._audio_buffer

        # Generate audio with Bark (wrap in thread to avoid blocking)
        try:
            # Extract emotion from recipe (default to neutral)
            emotion = audio_config.get("emotion", "neutral")

            audio = await asyncio.to_thread(
                bark.synthesize,
                text=script,
                voice_id=voice_id,
                emotion=emotion,
                output=self._audio_buffer,
                seed=seed,
            )
            return audio
        except ValueError as e:
            logger.warning(f"Bark synthesis failed: {e}, using zeroed buffer")
            self._audio_buffer.zero_()
            return self._audio_buffer
        except Exception as e:
            logger.error(f"Unexpected Bark error: {e}", exc_info=True)
            self._audio_buffer.zero_()
            return self._audio_buffer

    async def _generate_keyframes(
        self,
        script: Script,
        seed: int,
    ) -> list[torch.Tensor]:
        """Generate 3 keyframes with texture anchoring and fixed seed.

        Generates one keyframe per storyboard scene using Flux-Schnell.
        All keyframes use the SAME seed for subject consistency across scenes.

        Args:
            script: Script with storyboard scenes
            seed: Deterministic seed (used for ALL keyframes, not varied)

        Returns:
            List of keyframe tensors, each [C, H, W] in 0-1 range
        """
        assert self._model_registry is not None
        assert self._actor_buffer is not None

        self._model_registry.prepare_for_stage("keyframe")
        flux = self._model_registry.get_flux()

        keyframes = []

        for i, scene in enumerate(script.storyboard):
            # FIXED SEED for subject consistency - NOT seed + i
            scene_seed = seed

            # Build visual prompt with REINFORCED subject
            # Repeating subject_visual ensures Flux prioritizes character identity
            visual_prompt = (
                f"{script.subject_visual}. "  # First mention
                f"{script.subject_visual}, {scene}. "  # Second mention with scene
                f"{VISUAL_STYLE_PROMPT}"
            )

            logger.info(
                f"Generating keyframe {i+1}/{len(script.storyboard)}",
                extra={"prompt_preview": visual_prompt[:80], "seed": scene_seed},
            )

            # Generate keyframe using pre-allocated actor buffer
            keyframe = flux.generate(
                prompt=visual_prompt,
                seed=scene_seed,
                output=self._actor_buffer[0],  # Use first slot of buffer
            )

            # Clone to avoid buffer reuse issues
            keyframes.append(keyframe.clone())

        logger.info(f"Generated {len(keyframes)} keyframes at 720x480")
        return keyframes

    def _unload_flux(self) -> None:
        """Unload Flux to free VRAM before CogVideoX."""
        assert self._model_registry is not None

        flux = self._model_registry.get_flux()
        if flux.is_loaded:
            flux.unload()
            logger.info("Flux unloaded to free VRAM for CogVideoX")

    async def _generate_video(
        self,
        script: Script,
        keyframes: list[torch.Tensor],
        frames_per_scene: int,
        seed: int,
    ) -> torch.Tensor:
        """Generate video from keyframes using I2V.

        Animates each keyframe into a video clip using CogVideoX-5B-I2V,
        then concatenates clips with hard cuts.

        Args:
            script: Script with storyboard scenes
            keyframes: List of keyframe tensors from Flux
            frames_per_scene: Target frames per scene (from audio duration)
            seed: Deterministic seed

        Returns:
            Concatenated video tensor [total_frames, C, H, W]
        """
        assert self._model_registry is not None

        if len(keyframes) != len(script.storyboard):
            raise ValueError(
                f"Keyframes and storyboard length mismatch: "
                f"{len(keyframes)} keyframes vs {len(script.storyboard)} scenes"
            )

        # Build motion prompts from storyboard with motion style suffix
        motion_prompts = [
            f"{scene}{MOTION_STYLE_SUFFIX}"
            for scene in script.storyboard
        ]

        logger.info(f"Generating {len(keyframes)}-scene I2V montage...")
        self._model_registry.prepare_for_stage("video")
        cogvideox = self._model_registry.get_cogvideox()

        # Configure for montage (81 frames at 16fps per CogVideoX docs)
        config = VideoGenerationConfig(
            num_frames=81,
            guidance_scale=5.5,  # Higher for temporal stability (CogVideoX docs: 6.0 default)
            use_dynamic_cfg=True,
            fps=16,
        )

        # Trim frames: min of requested and 65 (to avoid tail degradation, ~4s at 16fps)
        trim_frames = min(frames_per_scene, 65)

        video = await cogvideox.generate_montage(
            keyframes=keyframes,
            prompts=motion_prompts,
            config=config,
            seed=seed,
            trim_frames=trim_frames,
        )

        logger.info(
            f"I2V Montage complete: {video.shape[0]} frames ({video.shape[0]/16:.1f}s)"
        )
        return video

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
        required_models = ["bark", "flux", "cogvideox", "clip_ensemble"]
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

        # Unload models if loaded
        if self._model_registry is not None:
            # Unload Flux
            try:
                flux = self._model_registry.get_flux()
                flux.unload()
            except Exception as e:
                logger.warning(f"Error unloading Flux: {e}")

            # Unload CogVideoX
            try:
                cogvideox = self._model_registry.get_cogvideox()
                cogvideox.unload()
            except Exception as e:
                logger.warning(f"Error unloading CogVideoX: {e}")

        # Clear model registry (triggers cleanup)
        self._model_registry = None

        # Clear pre-allocated buffers
        self._audio_buffer = None
        self._actor_buffer = None

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = False
        logger.info("DefaultRenderer shutdown complete")
