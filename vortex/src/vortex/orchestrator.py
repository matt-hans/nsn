"""ToonGen Video Orchestrator.

Main pipeline that coordinates:
1. Audio generation (F5-TTS/Kokoro)
2. Audio mixing (FFmpeg)
3. Visual generation (ComfyUI)
4. VRAM management (sequential execution)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vortex.core.audio import AudioEngine
from vortex.core.mixer import AudioCompositor, calculate_frame_count
from vortex.engine.client import ComfyClient
from vortex.engine.payload import WorkflowBuilder

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of video generation."""

    video_path: str
    clean_audio_path: str
    mixed_audio_path: str
    frame_count: int
    seed: int


class VideoOrchestrator:
    """Orchestrates the ToonGen video generation pipeline.

    Pipeline flow:
    1. Generate voice audio (F5-TTS or Kokoro)
    2. Mix with BGM/SFX (FFmpeg)
    3. Unload audio models (free VRAM)
    4. Build ComfyUI workflow payload
    5. Dispatch to ComfyUI and wait for completion
    6. Return paths to output files
    """

    def __init__(
        self,
        template_path: str = "templates/cartoon_workflow.json",
        assets_dir: str = "assets",
        output_dir: str = "outputs",
        comfy_host: str = "127.0.0.1",
        comfy_port: int = 8188,
        device: str = "cuda",
    ):
        """Initialize orchestrator.

        Args:
            template_path: Path to ComfyUI workflow JSON
            assets_dir: Root directory for audio assets
            output_dir: Directory for output files
            comfy_host: ComfyUI server hostname
            comfy_port: ComfyUI server port
            device: PyTorch device for audio models
        """
        self.template_path = template_path
        self.assets_dir = Path(assets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._audio_engine = AudioEngine(
            device=device,
            assets_dir=str(self.assets_dir / "voices"),
        )
        self._audio_mixer = AudioCompositor(
            assets_dir=str(self.assets_dir),
        )
        self._workflow_builder = WorkflowBuilder(template_path=template_path)
        self._comfy_client = ComfyClient(host=comfy_host, port=comfy_port)

    async def generate(
        self,
        prompt: str,
        script: str,
        voice_style: str | None = None,
        voice_id: str = "af_heart",
        engine: str = "auto",
        bgm_name: str | None = None,
        sfx_name: str | None = None,
        mix_ratio: float = 0.3,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate a video clip.

        Args:
            prompt: Visual prompt for Flux image generation
            script: Text to synthesize as speech
            voice_style: F5-TTS voice reference name
            voice_id: Kokoro voice ID
            engine: Audio engine selection ("auto", "f5_tts", "kokoro")
            bgm_name: Background music asset name
            sfx_name: Sound effect asset name
            mix_ratio: BGM volume relative to voice
            seed: Deterministic seed (random if not provided)

        Returns:
            Dict with video_path, clean_audio_path, mixed_audio_path, etc.
        """
        logger.info(f"Starting generation: prompt='{prompt[:50]}...'")

        # ===== PHASE 1: Audio Generation =====
        logger.info("Phase 1: Generating audio...")

        # Generate clean voice (for lip-sync)
        clean_audio_path = self._audio_engine.generate(
            script=script,
            engine=engine,
            voice_style=voice_style,
            voice_id=voice_id,
        )
        logger.info(f"Clean audio generated: {clean_audio_path}")

        # Mix with BGM/SFX (for broadcast)
        mixed_audio_path = self._audio_mixer.mix(
            voice_path=clean_audio_path,
            bgm_name=bgm_name,
            sfx_name=sfx_name,
            mix_ratio=mix_ratio,
        )
        logger.info(f"Mixed audio generated: {mixed_audio_path}")

        # Calculate frame count from audio duration
        frame_count = calculate_frame_count(clean_audio_path)
        logger.info(f"Calculated frame count: {frame_count}")

        # ===== VRAM HANDOFF =====
        # CRITICAL: Unload audio models BEFORE ComfyUI starts
        logger.info("Unloading audio models for VRAM handoff...")
        self._audio_engine.unload()

        # ===== PHASE 2: Visual Generation =====
        logger.info("Phase 2: Generating visuals via ComfyUI...")

        # Build workflow payload
        workflow = self._workflow_builder.build(
            prompt=prompt,
            audio_path=clean_audio_path,  # Clean audio for lip-sync
            seed=seed,
        )

        # Dispatch to ComfyUI
        video_path = await self._comfy_client.generate(workflow)
        logger.info(f"Video generated: {video_path}")

        # ===== PHASE 3: Post-processing =====
        # TODO: Mux mixed audio into final video (replace ComfyUI audio track)
        # For MVP, ComfyUI's VHS_VideoCombine uses the clean audio directly

        return {
            "video_path": video_path,
            "clean_audio_path": clean_audio_path,
            "mixed_audio_path": mixed_audio_path,
            "frame_count": frame_count,
            "seed": seed,
        }

    async def health_check(self) -> dict[str, bool]:
        """Check health of all components.

        Returns:
            Dict with component health status
        """
        comfy_ok = await self._comfy_client.check_health()

        return {
            "orchestrator": True,
            "comfyui": comfy_ok,
        }
