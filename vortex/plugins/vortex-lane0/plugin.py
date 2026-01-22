"""Vortex Lane 0 Plugin - Deterministic video generation for BFT consensus.

This plugin provides Lane 0 video generation through the Vortex pipeline:
- Flux-Schnell for keyframe image generation (NF4 quantized, ~6GB VRAM)
- CogVideoX for video generation (INT8 quantized, ~10-11GB VRAM)
- Kokoro TTS for audio synthesis
- Dual CLIP ensemble for semantic verification

All outputs are deterministic when given the same seed, enabling BFT consensus
verification across network validators.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VortexLane0Plugin:
    """Lane 0 video generation plugin using Vortex renderer system.

    This plugin wraps the VortexPipeline to provide deterministic video
    generation compatible with the sidecar plugin execution model.

    The plugin lazily initializes the pipeline on first use to avoid
    loading models during import (which would fail without GPU).
    """

    def __init__(self, manifest: dict[str, Any] | None = None):
        """Initialize plugin with optional manifest.

        Args:
            manifest: Plugin manifest dict (injected by plugin loader)
        """
        self.manifest = manifest
        self._pipeline = None
        self._device = os.environ.get("VORTEX_DEVICE", "cuda:0")

    async def _ensure_pipeline(self) -> Any:
        """Lazily initialize the VortexPipeline.

        Returns:
            Initialized VortexPipeline instance
        """
        if self._pipeline is None:
            logger.info(f"Initializing VortexPipeline on device: {self._device}")
            from vortex.pipeline import VortexPipeline

            self._pipeline = await VortexPipeline.create(device=self._device)
            logger.info(
                f"VortexPipeline initialized with renderer: {self._pipeline.renderer_name}"
            )
        return self._pipeline

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute video generation synchronously.

        This is the main entry point called by the plugin runner.

        Args:
            payload: Dict with keys:
                - recipe: Generation recipe dict
                - slot_id: Unique slot identifier
                - seed: Optional deterministic seed

        Returns:
            Dict with keys:
                - output_cid: Content identifier for generated output
                - video_data: Base64-encoded VP9/IVF video bytes
                - audio_waveform: Base64-encoded Opus/OGG audio bytes
                - clip_embedding: CLIP embedding as list
                - determinism_proof: Hex-encoded SHA256 hash
                - generation_time_ms: Total generation time
        """
        return asyncio.run(self._run_async(payload))

    async def _run_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute video generation asynchronously.

        Args:
            payload: Same as run()

        Returns:
            Same as run()

        Raises:
            RuntimeError: If generation fails
            ValueError: If payload is invalid
        """
        import base64

        start_time = time.time()

        # Validate payload
        recipe = payload.get("recipe")
        if not isinstance(recipe, dict):
            raise ValueError("payload 'recipe' must be a dict")

        slot_id = payload.get("slot_id")
        if not isinstance(slot_id, int):
            raise ValueError("payload 'slot_id' must be an integer")

        seed = payload.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("payload 'seed' must be an integer or null")

        # Initialize pipeline
        pipeline = await self._ensure_pipeline()

        logger.info(
            f"Starting Lane 0 generation for slot {slot_id}",
            extra={"slot_id": slot_id, "seed": seed, "renderer": pipeline.renderer_name},
        )

        # Generate video
        result = await pipeline.generate_slot(recipe, slot_id=slot_id, seed=seed)

        if not result.success:
            raise RuntimeError(f"Generation failed: {result.error_msg}")

        # Convert tensors to numpy
        video_np = result.video_frames.cpu().numpy()
        audio_np = result.audio_waveform.cpu().numpy()
        clip_np = result.clip_embedding.cpu().numpy()

        # Shape normalization: [T, C, H, W] → [T, H, W, C]
        if video_np.ndim == 4 and video_np.shape[1] == 3 and video_np.shape[3] != 3:
            video_np = np.transpose(video_np, (0, 2, 3, 1))

        # Ensure uint8 for encoder
        if video_np.dtype != np.uint8:
            # Assume float [0, 1] range
            video_np = (video_np * 255).clip(0, 255).astype(np.uint8)

        # Import encoder (lazy to avoid import errors without GPU)
        from vortex.utils.encoder import encode_audio, encode_video_frames

        # Offload CPU-bound encoding to thread pool
        fps = recipe.get("slot_params", {}).get("fps", 24)

        try:
            video_bytes, audio_bytes = await asyncio.gather(
                asyncio.to_thread(
                    encode_video_frames,
                    video_np,
                    fps=fps,
                    bitrate=3_000_000,
                    keyframe_interval=24,
                ),
                asyncio.to_thread(
                    encode_audio,
                    audio_np,
                    sample_rate=24000,
                    bitrate=64000,
                ),
            )
        except Exception as e:
            raise RuntimeError(f"Encoding pipeline failed: {str(e)}") from e

        # Free large arrays before base64 allocation
        del video_np
        del audio_np

        generation_time_ms = (time.time() - start_time) * 1000

        # Generate content identifier
        output_cid = f"local://{slot_id}/{result.determinism_proof.hex()[:16]}"

        logger.info(
            f"Lane 0 generation completed for slot {slot_id}",
            extra={
                "slot_id": slot_id,
                "generation_time_ms": generation_time_ms,
                "proof": result.determinism_proof.hex()[:16],
                "video_bytes": len(video_bytes),
                "audio_bytes": len(audio_bytes),
            },
        )

        return {
            "output_cid": output_cid,
            "video_data": base64.b64encode(video_bytes).decode("ascii"),
            "audio_waveform": base64.b64encode(audio_bytes).decode("ascii"),
            "clip_embedding": clip_np.tolist(),
            "determinism_proof": result.determinism_proof.hex(),
            "generation_time_ms": generation_time_ms,
        }
