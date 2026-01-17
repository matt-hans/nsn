"""Vortex Lane 0 Plugin - Deterministic video generation for BFT consensus.

This plugin provides Lane 0 video generation through the Vortex pipeline:
- Flux-Schnell for actor image generation (NF4 quantized, ~6GB VRAM)
- LivePortrait for video animation
- Kokoro TTS for audio synthesis
- Dual CLIP ensemble for semantic verification

All outputs are deterministic when given the same seed, enabling BFT consensus
verification across network validators.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
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
        self._output_dir = Path(os.environ.get("VORTEX_OUTPUT_PATH", "/tmp/vortex"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

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
                - video_path: Path to generated video frames
                - audio_path: Path to generated audio
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

        # Save outputs to files
        video_path = self._output_dir / f"slot_{slot_id}_video.npy"
        audio_path = self._output_dir / f"slot_{slot_id}_audio.npy"

        # Convert tensors to numpy and save
        video_np = result.video_frames.cpu().numpy()
        audio_np = result.audio_waveform.cpu().numpy()
        clip_np = result.clip_embedding.cpu().numpy()

        np.save(video_path, video_np)
        np.save(audio_path, audio_np)

        # Generate content identifier (MVP: use local path, production: IPFS CID)
        output_cid = f"local://{slot_id}/{result.determinism_proof.hex()[:16]}"

        generation_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Lane 0 generation completed for slot {slot_id}",
            extra={
                "slot_id": slot_id,
                "generation_time_ms": generation_time_ms,
                "proof": result.determinism_proof.hex()[:16],
                "video_shape": video_np.shape,
                "audio_samples": len(audio_np),
            },
        )

        return {
            "output_cid": output_cid,
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "clip_embedding": clip_np.tolist(),
            "determinism_proof": result.determinism_proof.hex(),
            "generation_time_ms": generation_time_ms,
            "video_shape": list(video_np.shape),
            "audio_samples": len(audio_np),
        }
