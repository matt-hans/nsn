"""Abstract base class for Lane 0 deterministic video renderers.

This module defines the DeterministicVideoRenderer interface that all Lane 0
video generation backends must implement. The interface guarantees:

1. Determinism: Same recipe + seed = identical output (byte-for-byte)
2. Resource compliance: Stay within declared VRAM budget
3. Latency compliance: Complete within max_latency_ms

Example implementation:
    class MyRenderer(DeterministicVideoRenderer):
        @property
        def manifest(self) -> RendererManifest:
            return self._manifest

        async def initialize(self, device: str, config: dict) -> None:
            # Load models to VRAM
            ...

        async def render(self, recipe, slot_id, seed, deadline) -> RenderResult:
            # Generate video deterministically
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vortex.renderers.types import RenderResult, RendererManifest


class DeterministicVideoRenderer(ABC):
    """Abstract base class for Lane 0 video renderers.

    Implementations MUST guarantee:
    1. Determinism: Same recipe + seed = identical output (byte-for-byte)
       This is required for BFT consensus verification.
    2. Resource compliance: Stay within declared VRAM budget
    3. Latency compliance: Complete within max_latency_ms

    The network validates these properties:
    - Determinism: Validators regenerate with same seed, compare determinism_proof
    - VRAM: On-chain registry rejects renderers exceeding 11.5GB
    - Latency: On-chain registry rejects renderers exceeding 15s for Lane 0
    """

    @property
    @abstractmethod
    def manifest(self) -> RendererManifest:
        """Return renderer manifest with resource declarations.

        The manifest declares:
        - VRAM requirements (must be <= 11.5GB for Lane 0)
        - Latency guarantee (must be <= 15000ms for Lane 0)
        - Model dependencies (validated against on-chain model registry)
        - Determinism flag (must be True for Lane 0)
        """
        ...

    @abstractmethod
    async def initialize(self, device: str, config: dict[str, Any]) -> None:
        """Initialize renderer and load models.

        Called once at pipeline startup. Must load all models to VRAM
        and pre-allocate buffers. After this method returns, the renderer
        must be ready to accept render() calls.

        Args:
            device: Target device (e.g., "cuda:0", "cpu")
            config: Configuration dict from config.yaml

        Raises:
            VortexInitializationError: If CUDA OOM or model loading fails
        """
        ...

    @abstractmethod
    async def render(
        self,
        recipe: dict[str, Any],
        slot_id: int,
        seed: int,
        deadline: float,
    ) -> RenderResult:
        """Render a single slot from recipe.

        This method must be deterministic: given the same recipe and seed,
        it must produce byte-for-byte identical output. This is enforced
        by BFT consensus where validators regenerate with the same seed.

        Args:
            recipe: Standardized recipe dict containing:
                - slot_params: {slot_id, duration_sec, fps}
                - audio_track: {script, voice_id, speed, emotion}
                - visual_track: {prompt, negative_prompt, expression_preset, ...}
                - semantic_constraints: {clip_threshold, ...}
            slot_id: Unique slot identifier
            seed: Deterministic seed for reproducibility (MUST be used)
            deadline: Absolute deadline timestamp (Unix epoch)

        Returns:
            RenderResult with video_frames, audio_waveform, clip_embedding,
            generation_time_ms, and determinism_proof

        Raises:
            DeadlineMissError: If deadline would be exceeded
            MemoryPressureError: If VRAM exceeds hard limit
        """
        ...

    @abstractmethod
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

        This proof is submitted to BFT consensus. Validators regenerate
        with the same recipe and seed, then compare proofs.

        Args:
            recipe: The recipe dict used for generation
            seed: The seed used for generation
            result: The RenderResult from generation

        Returns:
            32-byte SHA256 hash
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify renderer is healthy and models are loaded.

        Returns:
            True if renderer is ready to accept render() calls,
            False if models need reloading or GPU is unhealthy
        """
        ...

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<{self.__class__.__name__} name={self.manifest.name!r}>"
