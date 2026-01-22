"""Core types for Lane 0 video renderer system.

This module defines the data structures used by DeterministicVideoRenderer
implementations. Adapted from plugins/types.py with Lane 0-specific constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class RendererResources:
    """Resource requirements declared by a renderer.

    Attributes:
        vram_gb: Total VRAM required for all models (e.g., 11.8 GB)
        max_latency_ms: Maximum latency guarantee in milliseconds (e.g., 21000 ms)
    """

    vram_gb: float
    max_latency_ms: int


@dataclass(frozen=True)
class RendererManifest:
    """Renderer manifest metadata.

    Lane 0 renderers must declare deterministic=True and specify their
    model dependencies for on-chain registration validation.

    Attributes:
        schema_version: Manifest format version (e.g., "1.0")
        name: Unique renderer identifier (e.g., "default-narrative-chain")
        version: Renderer version (semantic versioning, e.g., "1.0.0")
        entrypoint: Module:Class path for loading (e.g., "renderer:DefaultRenderer")
        description: Human-readable description
        deterministic: Must be True for Lane 0 (BFT consensus requirement)
        resources: Resource requirements (VRAM, latency)
        model_dependencies: Model IDs this renderer depends on
    """

    schema_version: str
    name: str
    version: str
    entrypoint: str
    description: str
    deterministic: bool
    resources: RendererResources
    model_dependencies: tuple[str, ...]

    @staticmethod
    def _require_str(data: Mapping[str, Any], key: str) -> str:
        """Extract and validate a required string field."""
        value = data.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Manifest '{key}' must be a non-empty string")
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RendererManifest:
        """Create RendererManifest from dict with validation.

        Args:
            data: Dictionary loaded from manifest.yaml

        Returns:
            Validated RendererManifest instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        schema_version = cls._require_str(data, "schema_version")
        name = cls._require_str(data, "name")
        version = cls._require_str(data, "version")
        entrypoint = cls._require_str(data, "entrypoint")
        description = cls._require_str(data, "description")

        deterministic = data.get("deterministic")
        if not isinstance(deterministic, bool):
            raise ValueError("Manifest 'deterministic' must be a boolean")
        if not deterministic:
            raise ValueError("Lane 0 renderers must have 'deterministic: true'")

        resources = data.get("resources")
        if not isinstance(resources, Mapping):
            raise ValueError("Manifest 'resources' must be a mapping")
        vram_gb = resources.get("vram_gb")
        max_latency_ms = resources.get("max_latency_ms")
        if not isinstance(vram_gb, (int, float)) or vram_gb <= 0:
            raise ValueError("Manifest 'resources.vram_gb' must be > 0")
        if not isinstance(max_latency_ms, int) or max_latency_ms <= 0:
            raise ValueError("Manifest 'resources.max_latency_ms' must be > 0")

        deps = data.get("model_dependencies")
        if not isinstance(deps, Sequence) or isinstance(deps, (str, bytes)):
            raise ValueError("Manifest 'model_dependencies' must be a list")
        model_dependencies = tuple(str(dep) for dep in deps)

        return cls(
            schema_version=schema_version,
            name=name,
            version=version,
            entrypoint=entrypoint,
            description=description,
            deterministic=deterministic,
            resources=RendererResources(
                vram_gb=float(vram_gb),
                max_latency_ms=max_latency_ms,
            ),
            model_dependencies=model_dependencies,
        )


@dataclass
class RenderResult:
    """Result from a single render operation.

    Attributes:
        video_frames: Tensor of shape [T, C, H, W] (e.g., [1080, 3, 512, 512])
        audio_waveform: Tensor of shape [samples] at 24kHz (e.g., [1080000])
        clip_embedding: L2-normalized embedding from dual CLIP ensemble [512]
        generation_time_ms: Total generation time in milliseconds
        determinism_proof: SHA256 hash of (recipe + seed + output) for BFT
        success: Whether generation completed successfully
        error_msg: Error message if success=False
    """

    video_frames: torch.Tensor
    audio_waveform: torch.Tensor
    clip_embedding: torch.Tensor
    generation_time_ms: float
    determinism_proof: bytes
    success: bool = True
    error_msg: str | None = None


# Lane 0 constraints
LANE0_MAX_VRAM_GB = 11.5
LANE0_MAX_LATENCY_MS = 15000
