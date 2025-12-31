"""Core types for Vortex plugin system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


JsonSchema = Mapping[str, Any]


@dataclass(frozen=True)
class PluginResources:
    """Resource requirements declared by a plugin."""

    vram_gb: float
    max_latency_ms: int
    max_concurrency: int


@dataclass(frozen=True)
class PluginManifest:
    """Plugin manifest metadata and schemas."""

    schema_version: str
    name: str
    version: str
    entrypoint: str
    description: str
    supported_lanes: tuple[str, ...]
    deterministic: bool
    resources: PluginResources
    input_schema: JsonSchema
    output_schema: JsonSchema

    @staticmethod
    def _require_str(data: Mapping[str, Any], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Manifest '{key}' must be a non-empty string")
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PluginManifest":
        """Create PluginManifest from dict with validation."""
        schema_version = cls._require_str(data, "schema_version")
        name = cls._require_str(data, "name")
        version = cls._require_str(data, "version")
        entrypoint = cls._require_str(data, "entrypoint")
        description = cls._require_str(data, "description")

        lanes = data.get("supported_lanes")
        if not isinstance(lanes, Sequence) or isinstance(lanes, (str, bytes)):
            raise ValueError("Manifest 'supported_lanes' must be a list of lanes")
        supported_lanes = tuple(str(lane) for lane in lanes)
        if not supported_lanes:
            raise ValueError("Manifest 'supported_lanes' must not be empty")

        deterministic = data.get("deterministic")
        if not isinstance(deterministic, bool):
            raise ValueError("Manifest 'deterministic' must be a boolean")

        resources = data.get("resources")
        if not isinstance(resources, Mapping):
            raise ValueError("Manifest 'resources' must be a mapping")
        vram_gb = resources.get("vram_gb")
        max_latency_ms = resources.get("max_latency_ms")
        max_concurrency = resources.get("max_concurrency")
        if not isinstance(vram_gb, (int, float)) or vram_gb <= 0:
            raise ValueError("Manifest 'resources.vram_gb' must be > 0")
        if not isinstance(max_latency_ms, int) or max_latency_ms <= 0:
            raise ValueError("Manifest 'resources.max_latency_ms' must be > 0")
        if not isinstance(max_concurrency, int) or max_concurrency <= 0:
            raise ValueError("Manifest 'resources.max_concurrency' must be > 0")

        io_section = data.get("io")
        if not isinstance(io_section, Mapping):
            raise ValueError("Manifest 'io' must be a mapping")
        input_schema = io_section.get("input_schema")
        output_schema = io_section.get("output_schema")
        if not isinstance(input_schema, Mapping):
            raise ValueError("Manifest 'io.input_schema' must be a mapping")
        if not isinstance(output_schema, Mapping):
            raise ValueError("Manifest 'io.output_schema' must be a mapping")

        return cls(
            schema_version=schema_version,
            name=name,
            version=version,
            entrypoint=entrypoint,
            description=description,
            supported_lanes=supported_lanes,
            deterministic=deterministic,
            resources=PluginResources(
                vram_gb=float(vram_gb),
                max_latency_ms=max_latency_ms,
                max_concurrency=max_concurrency,
            ),
            input_schema=input_schema,
            output_schema=output_schema,
        )
