"""Unit tests for Vortex plugin system."""

from __future__ import annotations

from pathlib import Path

import pytest

from vortex.plugins.executor import PluginExecutor
from vortex.plugins.errors import ManifestError, PolicyViolationError, SchemaValidationError
from vortex.plugins.loader import load_manifest
from vortex.plugins.policy import PluginPolicy
from vortex.plugins.registry import PluginRegistry
from vortex.plugins.types import PluginManifest


def _write_plugin(tmp_path: Path, *, max_latency_ms: int = 5000) -> Path:
    plugin_dir = tmp_path / "example"
    plugin_dir.mkdir()

    (plugin_dir / "plugin.py").write_text(
        """
class ExampleRenderer:
    def __init__(self, manifest=None):
        self.manifest = manifest

    def run(self, payload):
        prompt = payload["prompt"]
        return {"output_cid": f"cid://{prompt}"}
"""
    )

    (plugin_dir / "manifest.yaml").write_text(
        f"""
schema_version: "1.0"
name: "example-renderer"
version: "0.1.0"
entrypoint: "plugin.py:ExampleRenderer"
description: "Example renderer plugin"
supported_lanes: ["lane1"]
deterministic: false
resources:
  vram_gb: 2.0
  max_latency_ms: {max_latency_ms}
  max_concurrency: 1
io:
  input_schema:
    type: object
    required: ["prompt"]
    properties:
      prompt:
        type: string
  output_schema:
    type: object
    required: ["output_cid"]
    properties:
      output_cid:
        type: string
"""
    )

    return plugin_dir


def test_manifest_loader_rejects_invalid(tmp_path: Path) -> None:
    """Manifest loader rejects missing required fields."""
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("name: bad")

    with pytest.raises(ManifestError, match="schema_version"):
        _ = load_manifest(manifest_path)


def test_policy_rejects_excess_vram(tmp_path: Path) -> None:
    """Policy enforcement rejects plugins that exceed VRAM limits."""
    _write_plugin(tmp_path)
    policy = PluginPolicy(
        max_vram_gb=1.0,
        lane0_max_latency_ms=15000,
        lane1_max_latency_ms=120000,
        allow_untrusted=True,
        allowlist=frozenset(),
    )

    with pytest.raises(PolicyViolationError, match="exceeds policy max"):
        PluginRegistry.from_directory(tmp_path, policy)


@pytest.mark.asyncio
async def test_registry_and_executor_round_trip(tmp_path: Path) -> None:
    """Registry loads plugin and executor enforces schemas."""
    _write_plugin(tmp_path)
    policy = PluginPolicy(
        max_vram_gb=10.0,
        lane0_max_latency_ms=15000,
        lane1_max_latency_ms=120000,
        allow_untrusted=True,
        allowlist=frozenset(),
    )

    registry = PluginRegistry.from_directory(tmp_path, policy)
    executor = PluginExecutor(registry)

    result = await executor.execute("example-renderer", {"prompt": "hello"})
    assert result.output["output_cid"] == "cid://hello"

    with pytest.raises(SchemaValidationError, match="missing required field"):
        await executor.execute("example-renderer", {})


def test_manifest_round_trip_from_dict() -> None:
    """Manifest from_dict returns stable typed structure."""
    manifest = PluginManifest.from_dict(
        {
            "schema_version": "1.0",
            "name": "unit-test",
            "version": "0.0.1",
            "entrypoint": "plugin.py:Plugin",
            "description": "unit",
            "supported_lanes": ["lane1"],
            "deterministic": False,
            "resources": {"vram_gb": 1.0, "max_latency_ms": 1000, "max_concurrency": 1},
            "io": {
                "input_schema": {"type": "object", "properties": {}, "required": []},
                "output_schema": {"type": "object", "properties": {}, "required": []},
            },
        }
    )

    assert manifest.name == "unit-test"
    assert manifest.resources.vram_gb == 1.0
