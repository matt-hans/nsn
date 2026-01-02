"""Integration tests for plugin sandboxing."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from vortex.plugins.executor import PluginExecutor
from vortex.plugins.policy import PluginPolicy
from vortex.plugins.registry import PluginRegistry
from vortex.plugins.sandbox import DockerSandboxRunner, SandboxConfig


def _docker_available(image: str) -> bool:
    if os.getenv("VORTEX_DOCKER_SANDBOX_TESTS") != "1":
        return False
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=2,
        )
    except Exception:
        return False
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=2,
        )
        return True
    except Exception:
        try:
            result = subprocess.run(
                ["docker", "image", "ls", image, "--format", "{{.Repository}}:{{.Tag}}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=2,
                text=True,
            )
        except Exception:
            return False
        return bool(result.stdout.strip())
    return True


def _write_sandbox_plugin(tmp_path: Path) -> Path:
    plugin_dir = tmp_path / "example"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.py").write_text(
        """
import urllib.request


class ExampleRenderer:
    def __init__(self, manifest=None):
        self.manifest = manifest

    def run(self, payload):
        result = {"read_ok": False, "network_ok": False}
        path = payload.get("path", "")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                handle.read()
            result["read_ok"] = True
        except Exception as exc:
            result["read_error"] = str(exc)

        try:
            urllib.request.urlopen("http://example.com", timeout=1).read()
            result["network_ok"] = True
        except Exception as exc:
            result["network_error"] = str(exc)
        return result
"""
    )
    (plugin_dir / "manifest.yaml").write_text(
        """
schema_version: "1.0"
name: "example-renderer"
version: "0.1.0"
entrypoint: "plugin.py:ExampleRenderer"
description: "Sandbox test plugin"
supported_lanes: ["lane1"]
deterministic: false
resources:
  vram_gb: 1.0
  max_latency_ms: 2000
  max_concurrency: 1
io:
  input_schema:
    type: object
    required: ["path"]
    properties:
      path:
        type: string
  output_schema:
    type: object
    required: ["read_ok", "network_ok"]
    properties:
      read_ok:
        type: boolean
      network_ok:
        type: boolean
"""
    )
    return plugin_dir


@pytest.mark.asyncio
async def test_docker_sandbox_blocks_host_access(tmp_path: Path) -> None:
    image = os.getenv("VORTEX_SANDBOX_IMAGE", "nsn-vortex:latest")
    if not _docker_available(image):
        pytest.skip("docker sandbox not available (set VORTEX_DOCKER_SANDBOX_TESTS=1)")

    _write_sandbox_plugin(tmp_path)
    secret_path = tmp_path / "host-secret.txt"
    secret_path.write_text("secret")

    policy = PluginPolicy(
        max_vram_gb=10.0,
        lane0_max_latency_ms=15000,
        lane1_max_latency_ms=120000,
        allow_untrusted=True,
        allowlist=frozenset(),
    )
    registry = PluginRegistry.from_directory(tmp_path, policy, load_plugins=False)

    config = SandboxConfig.from_dict(
        {
            "enabled": True,
            "engine": "docker",
            "docker_image": image,
            "network": "none",
            "memory_mb": 256,
            "cpu_cores": 1.0,
            "pids_limit": 128,
            "tmpfs_mb": 32,
        }
    )
    runner = DockerSandboxRunner(config)
    executor = PluginExecutor(registry, sandbox_runner=runner)

    result = await executor.execute("example-renderer", {"path": str(secret_path)})
    assert result.output["read_ok"] is False
    assert result.output["network_ok"] is False
