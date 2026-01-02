"""Run a plugin inside a sandboxed environment."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Mapping

from vortex.plugins.errors import PluginExecutionError
from vortex.plugins.loader import load_manifest, load_plugin


def _load_payload(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise PluginExecutionError("payload must be a JSON object")
    return payload


def _find_manifest(plugin_dir: Path, manifest_arg: str | None) -> Path:
    if manifest_arg:
        path = Path(manifest_arg)
        if path.exists():
            return path
    yaml_path = plugin_dir / "manifest.yaml"
    if yaml_path.exists():
        return yaml_path
    yml_path = plugin_dir / "manifest.yml"
    if yml_path.exists():
        return yml_path
    raise PluginExecutionError(f"Manifest not found in {plugin_dir}")


async def _run_with_timeout(plugin: Any, payload: Mapping[str, Any], budget_ms: int) -> Any:
    if hasattr(plugin, "run") and asyncio.iscoroutinefunction(plugin.run):
        return await asyncio.wait_for(plugin.run(payload), timeout=budget_ms / 1000)

    return await asyncio.wait_for(
        asyncio.to_thread(plugin.run, payload),
        timeout=budget_ms / 1000,
    )


def _serialize_result(output: dict[str, Any], duration_ms: float) -> str:
    return json.dumps({"output": output, "duration_ms": duration_ms})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vortex plugin in sandbox")
    parser.add_argument("--entrypoint", required=True, help="Plugin entrypoint module:Class")
    parser.add_argument("--plugin-dir", required=True, help="Plugin directory with manifest")
    parser.add_argument("--manifest", default=None, help="Optional manifest path")
    parser.add_argument("--payload", required=True, help="JSON payload string")
    parser.add_argument("--timeout-ms", type=int, default=0, help="Execution timeout in ms")
    args = parser.parse_args()

    plugin_dir = Path(args.plugin_dir)
    manifest_path = _find_manifest(plugin_dir, args.manifest)
    manifest = load_manifest(manifest_path)
    plugin = load_plugin(args.entrypoint, plugin_dir, manifest)
    payload = _load_payload(args.payload)
    budget_ms = args.timeout_ms or manifest.resources.max_latency_ms

    async def _run() -> str:
        start = time.monotonic()
        output = await _run_with_timeout(plugin, payload, budget_ms)
        duration_ms = (time.monotonic() - start) * 1000
        if not isinstance(output, dict):
            raise PluginExecutionError("Plugin output must be a dict")
        return _serialize_result(output, duration_ms)

    result = asyncio.run(_run())
    print(result)


if __name__ == "__main__":
    main()
