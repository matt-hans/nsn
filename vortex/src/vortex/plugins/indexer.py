"""Generate a plugin index file for reconciliation with the Rust sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from vortex.plugins.policy import policy_from_config
from vortex.plugins.registry import PluginRegistry


def build_index(registry: PluginRegistry) -> dict:
    plugins = []
    for manifest in registry.manifests():
        plugins.append(
            {
                "name": manifest.name,
                "version": manifest.version,
                "supported_lanes": list(manifest.supported_lanes),
                "deterministic": manifest.deterministic,
                "max_latency_ms": manifest.resources.max_latency_ms,
                "vram_gb": manifest.resources.vram_gb,
                "max_concurrency": manifest.resources.max_concurrency,
                "schema_version": manifest.schema_version,
            }
        )
    return {"plugins": plugins}


def _load_registry(config_path: Path) -> PluginRegistry:
    config = yaml.safe_load(config_path.read_text())
    plugin_cfg = config.get("plugins", {}) if isinstance(config, dict) else {}
    policy = policy_from_config(
        plugin_cfg.get("policy") if isinstance(plugin_cfg, dict) else None
    )
    base_path = config_path.parent
    return PluginRegistry.from_config(
        plugin_cfg,
        base_path,
        policy=policy,
        load_plugins=False,
    )


def write_index(output_path: Path, config_path: Path) -> None:
    registry = _load_registry(config_path)
    data = build_index(registry)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Vortex plugin index JSON")
    parser.add_argument(
        "--output",
        default="plugins/index.json",
        help="Output path for plugin index",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent.parent / "config.yaml"),
        help="Path to Vortex config.yaml",
    )
    args = parser.parse_args()
    write_index(Path(args.output), Path(args.config))


if __name__ == "__main__":
    main()
