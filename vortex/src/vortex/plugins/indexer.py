"""Generate a plugin index file for reconciliation with the Rust sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vortex.plugins.host import PluginHost


def build_index(host: PluginHost) -> dict:
    plugins = []
    for manifest in host.registry.manifests():
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


def write_index(output_path: Path) -> None:
    host = PluginHost.from_config()
    data = build_index(host)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Vortex plugin index JSON")
    parser.add_argument(
        "--output",
        default="plugins/index.json",
        help="Output path for plugin index",
    )
    args = parser.parse_args()
    write_index(Path(args.output))


if __name__ == "__main__":
    main()
