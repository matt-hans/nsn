# Vortex Plugins

Vortex plugins allow third-party renderers to run on the network as long as they
meet resource and latency guarantees. Plugins are discovered from a directory of
subfolders containing a `manifest.yaml` and a Python entrypoint.

## Manifest Schema (manifest.yaml)

Required keys:

- `schema_version`: string
- `name`: string (unique)
- `version`: string
- `entrypoint`: string (`module:Class` or `plugin.py:Class`)
- `description`: string
- `supported_lanes`: list (`lane0`, `lane1`)
- `deterministic`: bool (must be true for lane0)
- `resources`:
  - `vram_gb`: float
  - `max_latency_ms`: int
  - `max_concurrency`: int
- `io`:
  - `input_schema`: minimal JSON schema (type/object/properties/required)
  - `output_schema`: minimal JSON schema

Example:

```yaml
schema_version: "1.0"
name: "example-renderer"
version: "0.1.0"
entrypoint: "plugin.py:ExampleRenderer"
description: "Example renderer plugin"
supported_lanes: ["lane1"]
deterministic: false
resources:
  vram_gb: 4.5
  max_latency_ms: 20000
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
```

## Plugin Class Interface

Plugins must implement a `run(payload: dict) -> dict` method. It may be sync or
`async def`. The loader will enforce schema validation and latency budgets.

```python
from vortex.plugins.types import PluginManifest

class ExampleRenderer:
    def __init__(self, manifest: PluginManifest) -> None:
        self.manifest = manifest

    async def run(self, payload: dict) -> dict:
        prompt = payload["prompt"]
        return {"output_cid": f"cid://{prompt}"}
```

## Configuration

Enable plugins via `vortex/config.yaml`:

```yaml
plugins:
  enabled: true
  directory: "plugins"
  sandbox:
    # Enable sandboxed execution (required for allow_untrusted)
    enabled: true
    # Sandbox engine: "docker" (recommended) or "process" (dev only)
    engine: "docker"
    docker_image: "nsn-vortex:latest"
    docker_bin: "docker"
    network: "none"
    memory_mb: 4096
    cpu_cores: 2.0
    pids_limit: 256
    tmpfs_mb: 64
    gpus: null
    timeout_grace_ms: 5000
  policy:
    max_vram_gb: 11.5
    lane0_max_latency_ms: 15000
    lane1_max_latency_ms: 120000
    # SECURITY: Untrusted plugins are DISABLED by default.
    # Only plugins in the allowlist can execute.
    allow_untrusted: false
    allowlist: ["example-renderer"]
```

## Plugin Index (Reconciliation)

Generate a plugin index JSON file for the Rust sidecar to reconcile:

```bash
python3 -m vortex.plugins.indexer --output plugins/index.json
```

## Plugin Runner (Local Execution)

Run a plugin locally via CLI (used by the Rust sidecar fallback executor):

```bash
python3 -m vortex.plugins.runner --plugin example-renderer --payload '{"prompt":"hi"}'
```

When sandboxing is enabled, the runner dispatches execution into the sandbox
backend instead of running in-process.

## Notes

- Lane 0 plugins must be deterministic and meet the tighter latency budget.
- The executor enforces schema validation and latency; resource enforcement is
  based on declared manifest values.
- For production, set `allow_untrusted: false` and populate `allowlist` with
  approved plugin names.
- When `sandbox.enabled: true`, plugin code executes inside the sandbox (Docker
  by default) and never runs in-process on the host.
- Ensure the sandbox image is available (e.g., build `nsn-vortex:latest` via
  `docker/Dockerfile.vortex`).
