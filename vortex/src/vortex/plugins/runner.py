"""Run a Vortex plugin by name with JSON payload."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from vortex.plugins.host import PluginHost


def _load_payload(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    return payload


def _serialize_result(output: dict[str, Any], duration_ms: float) -> str:
    return json.dumps({"output": output, "duration_ms": duration_ms})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vortex plugin")
    parser.add_argument("--plugin", required=True, help="Plugin name to execute")
    parser.add_argument("--payload", required=True, help="JSON payload string")
    parser.add_argument("--timeout-ms", type=int, default=None, help="Optional timeout in ms")
    args = parser.parse_args()

    host = PluginHost.from_config()
    payload = _load_payload(args.payload)

    async def _run() -> str:
        result = await host.executor.execute(
            args.plugin,
            payload,
            timeout_ms=args.timeout_ms,
        )
        return _serialize_result(result.output, result.duration_ms)

    output = asyncio.run(_run())
    print(output)


if __name__ == "__main__":
    main()
