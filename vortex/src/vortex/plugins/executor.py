"""Execution helpers for plugin workloads."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Mapping

from vortex.plugins.errors import PluginExecutionError
from vortex.plugins.registry import PluginRegistry
from vortex.plugins.sandbox import SandboxRunner
from vortex.plugins.schema import validate_schema


@dataclass(frozen=True)
class PluginExecutionResult:
    output: dict[str, Any]
    duration_ms: float


class PluginExecutor:
    """Execute registered plugins with schema and latency enforcement."""

    def __init__(self, registry: PluginRegistry, sandbox_runner: SandboxRunner | None = None) -> None:
        self._registry = registry
        self._sandbox_runner = sandbox_runner

    async def execute(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        timeout_ms: int | None = None,
    ) -> PluginExecutionResult:
        manifest = self._registry.get_manifest(name)
        validate_schema(manifest.input_schema, payload, context="input")

        budget_ms = manifest.resources.max_latency_ms
        if timeout_ms is not None:
            budget_ms = min(budget_ms, timeout_ms)

        if self._sandbox_runner is None and self._registry.policy.allow_untrusted:
            raise PluginExecutionError(
                "Sandbox runner is required when allow_untrusted is enabled"
            )

        start_time = time.monotonic()
        try:
            if self._sandbox_runner is not None:
                if not self._sandbox_runner.is_available():
                    raise PluginExecutionError(
                        f"Sandbox runner '{self._sandbox_runner.engine}' is unavailable"
                    )
                plugin_root = self._registry.get_root(name)
                result = await self._sandbox_runner.run(
                    manifest, plugin_root, payload, budget_ms
                )
                output = result.output
                duration_ms = result.duration_ms
            else:
                plugin = self._registry.get_plugin(name)
                output = await _run_with_timeout(plugin, payload, budget_ms)
                duration_ms = (time.monotonic() - start_time) * 1000
        except asyncio.TimeoutError as exc:
            raise PluginExecutionError(
                f"Plugin '{name}' exceeded latency budget of {budget_ms}ms"
            ) from exc
        except PluginExecutionError:
            raise
        except Exception as exc:
            raise PluginExecutionError(f"Plugin '{name}' failed: {exc}") from exc

        if not isinstance(output, dict):
            raise PluginExecutionError(
                f"Plugin '{name}' returned {type(output).__name__}, expected dict"
            )

        validate_schema(manifest.output_schema, output, context="output")

        if self._sandbox_runner is not None:
            return PluginExecutionResult(output=output, duration_ms=duration_ms)

        duration_ms = (time.monotonic() - start_time) * 1000
        return PluginExecutionResult(output=output, duration_ms=duration_ms)


async def _run_with_timeout(plugin: Any, payload: Mapping[str, Any], budget_ms: int) -> Any:
    if hasattr(plugin, "run") and asyncio.iscoroutinefunction(plugin.run):
        return await asyncio.wait_for(plugin.run(payload), timeout=budget_ms / 1000)

    return await asyncio.wait_for(
        asyncio.to_thread(plugin.run, payload),
        timeout=budget_ms / 1000,
    )
