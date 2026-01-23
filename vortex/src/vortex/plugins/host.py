"""Convenience host to load plugin registry from Vortex config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vortex.plugins.errors import PluginExecutionError
from vortex.plugins.executor import PluginExecutor
from vortex.plugins.policy import policy_from_config
from vortex.plugins.registry import PluginRegistry
from vortex.plugins.sandbox import SandboxRunner, sandbox_from_config


@dataclass
class PluginHost:
    """Load plugins and provide execution utilities based on config."""

    registry: PluginRegistry
    executor: PluginExecutor
    config: dict[str, Any]
    sandbox: SandboxRunner | None

    @classmethod
    def from_config(cls, config_path: str | None = None) -> PluginHost:
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config.yaml")
        config = yaml.safe_load(Path(config_path).read_text())
        plugin_cfg = config.get("plugins", {}) if isinstance(config, dict) else {}
        base_path = Path(config_path).parent
        policy = policy_from_config(
            plugin_cfg.get("policy") if isinstance(plugin_cfg, dict) else None
        )
        sandbox_runner = sandbox_from_config(
            plugin_cfg.get("sandbox") if isinstance(plugin_cfg, dict) else None
        )
        load_plugins = sandbox_runner is None and not policy.allow_untrusted
        registry = PluginRegistry.from_config(
            plugin_cfg,
            base_path,
            policy=policy,
            load_plugins=load_plugins,
        )
        if sandbox_runner is not None and not sandbox_runner.is_available():
            raise PluginExecutionError("sandbox enabled but backend unavailable")
        if policy.allow_untrusted and sandbox_runner is None:
            raise PluginExecutionError("allow_untrusted requires sandbox.enabled=true")
        executor = PluginExecutor(registry, sandbox_runner=sandbox_runner)
        return cls(
            registry=registry,
            executor=executor,
            config=config,
            sandbox=sandbox_runner,
        )
