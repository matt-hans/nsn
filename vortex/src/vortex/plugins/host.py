"""Convenience host to load plugin registry from Vortex config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vortex.plugins.executor import PluginExecutor
from vortex.plugins.registry import PluginRegistry


@dataclass
class PluginHost:
    """Load plugins and provide execution utilities based on config."""

    registry: PluginRegistry
    executor: PluginExecutor
    config: dict[str, Any]

    @classmethod
    def from_config(cls, config_path: str | None = None) -> "PluginHost":
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config.yaml")
        config = yaml.safe_load(Path(config_path).read_text())
        plugin_cfg = config.get("plugins", {}) if isinstance(config, dict) else {}
        base_path = Path(config_path).parent
        registry = PluginRegistry.from_config(plugin_cfg, base_path)
        executor = PluginExecutor(registry)
        return cls(registry=registry, executor=executor, config=config)
