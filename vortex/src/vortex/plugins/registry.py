"""Registry for Vortex plugins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vortex.plugins.errors import PluginLoadError
from vortex.plugins.loader import load_manifest, load_plugin
from vortex.plugins.policy import PluginPolicy, policy_from_config
from vortex.plugins.types import PluginManifest


@dataclass
class PluginRecord:
    manifest: PluginManifest
    plugin: Any


class PluginRegistry:
    """Loads and stores plugins from a directory."""

    def __init__(self, policy: PluginPolicy) -> None:
        self._policy = policy
        self._records: dict[str, PluginRecord] = {}

    @property
    def policy(self) -> PluginPolicy:
        return self._policy

    def register(self, manifest: PluginManifest, plugin: Any) -> None:
        if manifest.name in self._records:
            raise PluginLoadError(f"Duplicate plugin name '{manifest.name}'")
        self._records[manifest.name] = PluginRecord(manifest=manifest, plugin=plugin)

    def list_plugins(self) -> list[str]:
        return sorted(self._records.keys())

    def get_manifest(self, name: str) -> PluginManifest:
        if name not in self._records:
            raise PluginLoadError(f"Plugin '{name}' not found")
        return self._records[name].manifest

    def get_plugin(self, name: str) -> Any:
        if name not in self._records:
            raise PluginLoadError(f"Plugin '{name}' not found")
        return self._records[name].plugin

    def manifests(self) -> list[PluginManifest]:
        return [record.manifest for record in self._records.values()]

    @classmethod
    def from_directory(cls, root: Path, policy: PluginPolicy) -> "PluginRegistry":
        registry = cls(policy=policy)
        if not root.exists():
            return registry

        manifest_paths = sorted(root.glob("*/manifest.yaml"))
        manifest_paths.extend(sorted(root.glob("*/manifest.yml")))
        for manifest_path in manifest_paths:
            manifest = load_manifest(manifest_path)
            registry.policy.check(manifest)
            plugin = load_plugin(manifest.entrypoint, manifest_path.parent, manifest)
            registry.register(manifest, plugin)
        return registry

    @classmethod
    def from_config(cls, config: dict[str, object] | None, base_path: Path) -> "PluginRegistry":
        cfg = config or {}
        policy = policy_from_config(cfg.get("policy") if isinstance(cfg, dict) else None)
        if isinstance(cfg, dict) and not bool(cfg.get("enabled", False)):
            return cls(policy=policy)
        root = cfg.get("directory", "plugins") if isinstance(cfg, dict) else "plugins"
        root_path = (base_path / str(root)).resolve()
        return cls.from_directory(root_path, policy)
