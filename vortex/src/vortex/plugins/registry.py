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
    root: Path
    plugin: Any | None


class PluginRegistry:
    """Loads and stores plugins from a directory."""

    def __init__(self, policy: PluginPolicy) -> None:
        self._policy = policy
        self._records: dict[str, PluginRecord] = {}

    @property
    def policy(self) -> PluginPolicy:
        return self._policy

    def register(self, manifest: PluginManifest, root: Path, plugin: Any | None) -> None:
        if manifest.name in self._records:
            raise PluginLoadError(f"Duplicate plugin name '{manifest.name}'")
        self._records[manifest.name] = PluginRecord(manifest=manifest, root=root, plugin=plugin)

    def list_plugins(self) -> list[str]:
        return sorted(self._records.keys())

    def get_manifest(self, name: str) -> PluginManifest:
        if name not in self._records:
            raise PluginLoadError(f"Plugin '{name}' not found")
        return self._records[name].manifest

    def get_plugin(self, name: str) -> Any:
        if name not in self._records:
            raise PluginLoadError(f"Plugin '{name}' not found")
        plugin = self._records[name].plugin
        if plugin is None:
            raise PluginLoadError(
                f"Plugin '{name}' is not loaded in-process (sandboxed execution enabled)"
            )
        return plugin

    def get_root(self, name: str) -> Path:
        if name not in self._records:
            raise PluginLoadError(f"Plugin '{name}' not found")
        return self._records[name].root

    def manifests(self) -> list[PluginManifest]:
        return [record.manifest for record in self._records.values()]

    @classmethod
    def from_directory(
        cls, root: Path, policy: PluginPolicy, *, load_plugins: bool = True
    ) -> PluginRegistry:
        if load_plugins and policy.allow_untrusted:
            raise PluginLoadError(
                "Refusing to load plugins in-process when allow_untrusted is enabled"
            )
        registry = cls(policy=policy)
        if not root.exists():
            return registry

        manifest_paths = sorted(root.glob("*/manifest.yaml"))
        manifest_paths.extend(sorted(root.glob("*/manifest.yml")))
        for manifest_path in manifest_paths:
            manifest = load_manifest(manifest_path)
            registry.policy.check(manifest)
            plugin = (
                load_plugin(manifest.entrypoint, manifest_path.parent, manifest)
                if load_plugins
                else None
            )
            registry.register(manifest, manifest_path.parent, plugin)
        return registry

    @classmethod
    def from_config(
        cls,
        config: dict[str, object] | None,
        base_path: Path,
        *,
        policy: PluginPolicy | None = None,
        load_plugins: bool = True,
    ) -> PluginRegistry:
        cfg = config or {}
        policy_cfg = cfg.get("policy") if isinstance(cfg, dict) else None
        policy_value = policy or policy_from_config(policy_cfg)
        if isinstance(cfg, dict) and not bool(cfg.get("enabled", False)):
            return cls(policy=policy_value)
        root = cfg.get("directory", "plugins") if isinstance(cfg, dict) else "plugins"
        root_path = (base_path / str(root)).resolve()
        return cls.from_directory(root_path, policy_value, load_plugins=load_plugins)
