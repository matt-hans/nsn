"""Loader utilities for plugin manifests and entrypoints."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from vortex.plugins.errors import ManifestError, PluginLoadError
from vortex.plugins.types import PluginManifest


class _PathInserter:
    """Temporarily insert a path to sys.path for module imports."""

    def __init__(self, path: Path) -> None:
        self._path = str(path)

    def __enter__(self) -> None:
        if self._path not in sys.path:
            sys.path.insert(0, self._path)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._path in sys.path:
            sys.path.remove(self._path)


def load_manifest(path: Path) -> PluginManifest:
    """Load and validate a plugin manifest from YAML/JSON."""
    try:
        raw = yaml.safe_load(path.read_text())
    except Exception as exc:
        raise ManifestError(f"Failed to parse manifest at {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ManifestError(f"Manifest at {path} must be a mapping")

    try:
        return PluginManifest.from_dict(raw)
    except ValueError as exc:
        raise ManifestError(f"Invalid manifest at {path}: {exc}") from exc


def _load_module_from_file(module_path: Path) -> ModuleType:
    module_name = f"vortex_plugin_{module_path.stem}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise PluginLoadError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    return module


def load_plugin(entrypoint: str, base_path: Path, manifest: PluginManifest) -> Any:
    """Load a plugin class from entrypoint and instantiate it."""
    if ":" not in entrypoint:
        raise PluginLoadError("Entrypoint must be in 'module:Class' format")

    module_ref, symbol = entrypoint.split(":", 1)
    module_path = base_path / module_ref

    try:
        if module_ref.endswith(".py"):
            if not module_path.exists():
                raise PluginLoadError(f"Plugin module file not found: {module_path}")
            module = _load_module_from_file(module_path)
        elif module_path.exists():
            module = _load_module_from_file(module_path)
        else:
            with _PathInserter(base_path):
                module = importlib.import_module(module_ref)
    except Exception as exc:
        raise PluginLoadError(f"Failed to import plugin module '{module_ref}': {exc}") from exc

    try:
        plugin_cls = getattr(module, symbol)
    except AttributeError as exc:
        raise PluginLoadError(
            f"Plugin entrypoint '{entrypoint}' missing symbol '{symbol}'"
        ) from exc

    try:
        plugin = plugin_cls(manifest)
    except TypeError:
        plugin = plugin_cls()

    if not hasattr(plugin, "run") or not callable(getattr(plugin, "run")):
        raise PluginLoadError(f"Plugin '{manifest.name}' missing required run() method")

    declared_manifest = getattr(plugin, "manifest", None)
    if declared_manifest is not None and declared_manifest != manifest:
        raise PluginLoadError(
            f"Plugin '{manifest.name}' manifest mismatch between code and manifest file"
        )

    return plugin
