"""Registry for Lane 0 video renderers.

This module provides the RendererRegistry class that discovers, loads,
and manages renderer implementations. Adapted from plugins/registry.py
with renderer-specific loading logic.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from vortex.renderers.base import DeterministicVideoRenderer
from vortex.renderers.errors import RendererLoadError, RendererNotFoundError
from vortex.renderers.policy import RendererPolicy, policy_from_config
from vortex.renderers.types import RendererManifest

logger = logging.getLogger(__name__)


class _PathInserter:
    """Temporarily insert a path to sys.path for module imports."""

    def __init__(self, path: Path) -> None:
        self._path = str(path)

    def __enter__(self) -> None:
        if self._path not in sys.path:
            sys.path.insert(0, self._path)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._path in sys.path:
            sys.path.remove(self._path)


def load_manifest(path: Path) -> RendererManifest:
    """Load and validate a renderer manifest from YAML.

    Args:
        path: Path to manifest.yaml file

    Returns:
        Validated RendererManifest

    Raises:
        RendererLoadError: If parsing or validation fails
    """
    try:
        raw = yaml.safe_load(path.read_text())
    except Exception as exc:
        raise RendererLoadError(f"Failed to parse manifest at {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RendererLoadError(f"Manifest at {path} must be a mapping")

    try:
        return RendererManifest.from_dict(raw)
    except ValueError as exc:
        raise RendererLoadError(f"Invalid manifest at {path}: {exc}") from exc


def _load_module_from_file(module_path: Path) -> ModuleType:
    """Load a Python module from file path."""
    module_name = f"vortex_renderer_{module_path.stem}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RendererLoadError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_renderer(
    entrypoint: str, base_path: Path, manifest: RendererManifest
) -> DeterministicVideoRenderer:
    """Load a renderer class from entrypoint and instantiate it.

    Args:
        entrypoint: Module:Class path (e.g., "renderer:DefaultRenderer")
        base_path: Base directory containing the module
        manifest: Renderer manifest for passing to constructor

    Returns:
        Instantiated DeterministicVideoRenderer

    Raises:
        RendererLoadError: If loading fails
    """
    if ":" not in entrypoint:
        raise RendererLoadError("Entrypoint must be in 'module:Class' format")

    module_ref, symbol = entrypoint.split(":", 1)
    module_path = base_path / module_ref

    try:
        if module_ref.endswith(".py"):
            if not module_path.exists():
                raise RendererLoadError(f"Renderer module file not found: {module_path}")
            module = _load_module_from_file(module_path)
        elif module_path.exists():
            module = _load_module_from_file(module_path)
        else:
            with _PathInserter(base_path):
                module = importlib.import_module(module_ref)
    except RendererLoadError:
        raise
    except Exception as exc:
        raise RendererLoadError(
            f"Failed to import renderer module '{module_ref}': {exc}"
        ) from exc

    try:
        renderer_cls = getattr(module, symbol)
    except AttributeError as exc:
        raise RendererLoadError(
            f"Renderer entrypoint '{entrypoint}' missing symbol '{symbol}'"
        ) from exc

    # Instantiate renderer
    try:
        renderer = renderer_cls(manifest)
    except TypeError:
        renderer = renderer_cls()

    # Validate it's a proper renderer
    if not isinstance(renderer, DeterministicVideoRenderer):
        raise RendererLoadError(
            f"Renderer '{manifest.name}' must inherit from DeterministicVideoRenderer"
        )

    return renderer


@dataclass
class RendererRecord:
    """Record of a registered renderer."""

    manifest: RendererManifest
    root: Path
    renderer: DeterministicVideoRenderer | None


class RendererRegistry:
    """Loads and stores renderers from a directory.

    Example:
        >>> registry = RendererRegistry.from_directory(Path("renderers"), policy)
        >>> renderer = registry.get("default-flux-liveportrait")
        >>> await renderer.initialize("cuda:0", config)
    """

    def __init__(self, policy: RendererPolicy) -> None:
        """Initialize registry with policy.

        Args:
            policy: Policy for validating renderers
        """
        self._policy = policy
        self._records: dict[str, RendererRecord] = {}

    @property
    def policy(self) -> RendererPolicy:
        """Return the policy used by this registry."""
        return self._policy

    def register(
        self,
        manifest: RendererManifest,
        root: Path,
        renderer: DeterministicVideoRenderer | None,
    ) -> None:
        """Register a renderer.

        Args:
            manifest: Renderer manifest
            root: Root directory containing the renderer
            renderer: Instantiated renderer (None if not loaded)

        Raises:
            RendererLoadError: If renderer with same name already exists
        """
        if manifest.name in self._records:
            raise RendererLoadError(f"Duplicate renderer name '{manifest.name}'")
        self._records[manifest.name] = RendererRecord(
            manifest=manifest, root=root, renderer=renderer
        )
        logger.info(f"Registered renderer: {manifest.name} v{manifest.version}")

    def list_renderers(self) -> list[str]:
        """Return sorted list of registered renderer names."""
        return sorted(self._records.keys())

    def get_manifest(self, name: str) -> RendererManifest:
        """Get manifest for a renderer by name.

        Args:
            name: Renderer name

        Returns:
            RendererManifest

        Raises:
            RendererNotFoundError: If renderer not found
        """
        if name not in self._records:
            raise RendererNotFoundError(f"Renderer '{name}' not found")
        return self._records[name].manifest

    def get(self, name: str) -> DeterministicVideoRenderer:
        """Get a renderer by name.

        Args:
            name: Renderer name

        Returns:
            DeterministicVideoRenderer instance

        Raises:
            RendererNotFoundError: If renderer not found
            RendererLoadError: If renderer was not loaded
        """
        if name not in self._records:
            raise RendererNotFoundError(f"Renderer '{name}' not found")
        renderer = self._records[name].renderer
        if renderer is None:
            raise RendererLoadError(f"Renderer '{name}' is not loaded")
        return renderer

    def get_root(self, name: str) -> Path:
        """Get root directory for a renderer.

        Args:
            name: Renderer name

        Returns:
            Path to renderer root directory

        Raises:
            RendererNotFoundError: If renderer not found
        """
        if name not in self._records:
            raise RendererNotFoundError(f"Renderer '{name}' not found")
        return self._records[name].root

    def manifests(self) -> list[RendererManifest]:
        """Return list of all registered manifests."""
        return [record.manifest for record in self._records.values()]

    @classmethod
    def from_directory(
        cls, root: Path, policy: RendererPolicy, *, load_renderers: bool = True
    ) -> RendererRegistry:
        """Create registry by scanning a directory for renderers.

        Args:
            root: Root directory to scan (looks for */manifest.yaml)
            policy: Policy for validating renderers
            load_renderers: Whether to load renderer classes

        Returns:
            Populated RendererRegistry
        """
        registry = cls(policy=policy)
        if not root.exists():
            logger.warning(f"Renderers directory does not exist: {root}")
            return registry

        manifest_paths = sorted(root.glob("*/manifest.yaml"))
        manifest_paths.extend(sorted(root.glob("*/manifest.yml")))

        for manifest_path in manifest_paths:
            try:
                manifest = load_manifest(manifest_path)
                policy.check(manifest)

                renderer = (
                    load_renderer(manifest.entrypoint, manifest_path.parent, manifest)
                    if load_renderers
                    else None
                )
                registry.register(manifest, manifest_path.parent, renderer)
            except Exception as exc:
                logger.error(f"Failed to load renderer from {manifest_path}: {exc}")
                raise

        return registry

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        base_path: Path,
        *,
        policy: RendererPolicy | None = None,
        load_renderers: bool = True,
    ) -> RendererRegistry:
        """Create registry from config dict.

        Args:
            config: Config dict with optional 'directory' and 'policy' keys
            base_path: Base path for resolving relative directories
            policy: Optional policy override
            load_renderers: Whether to load renderer classes

        Returns:
            Populated RendererRegistry
        """
        cfg = config or {}
        policy_value = policy or policy_from_config(
            cfg.get("policy") if isinstance(cfg, dict) else None
        )

        # Get renderers directory
        directory = cfg.get("directory", "renderers") if isinstance(cfg, dict) else "renderers"
        root_path = (base_path / str(directory)).resolve()

        return cls.from_directory(root_path, policy_value, load_renderers=load_renderers)
