"""Error types for Lane 0 video renderer system."""

from __future__ import annotations


class RendererError(Exception):
    """Base class for renderer errors."""

    pass


class RendererLoadError(RendererError):
    """Raised when renderer loading fails."""

    pass


class RendererNotFoundError(RendererError):
    """Raised when a requested renderer is not found."""

    pass


class RendererPolicyError(RendererError):
    """Raised when a renderer violates policy constraints."""

    pass


class RecipeValidationError(RendererError):
    """Raised when recipe validation fails."""

    pass


class DeterminismError(RendererError):
    """Raised when determinism verification fails."""

    pass
