"""Policy enforcement for Lane 0 video renderers.

This module defines the RendererPolicy class that enforces constraints
on renderers. Lane 0 has strict requirements:
- Determinism (required for BFT consensus)
- VRAM budget (must fit in 11.5GB)
- Latency budget (must complete in 15s)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from vortex.renderers.errors import RendererPolicyError
from vortex.renderers.types import LANE0_MAX_LATENCY_MS, LANE0_MAX_VRAM_GB, RendererManifest


@dataclass(frozen=True)
class RendererPolicy:
    """Policy constraints for Lane 0 renderers.

    Attributes:
        max_vram_gb: Maximum VRAM budget in GB (default: 11.5)
        max_latency_ms: Maximum latency budget in ms (default: 15000)
        require_determinism: Whether determinism is required (default: True)
        allowlist: Optional list of allowed renderer names (None = allow all)
    """

    max_vram_gb: float = LANE0_MAX_VRAM_GB
    max_latency_ms: int = LANE0_MAX_LATENCY_MS
    require_determinism: bool = True
    allowlist: tuple[str, ...] | None = None

    def check(self, manifest: RendererManifest) -> None:
        """Check if manifest satisfies policy constraints.

        Args:
            manifest: Renderer manifest to validate

        Raises:
            RendererPolicyError: If any constraint is violated
        """
        errors: list[str] = []

        # Check allowlist
        if self.allowlist is not None and manifest.name not in self.allowlist:
            errors.append(
                f"Renderer '{manifest.name}' not in allowlist: {self.allowlist}"
            )

        # Check determinism
        if self.require_determinism and not manifest.deterministic:
            errors.append(
                f"Renderer '{manifest.name}' must have deterministic=true for Lane 0"
            )

        # Check VRAM budget
        if manifest.resources.vram_gb > self.max_vram_gb:
            errors.append(
                f"Renderer '{manifest.name}' VRAM ({manifest.resources.vram_gb:.1f}GB) "
                f"exceeds max ({self.max_vram_gb:.1f}GB)"
            )

        # Check latency budget
        if manifest.resources.max_latency_ms > self.max_latency_ms:
            errors.append(
                f"Renderer '{manifest.name}' latency ({manifest.resources.max_latency_ms}ms) "
                f"exceeds max ({self.max_latency_ms}ms)"
            )

        if errors:
            raise RendererPolicyError("; ".join(errors))


def policy_from_config(config: Mapping[str, Any] | None) -> RendererPolicy:
    """Create RendererPolicy from config dict.

    Args:
        config: Optional config dict with policy overrides

    Returns:
        RendererPolicy instance
    """
    if config is None:
        return RendererPolicy()

    allowlist = config.get("allowlist")
    if allowlist is not None:
        if not isinstance(allowlist, (list, tuple)):
            raise ValueError("Policy 'allowlist' must be a list")
        allowlist = tuple(str(name) for name in allowlist)

    return RendererPolicy(
        max_vram_gb=float(config.get("max_vram_gb", LANE0_MAX_VRAM_GB)),
        max_latency_ms=int(config.get("max_latency_ms", LANE0_MAX_LATENCY_MS)),
        require_determinism=bool(config.get("require_determinism", True)),
        allowlist=allowlist,
    )
