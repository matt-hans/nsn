"""Policy enforcement for plugin resource and latency guarantees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vortex.plugins.errors import PolicyViolationError
from vortex.plugins.types import PluginManifest


@dataclass(frozen=True)
class PluginPolicy:
    """Policy constraints for accepting and executing plugins."""

    max_vram_gb: float
    lane0_max_latency_ms: int
    lane1_max_latency_ms: int
    allow_untrusted: bool
    allowlist: frozenset[str]

    def check(self, manifest: PluginManifest) -> None:
        """Validate manifest against policy constraints."""
        if manifest.resources.vram_gb > self.max_vram_gb:
            raise PolicyViolationError(
                f"Plugin '{manifest.name}' requires {manifest.resources.vram_gb:.2f}GB VRAM, "
                f"exceeds policy max {self.max_vram_gb:.2f}GB"
            )

        if not self.allow_untrusted and manifest.name not in self.allowlist:
            raise PolicyViolationError(
                f"Plugin '{manifest.name}' not in allowlist for this node"
            )

        for lane in manifest.supported_lanes:
            if lane == "lane0":
                if not manifest.deterministic:
                    raise PolicyViolationError(
                        f"Plugin '{manifest.name}' must be deterministic for lane0"
                    )
                if manifest.resources.max_latency_ms > self.lane0_max_latency_ms:
                    raise PolicyViolationError(
                        f"Plugin '{manifest.name}' latency {manifest.resources.max_latency_ms}ms "
                        f"exceeds lane0 max {self.lane0_max_latency_ms}ms"
                    )
            elif lane == "lane1":
                if manifest.resources.max_latency_ms > self.lane1_max_latency_ms:
                    raise PolicyViolationError(
                        f"Plugin '{manifest.name}' latency {manifest.resources.max_latency_ms}ms "
                        f"exceeds lane1 max {self.lane1_max_latency_ms}ms"
                    )
            else:
                raise PolicyViolationError(
                    f"Plugin '{manifest.name}' declares unsupported lane '{lane}'"
                )

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "PluginPolicy":
        """Create a policy from config dict."""
        max_vram_gb = float(config.get("max_vram_gb", 11.5))
        lane0_max_latency_ms = int(config.get("lane0_max_latency_ms", 15000))
        lane1_max_latency_ms = int(config.get("lane1_max_latency_ms", 120000))
        allow_untrusted = bool(config.get("allow_untrusted", False))
        allowlist = config.get("allowlist", [])
        if isinstance(allowlist, (list, tuple, set)):
            allowlist_set = frozenset(str(item) for item in allowlist)
        else:
            allowlist_set = frozenset()
        return cls(
            max_vram_gb=max_vram_gb,
            lane0_max_latency_ms=lane0_max_latency_ms,
            lane1_max_latency_ms=lane1_max_latency_ms,
            allow_untrusted=allow_untrusted,
            allowlist=allowlist_set,
        )


def policy_from_config(config: dict[str, object] | None) -> PluginPolicy:
    """Helper to create a policy from an optional config block."""
    if config is None:
        return PluginPolicy.from_config({})
    return PluginPolicy.from_config(config)


def normalize_allowlist(items: Iterable[str]) -> frozenset[str]:
    """Normalize allowlist entries into a frozenset of strings."""
    return frozenset(str(item) for item in items)
