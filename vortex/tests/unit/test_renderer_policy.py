"""Unit tests for RendererPolicy enforcement."""

import pytest

from vortex.renderers import (
    LANE0_MAX_LATENCY_MS,
    LANE0_MAX_VRAM_GB,
    RendererManifest,
    RendererPolicy,
    RendererPolicyError,
    RendererResources,
    policy_from_config,
)


def make_manifest(
    name: str = "test-renderer",
    vram_gb: float = 11.0,
    max_latency_ms: int = 15000,
    deterministic: bool = True,
) -> RendererManifest:
    """Helper to create a manifest for testing."""
    return RendererManifest(
        schema_version="1.0",
        name=name,
        version="1.0.0",
        entrypoint="renderer.py:TestRenderer",
        description="Test renderer",
        deterministic=deterministic,
        resources=RendererResources(vram_gb=vram_gb, max_latency_ms=max_latency_ms),
        model_dependencies=(),
    )


class TestRendererPolicy:
    """Tests for RendererPolicy constraint enforcement."""

    def test_default_policy_values(self):
        """Test default policy values match Lane 0 constraints."""
        policy = RendererPolicy()
        assert policy.max_vram_gb == LANE0_MAX_VRAM_GB
        assert policy.max_latency_ms == LANE0_MAX_LATENCY_MS
        assert policy.require_determinism is True
        assert policy.allowlist is None

    def test_valid_manifest_passes(self):
        """Test that a valid manifest passes policy check."""
        policy = RendererPolicy()
        manifest = make_manifest(vram_gb=11.0, max_latency_ms=15000)
        # Should not raise
        policy.check(manifest)

    def test_vram_exceeded_rejected(self):
        """Test that VRAM exceeding limit is rejected."""
        policy = RendererPolicy(max_vram_gb=11.5)
        manifest = make_manifest(vram_gb=12.0)  # Exceeds limit
        with pytest.raises(RendererPolicyError, match="VRAM"):
            policy.check(manifest)

    def test_latency_exceeded_rejected(self):
        """Test that latency exceeding limit is rejected."""
        policy = RendererPolicy(max_latency_ms=15000)
        manifest = make_manifest(max_latency_ms=20000)  # Exceeds limit
        with pytest.raises(RendererPolicyError, match="latency"):
            policy.check(manifest)

    def test_non_deterministic_rejected(self):
        """Test that non-deterministic renderers are rejected."""
        policy = RendererPolicy(require_determinism=True)
        manifest = make_manifest(deterministic=False)
        with pytest.raises(RendererPolicyError, match="deterministic"):
            policy.check(manifest)

    def test_allowlist_enforced(self):
        """Test that allowlist is enforced."""
        policy = RendererPolicy(allowlist=("allowed-renderer",))
        manifest = make_manifest(name="not-allowed")
        with pytest.raises(RendererPolicyError, match="allowlist"):
            policy.check(manifest)

    def test_allowlist_allows_listed(self):
        """Test that allowlisted renderers pass."""
        policy = RendererPolicy(allowlist=("allowed-renderer",))
        manifest = make_manifest(name="allowed-renderer")
        # Should not raise
        policy.check(manifest)

    def test_custom_limits(self):
        """Test custom policy limits."""
        policy = RendererPolicy(max_vram_gb=16.0, max_latency_ms=30000)
        manifest = make_manifest(vram_gb=15.0, max_latency_ms=25000)
        # Should not raise with higher limits
        policy.check(manifest)


class TestPolicyFromConfig:
    """Tests for policy_from_config factory function."""

    def test_none_config_returns_default(self):
        """Test that None config returns default policy."""
        policy = policy_from_config(None)
        assert policy.max_vram_gb == LANE0_MAX_VRAM_GB
        assert policy.max_latency_ms == LANE0_MAX_LATENCY_MS

    def test_custom_config(self):
        """Test custom config values."""
        config = {
            "max_vram_gb": 16.0,
            "max_latency_ms": 30000,
            "require_determinism": False,
        }
        policy = policy_from_config(config)
        assert policy.max_vram_gb == 16.0
        assert policy.max_latency_ms == 30000
        assert policy.require_determinism is False

    def test_allowlist_from_config(self):
        """Test allowlist from config."""
        config = {"allowlist": ["renderer-a", "renderer-b"]}
        policy = policy_from_config(config)
        assert policy.allowlist == ("renderer-a", "renderer-b")
