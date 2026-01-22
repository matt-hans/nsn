"""Unit tests for Lane 0 renderer interface.

Tests for:
- RendererManifest validation
- RendererResources constraints
- RenderResult dataclass
- DeterministicVideoRenderer ABC contract
"""

import pytest
import torch

from vortex.renderers import (
    LANE0_MAX_LATENCY_MS,
    LANE0_MAX_VRAM_GB,
    RendererManifest,
    RendererResources,
    RenderResult,
)


class TestRendererResources:
    """Tests for RendererResources dataclass."""

    def test_create_valid_resources(self):
        """Test creating valid resources."""
        resources = RendererResources(vram_gb=11.8, max_latency_ms=21000)
        assert resources.vram_gb == 11.8
        assert resources.max_latency_ms == 21000

    def test_resources_are_frozen(self):
        """Test that resources are immutable."""
        resources = RendererResources(vram_gb=11.8, max_latency_ms=21000)
        with pytest.raises(AttributeError):
            resources.vram_gb = 12.0


class TestRendererManifest:
    """Tests for RendererManifest validation."""

    def test_valid_manifest(self):
        """Test creating a valid manifest."""
        data = {
            "schema_version": "1.0",
            "name": "test-renderer",
            "version": "1.0.0",
            "entrypoint": "renderer.py:TestRenderer",
            "description": "A test renderer",
            "deterministic": True,
            "resources": {"vram_gb": 11.8, "max_latency_ms": 21000},
            "model_dependencies": ["flux-schnell", "cogvideox"],
        }
        manifest = RendererManifest.from_dict(data)
        assert manifest.name == "test-renderer"
        assert manifest.deterministic is True
        assert manifest.resources.vram_gb == 11.8
        assert len(manifest.model_dependencies) == 2

    def test_missing_required_field(self):
        """Test that missing required fields raise ValueError."""
        data = {
            "schema_version": "1.0",
            "name": "test-renderer",
            # Missing version, entrypoint, etc.
        }
        with pytest.raises(ValueError, match="version"):
            RendererManifest.from_dict(data)

    def test_empty_name_rejected(self):
        """Test that empty name is rejected."""
        data = {
            "schema_version": "1.0",
            "name": "",
            "version": "1.0.0",
            "entrypoint": "renderer.py:TestRenderer",
            "description": "Test",
            "deterministic": True,
            "resources": {"vram_gb": 11.8, "max_latency_ms": 21000},
            "model_dependencies": [],
        }
        with pytest.raises(ValueError, match="name"):
            RendererManifest.from_dict(data)

    def test_non_deterministic_rejected(self):
        """Test that non-deterministic renderers are rejected for Lane 0."""
        data = {
            "schema_version": "1.0",
            "name": "test-renderer",
            "version": "1.0.0",
            "entrypoint": "renderer.py:TestRenderer",
            "description": "Test",
            "deterministic": False,  # Lane 0 requires determinism
            "resources": {"vram_gb": 11.8, "max_latency_ms": 21000},
            "model_dependencies": [],
        }
        with pytest.raises(ValueError, match="deterministic"):
            RendererManifest.from_dict(data)

    def test_invalid_vram(self):
        """Test that invalid VRAM values are rejected."""
        data = {
            "schema_version": "1.0",
            "name": "test-renderer",
            "version": "1.0.0",
            "entrypoint": "renderer.py:TestRenderer",
            "description": "Test",
            "deterministic": True,
            "resources": {"vram_gb": 0, "max_latency_ms": 21000},
            "model_dependencies": [],
        }
        with pytest.raises(ValueError, match="vram_gb"):
            RendererManifest.from_dict(data)

    def test_invalid_latency(self):
        """Test that invalid latency values are rejected."""
        data = {
            "schema_version": "1.0",
            "name": "test-renderer",
            "version": "1.0.0",
            "entrypoint": "renderer.py:TestRenderer",
            "description": "Test",
            "deterministic": True,
            "resources": {"vram_gb": 11.8, "max_latency_ms": -1},
            "model_dependencies": [],
        }
        with pytest.raises(ValueError, match="max_latency_ms"):
            RendererManifest.from_dict(data)


class TestRenderResult:
    """Tests for RenderResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful render result."""
        result = RenderResult(
            video_frames=torch.randn(1080, 3, 512, 512),
            audio_waveform=torch.randn(1080000),
            clip_embedding=torch.randn(512),
            generation_time_ms=15000.0,
            determinism_proof=b"abc123",
            success=True,
        )
        assert result.success is True
        assert result.error_msg is None
        assert result.video_frames.shape == (1080, 3, 512, 512)

    def test_failed_result(self):
        """Test creating a failed render result."""
        result = RenderResult(
            video_frames=torch.empty(0),
            audio_waveform=torch.empty(0),
            clip_embedding=torch.empty(0),
            generation_time_ms=1000.0,
            determinism_proof=b"",
            success=False,
            error_msg="CUDA OOM",
        )
        assert result.success is False
        assert result.error_msg == "CUDA OOM"


class TestLane0Constants:
    """Tests for Lane 0 constraint constants."""

    def test_vram_limit(self):
        """Test VRAM limit constant."""
        assert LANE0_MAX_VRAM_GB == 11.5

    def test_latency_limit(self):
        """Test latency limit constant."""
        assert LANE0_MAX_LATENCY_MS == 15000
