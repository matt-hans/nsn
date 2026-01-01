"""Unit tests for vortex.utils.memory VRAM utilities."""

from unittest.mock import MagicMock, patch

import pytest

from vortex.utils.memory import (
    clear_cuda_cache,
    format_bytes,
    get_current_vram_usage,
    get_vram_stats,
    log_vram_snapshot,
)


class TestVRAMUtilities:
    """Test VRAM utility functions with mocked CUDA."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_current_vram_usage_no_cuda(self, mock_is_available):
        """Test VRAM usage returns 0 when CUDA unavailable."""
        usage = get_current_vram_usage()
        assert usage == 0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=6_000_000_000)  # 6 GB
    def test_get_current_vram_usage_with_cuda(self, mock_allocated, mock_is_available):
        """Test VRAM usage returns correct value with CUDA."""
        usage = get_current_vram_usage()
        assert usage == 6_000_000_000
        mock_allocated.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_vram_stats_no_cuda(self, mock_is_available):
        """Test VRAM stats returns zeros when CUDA unavailable."""
        stats = get_vram_stats()
        assert stats["allocated_gb"] == 0.0
        assert stats["reserved_gb"] == 0.0
        assert stats["max_allocated_gb"] == 0.0
        assert stats["total_gb"] == 0.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=6_000_000_000)
    @patch("torch.cuda.memory_reserved", return_value=6_500_000_000)
    @patch("torch.cuda.max_memory_allocated", return_value=7_000_000_000)
    @patch("torch.cuda.get_device_properties")
    def test_get_vram_stats_with_cuda(
        self,
        mock_props,
        mock_max_alloc,
        mock_reserved,
        mock_allocated,
        mock_is_available,
    ):
        """Test VRAM stats returns correct values with CUDA."""
        mock_props.return_value = MagicMock(total_memory=12_000_000_000)

        stats = get_vram_stats()
        assert stats["allocated_gb"] == pytest.approx(6.0, rel=0.01)
        assert stats["reserved_gb"] == pytest.approx(6.5, rel=0.01)
        assert stats["max_allocated_gb"] == pytest.approx(7.0, rel=0.01)
        assert stats["total_gb"] == pytest.approx(12.0, rel=0.01)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("vortex.utils.memory.get_vram_stats")
    @patch("vortex.utils.memory.logger")
    def test_log_vram_snapshot(self, mock_logger, mock_stats, mock_is_available):
        """Test VRAM snapshot logging."""
        mock_stats.return_value = {
            "allocated_gb": 6.2,
            "reserved_gb": 6.5,
            "max_allocated_gb": 7.0,
            "total_gb": 12.0,
        }

        log_vram_snapshot("test_snapshot")

        # Verify logger.log was called with correct arguments
        mock_logger.log.assert_called_once()
        call_args, call_kwargs = mock_logger.log.call_args
        # Check format string contains placeholders
        assert "VRAM snapshot" in call_args[1]
        # Check label is passed as argument
        assert "test_snapshot" in call_args

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    @patch("vortex.utils.memory.logger")
    def test_clear_cuda_cache(self, mock_logger, mock_empty_cache, mock_is_available):
        """Test emergency CUDA cache clearing."""
        clear_cuda_cache()
        mock_empty_cache.assert_called_once()
        # Should log warning
        mock_logger.warning.assert_called_once()

    def test_format_bytes(self):
        """Test byte formatting utility."""
        assert format_bytes(512) == "512 B"
        assert format_bytes(2048) == "2.00 KB"
        assert format_bytes(5_242_880) == "5.00 MB"
        assert format_bytes(6_442_450_944) == "6.00 GB"
