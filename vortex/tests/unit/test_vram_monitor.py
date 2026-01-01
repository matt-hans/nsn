"""Unit tests for VRAMMonitor class with mocked CUDA operations."""

from unittest.mock import MagicMock, patch

import pytest

from vortex.utils.exceptions import (
    MemoryLeakWarning,
    MemoryPressureError,
    MemoryPressureWarning,
)
from vortex.utils.memory import VRAMMonitor, VRAMSnapshot


class TestVRAMMonitor:
    """Test VRAMMonitor with mocked CUDA environment."""

    def test_init_with_valid_limits(self):
        """Test VRAMMonitor initialization with valid limits."""
        monitor = VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.5)

        assert monitor.soft_limit_bytes == int(11.0 * 1e9)
        assert monitor.hard_limit_bytes == int(11.5 * 1e9)
        assert monitor.emergency_cleanup is True
        assert monitor.soft_limit_violations == 0
        assert monitor.hard_limit_violations == 0
        assert monitor.emergency_cleanups == 0
        assert monitor.generation_count == 0
        assert monitor.baseline_usage is None

    def test_init_with_invalid_limits(self):
        """Test VRAMMonitor initialization fails if soft >= hard limit."""
        with pytest.raises(ValueError, match="must be less than"):
            VRAMMonitor(soft_limit_gb=11.5, hard_limit_gb=11.0)

        with pytest.raises(ValueError, match="must be less than"):
            VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.0)

    @patch("torch.cuda.is_available", return_value=False)
    def test_check_limits_no_cuda(self, mock_is_available):
        """Test check_limits does nothing when CUDA unavailable."""
        monitor = VRAMMonitor()
        # Should not raise any exceptions
        monitor.check_limits("test")
        assert monitor.soft_limit_violations == 0
        assert monitor.hard_limit_violations == 0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=int(10.0 * 1e9))
    def test_check_limits_within_limits(self, mock_allocated, mock_is_available):
        """Test check_limits with usage within soft limit."""
        monitor = VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.5)
        monitor.check_limits("within_limits")

        assert monitor.soft_limit_violations == 0
        assert monitor.hard_limit_violations == 0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=int(11.2 * 1e9))
    @patch("torch.cuda.empty_cache")
    @patch("vortex.utils.memory.logger")
    def test_check_limits_soft_limit_warning(
        self, mock_logger, mock_empty_cache, mock_allocated, mock_is_available
    ):
        """Test check_limits triggers warning at soft limit."""
        monitor = VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.5, emergency_cleanup=True)

        # Should raise MemoryPressureWarning but not error
        with pytest.warns(MemoryPressureWarning):
            monitor.check_limits("soft_limit_test")

        assert monitor.soft_limit_violations == 1
        assert monitor.hard_limit_violations == 0
        assert monitor.emergency_cleanups == 1

        # Verify logger.warning was called
        mock_logger.warning.assert_called_once()
        # Verify emergency cleanup was called
        mock_empty_cache.assert_called_once()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=int(11.6 * 1e9))
    @patch("vortex.utils.memory.logger")
    def test_check_limits_hard_limit_error(self, mock_logger, mock_allocated, mock_is_available):
        """Test check_limits raises MemoryPressureError at hard limit."""
        monitor = VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.5)

        with pytest.raises(MemoryPressureError) as exc_info:
            monitor.check_limits("hard_limit_test")

        assert monitor.hard_limit_violations == 1

        # Verify exception attributes
        exc = exc_info.value
        assert exc.current_gb == pytest.approx(11.6, rel=0.01)
        assert exc.hard_limit_gb == pytest.approx(11.5, rel=0.01)
        assert exc.delta_gb > 0

        # Verify logger.error was called
        mock_logger.error.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_log_snapshot_no_cuda(self, mock_is_available):
        """Test log_snapshot returns zero snapshot when CUDA unavailable."""
        monitor = VRAMMonitor()
        snapshot = monitor.log_snapshot("test_event", slot=123)

        assert isinstance(snapshot, VRAMSnapshot)
        assert snapshot.event == "test_event"
        assert snapshot.slot == 123
        assert snapshot.vram_usage_gb == 0.0
        assert snapshot.vram_allocated_gb == 0.0
        assert snapshot.vram_reserved_gb == 0.0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=int(10.5 * 1e9))
    @patch("torch.cuda.memory_reserved", return_value=int(11.0 * 1e9))
    @patch("vortex.utils.memory.logger")
    def test_log_snapshot_with_cuda(
        self, mock_logger, mock_reserved, mock_allocated, mock_is_available
    ):
        """Test log_snapshot with CUDA available."""
        monitor = VRAMMonitor()
        models = {"flux": 6.0, "liveportrait": 3.5, "kokoro": 0.4}
        snapshot = monitor.log_snapshot("post_generation", slot=12345, models=models)

        assert isinstance(snapshot, VRAMSnapshot)
        assert snapshot.event == "post_generation"
        assert snapshot.slot == 12345
        assert snapshot.vram_usage_gb == pytest.approx(10.5, rel=0.01)
        assert snapshot.vram_allocated_gb == pytest.approx(10.5, rel=0.01)
        assert snapshot.vram_reserved_gb == pytest.approx(11.0, rel=0.01)
        assert snapshot.models == models

        # Verify logger.info was called
        mock_logger.info.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_detect_memory_leak_no_cuda(self, mock_is_available):
        """Test detect_memory_leak returns False when CUDA unavailable."""
        monitor = VRAMMonitor()
        assert monitor.detect_memory_leak() is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=int(10.8 * 1e9))
    def test_detect_memory_leak_sets_baseline(self, mock_allocated, mock_is_available):
        """Test detect_memory_leak sets baseline on first call."""
        monitor = VRAMMonitor()
        assert monitor.baseline_usage is None

        result = monitor.detect_memory_leak()

        assert result is False
        assert monitor.baseline_usage == int(10.8 * 1e9)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("vortex.utils.memory.logger")
    def test_detect_memory_leak_within_threshold(self, mock_logger, mock_is_available):
        """Test detect_memory_leak with delta below threshold."""
        monitor = VRAMMonitor()
        monitor.baseline_usage = int(10.8 * 1e9)
        monitor.generation_count = 50

        # Current usage: 10.85GB (50MB delta, below 100MB threshold)
        with patch("torch.cuda.memory_allocated", return_value=int(10.85 * 1e9)):
            result = monitor.detect_memory_leak(threshold_mb=100)

        assert result is False
        mock_logger.warning.assert_not_called()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("vortex.utils.memory.logger")
    def test_detect_memory_leak_above_threshold(self, mock_logger, mock_is_available):
        """Test detect_memory_leak with delta above threshold."""
        monitor = VRAMMonitor()
        monitor.baseline_usage = int(10.8 * 1e9)
        monitor.generation_count = 100

        # Current usage: 11.0GB (200MB delta, above 100MB threshold)
        with patch("torch.cuda.memory_allocated", return_value=int(11.0 * 1e9)):
            with pytest.warns(MemoryLeakWarning):
                result = monitor.detect_memory_leak(threshold_mb=100)

        assert result is True
        mock_logger.warning.assert_called_once()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=int(10.8 * 1e9))
    def test_increment_generation_count_basic(self, mock_allocated, mock_is_available):
        """Test increment_generation_count increments counter."""
        monitor = VRAMMonitor()
        assert monitor.generation_count == 0

        monitor.increment_generation_count()
        assert monitor.generation_count == 1

        monitor.increment_generation_count()
        assert monitor.generation_count == 2

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=int(10.8 * 1e9))
    def test_increment_generation_count_auto_check(self, mock_allocated, mock_is_available):
        """Test increment_generation_count auto-checks leak at 100 generations."""
        monitor = VRAMMonitor()
        monitor.baseline_usage = int(10.8 * 1e9)

        # Increment to 99 (no auto-check)
        for _ in range(99):
            monitor.increment_generation_count()

        assert monitor.generation_count == 99

        # 100th increment should trigger auto-check
        with patch.object(monitor, "detect_memory_leak") as mock_detect:
            monitor.increment_generation_count()

        assert monitor.generation_count == 100
        mock_detect.assert_called_once()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.empty_cache")
    @patch("vortex.utils.memory.logger")
    def test_emergency_cleanup(
        self, mock_logger, mock_empty_cache, mock_allocated, mock_is_available
    ):
        """Test _emergency_cleanup frees memory and logs."""
        monitor = VRAMMonitor()

        # Before: 11.1GB, After: 10.95GB (150MB freed)
        mock_allocated.side_effect = [int(11.1 * 1e9), int(10.95 * 1e9)]

        monitor._emergency_cleanup()

        mock_empty_cache.assert_called_once()
        assert monitor.emergency_cleanups == 1
        mock_logger.info.assert_called_once()

        # Verify logged message contains freed amount
        log_args = mock_logger.info.call_args
        assert "freed" in log_args[0][0].lower()


class TestVRAMSnapshot:
    """Test VRAMSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test VRAMSnapshot creation with all fields."""
        snapshot = VRAMSnapshot(
            timestamp="2025-12-24T12:00:00Z",
            event="test_event",
            slot=12345,
            vram_usage_gb=10.95,
            vram_allocated_gb=10.85,
            vram_reserved_gb=11.2,
            models={"flux": 6.0, "liveportrait": 3.5},
        )

        assert snapshot.timestamp == "2025-12-24T12:00:00Z"
        assert snapshot.event == "test_event"
        assert snapshot.slot == 12345
        assert snapshot.vram_usage_gb == pytest.approx(10.95)
        assert snapshot.vram_allocated_gb == pytest.approx(10.85)
        assert snapshot.vram_reserved_gb == pytest.approx(11.2)
        assert snapshot.models == {"flux": 6.0, "liveportrait": 3.5}

    def test_snapshot_without_optional_fields(self):
        """Test VRAMSnapshot creation without optional fields."""
        snapshot = VRAMSnapshot(
            timestamp="2025-12-24T12:00:00Z",
            event="test_event",
            slot=None,
            vram_usage_gb=10.0,
            vram_allocated_gb=10.0,
            vram_reserved_gb=10.5,
        )

        assert snapshot.slot is None
        assert snapshot.models is None
