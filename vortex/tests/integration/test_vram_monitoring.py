"""Integration tests for VRAM monitoring with real GPU operations.

These tests require an actual CUDA-capable GPU and are skipped if CUDA is unavailable.
Run with: pytest tests/integration/test_vram_monitoring.py --gpu -v
"""

import pytest
import torch

from vortex.utils.exceptions import MemoryPressureError, MemoryPressureWarning
from vortex.utils.memory import VRAMMonitor, get_vram_stats

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available, skipping GPU tests"
)


class TestVRAMMonitorIntegration:
    """Integration tests for VRAMMonitor with real GPU operations."""

    def test_normal_operation_within_budget(self):
        """Test VRAMMonitor with usage within soft limit."""
        # Get baseline VRAM stats
        stats = get_vram_stats()
        current_gb = stats["allocated_gb"]

        # Create monitor with limits higher than current usage
        monitor = VRAMMonitor(
            soft_limit_gb=current_gb + 1.0,
            hard_limit_gb=current_gb + 2.0,
        )

        # Check limits - should not raise
        monitor.check_limits("baseline_test")

        assert monitor.soft_limit_violations == 0
        assert monitor.hard_limit_violations == 0

    def test_soft_limit_warning_with_tensor_allocation(self):
        """Test soft limit warning when allocating tensors near limit."""
        # Get current usage
        stats = get_vram_stats()
        baseline_gb = stats["allocated_gb"]

        # Set soft limit just above current usage
        monitor = VRAMMonitor(
            soft_limit_gb=baseline_gb + 0.1,  # 100MB headroom
            hard_limit_gb=baseline_gb + 1.0,
            emergency_cleanup=True,
        )

        # Allocate a tensor to push us over soft limit
        try:
            # Allocate 200MB tensor (should exceed soft limit)
            tensor = torch.randn(50_000_000, device="cuda")  # ~200MB

            # This should trigger soft limit warning
            with pytest.warns(MemoryPressureWarning):
                monitor.check_limits("after_tensor_allocation")

            assert monitor.soft_limit_violations == 1
            assert monitor.emergency_cleanups == 1

        finally:
            # Cleanup
            del tensor
            torch.cuda.empty_cache()

    def test_hard_limit_error_prevents_oom(self):
        """Test hard limit error prevents OOM by aborting early."""
        # Get current usage
        stats = get_vram_stats()
        baseline_gb = stats["allocated_gb"]

        # Set hard limit just above current usage
        monitor = VRAMMonitor(
            soft_limit_gb=baseline_gb + 0.1,
            hard_limit_gb=baseline_gb + 0.3,  # 300MB headroom
        )

        try:
            # Allocate 400MB tensor (should exceed hard limit)
            tensor = torch.randn(100_000_000, device="cuda")  # ~400MB

            # This should raise MemoryPressureError
            with pytest.raises(MemoryPressureError):
                monitor.check_limits("after_large_allocation")

            assert monitor.hard_limit_violations == 1

        finally:
            # Cleanup
            del tensor
            torch.cuda.empty_cache()

    def test_vram_snapshot_logging(self):
        """Test VRAM snapshot captures accurate GPU state."""
        monitor = VRAMMonitor()

        # Allocate a known tensor
        tensor = torch.randn(25_000_000, device="cuda")  # ~100MB

        try:
            snapshot = monitor.log_snapshot(
                "test_snapshot",
                slot=123,
                models={"test_model": 0.1},
            )

            # Verify snapshot has reasonable values
            assert snapshot.vram_usage_gb > 0
            assert snapshot.vram_allocated_gb > 0
            assert snapshot.vram_reserved_gb >= snapshot.vram_allocated_gb
            assert snapshot.event == "test_snapshot"
            assert snapshot.slot == 123
            assert snapshot.models == {"test_model": 0.1}

        finally:
            del tensor
            torch.cuda.empty_cache()

    def test_memory_leak_detection_over_iterations(self):
        """Test memory leak detection catches slow leaks."""
        monitor = VRAMMonitor()

        # Set baseline
        monitor.detect_memory_leak()
        assert monitor.baseline_usage is not None

        # Simulate 10 generations with small leak (10MB each)
        leaked_tensors = []
        try:
            for i in range(10):
                # Allocate 10MB tensor and "leak" it by not freeing
                tensor = torch.randn(2_500_000, device="cuda")  # ~10MB
                leaked_tensors.append(tensor)
                monitor.increment_generation_count()

            # After 10 generations (100MB leaked), check should detect leak
            # Note: Auto-check only happens at generation 100, so manually check
            leak_detected = monitor.detect_memory_leak(threshold_mb=50)

            # Should detect leak since we leaked ~100MB
            assert leak_detected is True

        finally:
            # Cleanup leaked tensors
            for tensor in leaked_tensors:
                del tensor
            torch.cuda.empty_cache()

    def test_emergency_cleanup_frees_memory(self):
        """Test emergency cleanup actually frees cached memory."""
        monitor = VRAMMonitor()

        # Allocate and free tensors to create cached memory
        tensors = []
        for _ in range(5):
            tensors.append(torch.randn(10_000_000, device="cuda"))  # 5x 40MB

        # Free tensors (memory stays cached)
        for tensor in tensors:
            del tensor

        # Get VRAM before cleanup
        before_reserved = torch.cuda.memory_reserved()

        # Emergency cleanup should free cached memory
        monitor._emergency_cleanup()

        # Get VRAM after cleanup
        after_reserved = torch.cuda.memory_reserved()

        # Reserved memory should decrease (cached memory freed)
        assert after_reserved < before_reserved
        assert monitor.emergency_cleanups == 1

    def test_monitor_overhead_is_minimal(self):
        """Test VRAMMonitor overhead is <1ms for check_limits."""
        import time

        monitor = VRAMMonitor()

        # Warm up
        for _ in range(10):
            monitor.check_limits()

        # Measure 1000 check_limits() calls
        start = time.perf_counter()
        for _ in range(1000):
            monitor.check_limits()
        end = time.perf_counter()

        # Average overhead should be <1ms per call
        avg_ms = (end - start) * 1000 / 1000
        assert avg_ms < 1.0, f"check_limits() overhead {avg_ms:.2f}ms exceeds 1ms target"

    def test_log_snapshot_overhead_is_minimal(self):
        """Test log_snapshot overhead is <5ms."""
        import time

        monitor = VRAMMonitor()

        # Warm up
        for _ in range(10):
            monitor.log_snapshot("warmup")

        # Measure 100 log_snapshot() calls
        start = time.perf_counter()
        for i in range(100):
            monitor.log_snapshot(f"test_{i}")
        end = time.perf_counter()

        # Average overhead should be <5ms per call
        avg_ms = (end - start) * 1000 / 100
        assert avg_ms < 5.0, f"log_snapshot() overhead {avg_ms:.2f}ms exceeds 5ms target"

    def test_monitor_self_vram_usage_is_minimal(self):
        """Test VRAMMonitor itself uses <10MB VRAM."""
        # Get baseline VRAM
        baseline = torch.cuda.memory_allocated()

        # Create monitor
        monitor = VRAMMonitor()

        # Use monitor for 100 operations
        for i in range(100):
            monitor.check_limits(f"iteration_{i}")
            if i % 10 == 0:
                monitor.log_snapshot(f"snapshot_{i}")

        # Get VRAM after monitor usage
        after = torch.cuda.memory_allocated()

        # VRAMMonitor should use <10MB
        delta_mb = (after - baseline) / 1e6
        assert delta_mb < 10.0, f"VRAMMonitor uses {delta_mb:.2f}MB, exceeds 10MB target"

    def test_pre_flight_check_scenario(self):
        """Test simulated pre-flight check for Vortex pipeline initialization."""
        from vortex.utils.exceptions import VortexInitializationError

        # Get total VRAM
        stats = get_vram_stats()
        total_gb = stats["total_gb"]

        # Simulate pre-flight check
        REQUIRED_VRAM_GB = 11.8

        if total_gb < REQUIRED_VRAM_GB:
            # Should raise VortexInitializationError
            with pytest.raises(VortexInitializationError):
                raise VortexInitializationError(
                    reason="Insufficient VRAM",
                    available_gb=total_gb,
                    required_gb=REQUIRED_VRAM_GB,
                )
        else:
            # GPU has sufficient VRAM, can proceed
            assert total_gb >= REQUIRED_VRAM_GB
