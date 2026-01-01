"""VRAM memory management utilities for Vortex pipeline.

Provides functions for:
- Querying current VRAM usage (torch.cuda.memory_allocated)
- Logging VRAM snapshots for debugging
- Emergency CUDA cache clearing
- Memory pressure monitoring with soft/hard limits
- Memory leak detection over time
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from .exceptions import MemoryLeakWarning, MemoryPressureError, MemoryPressureWarning

logger = logging.getLogger(__name__)


def get_current_vram_usage() -> int:
    """Get current VRAM usage in bytes on the default CUDA device.

    Returns:
        int: Bytes of VRAM currently allocated by PyTorch tensors.
             Returns 0 if CUDA is not available.

    Example:
        >>> usage = get_current_vram_usage()
        >>> print(f"Current VRAM: {usage / 1e9:.2f} GB")
    """
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated()


def get_vram_stats() -> dict[str, float]:
    """Get detailed VRAM statistics in GB.

    Returns:
        dict: Statistics with keys:
            - allocated_gb: Currently allocated VRAM
            - reserved_gb: Reserved by PyTorch memory allocator
            - max_allocated_gb: Peak allocation since process start
            - total_gb: Total VRAM capacity

    Example:
        >>> stats = get_vram_stats()
        >>> print(f"Allocated: {stats['allocated_gb']:.2f} GB")
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "total_gb": 0.0,
        }

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }


def log_vram_snapshot(label: str, level: int = logging.INFO) -> None:
    """Log current VRAM statistics with a descriptive label.

    Args:
        label: Descriptive label for this snapshot (e.g., "after_model_load")
        level: Logging level (default: INFO)

    Example:
        >>> log_vram_snapshot("after_flux_load")
        INFO - VRAM snapshot [after_flux_load]: allocated=6.2GB, reserved=6.5GB
    """
    stats = get_vram_stats()
    logger.log(
        level,
        "VRAM snapshot [%s]: allocated=%.2fGB, reserved=%.2fGB, max=%.2fGB, total=%.2fGB",
        label,
        stats["allocated_gb"],
        stats["reserved_gb"],
        stats["max_allocated_gb"],
        stats["total_gb"],
    )


def clear_cuda_cache() -> None:
    """Emergency CUDA cache clearing.

    Frees unused cached memory held by PyTorch allocator. This is a last-resort
    operation and should NOT be called during normal operation (defeats static
    VRAM residency pattern).

    Warning:
        This may cause performance degradation due to reallocations.
        Only use when memory pressure is critical.

    Example:
        >>> clear_cuda_cache()
        WARNING - Emergency CUDA cache cleared
    """
    if torch.cuda.is_available():
        logger.warning("Emergency CUDA cache cleared - this may impact performance")
        torch.cuda.empty_cache()


def reset_peak_memory_stats() -> None:
    """Reset peak memory statistics.

    Useful for benchmarking individual operations without contamination from
    previous allocations.

    Example:
        >>> reset_peak_memory_stats()
        >>> # Run operation
        >>> stats = get_vram_stats()
        >>> print(f"Peak for this operation: {stats['max_allocated_gb']:.2f} GB")
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def format_bytes(bytes_val: int) -> str:
    """Format byte count into human-readable string.

    Args:
        bytes_val: Number of bytes

    Returns:
        str: Formatted string (e.g., "6.24 GB", "512.00 MB")

    Example:
        >>> format_bytes(6543210987)
        '6.54 GB'
    """
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val / 1024:.2f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val / 1024**2:.2f} MB"
    else:
        return f"{bytes_val / 1024**3:.2f} GB"


@dataclass
class VRAMSnapshot:
    """VRAM usage snapshot at a point in time.

    Attributes:
        timestamp: ISO 8601 timestamp (UTC)
        event: Descriptive event label (e.g., "post_generation", "after_model_load")
        slot: Optional slot number for generation events
        vram_usage_gb: Current VRAM usage in GB
        vram_allocated_gb: VRAM allocated by PyTorch in GB
        vram_reserved_gb: VRAM reserved by PyTorch allocator in GB
        models: Optional per-model VRAM usage breakdown (model_name -> GB)

    Example:
        >>> snapshot = VRAMSnapshot(
        ...     timestamp="2025-12-24T12:00:00Z",
        ...     event="post_generation",
        ...     slot=12345,
        ...     vram_usage_gb=10.95,
        ...     vram_allocated_gb=10.85,
        ...     vram_reserved_gb=11.2,
        ...     models={"flux": 6.0, "liveportrait": 3.5, "kokoro": 0.4}
        ... )
    """

    timestamp: str
    event: str
    slot: int | None
    vram_usage_gb: float
    vram_allocated_gb: float
    vram_reserved_gb: float
    models: dict[str, float] | None = None


class VRAMMonitor:
    """Monitor VRAM usage and enforce soft/hard limits to prevent OOM.

    This class implements proactive memory pressure detection with:
    - Soft limit (default 11.0GB): Warning + optional emergency cleanup
    - Hard limit (default 11.5GB): Error raised, generation aborted
    - Memory leak detection over 100 generations (>100MB delta)
    - VRAM snapshots for debugging

    Attributes:
        soft_limit_bytes: Soft limit threshold in bytes
        hard_limit_bytes: Hard limit threshold in bytes
        emergency_cleanup: Whether to run emergency cleanup on soft limit
        soft_limit_violations: Counter for soft limit violations
        hard_limit_violations: Counter for hard limit violations
        emergency_cleanups: Counter for emergency cleanup operations
        baseline_usage: Baseline VRAM usage for leak detection (bytes)
        generation_count: Number of generations processed

    Example:
        >>> monitor = VRAMMonitor(soft_limit_gb=11.0, hard_limit_gb=11.5)
        >>> monitor.check_limits("after_model_load")  # May raise MemoryPressureError
        >>> snapshot = monitor.log_snapshot("post_generation", slot=12345)
        >>> monitor.increment_generation_count()  # Auto-checks leak every 100 gens
    """

    def __init__(
        self,
        soft_limit_gb: float = 11.0,
        hard_limit_gb: float = 11.5,
        emergency_cleanup: bool = True,
    ):
        """Initialize VRAM monitor with configurable limits.

        Args:
            soft_limit_gb: Soft limit in GB (default: 11.0)
            hard_limit_gb: Hard limit in GB (default: 11.5)
            emergency_cleanup: Enable automatic cleanup on soft limit (default: True)

        Raises:
            ValueError: If soft_limit >= hard_limit
        """
        if soft_limit_gb >= hard_limit_gb:
            raise ValueError(
                f"Soft limit ({soft_limit_gb}GB) must be less than hard limit ({hard_limit_gb}GB)"
            )

        self.soft_limit_bytes = int(soft_limit_gb * 1e9)
        self.hard_limit_bytes = int(hard_limit_gb * 1e9)
        self.emergency_cleanup = emergency_cleanup

        # Metrics
        self.soft_limit_violations = 0
        self.hard_limit_violations = 0
        self.emergency_cleanups = 0

        # Leak detection
        self.baseline_usage: int | None = None
        self.generation_count = 0

    def check_limits(self, context: str = "") -> None:
        """Check VRAM usage against soft and hard limits.

        Args:
            context: Descriptive context for logging (e.g., "after_model_load")

        Raises:
            MemoryPressureError: If hard limit exceeded
            MemoryPressureWarning: If soft limit exceeded (warning only)

        Example:
            >>> monitor.check_limits("pre_generation")
        """
        if not torch.cuda.is_available():
            return

        current_usage = torch.cuda.memory_allocated()

        # Hard limit (blocking error)
        if current_usage > self.hard_limit_bytes:
            self.hard_limit_violations += 1
            current_gb = current_usage / 1e9
            hard_limit_gb = self.hard_limit_bytes / 1e9
            delta_gb = (current_usage - self.hard_limit_bytes) / 1e9

            logger.error(
                "VRAM hard limit exceeded",
                extra={
                    "context": context,
                    "current_usage_gb": current_gb,
                    "hard_limit_gb": hard_limit_gb,
                    "delta_gb": delta_gb,
                },
            )

            raise MemoryPressureError(
                current_gb=current_gb,
                hard_limit_gb=hard_limit_gb,
                delta_gb=delta_gb,
            )

        # Soft limit (warning + optional cleanup)
        if current_usage > self.soft_limit_bytes:
            self.soft_limit_violations += 1
            current_gb = current_usage / 1e9
            soft_limit_gb = self.soft_limit_bytes / 1e9
            delta_gb = (current_usage - self.soft_limit_bytes) / 1e9

            logger.warning(
                "VRAM soft limit exceeded",
                extra={
                    "context": context,
                    "current_usage_gb": current_gb,
                    "soft_limit_gb": soft_limit_gb,
                    "delta_gb": delta_gb,
                },
            )

            # Emergency cleanup if enabled
            if self.emergency_cleanup:
                self._emergency_cleanup()

            # Raise warning (non-blocking)
            import warnings

            warnings.warn(
                MemoryPressureWarning(
                    current_gb=current_gb,
                    soft_limit_gb=soft_limit_gb,
                    delta_gb=delta_gb,
                ),
                stacklevel=2,
            )

    def log_snapshot(
        self,
        event: str,
        slot: int | None = None,
        models: dict[str, float] | None = None,
    ) -> VRAMSnapshot:
        """Log current VRAM usage snapshot.

        Args:
            event: Event label (e.g., "post_generation", "after_flux_load")
            slot: Optional slot number for generation events
            models: Optional per-model VRAM breakdown (model_name -> GB)

        Returns:
            VRAMSnapshot: Snapshot object with current VRAM state

        Example:
            >>> snapshot = monitor.log_snapshot(
            ...     "post_generation",
            ...     slot=12345,
            ...     models={"flux": 6.0, "liveportrait": 3.5}
            ... )
        """
        if not torch.cuda.is_available():
            # Return zero snapshot if CUDA unavailable
            snapshot = VRAMSnapshot(
                timestamp=datetime.now(UTC).isoformat(),
                event=event,
                slot=slot,
                vram_usage_gb=0.0,
                vram_allocated_gb=0.0,
                vram_reserved_gb=0.0,
                models=models or {},
            )
            return snapshot

        snapshot = VRAMSnapshot(
            timestamp=datetime.now(UTC).isoformat(),
            event=event,
            slot=slot,
            vram_usage_gb=torch.cuda.memory_allocated() / 1e9,
            vram_allocated_gb=torch.cuda.memory_allocated() / 1e9,
            vram_reserved_gb=torch.cuda.memory_reserved() / 1e9,
            models=models or {},
        )

        logger.info("VRAM snapshot", extra=snapshot.__dict__)
        return snapshot

    def _emergency_cleanup(self) -> None:
        """Emergency CUDA cache cleanup.

        Frees unused cached memory via torch.cuda.empty_cache().
        Logs before/after VRAM usage and amount freed.
        """
        if not torch.cuda.is_available():
            return

        before = torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()

        freed_mb = (before - after) / 1e6
        self.emergency_cleanups += 1

        logger.info(
            f"Emergency cleanup freed {freed_mb:.1f}MB",
            extra={
                "before_gb": before / 1e9,
                "after_gb": after / 1e9,
                "freed_mb": freed_mb,
            },
        )

    def detect_memory_leak(self, threshold_mb: float = 100) -> bool:
        """Check if VRAM usage has grown significantly over time.

        Args:
            threshold_mb: Growth threshold in MB to trigger leak warning (default: 100)

        Returns:
            bool: True if potential leak detected (delta > threshold)

        Example:
            >>> if monitor.detect_memory_leak(threshold_mb=100):
            ...     logger.warning("Potential memory leak detected")
        """
        if not torch.cuda.is_available():
            return False

        current = torch.cuda.memory_allocated()

        # Set baseline on first call
        if self.baseline_usage is None:
            self.baseline_usage = current
            return False

        delta_mb = (current - self.baseline_usage) / 1e6

        if delta_mb > threshold_mb:
            initial_gb = self.baseline_usage / 1e9
            current_gb = current / 1e9

            logger.warning(
                "Potential memory leak detected",
                extra={
                    "initial_usage_gb": initial_gb,
                    "current_usage_gb": current_gb,
                    "delta_mb": delta_mb,
                    "generations": self.generation_count,
                    "delta_per_generation_kb": delta_mb * 1000 / max(self.generation_count, 1),
                },
            )

            # Raise warning
            import warnings

            warnings.warn(
                MemoryLeakWarning(
                    initial_gb=initial_gb,
                    current_gb=current_gb,
                    delta_mb=delta_mb,
                    generations=self.generation_count,
                ),
                stacklevel=2,
            )

            return True

        return False

    def increment_generation_count(self) -> None:
        """Increment generation counter and check for leaks every 100 generations.

        Example:
            >>> monitor.increment_generation_count()
            # Auto-checks for leak at generation 100, 200, etc.
        """
        self.generation_count += 1

        # Check for leaks every 100 generations
        if self.generation_count % 100 == 0:
            self.detect_memory_leak()
