"""VRAM memory management utilities for Vortex pipeline.

Provides functions for:
- Querying current VRAM usage (torch.cuda.memory_allocated)
- Logging VRAM snapshots for debugging
- Emergency CUDA cache clearing
- Memory pressure monitoring
"""

import logging

import torch

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
