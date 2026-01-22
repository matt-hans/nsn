"""Vortex utility modules.

This package provides:
- Memory/VRAM monitoring and management
- Exception types for memory pressure handling
- Model offloading utilities

Legacy modules (lipsync, face_landmarks, liveportrait_features) have been removed
as they are no longer used in the Narrative Chain pipeline.
"""

from .exceptions import (
    MemoryLeakWarning,
    MemoryPressureError,
    MemoryPressureWarning,
    VortexInitializationError,
)
from .memory import (
    VRAMMonitor,
    VRAMSnapshot,
    clear_cuda_cache,
    format_bytes,
    get_current_vram_usage,
    get_vram_stats,
    log_vram_snapshot,
    reset_peak_memory_stats,
)

__all__ = [
    # Exceptions
    "MemoryPressureWarning",
    "MemoryPressureError",
    "VortexInitializationError",
    "MemoryLeakWarning",
    # Memory monitoring
    "VRAMMonitor",
    "VRAMSnapshot",
    "get_current_vram_usage",
    "get_vram_stats",
    "log_vram_snapshot",
    "clear_cuda_cache",
    "reset_peak_memory_stats",
    "format_bytes",
]
