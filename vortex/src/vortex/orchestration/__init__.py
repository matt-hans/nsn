"""Slot timing orchestration package.

Provides SlotScheduler for orchestrating Vortex generation pipeline with
deadline tracking, parallel execution, and timeout enforcement.

Key Classes:
- SlotScheduler: Main orchestration class
- SlotResult: Generation result with metadata
- GenerationBreakdown: Timing breakdown per stage
- SlotMetadata: Slot identification and timestamps
- DeadlineMissError: Exception for deadline violations

Example:
    >>> from vortex.orchestration import SlotScheduler
    >>> scheduler = SlotScheduler(pipeline, config)
    >>> result = await scheduler.execute(recipe, slot_id=12345)
"""

from vortex.orchestration.models import (
    GenerationBreakdown,
    SlotMetadata,
    SlotResult,
)
from vortex.orchestration.scheduler import DeadlineMissError, SlotScheduler

__all__ = [
    "SlotScheduler",
    "SlotResult",
    "GenerationBreakdown",
    "SlotMetadata",
    "DeadlineMissError",
]
