---
id: T019
title: VRAM Manager - Memory Pressure Monitoring & OOM Prevention
status: pending
priority: 2
agent: ai-ml
dependencies: [T014]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [vortex, ai-ml, python, gpu, monitoring, infrastructure, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - prd.md#section-12.1-static-resident-vram-layout
  - architecture.md#section-4.2.1-director-node-components

est_tokens: 11000
actual_tokens: null
---

## Description

Implement VRAM Manager system for proactive memory pressure detection, usage monitoring, and OOM (Out-Of-Memory) prevention in the Vortex pipeline. This component ensures all 5 AI models fit within the 11.8GB budget on RTX 3060 12GB cards.

**Critical Responsibilities**:
- Track VRAM usage across all resident models (Flux, LivePortrait, Kokoro, CLIP×2)
- Monitor memory pressure during generation (intermediate tensors, output buffers)
- Detect soft limit violations (11.0GB) and hard limit violations (11.5GB)
- Provide pre-allocation guarantees for output buffers
- Log VRAM snapshots for debugging and performance tuning
- Implement emergency cleanup mechanisms (cache clearing, buffer compaction)

**Integration Points**:
- Used by VortexPipeline at initialization (verify VRAM budget)
- Called before/after each generation (track memory delta)
- Alerts on memory pressure (warnings at 11.0GB, errors at 11.5GB)

## Business Context

**User Story**: As a Director node operator, I want real-time VRAM monitoring with early warnings, so that I can prevent OOM crashes that would cause slot misses and 150 ICN slashing penalties.

**Why This Matters**:
- CUDA OOM errors crash the entire Python process, requiring restart (5+ minute downtime)
- Slot misses result in DirectorSlotMissed events (150 ICN reputation penalty)
- VRAM fragmentation can cause slow degradation over time, leading to unpredictable failures
- Memory leaks compound over 100+ generations, eventually exceeding budget
- Proactive monitoring prevents crashes, enables graceful degradation, and supports root cause analysis

**What It Unblocks**:
- Production deployment of Director nodes (reliability requirement)
- Long-running Director processes (24/7 operation)
- Performance debugging and optimization
- Automated alerting and recovery

**Priority Justification**: Priority 2 (Important) - Not on critical path for initial implementation, but essential for production reliability. Without VRAM monitoring, Directors will experience unpredictable crashes and downtime.

## Acceptance Criteria

- [ ] VRAMMonitor class tracks VRAM usage via torch.cuda.memory_allocated()
- [ ] Soft limit (11.0GB) triggers MemoryPressureWarning (logged, generation continues)
- [ ] Hard limit (11.5GB) triggers MemoryPressureError (logged, generation aborted)
- [ ] Pre-flight check before VortexPipeline initialization verifies 12GB+ available
- [ ] Model registry tracks per-model VRAM usage (Flux: 6.0GB, LivePortrait: 3.5GB, etc.)
- [ ] Generation delta tracking: measure VRAM increase during slot generation
- [ ] VRAM snapshots logged at key points: startup, post-load, pre-generation, post-generation
- [ ] Emergency cleanup: clear_cuda_cache() callable on soft limit violations
- [ ] VRAM usage exposed as Prometheus metric (icn_vortex_vram_usage_bytes)
- [ ] Configuration: soft_limit, hard_limit, emergency_cleanup_enabled (from config.yaml)
- [ ] Buffer pre-allocation validation: ensure actor/video/audio buffers fit within budget
- [ ] Memory leak detection: alert if VRAM delta >100MB after 100 consecutive generations

## Test Scenarios

**Test Case 1: Normal Operation (Within Budget)**
- Given: VortexPipeline initialized with all 5 models loaded
- When: VRAM monitor checks usage
- Then: torch.cuda.memory_allocated() shows 10.5-11.3GB
  And soft limit not exceeded (11.0GB)
  And hard limit not exceeded (11.5GB)
  And no warnings or errors logged

**Test Case 2: Soft Limit Warning**
- Given: VRAM usage at 10.8GB
- When: Generation allocates 500MB intermediate tensor (total 11.3GB)
- Then: VRAM monitor detects usage >11.0GB
  And MemoryPressureWarning is logged with details: current_usage=11.3GB, soft_limit=11.0GB, delta=+0.3GB
  And generation continues (soft limit is non-blocking)
  And Prometheus metric icn_vortex_vram_soft_limit_violations_total increments

**Test Case 3: Hard Limit Error**
- Given: VRAM usage at 11.2GB
- When: Generation attempts to allocate 800MB tensor (would exceed 11.5GB)
- Then: VRAM monitor detects potential usage >11.5GB
  And MemoryPressureError is raised before allocation
  And error log includes: current_usage=11.2GB, requested=0.8GB, hard_limit=11.5GB
  And generation is aborted (no OOM crash)
  And emergency cleanup is triggered (clear CUDA cache)

**Test Case 4: Pre-Flight Check**
- Given: System with only 8GB GPU VRAM (below 12GB requirement)
- When: VortexPipeline initialization starts
- Then: Pre-flight check detects available VRAM <11.8GB required
  And VortexInitializationError is raised before model loading
  And error message: "Insufficient VRAM: 8.0GB available, 11.8GB required. Upgrade to RTX 3060 12GB or higher."

**Test Case 5: Memory Leak Detection**
- Given: VortexPipeline generates 100 consecutive slots
- When: VRAM monitor compares usage before slot 1 vs. after slot 100
- Then: VRAM delta is <100MB (acceptable)
  OR if delta >100MB, MemoryLeakWarning is logged
  And warning includes: initial_usage=10.8GB, current_usage=11.0GB, delta=+0.2GB, generations=100

**Test Case 6: Emergency Cleanup**
- Given: VRAM at 11.1GB (soft limit violated)
  And emergency_cleanup_enabled=true in config
- When: Soft limit warning is triggered
- Then: torch.cuda.empty_cache() is called
  And VRAM usage is measured again
  And cleanup result logged: "Emergency cleanup freed 150MB (11.1GB → 10.95GB)"

**Test Case 7: VRAM Snapshot Logging**
- Given: VortexPipeline running with debug logging enabled
- When: Slot generation completes
- Then: VRAM snapshot is logged with structure:
  ```json
  {
    "timestamp": "2025-12-24T12:00:00Z",
    "event": "post_generation",
    "slot": 12345,
    "vram_usage_gb": 10.95,
    "vram_allocated_gb": 10.85,
    "vram_reserved_gb": 11.2,
    "models": {
      "flux": 6.0,
      "liveportrait": 3.5,
      "kokoro": 0.4,
      "clip_b": 0.3,
      "clip_l": 0.6
    }
  }
  ```

## Technical Implementation

**Required Components**:

1. **vortex/utils/memory.py** (VRAM monitor core)
   - `VRAMMonitor` class with `check_limits()`, `log_snapshot()`, `emergency_cleanup()`
   - `get_current_vram_usage() -> int` (bytes)
   - `get_vram_breakdown() -> Dict[str, float]` (per-model usage)
   - `clear_cuda_cache()` (emergency cleanup)

2. **vortex/utils/exceptions.py** (Custom exceptions)
   - `MemoryPressureWarning` (soft limit, non-blocking)
   - `MemoryPressureError` (hard limit, blocking)
   - `VortexInitializationError` (pre-flight check failure)

3. **vortex/config.yaml** (Configuration)
   - `vram_limits: {soft_gb: 11.0, hard_gb: 11.5, emergency_cleanup: true}`
   - `monitoring: {snapshot_interval_s: 60, leak_detection_threshold_mb: 100}`

4. **vortex/tests/integration/test_vram_monitoring.py** (Integration tests)
   - Test cases 1-7 from above
   - Simulated memory pressure scenarios
   - Memory leak regression tests

5. **vortex/benchmarks/vram_stress_test.py** (Stress testing)
   - Generate 1000 consecutive slots
   - Track VRAM delta over time
   - Identify fragmentation patterns

**Validation Commands**:
```bash
# Unit tests (mocked CUDA)
pytest vortex/tests/unit/test_vram_monitor.py -v

# Integration test (real GPU)
pytest vortex/tests/integration/test_vram_monitoring.py --gpu -v

# VRAM profiling
python vortex/benchmarks/vram_profile.py

# Stress test (1000 generations)
python vortex/benchmarks/vram_stress_test.py --iterations 1000

# Memory leak detection
python vortex/tests/regression/memory_leak_detector.py --generations 500
```

**Code Patterns**:
```python
# From vortex/utils/memory.py
import torch
import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VRAMSnapshot:
    """VRAM usage snapshot at a point in time."""
    timestamp: str
    event: str
    slot: Optional[int]
    vram_usage_gb: float
    vram_allocated_gb: float
    vram_reserved_gb: float
    models: Dict[str, float]

class VRAMMonitor:
    """Monitor VRAM usage and enforce limits."""

    def __init__(self, soft_limit_gb: float = 11.0, hard_limit_gb: float = 11.5,
                 emergency_cleanup: bool = True):
        self.soft_limit_bytes = int(soft_limit_gb * 1e9)
        self.hard_limit_bytes = int(hard_limit_gb * 1e9)
        self.emergency_cleanup = emergency_cleanup

        # Metrics
        self.soft_limit_violations = 0
        self.hard_limit_violations = 0
        self.emergency_cleanups = 0

        # Leak detection
        self.baseline_usage = None
        self.generation_count = 0

    def check_limits(self, context: str = "") -> None:
        """Check VRAM usage against limits."""
        current_usage = torch.cuda.memory_allocated()

        # Soft limit (warning)
        if current_usage > self.soft_limit_bytes:
            self.soft_limit_violations += 1
            delta_gb = (current_usage - self.soft_limit_bytes) / 1e9
            logger.warning(f"VRAM soft limit exceeded", extra={
                "context": context,
                "current_usage_gb": current_usage / 1e9,
                "soft_limit_gb": self.soft_limit_bytes / 1e9,
                "delta_gb": delta_gb
            })

            # Emergency cleanup if enabled
            if self.emergency_cleanup:
                self._emergency_cleanup()

        # Hard limit (error)
        if current_usage > self.hard_limit_bytes:
            self.hard_limit_violations += 1
            delta_gb = (current_usage - self.hard_limit_bytes) / 1e9
            logger.error(f"VRAM hard limit exceeded", extra={
                "context": context,
                "current_usage_gb": current_usage / 1e9,
                "hard_limit_gb": self.hard_limit_bytes / 1e9,
                "delta_gb": delta_gb
            })
            raise MemoryPressureError(
                f"VRAM usage {current_usage/1e9:.2f}GB exceeds hard limit "
                f"{self.hard_limit_bytes/1e9:.2f}GB"
            )

    def log_snapshot(self, event: str, slot: Optional[int] = None,
                     models: Optional[Dict[str, float]] = None) -> VRAMSnapshot:
        """Log current VRAM usage snapshot."""
        snapshot = VRAMSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            event=event,
            slot=slot,
            vram_usage_gb=torch.cuda.memory_allocated() / 1e9,
            vram_allocated_gb=torch.cuda.memory_allocated() / 1e9,
            vram_reserved_gb=torch.cuda.memory_reserved() / 1e9,
            models=models or {}
        )

        logger.info("VRAM snapshot", extra=snapshot.__dict__)
        return snapshot

    def _emergency_cleanup(self) -> None:
        """Emergency CUDA cache cleanup."""
        before = torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()

        freed_mb = (before - after) / 1e6
        self.emergency_cleanups += 1

        logger.info(f"Emergency cleanup freed {freed_mb:.1f}MB", extra={
            "before_gb": before / 1e9,
            "after_gb": after / 1e9,
            "freed_mb": freed_mb
        })

    def detect_memory_leak(self, threshold_mb: float = 100) -> bool:
        """Check if VRAM usage has grown significantly over time."""
        if self.baseline_usage is None:
            self.baseline_usage = torch.cuda.memory_allocated()
            return False

        current = torch.cuda.memory_allocated()
        delta_mb = (current - self.baseline_usage) / 1e6

        if delta_mb > threshold_mb:
            logger.warning(f"Potential memory leak detected", extra={
                "initial_usage_gb": self.baseline_usage / 1e9,
                "current_usage_gb": current / 1e9,
                "delta_mb": delta_mb,
                "generations": self.generation_count,
                "delta_per_generation_kb": delta_mb * 1000 / max(self.generation_count, 1)
            })
            return True
        return False

    def increment_generation_count(self) -> None:
        """Increment generation counter for leak detection."""
        self.generation_count += 1

        # Check for leaks every 100 generations
        if self.generation_count % 100 == 0:
            self.detect_memory_leak()

def get_current_vram_usage() -> int:
    """Get current VRAM usage in bytes."""
    return torch.cuda.memory_allocated()

def clear_cuda_cache() -> int:
    """Clear CUDA cache, return freed bytes."""
    before = torch.cuda.memory_allocated()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    return before - after

class MemoryPressureWarning(UserWarning):
    """Soft VRAM limit exceeded (non-blocking)."""
    pass

class MemoryPressureError(RuntimeError):
    """Hard VRAM limit exceeded (blocking)."""
    pass
```

## Dependencies

**Hard Dependencies** (must be complete first):
- [T014] Vortex Core Pipeline - provides ModelRegistry that VRAM monitor tracks

**Soft Dependencies** (nice to have):
- Prometheus client library (for metrics export)

**External Dependencies**:
- Python 3.11
- PyTorch 2.1+ with CUDA 12.1+
- psutil 5.9.0+ (for system memory info)

## Design Decisions

**Decision 1: Soft vs. Hard Limits**
- **Rationale**: Soft limit (11.0GB) provides early warning without blocking. Hard limit (11.5GB) prevents catastrophic OOM crashes.
- **Alternatives**:
  - Single limit (rejected: no early warning)
  - Three-tier limits (rejected: too complex)
- **Trade-offs**: (+) Flexible, graceful degradation. (-) Soft limit might be ignored by developers.

**Decision 2: Emergency Cleanup (CUDA Cache Clear)**
- **Rationale**: torch.cuda.empty_cache() can free fragmented memory (50-200MB) without restarting process.
- **Alternatives**:
  - No cleanup (rejected: forces restart on soft limit)
  - Full restart (rejected: 5+ minute downtime)
- **Trade-offs**: (+) Fast recovery, no downtime. (-) Cache clear causes 1-2s pause in next generation.

**Decision 3: Per-Model VRAM Tracking**
- **Rationale**: Knowing which model is over budget (e.g., Flux at 6.5GB instead of 6.0GB) helps debugging.
- **Alternatives**:
  - Total usage only (rejected: hard to debug)
  - Per-tensor tracking (rejected: too granular, expensive)
- **Trade-offs**: (+) Useful for debugging. (-) Slight overhead to track per-model.

**Decision 4: Memory Leak Detection (100-generation threshold)**
- **Rationale**: Slow leaks (1-2MB/generation) compound to 100-200MB over 100 generations. Early detection prevents OOM.
- **Alternatives**:
  - No leak detection (rejected: silent failures)
  - Per-generation detection (rejected: noisy, false positives)
- **Trade-offs**: (+) Catches slow leaks. (-) May miss fast leaks between checks.

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| False positive soft limit warnings | Low (noise) | Medium (spiky usage) | Tune soft limit based on P99 usage, add hysteresis (only warn if sustained >30s) |
| Emergency cleanup pauses generation | Medium (latency spike) | Low (rare soft limit) | Monitor cleanup frequency, increase soft limit if triggered >1/hour |
| Memory leak detection false negatives | Medium (missed leaks) | Low (100-gen check) | Reduce check interval to 50 generations, add per-slot delta tracking |
| Per-model tracking overhead | Low (CPU/VRAM) | Low (simple dict) | Use weak references, only track on snapshot (not per-call) |
| VRAM monitor itself leaks memory | Medium (ironic) | Very low | Unit test VRAM monitor in isolation, verify no delta after 1000 calls |

## Progress Log

### [2025-12-24T00:00:00Z] - Task Created

**Created By:** task-creator agent
**Reason:** User request to create comprehensive Vortex Engine tasks per PRD sections 12.1-12.3
**Dependencies:** T014 (Vortex Core Pipeline)
**Estimated Complexity:** Standard (11,000 tokens estimated)

**Notes**: VRAM Manager is essential for production reliability. Prevents OOM crashes that cause slot misses and 150 ICN slashing penalties. Soft/hard limits enable graceful degradation.

## Completion Checklist

**Code Complete**:
- [ ] vortex/utils/memory.py implemented with VRAMMonitor class
- [ ] check_limits() with soft/hard limit detection
- [ ] log_snapshot() for VRAM usage logging
- [ ] emergency_cleanup() with CUDA cache clearing
- [ ] detect_memory_leak() with 100-generation threshold
- [ ] Custom exceptions: MemoryPressureWarning, MemoryPressureError
- [ ] Integration with VortexPipeline (check limits before/after generation)

**Testing**:
- [ ] Unit tests pass (mocked CUDA)
- [ ] Integration test triggers soft limit warning
- [ ] Integration test triggers hard limit error
- [ ] Pre-flight check test rejects insufficient VRAM
- [ ] Memory leak detection test catches slow leaks
- [ ] Emergency cleanup test verifies cache clearing
- [ ] VRAM snapshot logging test validates JSON structure

**Documentation**:
- [ ] Docstrings for VRAMMonitor, check_limits(), log_snapshot()
- [ ] vortex/utils/README.md updated with VRAM monitoring usage
- [ ] Soft/hard limit thresholds documented (11.0GB, 11.5GB)
- [ ] Emergency cleanup behavior explained
- [ ] Memory leak detection threshold justified (100MB/100 generations)

**Performance**:
- [ ] check_limits() overhead <1ms (measured via timeit)
- [ ] log_snapshot() overhead <5ms
- [ ] VRAM monitor itself uses <10MB VRAM
- [ ] No memory leaks in VRAM monitor after 1000 calls

**Definition of Done:**
Task is complete when ALL acceptance criteria met, ALL validations pass, and VRAM monitoring reliably detects soft/hard limit violations with <1ms overhead, preventing OOM crashes on RTX 3060 12GB.
