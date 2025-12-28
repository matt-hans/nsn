# Performance Verification Report - T014 (Vortex Core Pipeline)

**Date:** 2025-12-28
**Agent:** verify-performance (STAGE 4)
**Task:** T014 - Vortex Core Pipeline - Static VRAM Manager & Generation Orchestration
**Status:** WARN
**Score:** 72/100

---

## Executive Summary

The Vortex Core Pipeline implementation demonstrates good architectural patterns for static VRAM management and async orchestration. However, performance cannot be fully verified without real model implementations (deferred to T015-T018). The code foundation is solid but has several performance concerns that should be addressed before production deployment.

### Critical Gaps
- **No real models yet** - All models are MockModel placeholders with simulated VRAM allocation
- **No benchmarking data** - Performance targets (<15s generation) cannot be validated
- **Buffer cloning overhead** - Using `.clone()` on pre-allocated buffers creates unnecessary copies

---

## Response Time Analysis

### Baseline Status: NOT ESTABLISHED

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Cold start (model loading) | <30s | TBD | N/A (mocks) |
| Slot generation (P99) | <15s | ~0.35s | N/A (mocks) |
| VRAM usage | <=11.5GB | ~10.8GB | PASS (theoretical) |

### Issues

1. **[HIGH] `pipeline.py:428,441,455` - Buffer cloning creates unnecessary allocations**
   ```python
   async def _generate_audio(self, recipe: Dict) -> torch.Tensor:
       await asyncio.sleep(0.1)
       return self.audio_buffer.clone()  # <-- Creates 4.3MB copy per call
   ```
   **Impact:** Each generation clones 3 buffers (actor: 0.75MB, video: ~1.6GB, audio: 4.3MB), wasting ~1.6GB of memory bandwidth per slot.
   **Fix:** Return views or indices instead of clones. The buffers should be reused in-place.

2. **[MEDIUM] No actual GPU kernel execution timing**
   - Current mock uses `asyncio.sleep()` which does not measure CUDA kernel time
   - Real model inference may have different async characteristics
   **Fix:** Defer to T015-T018 for real measurements

---

## Memory & VRAM Analysis

### VRAM Budget Compliance: PASS (Theoretical)

| Component | Budget | Allocation | Status |
|-----------|--------|------------|--------|
| Flux-Schnell | 6.0 GB | 6.0 GB (mock) | PASS |
| LivePortrait | 3.5 GB | 3.5 GB (mock) | PASS |
| Kokoro-82M | 0.4 GB | 0.4 GB (mock) | PASS |
| CLIP-B-32 | 0.3 GB | 0.3 GB (mock) | PASS |
| CLIP-L-14 | 0.6 GB | 0.6 GB (mock) | PASS |
| Overhead | 1.0 GB | ~0.5 GB | PASS |
| **TOTAL** | **11.8 GB** | **~11.3 GB** | **PASS** |

### Memory Pressure Monitoring: PASS

- **Soft limit (11.0GB):** Correctly implemented with warning
- **Hard limit (11.5GB):** Correctly raises `MemoryPressureError`
- **Single warning flag:** Prevents log spam (good pattern)

### Issues

1. **[LOW] `models/__init__.py:42` - MockModel VRAM simulation is inefficient**
   ```python
   param_count = int(vram_gb * 268435456)  # Creates 1.6B parameters for 6GB model
   self.dummy_weight = nn.Parameter(torch.randn(param_count, dtype=torch.float32))
   ```
   **Impact:** Mock initialization takes significant time/memory on CPU
   **Fix:** Use smaller allocation for testing; real models will be different anyway

2. **[MEDIUM] Buffer allocation in `_allocate_buffers()` may fail silently on OOM**
   - No try-catch around buffer allocation
   - If model loading uses most VRAM, buffer allocation could fail
   **Fix:** Add VRAM check before buffer allocation, or allocate buffers before models

---

## Async Orchestration Analysis

### Parallelization: PASS

| Pattern | Implementation | Status |
|---------|---------------|--------|
| `asyncio.create_task()` | Used for audio and actor generation | PASS |
| `asyncio.gather()` | Waits for both tasks with timeout | PASS |
| Task cancellation | Properly handled with `CancelledError` | PASS |

### Code Review: `pipeline.py:369-377`

```python
audio_task = asyncio.create_task(self._generate_audio(recipe))
actor_task = asyncio.create_task(self._generate_actor(recipe))

audio_result, actor_result = await asyncio.wait_for(
    asyncio.gather(audio_task, actor_task),
    timeout=timeout,
)
```

**Assessment:** Correct pattern for parallel async execution. Timeout wrapper prevents hangs.

### Issues

1. **[LOW] No concurrent task limit**
   - If multiple slots are generated concurrently (not in spec but possible), no rate limiting
   - **Fix:** Add semaphore if concurrent generation is needed

2. **[MEDIUM] Sequential video generation waits for both audio + actor**
   - This is correct per spec, but video generation could theoretically start with partial audio
   - **Not an issue per current spec** - keeping as-is

---

## N+1 Query Analysis

### Database Queries: N/A (No database in Vortex)

The Vortex pipeline does not query databases. All data comes from:
1. Recipe input (passed as Dict)
2. Pre-loaded models (static VRAM)
3. Pre-allocated buffers

### Potential Issues: None

---

## Algorithmic Complexity Analysis

| Function | Complexity | Notes |
|----------|------------|-------|
| `ModelRegistry._load_all_models()` | O(1) - 5 models constant | PASS |
| `ModelRegistry.get_model()` | O(1) - dict lookup | PASS |
| `VRAMMonitor.check()` | O(1) - single syscall | PASS |
| `VortexPipeline.generate_slot()` | O(model_forward) | Defer to T015-T018 |
| `VortexPipeline._allocate_buffers()` | O(buffer_size) | One-time at init |

### Issues: None

All algorithms are O(1) or delegated to model implementations.

---

## Concurrency Analysis

### Race Conditions: PASS

- No shared mutable state between async tasks
- Each task uses its own buffer (pre-allocated)
- Results are combined after both complete (safe join pattern)

### Deadlock Risks: MINIMAL

- No circular dependencies
- Timeout on `asyncio.wait_for()` prevents hangs
- Cancellation handling is correct

### Issues

1. **[LOW] `pipeline.py:401-403` - CancelledError handling could orphan GPU work**
   ```python
   except asyncio.CancelledError:
       logger.warning(f"Slot {slot_id} generation cancelled")
       raise
   ```
   **Risk:** If cancellation happens mid-kernel, CUDA may continue processing orphaned work
   **Fix:** Add `torch.cuda.synchronize()` in cancellation handler (future enhancement)

---

## Database & I/O Analysis

### Configuration Loading: PASS

- YAML loaded once at init (`pipeline.py:258`)
- No repeated file I/O

### Logging: PASS

- Structured JSON logging configured
- VRAM stats included in logs per config

### Issues: None

---

## Caching Strategy

### Static VRAM Residency: PASS

The entire design is based on static residency (no swapping):
- Models loaded once at init
- Never unloaded
- Pre-allocated buffers

This IS the caching strategy for models.

### Issues: None

---

## Summary of Issues

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH | 1 |
| MEDIUM | 3 |
| LOW | 3 |
| INFO | 0 |

### Detailed Issues List

1. **[HIGH] `pipeline.py:428,441,455` - Unnecessary buffer cloning**
   - Cloning pre-allocated buffers defeats the purpose of pre-allocation
   - Creates ~1.6GB of unnecessary memory traffic per generation

2. **[MEDIUM] No real performance benchmarks**
   - Cannot validate <15s generation target with mocks
   - Defer to T015-T018 for actual model performance

3. **[MEDIUM] Buffer allocation order may cause OOM**
   - Models loaded first, then buffers
   - If models use 11.3GB, buffers may push over 11.5GB
   - Consider allocating buffers first or checking available VRAM

4. **[MEDIUM] MockModel VRAM simulation is inefficient**
   - Creates billions of parameters for testing
   - Slow initialization on CPU

5. **[LOW] No CUDA synchronization on cancellation**
   - Orphaned GPU work possible on task cancellation
   - Add `torch.cuda.synchronize()` in CancelledError handler

6. **[LOW] No concurrent task limiting**
   - If parallel slot generation is ever needed, add semaphore

7. **[LOW] VRAM check timing**
   - VRAM checked at start of generation but not during intermediate steps
   - Consider periodic checks during long-running operations

---

## Recommendations

### Before Production (T015-T018 Integration)

1. **Remove buffer cloning** - Return views or use buffer indices
2. **Benchmark on real GPU** - Validate <15s target with actual models
3. **Add per-step VRAM checks** - Monitor during video warping phase
4. **Add CUDA sync on cancellation** - Prevent orphaned work

### Future Enhancements

1. **Add generation metrics** - Track P50/P95/P99 latencies
2. **Memory profiler integration** - Automatic VRAM leak detection
3. **Adaptive timeout** - Adjust based on historical performance

---

## Performance Gates

| Gate | Status | Notes |
|------|--------|-------|
| VRAM <= 11.5GB | PASS | Theoretical budget met |
| Buffer pre-allocation | WARN | Cloning defeats purpose |
| Async parallelization | PASS | asyncio.create_task used correctly |
| No N+1 queries | PASS | No database queries |
| O(n^2) loops | PASS | All O(1) or delegated |
| Response time baseline | N/A | Requires real models |

---

## Final Verdict

**Decision:** WARN

**Rationale:**
- The architectural foundation is sound (static VRAM, pre-allocated buffers, async orchestration)
- VRAM budget compliance is theoretically correct
- However, buffer cloning introduces unnecessary memory overhead
- Real performance cannot be verified without actual model implementations (T015-T018)

**Blocking Issues:** None (code foundation is acceptable for T014 completion)

**Conditions for PASS:**
1. Address buffer cloning issue
2. Complete T015-T018 with real models
3. Benchmark on RTX 3060 showing <15s P99 generation

---

## Audit Metadata

- **Analysis Duration:** ~2 minutes
- **Files Analyzed:** 8
- **Lines of Code:** ~800
- **Test Coverage:** Unit tests present (mocked)
- **Configuration:** vortex/config.yaml reviewed

**Files Analyzed:**
- `/vortex/src/vortex/pipeline.py` (471 lines)
- `/vortex/src/vortex/utils/memory.py` (144 lines)
- `/vortex/src/vortex/models/__init__.py` (230 lines)
- `/vortex/config.yaml` (57 lines)
- `/vortex/tests/unit/test_pipeline.py` (322 lines)
- `/vortex/tests/unit/test_memory.py` (100 lines)
- `/vortex/README.md` (224 lines)
- `/.tasks/tasks/T014-vortex-core-pipeline.md` (295 lines)
