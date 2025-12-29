# Performance Verification Report - T016 (LivePortrait Integration)

**Stage:** 4 - Performance & Concurrency Verification
**Agent:** verify-performance
**Date:** 2025-12-28
**Task:** T016 - LivePortrait Integration - Audio-Driven Video Warping

---

## Executive Summary

**Decision:** WARN
**Score:** 62/100
**Critical Issues:** 0
**High Issues:** 3
**Medium Issues:** 4

The LivePortrait implementation has a solid foundation but contains several performance concerns that prevent a full PASS. The primary issues are: (1) Placeholder implementation without actual model weights, (2) Potential memory leaks in viseme sequence generation, (3) Non-vectorized frame-by-frame processing pattern, and (4) Missing baseline performance metrics. While the code structure is well-designed, the performance cannot be verified without real model weights.

---

## Response Time Analysis

### Target: <8s P99 for 45-second video (1080 frames @ 24fps)

**Status:** UNKNOWN (Placeholder implementation)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| P99 Latency | <8.0s | Unknown | BLOCKS |
| Throughput | >135 fps | Unknown | - |
| Warmup time | <3 iterations | Configured | OK |

**Issue 1: Placeholder Pipeline (`liveportrait.py:39-111`)**
- The `LivePortraitPipeline.warp_sequence()` method returns random tensors (`torch.rand()`) instead of performing actual video warping
- This makes latency benchmarks meaningless until real model is integrated
- Line 110: `return torch.rand(num_frames, 3, 512, 512)` - no actual inference

**Issue 2: Sequential Frame Processing Pattern (`liveportrait.py:271-276`)**
- The current pattern delegates to `warp_sequence()` which processes frames sequentially
- Even with real model, frame-by-frame generation is O(n) where n=1080 frames
- No batching or parallel processing implemented

### Recommendation
- BLOCK until real LivePortrait model weights are integrated
- Run actual benchmarks with `python benchmarks/liveportrait_latency.py --iterations 50 --profile-vram`

---

## Memory & VRAM Analysis

### Target: 3.0-4.0 GB VRAM budget

**Status:** PARTIALLY VERIFIED (allocation patterns correct)

**Positive Findings:**
1. Pre-allocated buffer support (`liveportrait.py:282-286`)
   ```python
   if output is not None:
       output[:num_frames].copy_(video)
       return output
   ```
2. FP16 precision configuration (`liveportrait.py:482-491`)
3. CUDA OOM handling with VRAM diagnostics (`liveportrait.py:517-535`)

**Issue 3: Memory Leak Risk - Expression Parameter Cloning (`liveportrait.py:327`)**
```python
return [params_tensor.clone() for _ in range(num_frames)]  # 1080 allocations
```
- Creates 1080 separate tensor allocations for expression parameters
- Each clone allocates new memory instead of reusing buffer
- For 45s video @ 24fps: 1080 tensors * ~32 bytes = ~35KB per call
- Under high concurrency, this could fragment GPU memory

**Fix:** Pre-allocate single tensor and update in-place:
```python
params_buffer = torch.zeros(num_frames, 4, dtype=torch.float32, device=device)
# Update in-place during iteration
```

**Issue 4: Viseme List Growth Pattern (`lipsync.py:127-136`)**
```python
raw_visemes = []
for i in range(num_frames):
    viseme = _audio_segment_to_viseme(frame_audio)
    raw_visemes.append(viseme)  # 1080 appends
```
- List append pattern could cause memory fragmentation
- Creates 1080 separate tensor objects before smoothing

**Fix:** Pre-allocate tensor array:
```python
raw_visemes = torch.zeros(num_frames, 3, dtype=torch.float32)
raw_visemes[i] = viseme
```

**Issue 5: Smoothing Window Tensor Stack (`lipsync.py:260-261`)**
```python
window_visemes = torch.stack(visemes[start:end])  # Creates new tensor per frame
```
- For 1080 frames with window_size=3: creates 1080 intermediate tensors
- Each stack operation allocates and deallocates memory

**Fix:** Use 1D convolution for smoothing:
```python
# Single kernel application instead of loop
smoothed = torch.nn.functional.conv1d(
    visemes.t().unsqueeze(0),
    smoothing_kernel,
    padding=1
)
```

---

## N+1 Query Pattern Analysis

### Status: N/A (No database queries in this component)

LivePortrait is a pure GPU inference pipeline with no external dependencies that would cause N+1 query patterns. However:

**Issue 6: Frame-by-Frame Audio Extraction (`lipsync.py:128-132`)**
```python
for i in range(num_frames):
    start_sample = int(i * sample_rate / fps)
    end_sample = int((i + 1) * sample_rate / fps)
    frame_audio = audio[start_sample:end_sample]  # 1080 slices
```
- Creates 1080 tensor views (slices) of the audio
- While efficient in PyTorch (views are cheap), the loop overhead is significant
- Could be vectorized using `unfold()` for sliding window

**Fix:** Vectorized audio framing:
```python
# Single tensor operation instead of loop
frame_size = sample_rate // fps
frames = audio.unfold(0, frame_size, frame_size)
```

---

## Concurrency & Race Conditions

### Status: LOW RISK (Single-threaded inference)

**Positive Findings:**
1. `@torch.no_grad()` decorator on `animate()` method (`liveportrait.py:173`)
2. Deterministic seed control for reproducibility (`liveportrait.py:237-239`)

**Issue 7: Non-Thread-Safe Seed Mutation (`liveportrait.py:238`)**
```python
if seed is not None:
    torch.manual_seed(seed)  # Global state mutation
```
- In concurrent environment, multiple threads calling `animate()` with seeds will race
- The global random state is shared across all threads

**Risk Level:** MEDIUM
- If multiple video generations run concurrently, determinism is lost
- Seed sets global state, but other threads may interleave operations

**Fix:** Use generator-based local seeding:
```python
if seed is not None:
    generator = torch.Generator(device=self.device)
    generator.manual_seed(seed)
    # Pass generator to all random operations
```

---

## Algorithmic Complexity Analysis

### Current Complexity: O(n) where n = num_frames

| Operation | Complexity | Bottleneck |
|-----------|------------|------------|
| Viseme generation | O(n) | Loop + slice |
| Viseme smoothing | O(n * w) | w = window_size |
| Expression interpolation | O(n * k) | k = sequence length |
| Video warping | O(n) | Sequential frames |

**Issue 8: Cubic Interpolation Per Frame (`liveportrait.py:393-394`)**
```python
t_smooth = 3 * t**2 - 2 * t**3  # Computed for every frame
```
- While mathematically correct for smooth transitions, computing this per-frame for 1080 frames is unnecessary overhead
- Could pre-compute interpolation weights

**Fix:** Pre-compute interpolation curve:
```python
# Once per animation call
t_values = torch.linspace(0, 1, num_frames)
t_smooth_values = 3 * t_values**2 - 2 * t_values**3
# Index during loop instead of computing
```

---

## Database & I/O Analysis

**Status:** N/A - No database or external I/O in this component

LivePortrait is a pure GPU inference pipeline. No N+1 query patterns detected.

---

## Cache Analysis

### Status: MISSING OPPORTUNITY

**Issue 9: Phoneme-to-Viseme Dictionary Lookups (`lipsync.py:219-227`)**
```python
if phoneme not in PHONEME_TO_VISEME:
    logger.warning(...)
    return torch.tensor([0.4, 0.5, 0.3], dtype=torch.float32)
params = PHONEME_TO_VISEME[phoneme]
return torch(params, dtype=torch.float32)  # New tensor allocation per call
```
- Each call to `phoneme_to_viseme()` allocates a new tensor
- Common phonemes are allocated repeatedly

**Fix:** Cache viseme tensors:
```python
_VISEME_CACHE = {name: torch.tensor(params) for name, params in PHONEME_TO_VISEME.items()}
def phoneme_to_viseme(phoneme: str) -> torch.Tensor:
    return _VISEME_CACHE.get(phoneme, _NEUTRAL_VISEME).clone()
```

---

## Performance Baselines

**Status:** NO BASELINE DATA

The task specification requires:
- P99 < 8s for 45s video on RTX 3060
- VRAM 3.0-4.0 GB
- Lip-sync accuracy within ±2 frames

**Current Reality:**
- Benchmark script exists (`benchmarks/liveportrait_latency.py`) but cannot run with placeholder model
- No historical performance data to compare against
- No regression detection mechanism

**Required Actions:**
1. Integrate real LivePortrait model weights
2. Run 50+ iteration benchmark to establish baseline
3. Store results in `.tasks/baselines/liveportrait.json`
4. Configure CI to run benchmarks on PRs

---

## Load Testing Assessment

**Status:** NOT PERFORMED

The implementation lacks:
1. Concurrent generation tests (multiple `animate()` calls in parallel)
2. Memory leak detection over 100+ iterations
3. VRAM pressure testing with other models loaded (Flux, Kokoro, CLIP)

**Test Required:**
```python
# Test memory stability over 100 generations
for i in range(100):
    model.animate(...)
    if torch.cuda.memory_allocated() > baseline * 1.5:
        raise MemoryLeakError(f"Leak detected at iteration {i}")
```

---

## Critical Path Analysis

LivePortrait is on the critical generation path:
```
Flux (6s) -> Kokoro (3s) -> LivePortrait (8s) -> CLIP (1s) = 18s total
```

**Issue 10: Sequential Dependency Blocks Parallelization**
- The pipeline must wait for LivePortrait to complete before CLIP verification
- No overlap possible between video warping and semantic verification
- LivePortrait is the bottleneck (44% of total time)

**Mitigation:**
- Pipeline cannot be restructured due to dependencies
- Only option is to optimize LivePortrait itself to <6s target

---

## Blocking Criteria Evaluation

| Criteria | Threshold | Status | Blocks |
|----------|-----------|--------|--------|
| Response time >2s | N/A | Unknown | No data |
| Response time regression >100% | N/A | N/A | No baseline |
| Memory leak (unbounded growth) | Any | Unknown | Not tested |
| Race condition | Critical path | MEDIUM risk | Issue 7 |
| N+1 query | Critical path | None | N/A |
| Missing database indexes | N/A | N/A | N/A |

**Result:** WARN due to lack of actual performance data

---

## Detailed Findings

### HIGH Issues

1. **`liveportrait.py:110` - Placeholder Implementation**
   - `warp_sequence()` returns random tensors instead of actual inference
   - Prevents any meaningful performance measurement
   - Fix: Integrate real LivePortrait model from HuggingFace

2. **`liveportrait.py:327` - Memory Leak Risk via Tensor Cloning**
   - Creates 1080 tensor clones for expression parameters
   - Causes GPU memory fragmentation under load
   - Fix: Pre-allocate buffer and update in-place

3. **`lipsync.py:260` - Inefficient Smoothing with Tensor Stack**
   - Creates new tensor for each frame's smoothing window
   - O(n * w) memory allocations instead of O(n)
   - Fix: Use 1D convolution for vectorized smoothing

### MEDIUM Issues

4. **`liveportrait.py:238` - Thread-Unsafe Global Seed Mutation**
   - `torch.manual_seed()` sets global state
   - Concurrent calls lose determinism
   - Fix: Use local `torch.Generator` per call

5. **`lipsync.py:128-132` - Non-Vectorized Audio Slicing**
   - Loop creates 1080 tensor views
   - Fix: Use `unfold()` for single tensor operation

6. **`lipsync.py:191-202` - Inefficient Spectral Centroid Calculation**
   - Full FFT computed per audio segment
   - Could use STFT or pre-computed features
   - Fix: Use librosa's pre-computed features or Wav2Vec2 embeddings

7. **Missing Performance Baseline**
   - No historical data to detect regression
   - Benchmark exists but cannot run with placeholder
   - Fix: Establish baseline after real model integration

### LOW Issues

8. **`liveportrait.py:394` - Repeated Cubic Interpolation Computation**
   - `3*t**2 - 2*t**3` computed per frame
   - Fix: Pre-compute interpolation weights

9. **`lipsync.py:226` - Repeated Tensor Allocation in phoneme_to_viseme**
   - Creates new tensor for each phoneme lookup
   - Fix: Cache viseme tensors as module-level constants

---

## Recommendations

### Immediate Actions (Before PASS)

1. **Integrate Real LivePortrait Model**
   - Replace `LivePortraitPipeline` placeholder with actual model
   - Download weights: `python vortex/scripts/download_liveportrait.py`
   - Verify with real inference test

2. **Fix Memory Leak Patterns**
   - Pre-allocate expression parameter buffer
   - Use vectorized smoothing (conv1d)
   - Cache phoneme-to-viseme tensors

3. **Establish Performance Baseline**
   ```bash
   python vortex/benchmarks/liveportrait_latency.py \
       --iterations 50 \
       --warmup 5 \
       --profile-vram \
       --output .tasks/baselines/liveportrait.json
   ```

4. **Fix Thread Safety**
   - Replace `torch.manual_seed()` with local `Generator`

### Future Optimizations

1. **TensorRT Integration** - 20-30% speedup potential
2. **Batch Processing** - If VRAM allows, process multiple frames
3. **Caching** - Cache expression sequences, common viseme patterns

---

## Test Coverage

| Test Type | Status | Coverage |
|-----------|--------|----------|
| Unit tests | PASS | Mocked interface |
| Integration tests | SKIP | Requires GPU (not run) |
| VRAM profiling | EXISTS | Not executed |
| Latency benchmark | EXISTS | Not executed (placeholder) |
| Memory leak test | MISSING | N/A |
| Concurrency test | MISSING | N/A |

---

## Conclusion

The LivePortrait integration code structure is well-designed with proper error handling, VRAM budgeting, and test infrastructure. However, the **placeholder implementation prevents performance verification**, and several **memory leak risks** exist in the viseme/expression generation patterns.

**Recommendation:** WARN until real model is integrated and benchmarks are run. The code quality is high, but performance cannot be verified without actual inference.

---

**Generated:** 2025-12-28T19:36:30Z
**Agent:** verify-performance (Stage 4)
**Task:** T016
