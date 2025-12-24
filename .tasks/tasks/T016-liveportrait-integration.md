---
id: T016
title: LivePortrait Integration - Audio-Driven Video Warping
status: pending
priority: 1
agent: ai-ml
dependencies: [T014, T015, T017]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [vortex, ai-ml, python, gpu, video, animation, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - prd.md#section-12.1-static-resident-vram-layout
  - prd.md#section-12.2-generation-pipeline
  - architecture.md#section-5.3-ai-ml-pipeline

est_tokens: 13000
actual_tokens: null
---

## Description

Integrate LivePortrait video warping model to animate static actor images (from Flux) into talking head videos driven by audio (from Kokoro TTS). This is the core video generation component that produces the final 45-second video slots.

**Critical Requirements**:
- VRAM budget: 3.5 GB (FP16 precision with TensorRT optimization)
- Input: 512×512 actor image + 24kHz audio waveform + expression presets
- Output: 24 FPS video (45 seconds = 1080 frames)
- Lip-sync accuracy: Audio-visual sync within ±2 frames (~83ms at 24fps)
- Latency target: <8s for 45-second video on RTX 3060

**Integration Points**:
- Loaded by VortexPipeline._load_live_portrait() at startup
- Called by VortexPipeline._warp_video() after audio + actor generation complete
- Outputs to pre-allocated video_buffer (1080×3×512×512 tensor)

## Business Context

**User Story**: As a Director node, I want to transform static actor images into realistic talking head videos synchronized with TTS audio, so that I can produce engaging content that passes CLIP semantic verification and earns reputation rewards.

**Why This Matters**:
- LivePortrait is the bottleneck stage of the generation pipeline (8s of the 12s budget)
- Lip-sync quality directly impacts CLIP scores and BFT consensus
- Poor animation quality leads to slot rejections (200 ICN reputation penalty)
- This component determines whether Director-generated content is "canon" or rejected

**What It Unblocks**:
- T018 (CLIP ensemble verification - validates LivePortrait output)
- T020 (Slot timing orchestration - depends on video generation latency)
- Full end-to-end video generation pipeline

**Priority Justification**: Priority 1 (Critical Path) - Blocks all video output functionality. Without LivePortrait, we have only static images, not videos, making ICN non-functional as a streaming network.

## Acceptance Criteria

- [ ] LivePortrait model loads with FP16 precision and TensorRT optimization at VortexPipeline initialization
- [ ] VRAM usage for LivePortrait is 3.0-4.0 GB (measured via torch.cuda.memory_allocated())
- [ ] animate() method accepts: source_image (512×512), driving_audio (45s @ 24kHz), expression_preset, fps=24, duration=45, output=video_buffer
- [ ] Output is 1080×3×512×512 float32 tensor (T×C×H×W) in range [0,1]
- [ ] Generation time is <8s P99 for 45-second video on RTX 3060
- [ ] Lip-sync accuracy: audio-visual alignment within ±2 frames (83ms tolerance)
- [ ] Expression presets supported: "neutral", "excited", "manic", "calm" (from Recipe schema)
- [ ] Expression transitions are smooth (no abrupt changes between keyframes)
- [ ] Outputs are written to pre-allocated video_buffer (no new allocations)
- [ ] Model supports batch_size=1 (single video per call)
- [ ] Error handling for: invalid image dimensions, audio length mismatch, CUDA OOM
- [ ] Model determinism: same inputs + seed = identical output frames

## Test Scenarios

**Test Case 1: Standard Video Generation**
- Given: VortexPipeline with LivePortrait loaded
  And actor_image from Flux (512×512×3)
  And audio_waveform from Kokoro (45s @ 24kHz)
- When: animate(source_image=actor, driving_audio=audio, expression="excited", fps=24, duration=45)
- Then: Output is 1080×3×512×512 tensor (1080 frames)
  And VRAM usage increases by <500MB during generation
  And generation completes in 6-8 seconds
  And video_buffer contains valid frames (all values in [0,1])

**Test Case 2: Lip-Sync Accuracy**
- Given: LivePortrait model loaded
  And test audio with distinct phonemes (e.g., "ba", "ma", "pa" at known timestamps)
- When: animate() is called with test audio
- Then: Mouth movements in output frames align with phonemes
  And alignment error is ≤±2 frames (measured via visual inspection or phoneme detector)
  And lip movements are anatomically plausible

**Test Case 3: Expression Preset Application**
- Given: Same actor image and audio
- When: animate() called with expression="neutral" vs expression="excited"
- Then: "excited" output has wider eyes, more mouth movement, faster head motion
  And "neutral" output has minimal facial movement
  And both maintain lip-sync accuracy

**Test Case 4: Expression Sequence Transitions**
- Given: Expression sequence ["neutral", "excited", "manic", "calm"]
- When: animate() applies sequence over 45-second duration
- Then: Transitions occur smoothly (no frame-to-frame jumps)
  And each expression lasts ~11.25 seconds
  And final expression is "calm"

**Test Case 5: VRAM Budget Compliance**
- Given: Fresh Python process with only LivePortrait loaded
- When: Model initialization completes
- Then: torch.cuda.memory_allocated() shows 3.0-4.0 GB
  And model layers use FP16 precision (torch.half)
  And TensorRT optimizations are applied (if available)

**Test Case 6: Audio-Video Length Mismatch**
- Given: Actor image and 60-second audio (exceeds 45s duration)
- When: animate(duration=45) is called
- Then: Audio is truncated to 45 seconds
  And warning logged: "Audio truncated from 60s to 45s"
  And generation succeeds with 1080 frames

## Technical Implementation

**Required Components**:

1. **vortex/models/liveportrait.py** (Model wrapper)
   - `load_liveportrait(device: str, precision: str = "fp16") -> LivePortraitModel`
   - `LivePortraitModel.animate(source_image, driving_audio, expression, fps, duration, output)`
   - TensorRT optimization loader (if available)
   - Lip-sync audio-to-viseme mapping

2. **vortex/models/configs/liveportrait_fp16.yaml** (Model config)
   - Model source: Hugging Face or GitHub release
   - Precision: FP16 (torch.half)
   - TensorRT engine path: `~/.cache/vortex/liveportrait_trt.engine`
   - Expression keyframe timings

3. **vortex/utils/lipsync.py** (Lip-sync utilities)
   - `audio_to_visemes(waveform: torch.Tensor) -> List[Viseme]`
   - Phoneme-to-mouth-shape mapping
   - Temporal alignment verification

4. **vortex/tests/integration/test_liveportrait_generation.py** (Integration tests)
   - Test cases 1-6 from above
   - Visual quality regression (compare against reference videos)
   - Lip-sync accuracy measurement

5. **vortex/benchmarks/liveportrait_latency.py** (Performance benchmark)
   - Measure generation time over 50 iterations
   - Profile frame generation bottlenecks
   - TensorRT vs native PyTorch comparison

**Validation Commands**:
```bash
# Install LivePortrait dependencies
pip install liveportrait==1.0.0  # hypothetical package
pip install onnx==1.15.0 tensorrt==8.6.1  # for TensorRT optimization

# Download model weights
python vortex/scripts/download_liveportrait.py

# Unit tests (mocked LivePortrait)
pytest vortex/tests/unit/test_liveportrait.py -v

# Integration test (real model, requires GPU)
pytest vortex/tests/integration/test_liveportrait_generation.py --gpu -v

# VRAM profiling
python vortex/benchmarks/liveportrait_vram_profile.py

# Latency benchmark
python vortex/benchmarks/liveportrait_latency.py --iterations 50

# Lip-sync accuracy test
python vortex/tests/visual/lipsync_accuracy.py --audio test_phonemes.wav --output /tmp/lipsync_test.mp4
```

**Code Patterns**:
```python
# From vortex/models/liveportrait.py
import torch
from typing import List, Optional

class LivePortraitModel:
    def __init__(self, model, device: str = "cuda"):
        self.model = model.half().to(device)  # FP16
        self.device = device

    @torch.no_grad()
    def animate(
        self,
        source_image: torch.Tensor,  # [3, 512, 512]
        driving_audio: torch.Tensor,  # [samples] at 24kHz
        expression_preset: str = "neutral",
        fps: int = 24,
        duration: int = 45,
        output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate animated video from static image + audio."""
        # Validate inputs
        assert source_image.shape == (3, 512, 512), f"Invalid image shape: {source_image.shape}"
        expected_samples = duration * 24000  # 24kHz
        if driving_audio.shape[0] > expected_samples:
            driving_audio = driving_audio[:expected_samples]
            logger.warning(f"Audio truncated to {duration}s")

        # Extract audio features (visemes)
        visemes = self._audio_to_visemes(driving_audio, fps)

        # Generate expression keyframes
        expression_params = self._get_expression_params(expression_preset)

        # Warp source image frame-by-frame
        num_frames = fps * duration  # 1080 frames
        frames = []

        for i in range(num_frames):
            # Interpolate expression + viseme at current frame
            current_viseme = visemes[i]
            current_expression = self._interpolate_expression(expression_params, i, num_frames)

            # Warp source image
            warped_frame = self.model.warp(
                source_image,
                viseme=current_viseme,
                expression=current_expression
            )
            frames.append(warped_frame)

        # Stack into [T, C, H, W]
        video = torch.stack(frames, dim=0)

        # Write to pre-allocated buffer
        if output is not None:
            output[:num_frames].copy_(video)
            return output
        return video

    def _audio_to_visemes(self, audio: torch.Tensor, fps: int) -> List[torch.Tensor]:
        """Convert audio waveform to per-frame viseme parameters."""
        # Simplified: use Wav2Vec2 or similar for phoneme detection
        # Then map phonemes to mouth shapes
        # Returns list of viseme tensors (one per frame)
        pass  # Implementation details
```

## Dependencies

**Hard Dependencies** (must be complete first):
- [T014] Vortex Core Pipeline - provides ModelRegistry, VRAMMonitor, video_buffer
- [T015] Flux-Schnell Integration - provides actor image input
- [T017] Kokoro-82M TTS Integration - provides driving audio input

**Soft Dependencies** (nice to have):
- TensorRT 8.6+ (for optimization, optional but recommended)

**External Dependencies**:
- Python 3.11
- PyTorch 2.1+ with CUDA 12.1+
- liveportrait library (hypothetical, may need custom build)
- ONNX 1.15.0+ (for TensorRT export)
- TensorRT 8.6.1+ (optional, for FP16 optimization)
- librosa 0.10.0+ (for audio feature extraction)

## Design Decisions

**Decision 1: FP16 Precision with TensorRT**
- **Rationale**: FP32 LivePortrait uses ~7GB VRAM, exceeding budget. FP16 reduces to ~3.5GB with <2% quality loss. TensorRT adds 20-30% speedup.
- **Alternatives**:
  - FP32 (rejected: too much VRAM)
  - INT8 quantization (rejected: significant quality loss for facial details)
- **Trade-offs**: (+) Fits budget, fast. (-) Requires TensorRT setup, slight quality loss.

**Decision 2: 24 FPS Output**
- **Rationale**: 24 FPS is cinematic standard, balances quality and compute. 30 FPS increases generation time by 25% with minimal perceived quality gain.
- **Alternatives**:
  - 30 FPS (rejected: 25% slower, 25% more VRAM for buffer)
  - 15 FPS (rejected: choppy motion)
- **Trade-offs**: (+) Standard, smooth. (-) Not as smooth as 60 FPS (but 60 FPS is overkill for talking heads).

**Decision 3: Expression Presets vs. Continuous Control**
- **Rationale**: Presets ("neutral", "excited", "manic", "calm") are simple for Recipe authors, easier to validate, and cacheable. Continuous control (e.g., arousal/valence) is complex.
- **Alternatives**:
  - Continuous emotion parameters (rejected: harder to specify, cache, validate)
- **Trade-offs**: (+) Simple API, predictable. (-) Less expressive than continuous control.

**Decision 4: Sequential Frame Generation (Not Batched)**
- **Rationale**: Batching 1080 frames requires ~40GB VRAM. Sequential generation uses constant ~4GB.
- **Alternatives**:
  - Batch generation (rejected: exceeds VRAM budget)
- **Trade-offs**: (+) Fits budget. (-) Slower than batched (but still meets <8s target).

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Lip-sync drift over 45 seconds | High (poor quality) | Medium (accumulation error) | Phoneme-based keyframe correction every 5s, validate sync via automated test |
| TensorRT compilation failures | Medium (slower fallback) | Medium (driver/version issues) | Graceful fallback to native PyTorch FP16, document TensorRT setup |
| Expression transitions cause artifacts | Medium (visual glitches) | Low (tested interpolation) | Smooth cubic interpolation, visual regression tests |
| VRAM usage exceeds 4.0GB | High (OOM crashes) | Low (well-tested) | Monitor VRAM per frame, log spikes, cap buffer size |
| LivePortrait model unavailable | High (pipeline broken) | Low (cache weights) | Local weight caching, provide download script, fallback to simpler warping |

## Progress Log

### [2025-12-24T00:00:00Z] - Task Created

**Created By:** task-creator agent
**Reason:** User request to create comprehensive Vortex Engine tasks per PRD sections 12.1-12.3
**Dependencies:** T014 (Core Pipeline), T015 (Flux), T017 (Kokoro)
**Estimated Complexity:** Standard (13,000 tokens estimated)

**Notes**: LivePortrait is the most compute-intensive stage of the pipeline (8s target). FP16 + TensorRT optimization is critical. Lip-sync accuracy directly impacts CLIP scores and consensus.

## Completion Checklist

**Code Complete**:
- [ ] vortex/models/liveportrait.py implemented with load_liveportrait(), LivePortraitModel.animate()
- [ ] FP16 precision config verified
- [ ] TensorRT optimization loader (with native PyTorch fallback)
- [ ] Expression preset interpolation
- [ ] Audio-to-viseme pipeline
- [ ] Integration with VortexPipeline._warp_video()
- [ ] Output to pre-allocated video_buffer

**Testing**:
- [ ] Unit tests pass (mocked LivePortrait)
- [ ] Integration test generates 1080-frame video
- [ ] VRAM profiling shows 3.0-4.0GB usage
- [ ] Latency benchmark P99 <8s on RTX 3060
- [ ] Lip-sync accuracy test passes (±2 frames)
- [ ] Expression preset test shows visible differences
- [ ] Audio-video length mismatch handled gracefully

**Documentation**:
- [ ] Docstrings for load_liveportrait(), animate()
- [ ] vortex/models/README.md updated with LivePortrait usage
- [ ] VRAM budget documented (3.5GB target)
- [ ] TensorRT setup instructions (optional but recommended)
- [ ] Expression preset definitions documented

**Performance**:
- [ ] P99 generation latency <8s for 45s video
- [ ] VRAM usage stable at ~3.5GB across 100 generations
- [ ] No memory leaks (VRAM delta <50MB after 100 calls)
- [ ] Lip-sync accuracy >95% (±2 frames) on test dataset

**Definition of Done:**
Task is complete when ALL acceptance criteria met, ALL validations pass, and LivePortrait integration produces lip-synced videos within 3.5GB VRAM budget and <8s P99 latency on RTX 3060.
