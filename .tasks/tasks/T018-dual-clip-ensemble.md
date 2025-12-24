---
id: T018
title: Dual CLIP Ensemble - Semantic Verification with Self-Check
status: pending
priority: 1
agent: ai-ml
dependencies: [T014, T015, T016]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [vortex, ai-ml, python, gpu, clip, verification, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - prd.md#section-12.2-dual-clip-ensemble
  - prd.md#section-8.0.1-enhancements
  - architecture.md#section-3.4-clip-adversarial-hardening

est_tokens: 14000
actual_tokens: null
---

## Description

Implement dual CLIP model ensemble (ViT-B-32 + ViT-L-14) for semantic verification of generated video content. This system provides both director self-checking (before BFT submission) and validator verification (during consensus).

**Critical v8.0.1 Enhancement**: Directors run CLIP self-verification before broadcasting to BFT. This reduces off-chain disputes by ~40% by catching borderline content early.

**Critical Requirements**:
- VRAM budget: 0.9 GB total (0.3GB for ViT-B-32, 0.6GB for ViT-L-14, both INT8 quantized)
- Models: CLIP-ViT-B-32 (weight 0.4) + CLIP-ViT-L-14 (weight 0.6)
- Input: Video frames (5 keyframes sampled from 1080 frames) + text prompt
- Output: DualClipResult with ensemble_score, individual scores, self_check_passed flag
- Thresholds: score_b ≥0.70, score_l ≥0.72 for self-check pass
- Latency target: <1s P99 for 5-frame verification on RTX 3060

**Integration Points**:
- Loaded by VortexPipeline._load_clip_b() and _load_clip_l() at startup
- Called by VortexPipeline._compute_clip_embedding() after video generation
- Used for director self-verification AND validator off-chain verification

## Business Context

**User Story**: As a Director node, I want to verify my generated video semantically matches the Recipe prompt before submitting to BFT, so that I avoid consensus failures and reputation penalties (200 ICN) from borderline content.

**Why This Matters**:
- Dual CLIP ensemble reduces false positives/negatives by 40% vs single model
- Self-check prevents wasted BFT rounds on low-quality content
- CLIP embeddings are used for BFT consensus (3-of-5 agreement on canonical_hash)
- Validator nodes use CLIP to verify director outputs and cast attestations
- Failed CLIP checks result in 200 ICN reputation penalty (DirectorSlotRejected)

**What It Unblocks**:
- Off-chain BFT coordination (directors exchange CLIP embeddings)
- Validator attestation logic (verify director outputs)
- Content policy enforcement (banned concepts detection)
- T020 (Slot timing orchestration - CLIP is final pipeline stage)

**Priority Justification**: Priority 1 (Critical Path) - Without CLIP verification, no BFT consensus can occur, making the entire ICN protocol non-functional. This is the trust anchor for content quality.

## Acceptance Criteria

- [ ] CLIP-ViT-B-32 model loads with INT8 quantization at VortexPipeline initialization
- [ ] CLIP-ViT-L-14 model loads with INT8 quantization at VortexPipeline initialization
- [ ] Combined VRAM usage for both CLIP models is 0.8-1.0 GB
- [ ] verify() method accepts: video_frames (T×C×H×W), prompt, threshold (default 0.75)
- [ ] Output is DualClipResult dataclass with: embedding, score_clip_b, score_clip_l, ensemble_score, self_check_passed
- [ ] Keyframe sampling: 5 evenly spaced frames from video (reduces compute by 216×)
- [ ] Ensemble scoring: weighted average (0.4 × score_b + 0.6 × score_l)
- [ ] Self-check thresholds: score_b ≥0.70 AND score_l ≥0.72 for pass
- [ ] Embedding normalization: L2 norm = 1.0 (for cosine similarity in BFT)
- [ ] Verification time is <1s P99 for 5 frames on RTX 3060
- [ ] Outlier detection: flag if |score_b - score_l| >0.15 (adversarial indicator)
- [ ] Error handling for: invalid frames, empty prompts, CUDA OOM
- [ ] Model determinism: same frames + prompt = identical embedding

## Test Scenarios

**Test Case 1: Standard Semantic Verification**
- Given: VortexPipeline with both CLIP models loaded
  And video_frames from LivePortrait (1080×3×512×512)
  And prompt "manic scientist, blue spiked hair, white lab coat"
- When: verify(video_frames, prompt, threshold=0.75)
- Then: 5 keyframes are sampled (indices 0, 270, 540, 810, 1079)
  And score_clip_b is computed (e.g., 0.82)
  And score_clip_l is computed (e.g., 0.85)
  And ensemble_score = 0.82×0.4 + 0.85×0.6 = 0.838
  And self_check_passed = True (both scores above thresholds)
  And embedding is 512-dim L2-normalized tensor

**Test Case 2: Self-Check Rejection**
- Given: CLIP models loaded
  And poor-quality video (e.g., blurry, off-prompt)
- When: verify(poor_video, prompt)
- Then: score_clip_b <0.70 OR score_clip_l <0.72
  And self_check_passed = False
  And warning logged: "Self-check failed: score_b=0.65, score_l=0.68"
  And Director should regenerate or skip slot

**Test Case 3: Ensemble vs. Single Model**
- Given: Same video + prompt
- When: verify() with dual ensemble vs. single ViT-B-32
- Then: Ensemble score variance is lower across 100 test videos
  And edge cases (adversarial, borderline) show 40% fewer false positives/negatives
  And ensemble is more robust to prompt variations

**Test Case 4: VRAM Budget Compliance**
- Given: Fresh Python process with both CLIP models loaded
- When: Model initialization completes
- Then: torch.cuda.memory_allocated() shows 0.8-1.0 GB total
  And ViT-B-32 uses ~0.3GB (INT8)
  And ViT-L-14 uses ~0.6GB (INT8)

**Test Case 5: Keyframe Sampling Efficiency**
- Given: 1080-frame video (45s @ 24fps)
- When: verify() is called
- Then: Only 5 frames are encoded (indices 0, 270, 540, 810, 1079)
  And verification time is <1s (vs ~10s for all 1080 frames)
  And quality is comparable to full-frame verification (CLIP score delta <0.02)

**Test Case 6: Outlier Detection (Adversarial Indicator)**
- Given: Adversarial video (e.g., prompt injection, subtle manipulation)
- When: verify() is called
- Then: |score_clip_b - score_clip_l| >0.15 (e.g., 0.45 vs 0.75)
  And outlier flag is set in DualClipResult
  And warning logged: "Score divergence detected (potential adversarial): Δ=0.30"
  And Director/Validator should escalate for manual review

**Test Case 7: Deterministic Embedding**
- Given: CLIP models with torch.manual_seed(42)
- When: verify(frames, prompt, seed=42) called twice
- Then: Both embeddings are identical (torch.equal() returns True)
  And scores are identical

## Technical Implementation

**Required Components**:

1. **vortex/models/clip_ensemble.py** (Dual model wrapper)
   - `load_clip_ensemble(device: str) -> ClipEnsemble`
   - `ClipEnsemble.verify(video_frames, prompt, threshold, seed)`
   - `DualClipResult` dataclass
   - Keyframe sampling logic
   - Outlier detection

2. **vortex/models/configs/clip_int8.yaml** (Quantization config)
   - INT8 quantization for both models
   - `ViT-B-32: {precision: "int8", cache: "~/.cache/vortex/clip_b32_int8.pt"}`
   - `ViT-L-14: {precision: "int8", cache: "~/.cache/vortex/clip_l14_int8.pt"}`

3. **vortex/utils/clip_utils.py** (Utilities)
   - `sample_keyframes(video: Tensor, num_frames: int) -> Tensor`
   - `normalize_embedding(emb: Tensor) -> Tensor` (L2 norm)
   - `compute_cosine_similarity(emb1, emb2) -> float`

4. **vortex/tests/integration/test_clip_ensemble.py** (Integration tests)
   - Test cases 1-7 from above
   - Adversarial robustness tests
   - CLIP score regression tests (vs reference videos)

5. **vortex/benchmarks/clip_latency.py** (Performance benchmark)
   - Measure verification time over 100 iterations
   - Compare 5-frame vs full-video verification
   - Profile encoding bottlenecks

**Validation Commands**:
```bash
# Install CLIP dependencies
pip install open-clip-torch==2.23.0 transformers==4.36.0

# Download and quantize models (one-time)
python vortex/scripts/download_and_quantize_clip.py

# Unit tests (mocked CLIP)
pytest vortex/tests/unit/test_clip_ensemble.py -v

# Integration test (real models, requires GPU)
pytest vortex/tests/integration/test_clip_ensemble.py --gpu -v

# VRAM profiling
python vortex/benchmarks/clip_vram_profile.py

# Latency benchmark
python vortex/benchmarks/clip_latency.py --iterations 100

# Adversarial robustness test
python vortex/tests/adversarial/clip_robustness.py

# Visual verification (manual check)
python vortex/scripts/clip_visual_check.py --video /tmp/test.mp4 --prompt "scientist"
```

**Code Patterns**:
```python
# From vortex/models/clip_ensemble.py
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DualClipResult:
    """Result from dual CLIP self-verification."""
    embedding: torch.Tensor      # Combined embedding for BFT (512-dim, L2-normalized)
    score_clip_b: float          # CLIP-ViT-B-32 score
    score_clip_l: float          # CLIP-ViT-L-14 score
    ensemble_score: float        # Weighted average (0.4×B + 0.6×L)
    self_check_passed: bool      # Whether content passes self-check
    outlier_detected: bool       # Whether scores diverge >0.15 (adversarial indicator)

class ClipEnsemble:
    def __init__(self, clip_b, clip_l, device: str = "cuda"):
        self.clip_b = clip_b.to(device)  # INT8
        self.clip_l = clip_l.to(device)  # INT8
        self.device = device

        # Ensemble weights (from PRD)
        self.weight_b = 0.4
        self.weight_l = 0.6

        # Self-check thresholds (v8.0.1)
        self.threshold_b = 0.70
        self.threshold_l = 0.72

    @torch.no_grad()
    def verify(
        self,
        video_frames: torch.Tensor,  # [T, C, H, W]
        prompt: str,
        threshold: float = 0.75,
        seed: int = None
    ) -> DualClipResult:
        """Verify video semantically matches prompt using dual CLIP ensemble."""
        if seed is not None:
            torch.manual_seed(seed)

        # Sample 5 keyframes for efficiency (216× speedup)
        keyframes = self._sample_keyframes(video_frames, num_frames=5)

        # CLIP-ViT-B-32 (primary, fast)
        image_features_b = self.clip_b.encode_image(keyframes)
        text_features_b = self.clip_b.encode_text(prompt)
        score_b = F.cosine_similarity(
            image_features_b.mean(dim=0),
            text_features_b
        ).item()

        # CLIP-ViT-L-14 (secondary, higher quality)
        image_features_l = self.clip_l.encode_image(keyframes)
        text_features_l = self.clip_l.encode_text(prompt)
        score_l = F.cosine_similarity(
            image_features_l.mean(dim=0),
            text_features_l
        ).item()

        # Ensemble score (weighted average)
        ensemble_score = score_b * self.weight_b + score_l * self.weight_l

        # Self-check: reject if either model scores below threshold
        self_check_passed = score_b >= self.threshold_b and score_l >= self.threshold_l

        # Outlier detection (adversarial indicator)
        outlier_detected = abs(score_b - score_l) > 0.15
        if outlier_detected:
            logger.warning(f"Score divergence detected (potential adversarial): "
                         f"score_b={score_b:.3f}, score_l={score_l:.3f}, Δ={abs(score_b-score_l):.3f}")

        # Combined embedding for BFT exchange
        combined_b = (image_features_b.mean(dim=0) + text_features_b) / 2
        combined_l = (image_features_l.mean(dim=0) + text_features_l) / 2

        # Weighted combination for final embedding
        final_embedding = combined_b * self.weight_b + combined_l * self.weight_l
        final_embedding = final_embedding / final_embedding.norm()  # L2 normalize

        return DualClipResult(
            embedding=final_embedding,
            score_clip_b=score_b,
            score_clip_l=score_l,
            ensemble_score=ensemble_score,
            self_check_passed=self_check_passed,
            outlier_detected=outlier_detected
        )

    def _sample_keyframes(self, video: torch.Tensor, num_frames: int = 5) -> torch.Tensor:
        """Sample evenly spaced keyframes from video."""
        T = video.shape[0]
        indices = torch.linspace(0, T-1, num_frames).long()
        return video[indices]

def load_clip_ensemble(device: str = "cuda") -> ClipEnsemble:
    """Load dual CLIP models with INT8 quantization."""
    import open_clip

    # Load ViT-B-32 (INT8)
    clip_b, _, preprocess_b = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='openai',
        precision='int8',
        device=device
    )

    # Load ViT-L-14 (INT8)
    clip_l, _, preprocess_l = open_clip.create_model_and_transforms(
        'ViT-L-14',
        pretrained='openai',
        precision='int8',
        device=device
    )

    return ClipEnsemble(clip_b, clip_l, device)
```

## Dependencies

**Hard Dependencies** (must be complete first):
- [T014] Vortex Core Pipeline - provides ModelRegistry, VRAMMonitor
- [T015] Flux-Schnell Integration - generates images to verify
- [T016] LivePortrait Integration - generates videos to verify

**Soft Dependencies** (nice to have):
- None

**External Dependencies**:
- Python 3.11
- PyTorch 2.1+ with CUDA 12.1+
- open-clip-torch 2.23.0+ (OpenAI CLIP models)
- transformers 4.36.0+ (for tokenization)
- Pillow 10.1.0+ (for image preprocessing)

## Design Decisions

**Decision 1: Dual Ensemble (ViT-B-32 + ViT-L-14) vs. Single Model**
- **Rationale**: Single CLIP model has ~5-10% false positive/negative rate. Dual ensemble with different architectures reduces this by 40% (from PRD §12.2).
- **Alternatives**:
  - Single ViT-B-32 (rejected: lower accuracy)
  - Single ViT-L-14 (rejected: only 0.6GB VRAM savings, loses diversity)
  - Triple ensemble with RN50 (future: +0.4GB VRAM, +50% latency)
- **Trade-offs**: (+) 40% fewer disputes, robust to adversarial. (-) +0.9GB VRAM, +50% verification time.

**Decision 2: INT8 Quantization for Both Models**
- **Rationale**: FP16 CLIP uses ~1.5GB total. INT8 reduces to ~0.9GB with <3% accuracy loss (acceptable for verification).
- **Alternatives**:
  - FP16 (rejected: 0.6GB more VRAM)
  - FP32 (rejected: 1.2GB more VRAM)
- **Trade-offs**: (+) Fits VRAM budget. (-) Slight accuracy loss (~2-3% CLIP score drop).

**Decision 3: 5 Keyframe Sampling vs. Full Video**
- **Rationale**: Full 1080-frame encoding takes ~10s. 5 keyframes (216× reduction) takes <1s with <2% accuracy difference.
- **Alternatives**:
  - All 1080 frames (rejected: 10× slower, minimal quality gain)
  - 1 keyframe (rejected: unstable, misses temporal consistency)
  - 10 keyframes (considered: only 2× slower than 5, but marginal accuracy gain)
- **Trade-offs**: (+) Fast, acceptable quality. (-) Misses some temporal details.

**Decision 4: Weighted Ensemble (0.4×B + 0.6×L) vs. Simple Average**
- **Rationale**: ViT-L-14 is higher quality (larger model). Weighting it more (0.6) improves overall accuracy by ~3% vs. equal weighting.
- **Alternatives**:
  - Equal weighting (0.5×B + 0.5×L) (rejected: slightly lower accuracy)
  - Adaptive weighting (rejected: adds complexity)
- **Trade-offs**: (+) Better accuracy. (-) Slightly more biased toward L-14 errors (mitigated by dual threshold check).

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Adversarial prompt injection | High (bypass verification) | Medium (user-provided prompts) | Outlier detection (score divergence >0.15), sanitize prompts, log suspicious cases |
| INT8 quantization quality loss | Medium (false rejections) | Low (tested <3% drop) | A/B test with FP16 baseline, monitor false positive rate, fallback to FP16 if needed |
| Keyframe sampling misses critical frames | Medium (inaccurate verification) | Low (evenly spaced) | Validate sampling on diverse videos, consider saliency-based sampling (future) |
| VRAM usage exceeds 1.0GB | Low (within budget) | Low (well-tested) | Monitor VRAM per verification, log spikes |
| CLIP model download timeout | Medium (startup fails) | Medium (network issues) | Cache models locally, provide offline fallback, retry with exponential backoff |

## Progress Log

### [2025-12-24T00:00:00Z] - Task Created

**Created By:** task-creator agent
**Reason:** User request to create comprehensive Vortex Engine tasks per PRD sections 12.1-12.3
**Dependencies:** T014 (Core Pipeline), T015 (Flux), T016 (LivePortrait)
**Estimated Complexity:** Standard (14,000 tokens estimated)

**Notes**: Dual CLIP ensemble is a v8.0.1 enhancement that reduces off-chain disputes by 40%. Self-check thresholds (0.70 for B-32, 0.72 for L-14) prevent wasted BFT rounds. This is the trust anchor for all content quality verification.

## Completion Checklist

**Code Complete**:
- [ ] vortex/models/clip_ensemble.py implemented with load_clip_ensemble(), ClipEnsemble.verify()
- [ ] DualClipResult dataclass with all fields
- [ ] INT8 quantization for both models
- [ ] Keyframe sampling logic (5 frames)
- [ ] Ensemble scoring (weighted average)
- [ ] Self-check thresholds (0.70, 0.72)
- [ ] Outlier detection (score divergence >0.15)
- [ ] Integration with VortexPipeline._compute_clip_embedding()

**Testing**:
- [ ] Unit tests pass (mocked CLIP models)
- [ ] Integration test verifies video semantics
- [ ] VRAM profiling shows 0.8-1.0GB usage
- [ ] Latency benchmark P99 <1s on RTX 3060
- [ ] Self-check rejection test passes
- [ ] Ensemble vs. single model comparison shows 40% improvement
- [ ] Keyframe sampling efficiency verified (216× speedup)
- [ ] Outlier detection test flags adversarial examples
- [ ] Deterministic embedding test passes

**Documentation**:
- [ ] Docstrings for load_clip_ensemble(), verify()
- [ ] vortex/models/README.md updated with CLIP usage
- [ ] VRAM budget documented (0.9GB target)
- [ ] Ensemble weighting rationale explained
- [ ] Self-check thresholds documented
- [ ] Outlier detection threshold (0.15) justified

**Performance**:
- [ ] P99 verification latency <1s for 5 frames
- [ ] VRAM usage stable at ~0.9GB across 100 verifications
- [ ] No memory leaks (VRAM delta <20MB after 100 calls)
- [ ] Accuracy improvement: 40% fewer false positives/negatives vs. single model

**Definition of Done:**
Task is complete when ALL acceptance criteria met, ALL validations pass, and dual CLIP ensemble provides robust semantic verification within 0.9GB VRAM budget and <1s P99 latency on RTX 3060, with 40% dispute reduction vs. single model.
