---
id: T015
title: Flux-Schnell Integration - NF4 Quantized Image Generation
status: pending
priority: 1
agent: ai-ml
dependencies: [T014]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [vortex, ai-ml, python, gpu, flux, image-generation, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - prd.md#section-12.1-static-resident-vram-layout
  - prd.md#section-12.2-generation-pipeline
  - architecture.md#section-5.3-ai-ml-pipeline

est_tokens: 12000
actual_tokens: null
---

## Description

Integrate Flux-Schnell image generation model with NF4 (4-bit) quantization into the Vortex pipeline. Flux generates the initial actor image from text prompts, which LivePortrait then animates into video.

**Critical Requirements**:
- VRAM budget: 6.0 GB (must fit within 11.8GB total budget)
- Precision: NF4 (4-bit quantization via bitsandbytes)
- Inference: 4 steps (Schnell = fast variant)
- Resolution: 512×512 output
- Guidance scale: 0.0 (unconditional generation for speed)
- Latency target: <12s P99 on RTX 3060

**Integration Points**:
- Loaded by VortexPipeline._load_flux() at startup
- Called by VortexPipeline._generate_actor() during slot generation
- Outputs to pre-allocated actor_buffer (512×512×3 tensor)

## Business Context

**User Story**: As a Director node, I want to generate unique actor images from text prompts in <12 seconds, so that I can proceed to video warping and meet my 45-second slot deadline.

**Why This Matters**:
- Flux-Schnell is the first stage of the generation pipeline and sets the visual quality baseline
- NF4 quantization is essential to fit within 6GB VRAM budget (vs 24GB for FP16)
- 4-step inference is the minimum for acceptable quality (2 steps produces artifacts)
- This component directly impacts Director earnings: slow generation = slot misses = 150 ICN slashing

**What It Unblocks**:
- T016 (LivePortrait video warping - depends on actor image)
- T020 (Slot timing orchestration)
- Full end-to-end Vortex pipeline testing

**Priority Justification**: Priority 1 (Critical Path) - Blocks all video generation functionality. Without Flux, no actor images can be created, making the pipeline non-functional.

## Acceptance Criteria

- [ ] Flux-Schnell model loads with NF4 quantization via bitsandbytes at VortexPipeline initialization
- [ ] VRAM usage for Flux model is 5.5-6.5 GB (measured via torch.cuda.memory_allocated())
- [ ] generate() method accepts: prompt, negative_prompt, num_inference_steps=4, guidance_scale=0.0, output=actor_buffer
- [ ] Output is 512×512×3 float32 tensor in range [0,1] normalized
- [ ] Generation time is <12s P99 on RTX 3060 12GB
- [ ] Model supports batch_size=1 (single image per call)
- [ ] Outputs are written to pre-allocated actor_buffer (no new allocations)
- [ ] Prompt length limit is 77 tokens (standard CLIP text encoder limit)
- [ ] Negative prompts are supported for quality control ("blurry", "low quality", "watermark")
- [ ] Model determinism: same seed + prompt = identical output (for testing)
- [ ] Error handling for invalid prompts, OOM, CUDA errors
- [ ] Model weights cached locally (no re-download on restart)

## Test Scenarios

**Test Case 1: Standard Actor Generation**
- Given: VortexPipeline is initialized with Flux loaded
- When: generate(prompt="manic scientist, blue spiked hair, white lab coat", steps=4, guidance=0.0)
- Then: Output is 512×512×3 tensor
  And VRAM usage increases by <500MB during generation
  And generation completes in 8-12 seconds
  And actor_buffer contains valid image (all values in [0,1])

**Test Case 2: Negative Prompt Application**
- Given: Flux model loaded
- When: generate(prompt="scientist", negative_prompt="blurry, low quality, watermark")
- Then: Output quality is visually higher than without negative prompt
  And generation time is not significantly impacted (<5% slower)

**Test Case 3: VRAM Budget Compliance**
- Given: Fresh Python process
- When: Flux model is loaded with NF4 quantization
- Then: torch.cuda.memory_allocated() shows 5.5-6.5 GB
  And model.dtype shows bitsandbytes.nn.Linear4bit for quantized layers
  And total VRAM (all Vortex models) remains ≤11.8 GB

**Test Case 4: Deterministic Output**
- Given: Flux model with torch.manual_seed(42)
- When: generate(prompt="scientist", steps=4, seed=42) called twice
- Then: Both outputs are bit-identical (torch.equal() returns True)

**Test Case 5: Long Prompt Handling**
- Given: Prompt exceeds 77 tokens (CLIP limit)
- When: generate(prompt="very long prompt..." × 100)
- Then: Prompt is truncated to 77 tokens
  And warning is logged: "Prompt truncated to 77 tokens"
  And generation succeeds

**Test Case 6: CUDA OOM Recovery**
- Given: System VRAM artificially limited to 5GB (below 6GB requirement)
- When: Flux model load is attempted
- Then: torch.cuda.OutOfMemoryError is caught
  And structured error log includes: model_name="flux-schnell", required_vram=6.0GB, available_vram=5.0GB
  And VortexInitializationError is raised with remediation message

## Technical Implementation

**Required Components**:

1. **vortex/models/flux.py** (Model wrapper)
   - `load_flux_schnell(device: str, quantization: str = "nf4") -> FluxModel`
   - `FluxModel.generate(prompt, negative_prompt, steps, guidance, output, seed)`
   - VRAM monitoring during load and inference

2. **vortex/models/configs/flux_schnell_nf4.yaml** (Model config)
   - Model ID: `black-forest-labs/FLUX.1-schnell`
   - Quantization config: `load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16`
   - Cache directory: `~/.cache/huggingface/hub/`
   - Safety checker: disabled (for performance)

3. **vortex/tests/integration/test_flux_generation.py** (Integration tests)
   - Test cases 1-6 from above
   - Visual quality regression tests (compare against reference images)
   - VRAM profiling

4. **vortex/benchmarks/flux_latency.py** (Performance benchmark)
   - Measure generation time over 50 iterations
   - Plot latency distribution (P50, P99, P99.9)
   - Identify outliers

**Validation Commands**:
```bash
# Install Flux dependencies
pip install diffusers==0.25.0 transformers==4.36.0 bitsandbytes==0.41.3

# Download model weights (one-time, ~12GB)
python vortex/scripts/download_flux.py

# Unit tests (mocked Flux)
pytest vortex/tests/unit/test_flux.py -v

# Integration test (real Flux, requires GPU)
pytest vortex/tests/integration/test_flux_generation.py --gpu -v

# VRAM profiling
python vortex/benchmarks/flux_vram_profile.py

# Latency benchmark
python vortex/benchmarks/flux_latency.py --iterations 50

# Visual quality check (manual inspection)
python vortex/scripts/visual_check_flux.py --prompt "scientist" --output /tmp/flux_test.png
```

**Code Patterns**:
```python
# From vortex/models/flux.py
import torch
from diffusers import FluxPipeline
from transformers import BitsAndBytesConfig

def load_flux_schnell(device: str = "cuda", quantization: str = "nf4") -> FluxPipeline:
    """Load Flux-Schnell with NF4 quantization."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4"
    )

    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        use_safetensors=True
    )

    pipeline.safety_checker = None  # Disable for performance
    pipeline.to(device)

    return pipeline

class FluxModel:
    def __init__(self, pipeline, device):
        self.pipeline = pipeline
        self.device = device

    @torch.no_grad()
    def generate(self, prompt: str, negative_prompt: str = "",
                 num_inference_steps: int = 4, guidance_scale: float = 0.0,
                 output: torch.Tensor = None, seed: int = None) -> torch.Tensor:
        """Generate 512x512 actor image."""
        if seed is not None:
            torch.manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
            output_type="pt"  # PyTorch tensor
        ).images[0]

        # Write to pre-allocated buffer
        if output is not None:
            output.copy_(result)
            return output
        return result
```

## Dependencies

**Hard Dependencies** (must be complete first):
- [T014] Vortex Core Pipeline - provides ModelRegistry, VRAMMonitor, actor_buffer

**Soft Dependencies** (nice to have):
- None

**External Dependencies**:
- Python 3.11
- PyTorch 2.1+ with CUDA 12.1+
- diffusers 0.25.0+ (Hugging Face)
- transformers 4.36.0+
- bitsandbytes 0.41.3+ (for NF4 support)
- accelerate 0.25.0+ (for model loading)
- safetensors 0.4.1+ (for safe model weights)

## Design Decisions

**Decision 1: NF4 Quantization vs FP16**
- **Rationale**: FP16 Flux requires ~24GB VRAM, exceeding RTX 3060 budget. NF4 reduces to ~6GB with minimal quality loss (<5% CLIP score drop).
- **Alternatives**:
  - FP16 (rejected: too much VRAM)
  - INT8 (considered: only ~10GB VRAM, but 15% quality loss)
  - Model distillation (future: could reduce to 4GB)
- **Trade-offs**: (+) Fits budget, acceptable quality. (-) Slightly slower inference than FP16 (~20% slower).

**Decision 2: 4 Inference Steps (Schnell Fast Variant)**
- **Rationale**: Schnell is optimized for 4-step inference. 2 steps produces artifacts, 8 steps doubles time without significant quality gain.
- **Alternatives**:
  - 2 steps (rejected: visible artifacts)
  - 8 steps (rejected: 24s generation time, exceeds budget)
- **Trade-offs**: (+) Fast, good quality. (-) Not as high quality as full Flux.1-dev (50 steps).

**Decision 3: Guidance Scale 0.0 (Unconditional)**
- **Rationale**: Classifier-free guidance (scale >0) requires 2× forward passes, doubling generation time. Scale 0.0 is faster with minimal quality loss for our use case.
- **Alternatives**:
  - Guidance scale 7.5 (rejected: 24s generation)
- **Trade-offs**: (+) 2× faster. (-) Slightly less prompt adherence (~5%).

**Decision 4: Disable Safety Checker**
- **Rationale**: Safety checker adds ~500ms overhead. ICN content policy is enforced via CLIP semantic verification (T018), so redundant here.
- **Alternatives**:
  - Keep safety checker (rejected: performance cost)
- **Trade-offs**: (+) Faster. (-) No NSFW filtering at this stage (handled by CLIP later).

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| NF4 quantization quality loss | Medium (poor visual quality) | Low (tested <5% CLIP drop) | A/B test with human evaluators, fall back to INT8 if needed |
| VRAM usage exceeds 6.5GB | High (OOM crashes) | Low (well-tested) | Monitor VRAM per generation, log outliers, restart if trend upward |
| Hugging Face model download timeout | Medium (startup fails) | Medium (network issues) | Cache model weights locally, provide offline fallback, retry with exponential backoff |
| bitsandbytes driver incompatibility | High (crashes) | Low (pinned versions) | CI matrix tests CUDA 12.1, 12.2, 12.3, document minimum driver (535+) |
| Prompt injection attacks | Low (no security impact) | Medium (user-provided prompts) | Sanitize prompts (remove special characters), limit length to 77 tokens |

## Progress Log

### [2025-12-24T00:00:00Z] - Task Created

**Created By:** task-creator agent
**Reason:** User request to create comprehensive Vortex Engine tasks per PRD sections 12.1-12.3
**Dependencies:** T014 (Vortex Core Pipeline)
**Estimated Complexity:** Standard (12,000 tokens estimated)

**Notes**: Flux-Schnell is the image generation backbone of ICN. NF4 quantization is critical to meet VRAM budget. Integration with VortexPipeline via _load_flux() and _generate_actor().

## Completion Checklist

**Code Complete**:
- [ ] vortex/models/flux.py implemented with load_flux_schnell(), FluxModel.generate()
- [ ] NF4 quantization config verified (bitsandbytes)
- [ ] Model weights cached locally (~/.cache/huggingface/)
- [ ] Integration with VortexPipeline._load_flux()
- [ ] Output to pre-allocated actor_buffer

**Testing**:
- [ ] Unit tests pass (mocked Flux pipeline)
- [ ] Integration test generates valid 512×512 image
- [ ] VRAM profiling shows 5.5-6.5GB usage
- [ ] Latency benchmark P99 <12s on RTX 3060
- [ ] Deterministic output test passes (same seed = same image)
- [ ] Long prompt truncation test passes

**Documentation**:
- [ ] Docstrings for load_flux_schnell(), FluxModel.generate()
- [ ] vortex/models/README.md updated with Flux usage
- [ ] VRAM budget documented (6.0GB target)
- [ ] Quantization config explained in comments

**Performance**:
- [ ] P99 generation latency <12s
- [ ] VRAM usage stable at ~6GB across 100 generations
- [ ] No memory leaks (VRAM delta <50MB after 100 calls)
- [ ] Visual quality acceptable (CLIP score >0.70 vs reference)

**Definition of Done:**
Task is complete when ALL acceptance criteria met, ALL validations pass, and Flux integration succeeds within 6GB VRAM budget with <12s P99 latency on RTX 3060.
