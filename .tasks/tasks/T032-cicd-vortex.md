---
id: T032
title: CI/CD Pipeline for Vortex AI Engine
status: pending
priority: 2
agent: infrastructure
dependencies: [T010]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [devops, cicd, vortex, ai, testing, phase1]

context_refs:
  - context/project.md
  - context/architecture.md

docs_refs:
  - PRD Section 16 (CI/CD - Vortex Testing)

est_tokens: 7000
actual_tokens: null
---

## Description

Implement CI/CD pipeline for Vortex AI engine using GitHub Actions with self-hosted GPU runner. Pipeline includes Python 3.11 setup, pytest with coverage, model loading tests (VRAM <12GB), generation benchmarks (5 slots in <15s each), and CLIP verification accuracy tests.

**Technical Approach:**
- Self-hosted runner with RTX 3060/4090 GPU
- Python 3.11, PyTorch 2.1+ with CUDA support
- pytest with coverage plugin
- Benchmark suite for slot generation performance
- Model checksum verification

**Integration Points:**
- Triggered on vortex/** code changes
- GPU required (cannot use standard GHA runners)
- Artifacts: Benchmark reports, generated samples

## Acceptance Criteria

- [ ] GitHub Actions workflow `.github/workflows/vortex.yml` exists
- [ ] Self-hosted GPU runner configured with label `gpu`
- [ ] Workflow triggers on vortex/** path changes
- [ ] Python dependencies install from requirements.txt
- [ ] pytest runs with >85% coverage target
- [ ] Model loading test asserts VRAM <11.8GB
- [ ] Benchmark generates 5 slots in <75s total (15s avg)
- [ ] CLIP verification accuracy >95% on test dataset
- [ ] Test failures include sample outputs for debugging

## Test Scenarios

**Test Case 1: Model Loading**
- When: Pipeline runs model_loading_test.py
- Then: All 5 models load, VRAM <11.8GB, no OOM errors

**Test Case 2: Generation Benchmark**
- When: Benchmark runs 5 slot generations
- Then: Average time <15s, P99 <18s, all CLIP scores >0.75

**Test Case 3: CLIP Ensemble Accuracy**
- When: Test 100 known image/text pairs
- Then: Dual CLIP ensemble accuracy >95%, false positive rate <2%

## Technical Implementation

**File:** `.github/workflows/vortex.yml`

```yaml
name: Vortex Engine CI

on:
  push:
    paths: ['vortex/**']
  pull_request:
    paths: ['vortex/**']

jobs:
  test-vortex:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install Dependencies
        run: pip install -r vortex/requirements.txt

      - name: Pytest Unit Tests
        run: pytest vortex/tests/unit --cov=vortex --cov-report=xml

      - name: Model Loading Test
        run: |
          python vortex/tests/test_model_loading.py
          # Asserts VRAM <11.8GB

      - name: Generation Benchmark
        timeout-minutes: 3
        run: |
          python vortex/benchmarks/slot_generation.py \
            --slots 5 \
            --max-time 15 \
            --output-report benchmarks/report.json

      - name: CLIP Accuracy Test
        run: pytest vortex/tests/test_clip_accuracy.py

      - name: Upload Artifacts
        uses: actions/upload-artifact@v3
        with:
          name: vortex-benchmarks
          path: benchmarks/
```

**File:** `vortex/tests/test_model_loading.py`

```python
import torch
from vortex.pipeline import VortexPipeline

def test_vram_usage():
    pipeline = VortexPipeline()
    vram_gb = torch.cuda.memory_allocated() / 1e9
    assert vram_gb < 11.8, f"VRAM {vram_gb:.2f} GB exceeds 11.8 GB limit"
    print(f"✅ VRAM usage: {vram_gb:.2f} GB")
```

**File:** `vortex/benchmarks/slot_generation.py`

```python
import argparse
import time
from vortex.pipeline import VortexPipeline

def benchmark(slots: int, max_time: int):
    pipeline = VortexPipeline()
    times = []

    for i in range(slots):
        start = time.monotonic()
        result = pipeline.generate_slot(test_recipe())
        elapsed = time.monotonic() - start
        times.append(elapsed)
        assert elapsed < max_time, f"Slot {i} took {elapsed:.2f}s > {max_time}s"

    avg = sum(times) / len(times)
    print(f"Average: {avg:.2f}s, P99: {max(times):.2f}s")
```

## Dependencies

**Hard Dependencies:**
- [T010] Vortex Engine Implementation

**External Dependencies:**
- Self-hosted GitHub Actions runner with GPU
- NVIDIA drivers 535+, CUDA 12.1+

## Design Decisions

**Decision 1: Self-Hosted Runner vs. Cloud GPU**
- **Rationale:** GitHub Actions doesn't provide GPU runners; self-hosted is only option
- **Trade-offs:** (+) Full control. (-) Maintenance overhead, uptime dependency

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| GPU runner offline | High | Medium | Monitor with alerts, backup runner |
| CUDA version mismatch | Medium | Low | Pin CUDA version in runner setup |

## Progress Log

### [2025-12-24] - Task Created
**Created By:** task-creator agent
**Dependencies:** T010
**Estimated Complexity:** Standard

## Completion Checklist

- [ ] vortex.yml workflow created
- [ ] Self-hosted GPU runner configured
- [ ] Model loading test passes
- [ ] Benchmark suite runs in <3 minutes
- [ ] Coverage report generated

**Definition of Done:**
Vortex CI/CD pipeline runs on every vortex/** code change, verifies model loading, runs benchmarks, and ensures generation performance meets <15s/slot target.
