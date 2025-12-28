#!/usr/bin/env python
"""VRAM profiling for Flux-Schnell model.

Measures VRAM usage at different stages:
1. Baseline (CUDA initialized)
2. After model loading
3. During generation (peak)
4. After generation (cleanup)

Usage:
    python vortex/benchmarks/flux_vram_profile.py

Expected Output:
    - Flux model VRAM: 5.5-6.5 GB
    - Generation overhead: <500 MB
    - Total VRAM with buffers: <7.0 GB
"""

import sys
from pathlib import Path

import torch

# Add vortex to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vortex.models.flux import load_flux_schnell


def format_vram(bytes_value: int) -> str:
    """Format bytes as GB."""
    return f"{bytes_value / 1e9:.3f} GB"


def main():
    """Run VRAM profiling."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for VRAM profiling")
        sys.exit(1)

    device = "cuda:0"
    print("=" * 80)
    print("Flux-Schnell VRAM Profiling")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    print("=" * 80)

    # Stage 1: Baseline (CUDA initialized)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)

    print("\n[1] BASELINE (CUDA initialized)")
    print(f"    Allocated: {format_vram(baseline_allocated)}")
    print(f"    Reserved:  {format_vram(baseline_reserved)}")

    # Stage 2: After model loading
    print("\n[2] LOADING FLUX-SCHNELL (this may take 30-60 seconds)...")

    flux_model = load_flux_schnell(device=device, quantization="nf4")

    loaded_allocated = torch.cuda.memory_allocated(device)
    loaded_reserved = torch.cuda.memory_reserved(device)
    model_vram = loaded_allocated - baseline_allocated

    print(f"    Allocated: {format_vram(loaded_allocated)}")
    print(f"    Reserved:  {format_vram(loaded_reserved)}")
    print(f"    Model VRAM: {format_vram(model_vram)} ⬅️  Target: 5.5-6.5 GB")

    # Check if within budget
    model_vram_gb = model_vram / 1e9
    if 5.5 <= model_vram_gb <= 6.5:
        print("    ✅ PASS - Model VRAM within budget")
    elif model_vram_gb < 5.5:
        print("    ⚠️  WARNING - Model VRAM suspiciously low")
    else:
        print("    ❌ FAIL - Model VRAM exceeds 6.5 GB budget")

    # Stage 3: During generation (peak)
    print("\n[3] GENERATING IMAGE...")

    prompt = "a scientist in a laboratory"
    result = flux_model.generate(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
    )

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    generation_overhead = peak_allocated - loaded_allocated

    print(f"    Peak allocated: {format_vram(peak_allocated)}")
    print(f"    Peak reserved:  {format_vram(peak_reserved)}")
    print(f"    Generation overhead: {format_vram(generation_overhead)} ⬅️  Target: <500 MB")

    # Check generation overhead
    generation_overhead_mb = generation_overhead / 1e6
    if generation_overhead_mb < 500:
        print("    ✅ PASS - Generation overhead within budget")
    else:
        print(f"    ⚠️  WARNING - Generation overhead high ({generation_overhead_mb:.1f} MB)")

    # Stage 4: After generation (cleanup)
    del result
    torch.cuda.empty_cache()

    after_gen_allocated = torch.cuda.memory_allocated(device)
    after_gen_reserved = torch.cuda.memory_reserved(device)
    cleanup_released = loaded_allocated - after_gen_allocated

    print("\n[4] AFTER GENERATION (cleanup)")
    print(f"    Allocated: {format_vram(after_gen_allocated)}")
    print(f"    Reserved:  {format_vram(after_gen_reserved)}")
    print(f"    Cleanup released: {format_vram(cleanup_released)}")

    # Summary table
    print("\n" + "=" * 80)
    print("VRAM BUDGET COMPLIANCE SUMMARY")
    print("=" * 80)

    print(f"\n{'Component':<30} {'VRAM':>12} {'Budget':>12} {'Status':>10}")
    print("-" * 80)
    print(
        f"{'Flux-Schnell Model':<30} {format_vram(model_vram):>12} "
        f"{'5.5-6.5 GB':>12} "
        f"{'✅ PASS' if 5.5 <= model_vram_gb <= 6.5 else '❌ FAIL':>10}"
    )
    print(
        f"{'Generation Overhead':<30} {format_vram(generation_overhead):>12} "
        f"{'<500 MB':>12} "
        f"{'✅ PASS' if generation_overhead_mb < 500 else '⚠️  WARN':>10}"
    )
    print(
        f"{'Total Peak VRAM':<30} {format_vram(peak_allocated):>12} "
        f"{'<7.0 GB':>12} "
        f"{'✅ PASS' if peak_allocated / 1e9 < 7.0 else '❌ FAIL':>10}"
    )

    print("\n" + "=" * 80)

    # Final verdict
    all_pass = (
        5.5 <= model_vram_gb <= 6.5
        and generation_overhead_mb < 500
        and peak_allocated / 1e9 < 7.0
    )

    if all_pass:
        print("✅ VERDICT: All VRAM budgets PASSED")
        sys.exit(0)
    else:
        print("❌ VERDICT: Some VRAM budgets FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
