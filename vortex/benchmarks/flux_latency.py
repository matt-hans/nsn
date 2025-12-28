#!/usr/bin/env python
"""Latency benchmark for Flux-Schnell generation.

Measures generation time over multiple iterations to compute:
- P50 (median) latency
- P99 latency (target: <12s)
- P99.9 latency
- Mean and standard deviation

Usage:
    python vortex/benchmarks/flux_latency.py --iterations 50

Requirements:
    - CUDA GPU (RTX 3060 12GB or better)
    - Flux-Schnell model weights cached locally
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add vortex to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vortex.models.flux import load_flux_schnell


def compute_percentiles(latencies: list[float]) -> dict:
    """Compute latency percentiles."""
    arr = np.array(latencies)
    return {
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99),
        "p99.9": np.percentile(arr, 99.9),
        "mean": np.mean(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
    }


def plot_histogram(latencies: list[float], output_path: str = None):
    """Plot latency distribution histogram."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.hist(latencies, bins=30, edgecolor="black", alpha=0.7)
        plt.axvline(12.0, color="red", linestyle="--", label="P99 Target (12s)")
        plt.xlabel("Generation Time (seconds)")
        plt.ylabel("Frequency")
        plt.title("Flux-Schnell Generation Latency Distribution")
        plt.legend()
        plt.grid(alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"\n📊 Histogram saved to: {output_path}")
        else:
            plt.show()

    except ImportError:
        print("\n⚠️  matplotlib not installed, skipping histogram plot")


def main():
    """Run latency benchmark."""
    parser = argparse.ArgumentParser(description="Flux-Schnell latency benchmark")
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of generation iterations (default: 50)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot latency distribution histogram",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for histogram (default: show plot)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for latency benchmark")
        sys.exit(1)

    device = "cuda:0"
    print("=" * 80)
    print("Flux-Schnell Latency Benchmark")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Iterations: {args.iterations}")
    print("Target P99: <12.0 seconds")
    print("=" * 80)

    # Load model
    print("\n⏳ Loading Flux-Schnell model (30-60 seconds)...")
    flux_model = load_flux_schnell(device=device, quantization="nf4")
    print("✅ Model loaded\n")

    # Warmup (first generation is always slower)
    print("🔥 Warmup run...")
    warmup_start = time.time()
    _ = flux_model.generate(
        prompt="warmup",
        num_inference_steps=4,
        guidance_scale=0.0,
    )
    warmup_time = time.time() - warmup_start
    print(f"   Warmup time: {warmup_time:.2f}s\n")

    # Benchmark iterations
    latencies = []
    prompts = [
        "a scientist in a laboratory",
        "manic scientist, blue spiked hair, white lab coat",
        "rick sanchez from rick and morty",
        "cartoon scientist character",
        "scientist with wild hair in lab",
    ]

    print(f"🏃 Running {args.iterations} benchmark iterations...\n")

    for i in range(args.iterations):
        # Rotate prompts
        prompt = prompts[i % len(prompts)]

        # Time generation
        start_time = time.time()
        _ = flux_model.generate(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
        )
        end_time = time.time()

        latency = end_time - start_time
        latencies.append(latency)

        # Progress indicator
        if (i + 1) % 10 == 0:
            recent_mean = np.mean(latencies[-10:])
            print(f"   [{i+1:3d}/{args.iterations}] Recent avg: {recent_mean:.2f}s")

    # Compute statistics
    stats = compute_percentiles(latencies)

    print("\n" + "=" * 80)
    print("LATENCY STATISTICS")
    print("=" * 80)

    print(f"\n{'Percentile':<15} {'Latency (s)':>15} {'vs Target':>15}")
    print("-" * 80)
    print(f"{'P50 (median)':<15} {stats['p50']:>15.3f} {'-':>15}")
    print(f"{'P90':<15} {stats['p90']:>15.3f} {'-':>15}")
    print(f"{'P95':<15} {stats['p95']:>15.3f} {'-':>15}")

    p99_status = "✅ PASS" if stats['p99'] < 12.0 else "❌ FAIL"
    print(f"{'P99 (TARGET)':<15} {stats['p99']:>15.3f} {p99_status:>15}")

    print(f"{'P99.9':<15} {stats['p99.9']:>15.3f} {'-':>15}")

    print(f"\n{'Mean':<15} {stats['mean']:>15.3f}")
    print(f"{'Std Dev':<15} {stats['std']:>15.3f}")
    print(f"{'Min':<15} {stats['min']:>15.3f}")
    print(f"{'Max':<15} {stats['max']:>15.3f}")

    print("\n" + "=" * 80)

    # Outlier detection
    outlier_threshold = stats["mean"] + 3 * stats["std"]
    outliers = [lat for lat in latencies if lat > outlier_threshold]

    if outliers:
        print(f"\n⚠️  {len(outliers)} outliers detected (>{outlier_threshold:.2f}s):")
        for lat in outliers:
            print(f"    {lat:.3f}s")

    # Final verdict
    if stats["p99"] < 12.0:
        print("\n✅ VERDICT: P99 latency PASSED (<12s target)")
        verdict = 0
    else:
        print(f"\n❌ VERDICT: P99 latency FAILED ({stats['p99']:.2f}s > 12s target)")
        verdict = 1

    # Plot histogram
    if args.plot:
        plot_histogram(latencies, output_path=args.output)

    sys.exit(verdict)


if __name__ == "__main__":
    main()
