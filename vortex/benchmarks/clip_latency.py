#!/usr/bin/env python3
"""Benchmark CLIP ensemble verification latency.

Measures P50, P95, P99 latency for dual CLIP ensemble verification
with 5-frame keyframe sampling vs full video encoding.

Usage:
    python benchmarks/clip_latency.py --iterations 100 --device cuda

Target: <1s P99 for 5-frame verification on RTX 3060
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_clip_latency(
    iterations: int = 100,
    device: str = "cuda",
    num_frames: int = 5,
) -> dict:
    """Benchmark CLIP ensemble verification latency.

    Args:
        iterations: Number of benchmark iterations
        device: Target device ("cuda" or "cpu")
        num_frames: Number of keyframes to sample

    Returns:
        Dictionary with latency statistics
    """
    try:
        from vortex.models.clip_ensemble import load_clip_ensemble
    except ImportError as e:
        logger.error(f"Failed to import CLIP ensemble: {e}")
        sys.exit(1)

    # Load ensemble
    logger.info(f"Loading CLIP ensemble on {device}...")
    ensemble = load_clip_ensemble(device=device)

    # Generate synthetic video (1080 frames @ 512x512)
    logger.info("Generating synthetic video (1080 frames @ 512x512)...")
    video = torch.randn(1080, 3, 512, 512, dtype=torch.float32)

    prompt = "a scientist with blue hair wearing a white lab coat"

    # Warmup runs (5 iterations)
    logger.info("Warmup (5 iterations)...")
    for i in range(5):
        _ = ensemble.verify(video, prompt, seed=42 + i)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    logger.info(f"Benchmarking ({iterations} iterations)...")
    latencies = []

    for i in range(iterations):
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = ensemble.verify(video, prompt, seed=42 + i)

        if device == "cuda":
            torch.cuda.synchronize()

        latency = time.perf_counter() - start
        latencies.append(latency)

        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{iterations} iterations")

    # Compute statistics
    latencies = np.array(latencies) * 1000  # Convert to ms

    stats = {
        "mean_ms": np.mean(latencies),
        "median_ms": np.median(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "iterations": iterations,
        "device": device,
        "num_frames": num_frames,
    }

    # Print results
    logger.info("=" * 60)
    logger.info("CLIP Ensemble Latency Benchmark Results")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Iterations: {iterations}")
    logger.info(f"Keyframes sampled: {num_frames}")
    logger.info(f"Video size: 1080 frames @ 512x512")
    logger.info("-" * 60)
    logger.info(f"Mean:   {stats['mean_ms']:7.2f} ms")
    logger.info(f"Median: {stats['median_ms']:7.2f} ms")
    logger.info(f"Std:    {stats['std_ms']:7.2f} ms")
    logger.info(f"Min:    {stats['min_ms']:7.2f} ms")
    logger.info(f"Max:    {stats['max_ms']:7.2f} ms")
    logger.info("-" * 60)
    logger.info(f"P50:    {stats['p50_ms']:7.2f} ms")
    logger.info(f"P95:    {stats['p95_ms']:7.2f} ms")
    logger.info(f"P99:    {stats['p99_ms']:7.2f} ms")
    logger.info("=" * 60)

    # Check against target
    target_p99_ms = 1000  # 1 second
    if stats['p99_ms'] < target_p99_ms:
        logger.info(f"✓ P99 latency {stats['p99_ms']:.2f}ms meets target <{target_p99_ms}ms")
    else:
        logger.warning(f"✗ P99 latency {stats['p99_ms']:.2f}ms exceeds target {target_p99_ms}ms")

    # VRAM usage
    if device == "cuda":
        vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(f"Peak VRAM usage: {vram_gb:.2f} GB")

        if vram_gb <= 1.0:
            logger.info(f"✓ VRAM usage {vram_gb:.2f}GB within budget (≤1.0GB)")
        else:
            logger.warning(f"✗ VRAM usage {vram_gb:.2f}GB exceeds budget 1.0GB")

    return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark CLIP ensemble verification latency"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Target device (default: cuda if available)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Number of keyframes to sample (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results to JSON file (optional)"
    )

    args = parser.parse_args()

    try:
        stats = benchmark_clip_latency(
            iterations=args.iterations,
            device=args.device,
            num_frames=args.num_frames,
        )

        # Save results if output path provided
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
