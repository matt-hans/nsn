#!/usr/bin/env python3
"""Benchmark LivePortrait video generation latency.

This script measures generation time for 45-second videos over multiple
iterations to determine P50, P95, and P99 latency percentiles.

Usage:
    python benchmarks/liveportrait_latency.py --iterations 50
    python benchmarks/liveportrait_latency.py --iterations 100 --warmup 5
    python benchmarks/liveportrait_latency.py --output results.json

Target:
    P99 < 8.0s on RTX 3060 for 45s @ 24fps (1080 frames)
"""

import argparse
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Dict, List

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark LivePortrait video generation latency"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=45,
        help="Video duration in seconds (default: 45)",
    )
    parser.add_argument(
        "--fps", type=int, default=24, help="Video frame rate (default: 24)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Target device (default: cuda:0)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Model precision (default: fp16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (optional)",
    )
    parser.add_argument(
        "--profile-vram",
        action="store_true",
        help="Profile VRAM usage during generation",
    )
    return parser.parse_args()


def load_model(device: str, precision: str):
    """Load LivePortrait model.

    Args:
        device: Target device
        precision: Model precision

    Returns:
        Loaded LivePortraitModel
    """
    from vortex.models.liveportrait import load_liveportrait

    logger.info(f"Loading LivePortrait model (device={device}, precision={precision})")
    model = load_liveportrait(device=device, precision=precision)
    logger.info("Model loaded successfully")
    return model


def generate_sample_inputs(duration: int, device: str) -> tuple:
    """Generate sample actor image and audio.

    Args:
        duration: Audio duration in seconds
        device: Target device

    Returns:
        Tuple of (actor_image, driving_audio)
    """
    actor_image = torch.rand(3, 512, 512, device=device)
    driving_audio = torch.randn(int(duration * 24000), device=device)  # 24kHz
    return actor_image, driving_audio


def benchmark_generation(
    model,
    actor_image: torch.Tensor,
    driving_audio: torch.Tensor,
    fps: int,
    duration: int,
    iterations: int,
    warmup: int = 3,
    profile_vram: bool = False,
) -> Dict[str, float]:
    """Benchmark video generation latency.

    Args:
        model: LivePortraitModel instance
        actor_image: Source image
        driving_audio: Driving audio
        fps: Frame rate
        duration: Video duration
        iterations: Number of iterations
        warmup: Number of warmup iterations
        profile_vram: Profile VRAM usage

    Returns:
        Dict with latency statistics
    """
    logger.info(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        model.animate(
            source_image=actor_image,
            driving_audio=driving_audio,
            expression_preset="neutral",
            fps=fps,
            duration=duration,
        )

    logger.info(f"Running {iterations} benchmark iterations...")
    latencies = []
    vram_usage = []

    for i in range(iterations):
        if profile_vram and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()

        model.animate(
            source_image=actor_image,
            driving_audio=driving_audio,
            expression_preset="excited",
            fps=fps,
            duration=duration,
        )

        latency = time.perf_counter() - start_time
        latencies.append(latency)

        if profile_vram and torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
            vram_usage.append(peak_vram_gb)

        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{iterations} iterations")

    # Compute statistics
    latencies.sort()
    stats = {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min": min(latencies),
        "max": max(latencies),
        "p50": latencies[int(0.50 * len(latencies))],
        "p95": latencies[int(0.95 * len(latencies))],
        "p99": latencies[int(0.99 * len(latencies))],
    }

    if vram_usage:
        stats["vram_mean_gb"] = statistics.mean(vram_usage)
        stats["vram_max_gb"] = max(vram_usage)

    return stats


def print_results(stats: Dict[str, float], duration: int, fps: int):
    """Print benchmark results in formatted table.

    Args:
        stats: Statistics dictionary
        duration: Video duration
        fps: Frame rate
    """
    num_frames = duration * fps

    print("\n" + "=" * 60)
    print(f"LivePortrait Latency Benchmark Results")
    print("=" * 60)
    print(f"Video: {duration}s @ {fps}fps = {num_frames} frames")
    print("-" * 60)
    print(f"Mean:     {stats['mean']:.3f}s")
    print(f"Median:   {stats['median']:.3f}s")
    print(f"Stdev:    {stats['stdev']:.3f}s")
    print(f"Min:      {stats['min']:.3f}s")
    print(f"Max:      {stats['max']:.3f}s")
    print("-" * 60)
    print(f"P50:      {stats['p50']:.3f}s")
    print(f"P95:      {stats['p95']:.3f}s")
    print(f"P99:      {stats['p99']:.3f}s  (target: <8.0s)")
    print("-" * 60)

    # Throughput
    throughput_fps = num_frames / stats['p99']
    print(f"Throughput (P99): {throughput_fps:.1f} fps")

    # VRAM
    if "vram_mean_gb" in stats:
        print("-" * 60)
        print(f"VRAM Mean:  {stats['vram_mean_gb']:.2f}GB")
        print(f"VRAM Peak:  {stats['vram_max_gb']:.2f}GB  (target: 3.0-4.0GB)")

    print("=" * 60)

    # Check if targets met
    p99_target_met = stats["p99"] < 8.0
    vram_target_met = (
        "vram_max_gb" not in stats
        or 3.0 <= stats["vram_max_gb"] <= 4.0
    )

    print("\nTarget Status:")
    print(f"  P99 Latency: {'✓ PASS' if p99_target_met else '✗ FAIL'}")
    if "vram_max_gb" in stats:
        print(f"  VRAM Budget: {'✓ PASS' if vram_target_met else '✗ FAIL'}")
    print()


def save_results(stats: Dict[str, float], output_path: str, args):
    """Save benchmark results to JSON file.

    Args:
        stats: Statistics dictionary
        output_path: Output file path
        args: Command-line arguments
    """
    results = {
        "benchmark": "liveportrait_latency",
        "config": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "duration": args.duration,
            "fps": args.fps,
            "device": args.device,
            "precision": args.precision,
        },
        "stats": stats,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main():
    """Main benchmark function."""
    args = parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This benchmark requires GPU.")
        return 1

    # Load model
    try:
        model = load_model(args.device, args.precision)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Generate sample inputs
    actor_image, driving_audio = generate_sample_inputs(args.duration, args.device)

    # Run benchmark
    try:
        stats = benchmark_generation(
            model=model,
            actor_image=actor_image,
            driving_audio=driving_audio,
            fps=args.fps,
            duration=args.duration,
            iterations=args.iterations,
            warmup=args.warmup,
            profile_vram=args.profile_vram,
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

    # Print results
    print_results(stats, args.duration, args.fps)

    # Save results if requested
    if args.output:
        save_results(stats, args.output, args)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    exit(main())
