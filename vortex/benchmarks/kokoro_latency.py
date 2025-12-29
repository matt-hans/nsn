#!/usr/bin/env python3
"""Benchmark Kokoro TTS synthesis latency.

Measures synthesis time across different:
- Script lengths (10, 50, 100, 200 words)
- Voice IDs (rick_c137, morty, summer)
- Emotions (neutral, excited, manic)
- Speed settings (0.8, 1.0, 1.2)

Generates performance report with:
- Mean, P50, P95, P99 latencies
- Latency vs. script length plot
- VRAM usage tracking
- Throughput (characters/sec)

Usage:
    python benchmarks/kokoro_latency.py
    python benchmarks/kokoro_latency.py --iterations 100 --output report.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import torch
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_text(num_words: int) -> str:
    """Generate test text of specified length.

    Args:
        num_words: Number of words

    Returns:
        str: Test text
    """
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Testing text to speech synthesis performance with Kokoro. "
        "This is a benchmark script for measuring latency. "
    )
    words = base_text.split()

    # Repeat to get desired length
    full_text = []
    while len(full_text) < num_words:
        full_text.extend(words)

    return " ".join(full_text[:num_words])


def benchmark_synthesis(
    kokoro,
    text: str,
    voice_id: str,
    speed: float,
    emotion: str,
    iterations: int,
) -> Dict:
    """Benchmark synthesis for given parameters.

    Args:
        kokoro: KokoroWrapper instance
        text: Input text
        voice_id: Voice ID
        speed: Speed multiplier
        emotion: Emotion name
        iterations: Number of iterations

    Returns:
        dict: Benchmark results with latencies and metrics
    """
    latencies = []
    vram_usage = []

    # Warm-up run
    kokoro.synthesize(text=text, voice_id=voice_id, speed=speed, emotion=emotion)

    # Benchmark runs
    for i in range(iterations):
        # Reset VRAM stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()

        audio = kokoro.synthesize(
            text=text,
            voice_id=voice_id,
            speed=speed,
            emotion=emotion
        )

        end_time = time.perf_counter()

        latency = end_time - start_time
        latencies.append(latency)

        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
            vram_usage.append(peak_vram)

        if (i + 1) % 10 == 0:
            logger.info(f"  Iteration {i + 1}/{iterations}: {latency:.3f}s")

    # Calculate statistics
    latencies_sorted = sorted(latencies)
    results = {
        "text_length_words": len(text.split()),
        "text_length_chars": len(text),
        "voice_id": voice_id,
        "speed": speed,
        "emotion": emotion,
        "iterations": iterations,
        "latency_mean": np.mean(latencies),
        "latency_std": np.std(latencies),
        "latency_min": np.min(latencies),
        "latency_max": np.max(latencies),
        "latency_p50": np.percentile(latencies_sorted, 50),
        "latency_p95": np.percentile(latencies_sorted, 95),
        "latency_p99": np.percentile(latencies_sorted, 99),
        "throughput_chars_per_sec": len(text) / np.mean(latencies),
    }

    if vram_usage:
        results["vram_peak_gb"] = np.max(vram_usage)
        results["vram_mean_gb"] = np.mean(vram_usage)

    return results


def plot_latency_vs_length(results: List[Dict], output_path: Path):
    """Plot latency vs. script length.

    Args:
        results: List of benchmark results
        output_path: Output file path for plot
    """
    plt.figure(figsize=(10, 6))

    # Extract data
    lengths = [r["text_length_words"] for r in results]
    latencies_mean = [r["latency_mean"] for r in results]
    latencies_p99 = [r["latency_p99"] for r in results]

    # Plot
    plt.plot(lengths, latencies_mean, marker='o', label='Mean latency', linewidth=2)
    plt.plot(lengths, latencies_p99, marker='s', label='P99 latency', linewidth=2)

    # Target line
    plt.axhline(y=2.0, color='r', linestyle='--', label='Target (2.0s P99)')

    plt.xlabel('Script Length (words)', fontsize=12)
    plt.ylabel('Latency (seconds)', fontsize=12)
    plt.title('Kokoro TTS Synthesis Latency vs. Script Length', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    logger.info(f"Plot saved to {output_path}")


def main():
    """Main benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark Kokoro TTS latency")
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations per test"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="kokoro_benchmark_results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default="kokoro_latency_plot.png",
        help="Output plot file"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Kokoro TTS Latency Benchmark")
    logger.info("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU (will be slower)")

    # Load model
    logger.info("Loading Kokoro model...")
    from vortex.models.kokoro import load_kokoro

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kokoro = load_kokoro(device=device)

    logger.info(f"Model loaded on {device}")

    # Test configurations
    script_lengths = [10, 50, 100, 200]  # words
    voices = ["rick_c137"]  # Focus on one voice for latency testing
    emotions = ["neutral"]
    speeds = [1.0]

    all_results = []

    # Benchmark across script lengths
    logger.info("\nBenchmarking script length impact...")
    for length in script_lengths:
        text = generate_text(length)
        logger.info(f"\nScript length: {length} words ({len(text)} chars)")

        results = benchmark_synthesis(
            kokoro=kokoro,
            text=text,
            voice_id="rick_c137",
            speed=1.0,
            emotion="neutral",
            iterations=args.iterations,
        )

        all_results.append(results)

        logger.info(f"  Mean latency: {results['latency_mean']:.3f}s")
        logger.info(f"  P99 latency: {results['latency_p99']:.3f}s")
        logger.info(f"  Throughput: {results['throughput_chars_per_sec']:.1f} chars/sec")

        if "vram_peak_gb" in results:
            logger.info(f"  Peak VRAM: {results['vram_peak_gb']:.3f} GB")

    # Test voice variations
    logger.info("\nBenchmarking voice variations...")
    text = generate_text(100)
    for voice in ["rick_c137", "morty", "summer"]:
        logger.info(f"\nVoice: {voice}")

        results = benchmark_synthesis(
            kokoro=kokoro,
            text=text,
            voice_id=voice,
            speed=1.0,
            emotion="neutral",
            iterations=20,  # Fewer iterations for voice tests
        )

        logger.info(f"  P99 latency: {results['latency_p99']:.3f}s")

    # Test emotion variations
    logger.info("\nBenchmarking emotion variations...")
    for emotion in ["neutral", "excited", "manic"]:
        logger.info(f"\nEmotion: {emotion}")

        results = benchmark_synthesis(
            kokoro=kokoro,
            text=text,
            voice_id="rick_c137",
            speed=1.0,
            emotion=emotion,
            iterations=20,
        )

        logger.info(f"  P99 latency: {results['latency_p99']:.3f}s")

    # Save results
    output_data = {
        "benchmark_config": {
            "iterations": args.iterations,
            "device": device,
            "torch_version": torch.__version__,
        },
        "results": all_results,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n✓ Results saved to {args.output}")

    # Generate plot
    plot_latency_vs_length(all_results, args.plot)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Benchmark Summary")
    logger.info("=" * 60)

    max_p99 = max(r["latency_p99"] for r in all_results)
    logger.info(f"Maximum P99 latency: {max_p99:.3f}s")

    if max_p99 < 2.0:
        logger.info("✓ PASSES latency target (<2.0s P99)")
    else:
        logger.warning("✗ FAILS latency target (<2.0s P99)")

    # VRAM check
    if "vram_peak_gb" in all_results[0]:
        max_vram = max(r["vram_peak_gb"] for r in all_results)
        logger.info(f"Peak VRAM usage: {max_vram:.3f} GB")

        if 0.3 <= max_vram <= 0.5:
            logger.info("✓ PASSES VRAM budget (0.3-0.5 GB)")
        else:
            logger.warning(f"⚠ VRAM usage outside target range (0.3-0.5 GB)")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
