#!/usr/bin/env python3
"""End-to-end test for Montage pipeline.

Verifies the full pipeline produces animated video with audio matching target duration.
The montage architecture generates 3 independent 5-second clips (15s total at 8fps).

Pipeline: Showrunner (3-scene storyboard) -> Kokoro TTS -> Flux keyframe -> CogVideoX video -> CLIP

Usage:
    python scripts/e2e_narrative_test.py --theme "bizarre infomercial" --duration 15
    python scripts/e2e_narrative_test.py --seed 42 --output-dir /tmp/test_output
    python scripts/e2e_narrative_test.py --help

Manual Checklist After Running:
- [ ] Video shows actual motion (frames differ significantly)
- [ ] Audio has audible speech
- [ ] Duration matches target (+/- 3s)
- [ ] Style matches "Interdimensional Cable" aesthetic
- [ ] No warping/distortion artifacts
- [ ] Script has valid 3-scene storyboard
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TestResult:
    """Container for a single verification result."""

    def __init__(self, name: str, passed: bool, message: str, details: dict | None = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


def verify_video_exists(video_path: str) -> TestResult:
    """Verify video file exists and has non-zero size.

    Args:
        video_path: Path to video file

    Returns:
        TestResult with pass/fail status
    """
    path = Path(video_path)

    if not path.exists():
        return TestResult(
            name="Video File Exists",
            passed=False,
            message=f"Video file not found: {video_path}",
        )

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        return TestResult(
            name="Video File Exists",
            passed=False,
            message="Video file exists but is empty (0 bytes)",
        )

    size_mb = size_bytes / (1024 * 1024)
    return TestResult(
        name="Video File Exists",
        passed=True,
        message=f"Video file exists ({size_mb:.2f} MB)",
        details={"size_bytes": size_bytes, "size_mb": size_mb},
    )


def verify_video_animation(video_path: str, threshold: float = 0.01) -> TestResult:
    """Verify video has actual animation (frames differ).

    Extracts first 5 frames and computes average pixel difference between
    consecutive frames. Animation is detected if difference exceeds threshold.

    Args:
        video_path: Path to video file
        threshold: Minimum average frame difference to consider "animated" (0-1 scale)

    Returns:
        TestResult with pass/fail status and motion metrics
    """
    try:
        from PIL import Image

        # Extract first 5 frames using ffmpeg
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-vframes",
                    "5",
                    "-q:v",
                    "2",
                    f"{tmpdir}/frame_%03d.jpg",
                ],
                capture_output=True,
                check=False,
            )

            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                return TestResult(
                    name="Video Animation",
                    passed=False,
                    message=f"Failed to extract frames: {stderr[:200]}",
                )

            # Load extracted frames
            frames = []
            for i in range(1, 6):
                frame_path = Path(tmpdir) / f"frame_{i:03d}.jpg"
                if frame_path.exists():
                    img = Image.open(frame_path)
                    frames.append(np.array(img))

            if len(frames) < 2:
                return TestResult(
                    name="Video Animation",
                    passed=False,
                    message=f"Only {len(frames)} frame(s) extracted, need at least 2",
                    details={"frames_extracted": len(frames)},
                )

            # Calculate frame-to-frame differences
            diffs = []
            for i in range(len(frames) - 1):
                # Compute absolute pixel difference normalized to [0, 1]
                diff = np.abs(frames[i].astype(float) - frames[i + 1].astype(float))
                avg_diff = diff.mean() / 255.0
                diffs.append(avg_diff)

            avg_motion = float(np.mean(diffs))
            max_motion = float(np.max(diffs))
            min_motion = float(np.min(diffs))

            if avg_motion < threshold:
                return TestResult(
                    name="Video Animation",
                    passed=False,
                    message=f"Low motion detected: avg={avg_motion:.4f} < threshold={threshold}",
                    details={
                        "avg_motion": avg_motion,
                        "max_motion": max_motion,
                        "min_motion": min_motion,
                        "frame_diffs": diffs,
                        "threshold": threshold,
                    },
                )

            return TestResult(
                name="Video Animation",
                passed=True,
                message=f"Motion detected: avg={avg_motion:.4f}, max={max_motion:.4f}",
                details={
                    "avg_motion": avg_motion,
                    "max_motion": max_motion,
                    "min_motion": min_motion,
                    "frame_diffs": diffs,
                    "threshold": threshold,
                },
            )

    except ImportError as e:
        return TestResult(
            name="Video Animation",
            passed=False,
            message=f"Missing dependency: {e}",
        )
    except Exception as e:
        return TestResult(
            name="Video Animation",
            passed=False,
            message=f"Verification error: {e}",
        )


def verify_audio_exists(audio_path: str) -> TestResult:
    """Verify audio file exists and has non-zero size.

    Args:
        audio_path: Path to audio file

    Returns:
        TestResult with pass/fail status
    """
    path = Path(audio_path)

    if not path.exists():
        return TestResult(
            name="Audio File Exists",
            passed=False,
            message=f"Audio file not found: {audio_path}",
        )

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        return TestResult(
            name="Audio File Exists",
            passed=False,
            message="Audio file exists but is empty (0 bytes)",
        )

    size_kb = size_bytes / 1024
    return TestResult(
        name="Audio File Exists",
        passed=True,
        message=f"Audio file exists ({size_kb:.1f} KB)",
        details={"size_bytes": size_bytes, "size_kb": size_kb},
    )


def verify_audio_waveform(audio_path: str, min_samples: int = 1000) -> TestResult:
    """Verify audio file has audible waveform (not silence).

    Args:
        audio_path: Path to audio file
        min_samples: Minimum samples expected for valid audio

    Returns:
        TestResult with pass/fail status and audio metrics
    """
    try:
        import soundfile as sf

        audio, sr = sf.read(audio_path)

        if len(audio) < min_samples:
            return TestResult(
                name="Audio Waveform",
                passed=False,
                message=f"Audio too short: {len(audio)} samples < {min_samples} minimum",
                details={
                    "samples": len(audio),
                    "sample_rate": sr,
                    "min_samples": min_samples,
                },
            )

        # Compute RMS to detect silence
        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.max(np.abs(audio)))
        duration_sec = len(audio) / sr

        # RMS below 0.001 is effectively silent
        if rms < 0.001:
            return TestResult(
                name="Audio Waveform",
                passed=False,
                message=f"Audio appears silent: RMS={rms:.6f}",
                details={
                    "rms": rms,
                    "peak": peak,
                    "duration_sec": duration_sec,
                    "sample_rate": sr,
                    "samples": len(audio),
                },
            )

        return TestResult(
            name="Audio Waveform",
            passed=True,
            message=f"Audio OK: {duration_sec:.1f}s, RMS={rms:.4f}, peak={peak:.4f}",
            details={
                "rms": rms,
                "peak": peak,
                "duration_sec": duration_sec,
                "sample_rate": sr,
                "samples": len(audio),
            },
        )

    except ImportError as e:
        return TestResult(
            name="Audio Waveform",
            passed=False,
            message=f"Missing dependency: {e}. Install with: pip install soundfile",
        )
    except Exception as e:
        return TestResult(
            name="Audio Waveform",
            passed=False,
            message=f"Verification error: {e}",
        )


def verify_duration(video_path: str, target_duration: float, tolerance: float = 2.0) -> TestResult:
    """Verify video duration matches target within tolerance.

    Args:
        video_path: Path to video file
        target_duration: Expected duration in seconds
        tolerance: Allowed deviation in seconds (+/-)

    Returns:
        TestResult with pass/fail status and duration metrics
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            stderr = result.stderr or "Unknown error"
            return TestResult(
                name="Video Duration",
                passed=False,
                message=f"ffprobe failed: {stderr[:200]}",
            )

        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        diff = abs(duration - target_duration)

        if diff > tolerance:
            return TestResult(
                name="Video Duration",
                passed=False,
                message=(
                    f"Duration mismatch: {duration:.1f}s vs target {target_duration:.1f}s "
                    f"(diff={diff:.1f}s > tolerance={tolerance:.1f}s)"
                ),
                details={
                    "actual_duration": duration,
                    "target_duration": target_duration,
                    "difference": diff,
                    "tolerance": tolerance,
                },
            )

        return TestResult(
            name="Video Duration",
            passed=True,
            message=f"Duration OK: {duration:.1f}s (target {target_duration:.1f}s +/-{tolerance}s)",
            details={
                "actual_duration": duration,
                "target_duration": target_duration,
                "difference": diff,
                "tolerance": tolerance,
            },
        )

    except json.JSONDecodeError as e:
        return TestResult(
            name="Video Duration",
            passed=False,
            message=f"Failed to parse ffprobe output: {e}",
        )
    except Exception as e:
        return TestResult(
            name="Video Duration",
            passed=False,
            message=f"Verification error: {e}",
        )


def verify_storyboard(result) -> TestResult:
    """Verify script has valid 3-scene storyboard.

    Args:
        result: Generation result from orchestrator

    Returns:
        TestResult with pass/fail status
    """
    if not hasattr(result, "script") or result.script is None:
        return TestResult(
            name="Script Storyboard",
            passed=False,
            message="No script in result",
        )

    storyboard = getattr(result.script, "storyboard", None)
    if storyboard is None:
        return TestResult(
            name="Script Storyboard",
            passed=False,
            message="Script missing storyboard attribute",
        )

    if len(storyboard) != 3:
        return TestResult(
            name="Script Storyboard",
            passed=False,
            message=f"Expected 3 scenes, got {len(storyboard)}",
            details={"scene_count": len(storyboard)},
        )

    # Check each scene is non-empty
    for i, scene in enumerate(storyboard):
        if not scene or len(scene.strip()) < 10:
            return TestResult(
                name="Script Storyboard",
                passed=False,
                message=f"Scene {i+1} is too short or empty",
                details={"scene": scene, "scene_index": i},
            )

    return TestResult(
        name="Script Storyboard",
        passed=True,
        message=f"Storyboard OK: {len(storyboard)} scenes",
        details={"scenes": storyboard},
    )


def verify_frame_count(video_path: str, min_frames: int = 10) -> TestResult:
    """Verify video has minimum number of frames.

    Args:
        video_path: Path to video file
        min_frames: Minimum expected frames

    Returns:
        TestResult with pass/fail status and frame count
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-print_format",
                "json",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            # Fallback: estimate from duration and assume 8fps
            dur_result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    video_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if dur_result.returncode == 0:
                data = json.loads(dur_result.stdout)
                duration = float(data["format"]["duration"])
                estimated_frames = int(duration * 8)  # Assume 8fps
                return TestResult(
                    name="Frame Count",
                    passed=estimated_frames >= min_frames,
                    message=f"Estimated ~{estimated_frames} frames (from duration, assuming 8fps)",
                    details={"estimated_frames": estimated_frames, "method": "duration_estimate"},
                )
            return TestResult(
                name="Frame Count",
                passed=False,
                message="Could not determine frame count",
            )

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            return TestResult(
                name="Frame Count",
                passed=False,
                message="No video stream found",
            )

        frame_count = int(streams[0].get("nb_read_frames", 0))

        if frame_count < min_frames:
            return TestResult(
                name="Frame Count",
                passed=False,
                message=f"Too few frames: {frame_count} < {min_frames} minimum",
                details={"frame_count": frame_count, "min_frames": min_frames},
            )

        return TestResult(
            name="Frame Count",
            passed=True,
            message=f"Frame count OK: {frame_count} frames",
            details={"frame_count": frame_count, "min_frames": min_frames},
        )

    except Exception as e:
        return TestResult(
            name="Frame Count",
            passed=False,
            message=f"Verification error: {e}",
        )


async def run_e2e_test(
    theme: str,
    duration: float,
    seed: int | None,
    output_dir: str,
    config_path: str,
    verbose: bool = False,
) -> int:
    """Run the full E2E test.

    Args:
        theme: Theme for script generation
        duration: Target video duration in seconds
        seed: Random seed (None for random)
        output_dir: Output directory for generated files
        config_path: Path to config.yaml
        verbose: Enable verbose output

    Returns:
        0 for success, 1 for failure
    """
    print("=" * 70)
    print("MONTAGE PIPELINE E2E TEST")
    print("=" * 70)
    print(f"  Theme:           {theme}")
    print(f"  Target Duration: {duration}s")
    print(f"  Seed:            {seed or 'random'}")
    print(f"  Output Dir:      {output_dir}")
    print(f"  Config:          {config_path}")
    print()

    # Change to vortex directory for relative config paths
    vortex_dir = Path(__file__).parent.parent
    original_cwd = Path.cwd()

    try:
        import os

        os.chdir(vortex_dir)
        logger.info(f"Changed working directory to {vortex_dir}")
    except Exception as e:
        logger.warning(f"Could not change to vortex directory: {e}")

    # Step 1: Initialize orchestrator
    print("[1/6] Initializing orchestrator...")
    try:
        from vortex.orchestrator import VideoOrchestrator

        orchestrator = VideoOrchestrator(
            config_path=config_path,
            output_dir=output_dir,
            device="cuda",
        )

        await orchestrator.initialize()
        print("       Orchestrator initialized successfully")

    except ImportError as e:
        print(f"       ERROR: Import failed - {e}")
        print("       Ensure you're running from the vortex directory with dependencies installed")
        return 1
    except Exception as e:
        print(f"       ERROR: Failed to initialize - {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1

    # Step 2: Health check
    print("\n[2/6] Running health check...")
    try:
        health = await orchestrator.health_check()
        if health.get("orchestrator") and health.get("renderer"):
            print("       Health check passed")
        else:
            print(f"       WARNING: Health check incomplete - {health}")
    except Exception as e:
        print(f"       WARNING: Health check failed - {e}")

    # Step 3: Run generation
    print(f"\n[3/6] Generating video (target {duration}s)...")
    print("       This may take 1-3 minutes depending on hardware...")
    try:
        result = await orchestrator.generate(
            slot_id=9999,  # Test slot
            seed=seed,
            theme=theme,
            tone="absurd",
            target_duration=duration,
            deadline_sec=1800.0,  # 30 minute deadline (CogVideoX takes ~20 min)
        )
    except Exception as e:
        print(f"       ERROR: Generation failed - {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        await orchestrator.shutdown()
        return 1

    if not result.success:
        print(f"       ERROR: Generation failed - {result.error_msg}")
        await orchestrator.shutdown()
        return 1

    print(f"       Generation complete in {result.generation_time_ms / 1000:.1f}s")
    print(f"       Video: {result.video_path}")
    print(f"       Audio: {result.audio_path}")
    print(f"       Seed:  {result.seed}")

    # Step 4: Run verification tests
    print("\n[4/6] Running verification tests...")
    tests: list[TestResult] = []

    # Video file exists
    tests.append(verify_video_exists(result.video_path))
    print(f"       {tests[-1]}")

    # Audio file exists
    tests.append(verify_audio_exists(result.audio_path))
    print(f"       {tests[-1]}")

    # Video has animation
    tests.append(verify_video_animation(result.video_path))
    print(f"       {tests[-1]}")

    # Audio has waveform (not silence)
    tests.append(verify_audio_waveform(result.audio_path))
    print(f"       {tests[-1]}")

    # Duration matches target (montage has 3 independent clips, so wider tolerance)
    tests.append(verify_duration(result.video_path, duration, tolerance=3.0))
    print(f"       {tests[-1]}")

    # Frame count adequate
    # Montage: 3 scenes x 40 frames each = 120 frames @ 8fps = 15s
    min_frames = max(10, int(duration * 8))  # 8fps for montage
    tests.append(verify_frame_count(result.video_path, min_frames=min_frames))
    print(f"       {tests[-1]}")

    # Storyboard verification (if script available in result)
    if hasattr(result, "script"):
        tests.append(verify_storyboard(result))
        print(f"       {tests[-1]}")

    # Step 5: Cleanup
    print("\n[5/6] Shutting down orchestrator...")
    try:
        await orchestrator.shutdown()
        print("       Orchestrator shutdown complete")
    except Exception as e:
        print(f"       WARNING: Shutdown error - {e}")

    # Restore original working directory
    try:
        import os

        os.chdir(original_cwd)
    except Exception:
        pass

    # Step 6: Summary
    print("\n[6/6] Test Summary")
    passed_tests = [t for t in tests if t.passed]
    failed_tests = [t for t in tests if not t.passed]
    all_passed = len(failed_tests) == 0

    print()
    print("=" * 70)
    if all_passed:
        print("E2E TEST PASSED")
        print(f"  All {len(tests)} verification tests passed")
        print()
        print("Output Files:")
        print(f"  Video: {result.video_path}")
        print(f"  Audio: {result.audio_path}")
        print()
        print("Manual Review Checklist:")
        print("  [ ] Video shows actual motion (frames differ visually)")
        print("  [ ] Audio has audible speech (play the .wav file)")
        print("  [ ] Style matches 'Interdimensional Cable' aesthetic")
        print("  [ ] No warping/distortion artifacts")
        print("  [ ] Script has valid 3-scene storyboard")
    else:
        print("E2E TEST FAILED")
        print(f"  {len(passed_tests)}/{len(tests)} tests passed")
        print()
        print("Failed Tests:")
        for t in failed_tests:
            print(f"  - {t.name}: {t.message}")
            if verbose and t.details:
                for k, v in t.details.items():
                    print(f"      {k}: {v}")
    print("=" * 70)

    # Write test report to JSON for CI/CD integration
    report_path = Path(output_dir) / "e2e_test_report.json"
    try:
        report = {
            "success": all_passed,
            "theme": theme,
            "target_duration": duration,
            "seed": result.seed,
            "generation_time_ms": result.generation_time_ms,
            "video_path": result.video_path,
            "audio_path": result.audio_path,
            "tests": [
                {
                    "name": t.name,
                    "passed": t.passed,
                    "message": t.message,
                    "details": t.details,
                }
                for t in tests
            ],
        }
        if hasattr(result, "script") and result.script is not None:
            report["script"] = {
                "setup": getattr(result.script, "setup", ""),
                "punchline": getattr(result.script, "punchline", ""),
                "subject_visual": getattr(result.script, "subject_visual", ""),
                "storyboard": list(getattr(result.script, "storyboard", [])),
                "video_prompts": list(getattr(result.script, "video_prompts", [])),
            }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nTest report saved to: {report_path}")
    except Exception as e:
        logger.warning(f"Could not save test report: {e}")

    return 0 if all_passed else 1


def main() -> int:
    """Parse arguments and run E2E test."""
    parser = argparse.ArgumentParser(
        description="End-to-end test for Montage pipeline (3 scenes x 5s = 15s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test with default settings (15s montage)
    python scripts/e2e_narrative_test.py

    # Custom theme and duration
    python scripts/e2e_narrative_test.py --theme "weird cooking show" --duration 15

    # Reproducible test with fixed seed
    python scripts/e2e_narrative_test.py --seed 42 --output-dir /tmp/test_output

    # Verbose output for debugging
    python scripts/e2e_narrative_test.py --verbose
        """,
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="bizarre infomercial",
        help="Theme for script generation (default: 'bizarre infomercial')",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Target video duration in seconds (default: 15.0 for 3-scene montage)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/e2e_test",
        help="Output directory for generated files (default: outputs/e2e_test)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed error traces",
    )

    args = parser.parse_args()

    # Run test
    return asyncio.run(
        run_e2e_test(
            theme=args.theme,
            duration=args.duration,
            seed=args.seed,
            output_dir=args.output_dir,
            config_path=args.config,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
