#!/usr/bin/env python3
"""Final E2E test for animated cartoon generation.

Verifies the full pipeline produces ANIMATED (not static) cartoon video.
"""

import asyncio
import json
import sys
import wave
from pathlib import Path

import aiohttp
import numpy as np


def create_test_audio(path: str, duration: float = 2.0, sample_rate: int = 24000):
    """Create test audio."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = 0.3 * np.sin(2 * np.pi * 200 * t)
    envelope = np.ones_like(t)
    fade = int(0.1 * sample_rate)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    audio = (audio * envelope * 32767).astype(np.int16)

    with wave.open(path, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())


async def run_workflow(workflow: dict, timeout: float = 600.0) -> str:
    """Queue workflow and wait for output."""
    import websockets

    # Queue
    async with aiohttp.ClientSession() as session:
        payload = {"prompt": workflow, "client_id": "final_test"}
        async with session.post("http://127.0.0.1:8188/prompt", json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(await resp.text())
            prompt_id = (await resp.json())["prompt_id"]
            print(f"  Queued: {prompt_id}")

    # Wait
    output_path = None
    async with asyncio.timeout(timeout):
        async with websockets.connect("ws://127.0.0.1:8188/ws?clientId=final_test") as ws:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                data = msg.get("data", {})

                if data.get("prompt_id") and data["prompt_id"] != prompt_id:
                    continue

                if msg["type"] == "progress":
                    print(f"\r  Progress: {data.get('value')}/{data.get('max')}  ", end="", flush=True)
                elif msg["type"] == "executed":
                    if "gifs" in data.get("output", {}):
                        for v in data["output"]["gifs"]:
                            if v.get("filename"):
                                output_path = f"/home/matt/nsn/ComfyUI/output/{v.get('subfolder', '')}/{v['filename']}".replace("//", "/")
                                print(f"\n  Video: {output_path}")
                elif msg["type"] == "execution_error":
                    raise RuntimeError(data.get("exception_message", str(data)))
                elif msg["type"] == "execution_complete":
                    return output_path

    return output_path


def verify_animation(video_path: str) -> bool:
    """Verify video has actual animation (frames differ)."""
    import subprocess
    import tempfile
    from PIL import Image

    # Extract frames
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-vframes", "4",
            f"{tmpdir}/frame_%02d.png"
        ], capture_output=True, check=True)

        # Load and compare
        frames = [np.array(Image.open(f"{tmpdir}/frame_0{i}.png")) for i in range(1, 5)]

    total_changed = 0
    for i in range(3):
        diff = np.abs(frames[i].astype(float) - frames[i+1].astype(float))
        pixels_changed = (diff > 5).sum() // 3
        total_changed += pixels_changed
        print(f"    Frame {i+1}→{i+2}: {pixels_changed:,} pixels changed")

    is_animated = total_changed > 10000  # Significant pixel changes
    return is_animated


async def main():
    print("=" * 60)
    print("FINAL E2E TEST: Animated Cartoon Generation")
    print("=" * 60)

    # Setup
    print("\n[1/4] Setup...")
    create_test_audio("/tmp/test_audio.wav")
    print("  Audio created")

    # Load workflow
    print("\n[2/4] Loading cartoon_workflow.json...")
    with open(Path(__file__).parent.parent / "templates" / "cartoon_workflow.json") as f:
        workflow = json.load(f)

    import random
    seed = random.randint(1, 2**31 - 1)
    workflow["10"]["inputs"]["seed"] = seed

    # Custom prompt for wacky interdimensional cable style
    wacky_prompt = "a wacky cartoon alien creature hosting a TV show, surreal background, bright neon colors, absurd expressions, interdimensional cable style, animated cartoon"
    workflow["6"]["inputs"]["text"] = wacky_prompt
    print(f"  Prompt: '{wacky_prompt[:50]}...'")
    print(f"  Seed: {seed}")

    # Generate
    print("\n[3/4] Generating animated cartoon...")
    try:
        output_path = await run_workflow(workflow, timeout=600)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return 1

    if not output_path or not Path(output_path).exists():
        print("  ERROR: No output file")
        return 1

    # Verify animation
    print("\n[4/4] Verifying animation...")
    try:
        is_animated = verify_animation(output_path)
    except Exception as e:
        print(f"  Verification error: {e}")
        is_animated = False

    print(f"\n  Animation detected: {'YES' if is_animated else 'NO'}")

    if is_animated:
        size_kb = Path(output_path).stat().st_size / 1024
        print("\n" + "=" * 60)
        print("TEST PASSED - ANIMATED CARTOON GENERATED!")
        print(f"  Output: {output_path}")
        print(f"  Size: {size_kb:.1f} KB")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("TEST FAILED - Video is NOT animated")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
