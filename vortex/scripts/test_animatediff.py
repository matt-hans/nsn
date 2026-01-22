#!/usr/bin/env python3
"""Test AnimateDiff cartoon generation.

Usage:
    python scripts/test_animatediff.py
"""

import asyncio
import json
import sys
from pathlib import Path

import aiohttp
import numpy as np


def create_test_audio(path: str, duration: float = 2.0, sample_rate: int = 24000):
    """Create test audio file."""
    import wave

    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = 0.3 * np.sin(2 * np.pi * 200 * t)
    audio += 0.2 * np.sin(2 * np.pi * 400 * t)

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


async def queue_prompt(workflow: dict) -> str:
    """Queue workflow."""
    async with aiohttp.ClientSession() as session:
        payload = {"prompt": workflow, "client_id": "test_ad"}
        async with session.post("http://127.0.0.1:8188/prompt", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Queue failed: {text}")
            data = await resp.json()
            return data["prompt_id"]


async def wait_for_completion(prompt_id: str, timeout: float = 600.0) -> str:
    """Wait for completion via WebSocket."""
    import websockets

    output_path = None
    last_progress = ""

    async with asyncio.timeout(timeout):
        async with websockets.connect("ws://127.0.0.1:8188/ws?clientId=test_ad") as ws:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg = json.loads(raw)
                msg_type = msg.get("type")
                data = msg.get("data", {})

                if data.get("prompt_id") and data["prompt_id"] != prompt_id:
                    continue

                if msg_type == "progress":
                    progress = f"{data.get('value', 0)}/{data.get('max', 100)}"
                    if progress != last_progress:
                        print(f"\r  Progress: {progress}    ", end="", flush=True)
                        last_progress = progress

                elif msg_type == "executed":
                    output = data.get("output", {})
                    if "gifs" in output:
                        for vid in output["gifs"]:
                            filename = vid.get("filename")
                            if filename:
                                subfolder = vid.get("subfolder", "")
                                output_path = f"/home/matt/nsn/ComfyUI/output/{subfolder}/{filename}".replace("//", "/")
                                print(f"\n  Output: {output_path}")

                elif msg_type == "execution_error":
                    error = data.get("exception_message", str(data))
                    print(f"\n  ERROR: {error}")
                    # Print more details
                    if "node_errors" in data:
                        for node_id, node_err in data.get("node_errors", {}).items():
                            print(f"    Node {node_id}: {node_err}")
                    raise RuntimeError(f"Execution error: {error}")

                elif msg_type == "execution_complete":
                    print("\n  Complete!")
                    return output_path

    return output_path


async def main():
    print("=" * 60)
    print("AnimateDiff Cartoon Generation Test")
    print("=" * 60)

    # Create test audio
    print("\n[1/4] Creating test audio...")
    create_test_audio("/tmp/test_audio.wav", duration=2.0)
    print("  Created /tmp/test_audio.wav")

    # Load workflow
    print("\n[2/4] Loading AnimateDiff workflow...")
    template_path = Path(__file__).parent.parent / "templates" / "animatediff_cartoon.json"
    with open(template_path) as f:
        workflow = json.load(f)

    # Randomize seed
    import random
    seed = random.randint(1, 2**31 - 1)
    workflow["10"]["inputs"]["seed"] = seed
    print(f"  Seed: {seed}")
    print(f"  Frames: {workflow['3']['inputs']['batch_size']}")

    # Queue and execute
    print("\n[3/4] Generating animation (this takes 1-3 minutes)...")
    try:
        prompt_id = await queue_prompt(workflow)
        print(f"  Queued: {prompt_id}")
        output_path = await wait_for_completion(prompt_id, timeout=600)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Verify
    print("\n[4/4] Verifying output...")
    if output_path:
        out_file = Path(output_path)
        if out_file.exists():
            size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"  File: {output_path}")
            print(f"  Size: {size_mb:.2f} MB")

            # Check frame count with ffprobe
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                 "-count_frames", "-show_entries", "stream=nb_read_frames",
                 "-of", "csv=p=0", output_path],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                frames = result.stdout.strip()
                print(f"  Frames: {frames}")

            print("\n" + "=" * 60)
            print("ANIMATEDIFF TEST PASSED!")
            print(f"Output: {output_path}")
            print("=" * 60)
            return 0

    print("  ERROR: No output file")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
