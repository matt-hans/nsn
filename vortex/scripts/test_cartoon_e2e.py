#!/usr/bin/env python3
"""End-to-end test for cartoon video generation via ComfyUI.

Usage:
    python scripts/test_cartoon_e2e.py

Requires:
    - ComfyUI running on localhost:8188
    - Flux, CLIP, VAE models in ComfyUI models directory
"""

import asyncio
import json
import sys
from pathlib import Path

import aiohttp
import numpy as np

# Audio generation config
AUDIO_DURATION_SEC = 3.0
AUDIO_SAMPLE_RATE = 24000
AUDIO_PATH = "/tmp/test_audio.wav"


def create_test_audio(path: str, duration: float = 3.0, sample_rate: int = 24000):
    """Create a simple test audio file.

    Uses simple sine waves to create a voice-like audio pattern.
    """
    import wave
    import struct

    # Generate simple sine wave audio
    t = np.linspace(0, duration, int(duration * sample_rate))

    # Mix multiple frequencies for a more complex sound
    audio = 0.3 * np.sin(2 * np.pi * 200 * t)  # Base tone
    audio += 0.2 * np.sin(2 * np.pi * 400 * t)  # Harmonic
    audio += 0.1 * np.sin(2 * np.pi * 800 * t)  # Higher harmonic

    # Add envelope (fade in/out)
    envelope = np.ones_like(t)
    fade_samples = int(0.1 * sample_rate)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    audio = audio * envelope

    # Normalize to int16 range
    audio = (audio * 32767).astype(np.int16)

    # Write WAV file
    with wave.open(path, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())

    print(f"  Created test audio: {path} ({duration}s @ {sample_rate}Hz)")


async def queue_prompt(workflow: dict) -> str:
    """Queue workflow and return prompt_id."""
    async with aiohttp.ClientSession() as session:
        payload = {"prompt": workflow, "client_id": "test_cartoon"}
        async with session.post(
            "http://127.0.0.1:8188/prompt",
            json=payload,
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Failed to queue: {text}")
            data = await response.json()
            return data["prompt_id"]


async def wait_for_completion(prompt_id: str, timeout: float = 600.0) -> str:
    """Wait for job completion via WebSocket."""
    import websockets

    ws_url = "ws://127.0.0.1:8188/ws?clientId=test_cartoon"
    output_path = None

    async with asyncio.timeout(timeout):
        async with websockets.connect(ws_url) as ws:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue

                msg = json.loads(raw)
                msg_type = msg.get("type")
                data = msg.get("data", {})

                # Filter to our prompt
                if data.get("prompt_id") and data["prompt_id"] != prompt_id:
                    continue

                if msg_type == "progress":
                    value = data.get("value", 0)
                    max_val = data.get("max", 100)
                    print(f"\r  Progress: {value}/{max_val}", end="", flush=True)

                elif msg_type == "executed":
                    output = data.get("output", {})
                    # Check for video output (VHS_VideoCombine outputs to "gifs")
                    if "gifs" in output:
                        for vid in output["gifs"]:
                            filename = vid.get("filename")
                            subfolder = vid.get("subfolder", "")
                            if filename:
                                base = "/home/matt/nsn/ComfyUI/output"
                                output_path = f"{base}/{subfolder}/{filename}".replace("//", "/")
                                print(f"\n  Video output: {output_path}")

                elif msg_type == "execution_error":
                    error = data.get("exception_message", str(data))
                    raise RuntimeError(f"Execution error: {error}")

                elif msg_type == "execution_complete":
                    print("\n  Execution complete!")
                    return output_path

    return output_path


async def main():
    print("=" * 60)
    print("Cartoon Video Generation E2E Test")
    print("=" * 60)

    # Step 1: Check ComfyUI health
    print("\n[1/5] Checking ComfyUI connection...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://127.0.0.1:8188/system_stats",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status != 200:
                    print("  ERROR: ComfyUI not responding")
                    return 1
                data = await response.json()
                vram = data["devices"][0]["vram_total"] / (1024**3)
                print(f"  ComfyUI: OK (GPU: {data['devices'][0]['name'].split(':')[1].strip()}, {vram:.1f}GB)")
    except Exception as e:
        print(f"  ERROR: Cannot connect to ComfyUI: {e}")
        return 1

    # Step 2: Generate test audio
    print("\n[2/5] Generating test audio...")
    create_test_audio(AUDIO_PATH, duration=AUDIO_DURATION_SEC, sample_rate=AUDIO_SAMPLE_RATE)

    # Step 3: Load and configure workflow
    print("\n[3/5] Loading workflow template...")
    template_path = Path(__file__).parent.parent / "templates" / "cartoon_workflow.json"

    with open(template_path) as f:
        workflow = json.load(f)

    # Inject test parameters
    import random
    seed = random.randint(1, 2**31 - 1)

    # Update prompt
    prompt = "A happy cartoon cat scientist with glasses in a colorful laboratory, bright colors, anime style"
    workflow["6"]["inputs"]["clip_l"] = prompt
    workflow["6"]["inputs"]["t5xxl"] = prompt

    # Update seed
    workflow["10"]["inputs"]["seed"] = seed

    # Update audio path
    workflow["40"]["inputs"]["audio_file"] = AUDIO_PATH

    # Calculate frame count from audio duration (24 fps)
    frame_count = int(AUDIO_DURATION_SEC * 24)
    workflow["20"]["inputs"]["amount"] = frame_count

    print(f"  Prompt: '{prompt[:50]}...'")
    print(f"  Seed: {seed}")
    print(f"  Frames: {frame_count}")

    # Step 4: Queue and execute
    print("\n[4/5] Executing workflow (this may take 1-2 minutes)...")
    try:
        prompt_id = await queue_prompt(workflow)
        print(f"  Queued: {prompt_id}")

        output_path = await wait_for_completion(prompt_id, timeout=600)
    except Exception as e:
        print(f"  ERROR: Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 5: Verify output
    print("\n[5/5] Verifying output...")
    if output_path:
        out_file = Path(output_path)
        if out_file.exists():
            size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"  File: {output_path}")
            print(f"  Size: {size_mb:.2f} MB")

            # Basic validation
            if size_mb < 0.01:
                print("  WARNING: Video file seems too small")
            else:
                print("\n" + "=" * 60)
                print("E2E TEST PASSED!")
                print(f"Output: {output_path}")
                print("=" * 60)
                return 0
        else:
            print(f"  ERROR: Output file not found: {output_path}")
            return 1
    else:
        print("  ERROR: No output path returned")
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
