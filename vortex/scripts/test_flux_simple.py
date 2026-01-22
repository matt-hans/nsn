#!/usr/bin/env python3
"""Quick test for Flux image generation via ComfyUI.

Usage:
    python scripts/test_flux_simple.py
"""

import asyncio
import json
import sys
from pathlib import Path

import aiohttp


async def queue_prompt(workflow: dict) -> str:
    """Queue workflow and return prompt_id."""
    async with aiohttp.ClientSession() as session:
        payload = {"prompt": workflow, "client_id": "test_flux"}
        async with session.post(
            "http://127.0.0.1:8188/prompt",
            json=payload,
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Failed to queue: {text}")
            data = await response.json()
            return data["prompt_id"]


async def wait_for_completion(prompt_id: str, timeout: float = 300.0) -> str:
    """Wait for job completion via WebSocket."""
    import websockets

    ws_url = f"ws://127.0.0.1:8188/ws?clientId=test_flux"
    output_path = None

    async with asyncio.timeout(timeout):
        async with websockets.connect(ws_url) as ws:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue  # Skip binary

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
                    if "images" in output:
                        for img in output["images"]:
                            filename = img.get("filename")
                            subfolder = img.get("subfolder", "")
                            if filename:
                                output_path = f"/home/matt/nsn/ComfyUI/output/{subfolder}/{filename}".replace("//", "/")
                                print(f"\n  Output: {output_path}")

                elif msg_type == "execution_error":
                    error = data.get("exception_message", str(data))
                    raise RuntimeError(f"Execution error: {error}")

                elif msg_type == "execution_complete":
                    print("\n  Execution complete!")
                    return output_path

    return output_path


async def main():
    print("=" * 60)
    print("Flux Image Generation Test")
    print("=" * 60)

    # Load workflow
    template_path = Path(__file__).parent.parent / "templates" / "flux_simple.json"
    print(f"\n[1/4] Loading workflow from {template_path}")

    with open(template_path) as f:
        workflow = json.load(f)

    # Modify seed for randomness
    import random
    seed = random.randint(1, 2**31 - 1)
    workflow["7"]["inputs"]["seed"] = seed
    print(f"  Seed: {seed}")

    # Queue prompt
    print("\n[2/4] Queuing workflow...")
    try:
        prompt_id = await queue_prompt(workflow)
        print(f"  Prompt ID: {prompt_id}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1

    # Wait for completion
    print("\n[3/4] Generating image (this may take 30-60 seconds on first run)...")
    try:
        output_path = await wait_for_completion(prompt_id, timeout=300)
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1

    # Verify output
    print("\n[4/4] Verifying output...")
    if output_path:
        out_file = Path(output_path)
        if out_file.exists():
            size_kb = out_file.stat().st_size / 1024
            print(f"  File: {output_path}")
            print(f"  Size: {size_kb:.1f} KB")
            print("\n" + "=" * 60)
            print("FLUX TEST PASSED!")
            print("=" * 60)
            return 0
        else:
            print(f"  ERROR: Output file not found: {output_path}")
            return 1
    else:
        print("  ERROR: No output path returned")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
