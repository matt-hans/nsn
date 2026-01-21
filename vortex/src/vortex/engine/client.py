"""ComfyUI WebSocket client for job dispatch and monitoring.

Provides async interface for:
- Queuing workflow prompts
- Monitoring execution progress
- Retrieving output file paths
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class ComfyClient:
    """Async client for ComfyUI WebSocket API.

    Handles job queuing, progress monitoring, and result retrieval.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        timeout: float = 300.0,
    ):
        """Initialize client.

        Args:
            host: ComfyUI server hostname
            port: ComfyUI server port
            timeout: Maximum seconds to wait for job completion
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

        self._base_url = f"http://{host}:{port}"
        self._ws_url = f"ws://{host}:{port}/ws?clientId={self.client_id}"

    async def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """Queue a workflow for execution.

        Args:
            workflow: ComfyUI workflow dict (API format)

        Returns:
            Prompt ID for tracking execution

        Raises:
            RuntimeError: If queuing fails
        """
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/prompt",
                json=payload,
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"Failed to queue prompt: {text}")

                data = await response.json()
                prompt_id = data.get("prompt_id")

                if not prompt_id:
                    raise RuntimeError(f"No prompt_id in response: {data}")

                logger.info(f"Queued prompt: {prompt_id}")
                return prompt_id

    def _parse_message(self, raw: str) -> dict[str, Any] | None:
        """Parse WebSocket message.

        Args:
            raw: Raw message string

        Returns:
            Parsed message dict or None if not JSON
        """
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Non-JSON message: {raw[:100]}")
            return None

    async def _listen_progress(
        self,
        prompt_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Listen for execution progress via WebSocket.

        Args:
            prompt_id: Prompt ID to monitor

        Yields:
            Progress update dicts
        """
        import websockets

        async with websockets.connect(self._ws_url) as ws:
            async for raw_message in ws:
                if isinstance(raw_message, bytes):
                    # Binary message (preview image) - skip
                    continue

                message = self._parse_message(raw_message)
                if message is None:
                    continue

                msg_type = message.get("type")
                data = message.get("data", {})

                # Filter to our prompt
                if data.get("prompt_id") and data["prompt_id"] != prompt_id:
                    continue

                yield message

                # Check for completion
                if msg_type == "executed":
                    logger.info(f"Execution completed for node: {data.get('node')}")
                elif msg_type == "execution_error":
                    raise RuntimeError(f"Execution error: {data}")
                elif msg_type == "execution_complete":
                    logger.info("Workflow execution complete")
                    return

    async def wait_for_completion(
        self,
        prompt_id: str,
        output_node: str = "99",
    ) -> str:
        """Wait for job completion and return output path.

        Args:
            prompt_id: Prompt ID to wait for
            output_node: Node ID that produces the output file

        Returns:
            Path to output file (MP4)

        Raises:
            TimeoutError: If job doesn't complete within timeout
            RuntimeError: If execution fails
        """
        output_path = None

        try:
            async with asyncio.timeout(self.timeout):
                async for message in self._listen_progress(prompt_id):
                    msg_type = message.get("type")
                    data = message.get("data", {})

                    if msg_type == "progress":
                        value = data.get("value", 0)
                        max_val = data.get("max", 100)
                        logger.debug(f"Progress: {value}/{max_val}")

                    elif msg_type == "executed":
                        output = data.get("output", {})

                        # Check for video output (VHS_VideoCombine)
                        if "gifs" in output:
                            for gif in output["gifs"]:
                                filename = gif.get("filename")
                                subfolder = gif.get("subfolder", "")
                                if filename:
                                    output_path = str(
                                        Path(self._get_output_dir())
                                        / subfolder
                                        / filename
                                    )
                                    logger.info(f"Output file: {output_path}")

                    elif msg_type == "execution_complete":
                        break

        except TimeoutError:
            raise TimeoutError(
                f"Job {prompt_id} did not complete within {self.timeout}s"
            )

        if output_path is None:
            raise RuntimeError(f"No output file found for job {prompt_id}")

        return output_path

    def _get_output_dir(self) -> str:
        """Get ComfyUI output directory path."""
        # Default ComfyUI output location
        if self.host == "127.0.0.1":
            return "/home/matt/nsn/ComfyUI/output"
        return "/ComfyUI/output"

    async def generate(
        self,
        workflow: dict[str, Any],
        output_node: str = "99",
    ) -> str:
        """Queue workflow and wait for output.

        Convenience method combining queue_prompt and wait_for_completion.

        Args:
            workflow: ComfyUI workflow dict
            output_node: Node ID for output

        Returns:
            Path to output file
        """
        prompt_id = await self.queue_prompt(workflow)
        return await self.wait_for_completion(prompt_id, output_node)

    async def check_health(self) -> bool:
        """Check if ComfyUI server is reachable.

        Returns:
            True if server responds, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}/system_stats",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"ComfyUI health check failed: {e}")
            return False
