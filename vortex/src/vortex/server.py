"""Vortex AI video generation server.

This server provides HTTP JSON API for Lane 0 video generation:
- POST /generate: Generate video from recipe
- GET /health: Health check endpoint
- GET /vram: VRAM status endpoint
- GET /metrics: Prometheus metrics endpoint

The server wraps the VortexPipeline and provides a simple HTTP interface
for standalone operation or direct integration testing.

Example:
    $ python -m vortex.server --host 0.0.0.0 --port 50051 --device cuda:0

    $ curl -X POST http://localhost:50051/generate \\
        -H "Content-Type: application/json" \\
        -d '{"recipe": {...}, "slot_id": 1, "seed": 42}'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import traceback
from http import HTTPStatus
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VortexServer:
    """HTTP JSON API server for Vortex video generation.

    Provides endpoints for video generation, health checks, and VRAM monitoring.
    Lazily initializes the VortexPipeline on first generation request.
    """

    def __init__(self, device: str, output_path: Path):
        """Initialize server.

        Args:
            device: CUDA device for GPU inference (e.g., "cuda:0")
            output_path: Directory for output files
        """
        self.device = device
        self.output_path = output_path
        self._pipeline = None
        self._initialized = False
        self._init_error: str | None = None

    async def ensure_pipeline(self) -> bool:
        """Lazily initialize the VortexPipeline.

        Returns:
            True if pipeline is ready, False if initialization failed
        """
        if self._initialized:
            return self._pipeline is not None

        try:
            logger.info(f"Initializing VortexPipeline on device: {self.device}")
            from vortex.pipeline import VortexPipeline

            self._pipeline = await VortexPipeline.create(device=self.device)
            self._initialized = True
            logger.info(
                f"VortexPipeline initialized with renderer: {self._pipeline.renderer_name}"
            )
            return True
        except Exception as e:
            self._init_error = str(e)
            self._initialized = True
            logger.error(f"Failed to initialize VortexPipeline: {e}")
            return False

    async def handle_health(self) -> tuple[int, dict[str, Any]]:
        """Handle health check request.

        Returns:
            Tuple of (HTTP status code, response dict)
        """
        status = "healthy" if self._pipeline is not None else "initializing"
        if self._init_error:
            status = "unhealthy"

        response = {
            "status": status,
            "device": self.device,
            "initialized": self._initialized,
        }

        if self._pipeline is not None:
            response["renderer"] = self._pipeline.renderer_name
            response["renderer_version"] = self._pipeline.renderer_version

        if self._init_error:
            response["error"] = self._init_error

        return HTTPStatus.OK, response

    async def handle_vram(self) -> tuple[int, dict[str, Any]]:
        """Handle VRAM status request.

        Returns:
            Tuple of (HTTP status code, response dict)
        """
        if not torch.cuda.is_available():
            return HTTPStatus.OK, {
                "available": False,
                "reason": "CUDA not available",
            }

        try:
            device_idx = (
                int(self.device.split(":")[-1]) if ":" in self.device else 0
            )
            props = torch.cuda.get_device_properties(device_idx)
            allocated = torch.cuda.memory_allocated(device_idx)
            reserved = torch.cuda.memory_reserved(device_idx)
            total = props.total_memory

            return HTTPStatus.OK, {
                "available": True,
                "device": self.device,
                "gpu_name": props.name,
                "total_gb": total / 1e9,
                "allocated_gb": allocated / 1e9,
                "reserved_gb": reserved / 1e9,
                "free_gb": (total - reserved) / 1e9,
            }
        except Exception as e:
            return HTTPStatus.INTERNAL_SERVER_ERROR, {
                "available": False,
                "error": str(e),
            }

    async def handle_metrics(self) -> tuple[int, str]:
        """Handle Prometheus metrics request.

        Returns Prometheus-formatted metrics for GPU monitoring.

        Returns:
            Tuple of (HTTP status code, response text)
        """
        lines = []

        # Add metric metadata
        lines.append("# HELP nvidia_gpu_memory_used_bytes GPU memory currently used in bytes")
        lines.append("# TYPE nvidia_gpu_memory_used_bytes gauge")
        lines.append("# HELP nvidia_gpu_memory_total_bytes Total GPU memory in bytes")
        lines.append("# TYPE nvidia_gpu_memory_total_bytes gauge")
        lines.append("# HELP nvidia_gpu_temperature_celsius GPU temperature in Celsius")
        lines.append("# TYPE nvidia_gpu_temperature_celsius gauge")
        lines.append("# HELP vortex_pipeline_initialized Whether the pipeline is initialized")
        lines.append("# TYPE vortex_pipeline_initialized gauge")

        # Pipeline status
        initialized = 1 if self._pipeline is not None else 0
        lines.append(f'vortex_pipeline_initialized{{device="{self.device}"}} {initialized}')

        if torch.cuda.is_available():
            try:
                device_idx = (
                    int(self.device.split(":")[-1]) if ":" in self.device else 0
                )
                props = torch.cuda.get_device_properties(device_idx)
                allocated = torch.cuda.memory_allocated(device_idx)
                total = props.total_memory
                gpu_name = props.name.replace('"', '\\"')

                # Memory metrics
                gpu_label = f'gpu="{device_idx}",name="{gpu_name}"'
                lines.append(f"nvidia_gpu_memory_used_bytes{{{gpu_label}}} {allocated}")
                lines.append(f"nvidia_gpu_memory_total_bytes{{{gpu_label}}} {total}")

                # Try to get temperature via pynvml
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    lines.append(f"nvidia_gpu_temperature_celsius{{{gpu_label}}} {temp}")
                    pynvml.nvmlShutdown()
                except Exception:
                    # pynvml not available or failed - skip temperature metric
                    pass

            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")

        return HTTPStatus.OK, "\n".join(lines) + "\n"

    async def handle_generate(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Handle video generation request.

        Args:
            body: Request body with keys:
                - recipe: Generation recipe dict
                - slot_id: Unique slot identifier
                - seed: Optional deterministic seed

        Returns:
            Tuple of (HTTP status code, response dict)
        """
        # Validate request
        recipe = body.get("recipe")
        if not isinstance(recipe, dict):
            return HTTPStatus.BAD_REQUEST, {"error": "recipe must be a dict"}

        slot_id = body.get("slot_id")
        if not isinstance(slot_id, int):
            return HTTPStatus.BAD_REQUEST, {"error": "slot_id must be an integer"}

        seed = body.get("seed")
        if seed is not None and not isinstance(seed, int):
            return HTTPStatus.BAD_REQUEST, {"error": "seed must be an integer or null"}

        # Ensure pipeline is initialized
        if not await self.ensure_pipeline():
            return HTTPStatus.SERVICE_UNAVAILABLE, {
                "error": f"Pipeline initialization failed: {self._init_error}"
            }

        try:
            start_time = time.time()

            logger.info(f"Starting generation for slot {slot_id}", extra={"seed": seed})

            result = await self._pipeline.generate_slot(
                recipe=recipe, slot_id=slot_id, seed=seed
            )

            if not result.success:
                return HTTPStatus.INTERNAL_SERVER_ERROR, {
                    "error": result.error_msg,
                    "slot_id": slot_id,
                }

            # Save outputs
            video_path = self.output_path / f"slot_{slot_id}_video.npy"
            audio_path = self.output_path / f"slot_{slot_id}_audio.npy"

            video_np = result.video_frames.cpu().numpy()
            audio_np = result.audio_waveform.cpu().numpy()
            clip_np = result.clip_embedding.cpu().numpy()

            np.save(video_path, video_np)
            np.save(audio_path, audio_np)

            generation_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Generation completed for slot {slot_id}",
                extra={
                    "generation_time_ms": generation_time_ms,
                    "proof": result.determinism_proof.hex()[:16],
                },
            )

            return HTTPStatus.OK, {
                "success": True,
                "slot_id": slot_id,
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "video_shape": list(video_np.shape),
                "audio_samples": len(audio_np),
                "clip_embedding_shape": list(clip_np.shape),
                "determinism_proof": result.determinism_proof.hex(),
                "generation_time_ms": generation_time_ms,
            }

        except Exception as e:
            logger.error(f"Generation failed for slot {slot_id}: {e}")
            return HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    async def handle_request(
        self, method: str, path: str, body: bytes
    ) -> tuple[int, dict[str, Any]]:
        """Route and handle HTTP request.

        Args:
            method: HTTP method (GET, POST)
            path: Request path
            body: Request body bytes

        Returns:
            Tuple of (HTTP status code, response dict)
        """
        if path == "/health" and method == "GET":
            return await self.handle_health()
        elif path == "/vram" and method == "GET":
            return await self.handle_vram()
        elif path == "/metrics" and method == "GET":
            # Metrics returns (status, str) not (status, dict)
            return await self.handle_metrics()  # type: ignore[return-value]
        elif path == "/generate" and method == "POST":
            try:
                parsed_body = json.loads(body) if body else {}
            except json.JSONDecodeError as e:
                return HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {e}"}
            return await self.handle_generate(parsed_body)
        else:
            return HTTPStatus.NOT_FOUND, {
                "error": f"Unknown endpoint: {method} {path}",
                "available_endpoints": [
                    "GET /health",
                    "GET /vram",
                    "GET /metrics",
                    "POST /generate",
                ],
            }


class HTTPHandler:
    """Simple HTTP request handler for asyncio."""

    def __init__(self, server: VortexServer):
        self.server = server

    async def handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming HTTP connection."""
        try:
            # Read request line
            request_line = await reader.readline()
            if not request_line:
                return

            parts = request_line.decode().strip().split()
            if len(parts) < 2:
                return

            method = parts[0]
            path = parts[1].split("?")[0]  # Strip query string

            # Read headers
            content_length = 0
            while True:
                line = await reader.readline()
                if line == b"\r\n" or line == b"\n" or not line:
                    break
                header = line.decode().strip().lower()
                if header.startswith("content-length:"):
                    content_length = int(header.split(":")[1].strip())

            # Read body
            body = b""
            if content_length > 0:
                body = await reader.read(content_length)

            # Handle request
            status, response = await self.server.handle_request(method, path, body)

            # Send response - metrics returns text/plain, others return JSON
            if path == "/metrics":
                response_body = response  # Already a string
                content_type = "text/plain; charset=utf-8"
            else:
                response_body = json.dumps(response)
                content_type = "application/json"

            http_response = (
                f"HTTP/1.1 {status} {status.phrase}\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Length: {len(response_body)}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
                f"{response_body}"
            )

            writer.write(http_response.encode())
            await writer.drain()

        except Exception as e:
            logger.error(f"Error handling request: {e}")
        finally:
            writer.close()
            await writer.wait_closed()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vortex AI video generation server")
    parser.add_argument(
        "--host",
        default=os.getenv("VORTEX_HOST", "0.0.0.0"),
        help="Host address to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("VORTEX_PORT", "50051")),
        help="Port to listen on (default: 50051)",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=None,
        help="Deprecated: use --port instead (kept for backward compatibility)",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("VORTEX_DEVICE", "cuda:0"),
        help="CUDA device for GPU inference (default: cuda:0)",
    )
    parser.add_argument(
        "--models-path",
        default=os.getenv("VORTEX_MODELS_PATH", "/models"),
        help="Path to model weights directory (default: /models)",
    )
    parser.add_argument(
        "--output-path",
        default=os.getenv("VORTEX_OUTPUT_PATH", "/output"),
        help="Path to output directory (default: /output)",
    )
    parser.add_argument(
        "--eager-init",
        action="store_true",
        help="Initialize pipeline eagerly at startup (default: lazy init on first request)",
    )
    args = parser.parse_args()
    # Handle deprecated --grpc-port for backward compatibility
    if args.grpc_port is not None:
        logger.warning("--grpc-port is deprecated, use --port instead")
        args.port = args.grpc_port
    return args


async def _run_server(
    host: str, port: int, device: str, output_path: Path, eager_init: bool
) -> None:
    """Run the HTTP server."""
    server = VortexServer(device=device, output_path=output_path)
    handler = HTTPHandler(server)

    # Eager initialization if requested
    if eager_init:
        logger.info("Performing eager pipeline initialization...")
        if await server.ensure_pipeline():
            logger.info("Pipeline initialized successfully")
        else:
            logger.error(f"Pipeline initialization failed: {server._init_error}")

    tcp_server = await asyncio.start_server(
        handler.handle_connection, host=host, port=port
    )

    addresses = ", ".join(str(sock.getsockname()) for sock in tcp_server.sockets or [])
    logger.info(f"Vortex server listening on {addresses}")

    async with tcp_server:
        await tcp_server.serve_forever()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args()

    models_path = Path(args.models_path)
    output_path = Path(args.output_path)

    models_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Vortex server starting: host={args.host} port={args.port} device={args.device}"
    )
    logger.info(f"Models path: {models_path}")
    logger.info(f"Output path: {output_path}")

    asyncio.run(
        _run_server(
            host=args.host,
            port=args.port,
            device=args.device,
            output_path=output_path,
            eager_init=args.eager_init,
        )
    )


if __name__ == "__main__":
    main()
