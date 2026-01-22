"""Vortex AI video generation server (ToonGen).

This server provides HTTP JSON API for Lane 0 video generation using ToonGen:
- POST /generate: Generate video from recipe via ComfyUI
- POST /unload: Manually unload models to free GPU memory
- GET /health: Health check endpoint (includes ComfyUI status)
- GET /vram: VRAM status endpoint
- GET /metrics: Prometheus metrics endpoint

The server wraps the VideoOrchestrator and provides a simple HTTP interface
for standalone operation or direct integration testing.

GPU Memory Management:
    The orchestrator manages VRAM handoff between audio and visual models:
    - Audio models (F5-TTS/Kokoro) load for voice generation
    - Audio models unload before ComfyUI visual generation
    - ComfyUI manages its own models (Flux, CogVideoX, etc.)

    Models are loaded lazily on first /generate request and remain in VRAM
    for fast subsequent generations. To free GPU memory for other processes:

    1. Automatic: Models unload after --idle-timeout seconds of inactivity
       (default: 300 seconds / 5 minutes, set to 0 to disable)

    2. Manual: POST /unload to immediately free GPU memory

    Models will automatically reload on the next /generate request.

Example:
    $ python -m vortex.server --host 0.0.0.0 --port 50051 --device cuda:0

    # With 10-minute idle timeout
    $ python -m vortex.server --idle-timeout 600

    # Disable auto-unload (models stay loaded forever)
    $ python -m vortex.server --idle-timeout 0

    $ curl -X POST http://localhost:50051/generate \\
        -H "Content-Type: application/json" \\
        -d '{"recipe": {...}, "slot_id": 1, "seed": 42}'

    # Manually unload models to free GPU
    $ curl -X POST http://localhost:50051/unload
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

import torch

logger = logging.getLogger(__name__)


class VortexServer:
    """HTTP JSON API server for Vortex video generation.

    Provides endpoints for video generation, health checks, and VRAM monitoring.
    Lazily initializes the VideoOrchestrator on first generation request.

    GPU Memory Management:
        - Models load lazily on first /generate request
        - Models unload automatically after idle_timeout_sec of inactivity
        - Manual unload available via /unload endpoint
        - Models reload automatically on next /generate request
    """

    def __init__(
        self,
        device: str,
        output_path: Path,
        idle_timeout_sec: float = 300.0,
        comfy_host: str = "127.0.0.1",
        comfy_port: int = 8188,
        template_path: str = "templates/cartoon_workflow.json",
        assets_dir: str = "assets",
    ):
        """Initialize server.

        Args:
            device: CUDA device for GPU inference (e.g., "cuda:0")
            output_path: Directory for output files
            idle_timeout_sec: Seconds of inactivity before unloading models.
                              Set to 0 to disable auto-unload. Default: 300 (5 min)
            comfy_host: ComfyUI server hostname
            comfy_port: ComfyUI server port
            template_path: Path to ComfyUI workflow template JSON
            assets_dir: Root directory for audio assets
        """
        self.device = device
        self.output_path = output_path
        self.idle_timeout_sec = idle_timeout_sec
        self._comfy_host = comfy_host
        self._comfy_port = comfy_port
        self._template_path = template_path
        self._assets_dir = assets_dir
        self._orchestrator = None
        self._initialized = False
        self._init_error: str | None = None
        self._last_activity_time: float = 0.0
        self._idle_monitor_task: asyncio.Task | None = None
        self._unload_count: int = 0  # Track number of unloads for metrics

    async def ensure_orchestrator(self) -> bool:
        """Lazily initialize the VideoOrchestrator.

        Returns:
            True if orchestrator is ready, False if initialization failed
        """
        # Update activity time
        self._last_activity_time = time.time()

        # If already initialized and orchestrator exists, we're good
        if self._initialized and self._orchestrator is not None:
            return True

        # If previously failed, don't retry
        if self._initialized and self._init_error is not None:
            return False

        try:
            logger.info(f"Initializing VideoOrchestrator on device: {self.device}")
            from vortex.orchestrator import VideoOrchestrator

            self._orchestrator = VideoOrchestrator(
                template_path=self._template_path,
                assets_dir=self._assets_dir,
                output_dir=str(self.output_path),
                comfy_host=self._comfy_host,
                comfy_port=self._comfy_port,
                device=self.device,
            )
            self._initialized = True
            self._init_error = None
            logger.info(
                f"VideoOrchestrator initialized: ComfyUI at {self._comfy_host}:{self._comfy_port}"
            )

            # Start idle monitor if timeout is enabled
            if self.idle_timeout_sec > 0 and self._idle_monitor_task is None:
                self._idle_monitor_task = asyncio.create_task(self._idle_monitor())
                logger.info(
                    f"Idle monitor started (timeout: {self.idle_timeout_sec}s)"
                )

            return True
        except Exception as e:
            self._init_error = str(e)
            self._initialized = True
            logger.error(f"Failed to initialize VideoOrchestrator: {e}")
            return False

    async def unload_orchestrator(self) -> bool:
        """Unload the orchestrator and free GPU memory.

        Returns:
            True if orchestrator was unloaded, False if nothing to unload
        """
        if self._orchestrator is None:
            logger.debug("No orchestrator to unload")
            return False

        try:
            logger.info("Unloading VideoOrchestrator to free GPU memory...")

            # Get VRAM before unload for logging
            if torch.cuda.is_available():
                device_idx = (
                    int(self.device.split(":")[-1]) if ":" in self.device else 0
                )
                vram_before = torch.cuda.memory_allocated(device_idx) / 1e9

            # Delete orchestrator and clear references
            del self._orchestrator
            self._orchestrator = None
            self._initialized = False
            self._init_error = None

            # Force CUDA memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                vram_after = torch.cuda.memory_allocated(device_idx) / 1e9
                freed = vram_before - vram_after
                logger.info(
                    f"Orchestrator unloaded: freed {freed:.2f} GB VRAM "
                    f"({vram_before:.2f} GB -> {vram_after:.2f} GB)"
                )
            else:
                logger.info("Orchestrator unloaded (CUDA not available)")

            self._unload_count += 1
            return True

        except Exception as e:
            logger.error(f"Error unloading orchestrator: {e}")
            return False

    async def _idle_monitor(self) -> None:
        """Background task that monitors for idle timeout and unloads models."""
        logger.debug("Idle monitor task started")
        check_interval = min(30.0, self.idle_timeout_sec / 2)  # Check at least every 30s

        try:
            while True:
                await asyncio.sleep(check_interval)

                # Skip if orchestrator not loaded
                if self._orchestrator is None:
                    continue

                # Check if idle timeout exceeded
                idle_time = time.time() - self._last_activity_time
                if idle_time >= self.idle_timeout_sec:
                    logger.info(
                        f"Idle timeout reached ({idle_time:.0f}s >= {self.idle_timeout_sec}s), "
                        "unloading models..."
                    )
                    await self.unload_orchestrator()

        except asyncio.CancelledError:
            logger.debug("Idle monitor task cancelled")
        except Exception as e:
            logger.error(f"Idle monitor error: {e}")

    def _update_activity(self) -> None:
        """Update last activity timestamp."""
        self._last_activity_time = time.time()

    async def handle_health(self) -> tuple[int, dict[str, Any]]:
        """Handle health check request.

        Returns:
            Tuple of (HTTP status code, response dict)
        """
        if self._orchestrator is not None:
            status = "healthy"
        elif self._init_error:
            status = "unhealthy"
        else:
            status = "idle"  # Not loaded, waiting for first request

        response: dict[str, Any] = {
            "status": status,
            "device": self.device,
            "models_loaded": self._orchestrator is not None,
            "comfyui_host": self._comfy_host,
            "comfyui_port": self._comfy_port,
        }

        # Check ComfyUI health if orchestrator is available
        comfyui_connected = False
        if self._orchestrator is not None:
            try:
                health = await self._orchestrator.health_check()
                comfyui_connected = health.get("comfyui", False)
            except Exception as e:
                logger.warning(f"ComfyUI health check failed: {e}")
                comfyui_connected = False

            # Include idle time info
            idle_time = time.time() - self._last_activity_time
            response["idle_seconds"] = round(idle_time, 1)
            if self.idle_timeout_sec > 0:
                response["idle_timeout_seconds"] = self.idle_timeout_sec
                response["unload_in_seconds"] = max(
                    0, round(self.idle_timeout_sec - idle_time, 1)
                )

        response["comfyui_connected"] = comfyui_connected

        if self._init_error:
            response["error"] = self._init_error

        # Include unload stats
        if self._unload_count > 0:
            response["unload_count"] = self._unload_count

        return HTTPStatus.OK, response

    async def handle_unload(self) -> tuple[int, dict[str, Any]]:
        """Handle manual model unload request.

        Returns:
            Tuple of (HTTP status code, response dict)
        """
        if self._orchestrator is None:
            return HTTPStatus.OK, {
                "success": True,
                "message": "Models already unloaded",
                "models_loaded": False,
            }

        # Get VRAM before for response
        vram_before = 0.0
        if torch.cuda.is_available():
            device_idx = (
                int(self.device.split(":")[-1]) if ":" in self.device else 0
            )
            vram_before = torch.cuda.memory_allocated(device_idx) / 1e9

        success = await self.unload_orchestrator()

        # Get VRAM after
        vram_after = 0.0
        if torch.cuda.is_available():
            vram_after = torch.cuda.memory_allocated(device_idx) / 1e9

        if success:
            return HTTPStatus.OK, {
                "success": True,
                "message": "Models unloaded successfully",
                "models_loaded": False,
                "vram_freed_gb": round(vram_before - vram_after, 2),
                "vram_before_gb": round(vram_before, 2),
                "vram_after_gb": round(vram_after, 2),
            }
        else:
            return HTTPStatus.INTERNAL_SERVER_ERROR, {
                "success": False,
                "message": "Failed to unload models",
                "models_loaded": self._orchestrator is not None,
            }

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
        lines.append(
            "# HELP vortex_orchestrator_initialized Whether the orchestrator is initialized"
        )
        lines.append("# TYPE vortex_orchestrator_initialized gauge")
        lines.append("# HELP vortex_idle_seconds Seconds since last generation activity")
        lines.append("# TYPE vortex_idle_seconds gauge")
        lines.append("# HELP vortex_idle_timeout_seconds Configured idle timeout (0=disabled)")
        lines.append("# TYPE vortex_idle_timeout_seconds gauge")
        lines.append("# HELP vortex_unload_total Total number of model unloads")
        lines.append("# TYPE vortex_unload_total counter")
        lines.append("# HELP vortex_comfyui_connected Whether ComfyUI is connected")
        lines.append("# TYPE vortex_comfyui_connected gauge")

        device_label = f'device="{self.device}"'

        # Orchestrator status
        initialized = 1 if self._orchestrator is not None else 0
        lines.append(f"vortex_orchestrator_initialized{{{device_label}}} {initialized}")

        # ComfyUI connection status
        comfyui_connected = 0
        if self._orchestrator is not None:
            try:
                health = await self._orchestrator.health_check()
                comfyui_connected = 1 if health.get("comfyui", False) else 0
            except Exception:
                pass
        lines.append(f"vortex_comfyui_connected{{{device_label}}} {comfyui_connected}")

        # Idle metrics
        if self._last_activity_time > 0:
            idle_seconds = time.time() - self._last_activity_time
            lines.append(f"vortex_idle_seconds{{{device_label}}} {idle_seconds:.1f}")
        lines.append(f"vortex_idle_timeout_seconds{{{device_label}}} {self.idle_timeout_sec}")
        lines.append(f"vortex_unload_total{{{device_label}}} {self._unload_count}")

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
                - recipe: Generation recipe dict containing:
                    - slot_params: Basic slot parameters
                    - audio_track: Script, voice settings
                    - audio_environment: BGM, SFX, mix_ratio
                    - visual_track: Prompt for image generation
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

        # Ensure orchestrator is initialized
        if not await self.ensure_orchestrator():
            return HTTPStatus.SERVICE_UNAVAILABLE, {
                "error": f"Orchestrator initialization failed: {self._init_error}"
            }

        try:
            start_time = time.time()

            # Extract recipe components
            audio_track = recipe.get("audio_track", {})
            audio_environment = recipe.get("audio_environment")
            visual_track = recipe.get("visual_track", {})

            logger.info(f"Starting generation for slot {slot_id}", extra={"seed": seed})

            # Call orchestrator with extracted parameters
            result = await self._orchestrator.generate(
                prompt=visual_track.get("prompt", ""),
                script=audio_track.get("script", ""),
                voice_style=audio_track.get("voice_style"),
                voice_id=audio_track.get("voice_id", "af_heart"),
                engine=audio_track.get("engine", "auto"),
                bgm_name=audio_environment.get("bgm") if audio_environment else None,
                sfx_name=audio_environment.get("sfx") if audio_environment else None,
                mix_ratio=audio_environment.get("mix_ratio", 0.3) if audio_environment else 0.3,
                seed=seed,
            )

            generation_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Generation completed for slot {slot_id}",
                extra={
                    "generation_time_ms": generation_time_ms,
                    "frame_count": result.get("frame_count"),
                },
            )

            return HTTPStatus.OK, {
                "success": True,
                "slot_id": slot_id,
                "video_path": result["video_path"],
                "clean_audio_path": result["clean_audio_path"],
                "mixed_audio_path": result["mixed_audio_path"],
                "frame_count": result["frame_count"],
                "seed": result["seed"],
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
        elif path == "/unload" and method == "POST":
            return await self.handle_unload()
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
                    "POST /unload",
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
    parser = argparse.ArgumentParser(description="Vortex AI video generation server (ToonGen)")
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
        help="Initialize orchestrator eagerly at startup (default: lazy init on first request)",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=float(os.getenv("VORTEX_IDLE_TIMEOUT", "300")),
        help=(
            "Seconds of inactivity before unloading models to free GPU memory. "
            "Set to 0 to disable auto-unload. (default: 300 = 5 minutes)"
        ),
    )
    parser.add_argument(
        "--comfy-host",
        default=os.getenv("COMFY_HOST", "127.0.0.1"),
        help="ComfyUI server hostname (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--comfy-port",
        type=int,
        default=int(os.getenv("COMFY_PORT", "8188")),
        help="ComfyUI server port (default: 8188)",
    )
    parser.add_argument(
        "--template-path",
        default=os.getenv("VORTEX_TEMPLATE_PATH", "templates/cartoon_workflow.json"),
        help="Path to ComfyUI workflow template JSON",
    )
    parser.add_argument(
        "--assets-dir",
        default=os.getenv("VORTEX_ASSETS_DIR", "assets"),
        help="Root directory for audio assets (default: assets)",
    )
    args = parser.parse_args()
    # Handle deprecated --grpc-port for backward compatibility
    if args.grpc_port is not None:
        logger.warning("--grpc-port is deprecated, use --port instead")
        args.port = args.grpc_port
    return args


async def _run_server(
    host: str,
    port: int,
    device: str,
    output_path: Path,
    eager_init: bool,
    idle_timeout_sec: float,
    comfy_host: str,
    comfy_port: int,
    template_path: str,
    assets_dir: str,
) -> None:
    """Run the HTTP server."""
    server = VortexServer(
        device=device,
        output_path=output_path,
        idle_timeout_sec=idle_timeout_sec,
        comfy_host=comfy_host,
        comfy_port=comfy_port,
        template_path=template_path,
        assets_dir=assets_dir,
    )
    handler = HTTPHandler(server)

    # Eager initialization if requested
    if eager_init:
        logger.info("Performing eager orchestrator initialization...")
        if await server.ensure_orchestrator():
            logger.info("Orchestrator initialized successfully")
        else:
            logger.error(f"Orchestrator initialization failed: {server._init_error}")

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
    logger.info(f"ComfyUI: {args.comfy_host}:{args.comfy_port}")
    logger.info(f"Template: {args.template_path}")
    logger.info(f"Assets: {args.assets_dir}")
    if args.idle_timeout > 0:
        logger.info(f"Idle timeout: {args.idle_timeout}s (models will unload after inactivity)")
    else:
        logger.info("Idle timeout: disabled (models stay loaded until manual unload)")

    asyncio.run(
        _run_server(
            host=args.host,
            port=args.port,
            device=args.device,
            output_path=output_path,
            eager_init=args.eager_init,
            idle_timeout_sec=args.idle_timeout,
            comfy_host=args.comfy_host,
            comfy_port=args.comfy_port,
            template_path=args.template_path,
            assets_dir=args.assets_dir,
        )
    )


if __name__ == "__main__":
    main()
