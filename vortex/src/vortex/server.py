"""Minimal Vortex gRPC placeholder server.

This keeps the container alive and listening on the gRPC port until the
full service implementation lands.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vortex placeholder server")
    parser.add_argument("--models-path", default=os.getenv("VORTEX_MODELS_PATH", "/models"))
    parser.add_argument("--output-path", default=os.getenv("VORTEX_OUTPUT_PATH", "/output"))
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=int(os.getenv("VORTEX_GRPC_PORT", "50051")),
    )
    return parser.parse_args()


async def _handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    _ = reader
    writer.close()
    await writer.wait_closed()


async def _serve(port: int) -> None:
    server = await asyncio.start_server(_handle_connection, host="0.0.0.0", port=port)
    addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    logging.info("Vortex placeholder listening on %s", addresses)
    async with server:
        await server.serve_forever()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Path(args.models_path).mkdir(parents=True, exist_ok=True)
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    logging.warning("Vortex gRPC server not implemented; running placeholder only.")
    asyncio.run(_serve(args.grpc_port))


if __name__ == "__main__":
    main()
