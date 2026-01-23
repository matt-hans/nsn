"""Sandboxed execution helpers for Vortex plugins."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from vortex.plugins.errors import PluginExecutionError
from vortex.plugins.types import PluginManifest

LOG = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram
except ImportError:  # pragma: no cover - optional metrics
    Counter = None
    Histogram = None

if Counter is not None and Histogram is not None:
    SANDBOX_RUNS = Counter(
        "vortex_plugin_sandbox_runs_total",
        "Total number of sandbox executions",
        ["engine"],
    )
    SANDBOX_FAILURES = Counter(
        "vortex_plugin_sandbox_failures_total",
        "Total sandbox execution failures",
        ["engine", "reason"],
    )
    SANDBOX_DURATION = Histogram(
        "vortex_plugin_sandbox_duration_ms",
        "Sandbox execution duration in ms",
        ["engine"],
    )
else:  # pragma: no cover - metrics disabled
    SANDBOX_RUNS = None
    SANDBOX_FAILURES = None
    SANDBOX_DURATION = None


@dataclass(frozen=True)
class SandboxResult:
    """Structured result returned by a sandbox runner."""

    output: dict[str, Any]
    duration_ms: float
    stdout: str
    stderr: str
    exit_code: int
    engine: str


class SandboxRunner(Protocol):
    """Protocol for sandbox execution backends."""

    engine: str

    def is_available(self) -> bool:
        """Return True if the sandbox backend is available."""

    async def run(
        self,
        manifest: PluginManifest,
        plugin_root: Path,
        payload: Mapping[str, Any],
        budget_ms: int,
    ) -> SandboxResult:
        """Execute a plugin inside a sandbox and return the structured result."""


@dataclass(frozen=True)
class SandboxConfig:
    enabled: bool
    engine: str
    docker_image: str
    docker_bin: str
    network: str
    memory_mb: int
    cpu_cores: float
    pids_limit: int
    tmpfs_mb: int
    gpus: str | None
    timeout_grace_ms: int
    allow_insecure: bool
    seccomp_profile: str | None

    @classmethod
    def from_dict(cls, raw: Mapping[str, object] | None) -> SandboxConfig:
        data = raw or {}
        enabled = bool(data.get("enabled", True))
        engine = str(data.get("engine", "docker")).lower()
        docker_image = str(data.get("docker_image", "nsn-vortex:latest"))
        docker_bin = str(data.get("docker_bin", "docker"))
        network = str(data.get("network", "none"))
        memory_mb = int(data.get("memory_mb", 4096))
        cpu_cores = float(data.get("cpu_cores", 2.0))
        pids_limit = int(data.get("pids_limit", 256))
        tmpfs_mb = int(data.get("tmpfs_mb", 64))
        gpus = data.get("gpus")
        gpus_value = str(gpus) if isinstance(gpus, (str, int, float)) else None
        timeout_grace_ms = int(data.get("timeout_grace_ms", 5000))
        allow_insecure = bool(data.get("allow_insecure", False)) or bool(
            os.getenv("VORTEX_ALLOW_UNSAFE_SANDBOX")
        )
        seccomp_profile = data.get("seccomp_profile")
        seccomp_value = str(seccomp_profile) if isinstance(seccomp_profile, str) else "default"

        if memory_mb <= 0 or cpu_cores <= 0 or pids_limit <= 0:
            raise PluginExecutionError("Sandbox resource limits must be > 0")
        return cls(
            enabled=enabled,
            engine=engine,
            docker_image=docker_image,
            docker_bin=docker_bin,
            network=network,
            memory_mb=memory_mb,
            cpu_cores=cpu_cores,
            pids_limit=pids_limit,
            tmpfs_mb=tmpfs_mb,
            gpus=gpus_value,
            timeout_grace_ms=timeout_grace_ms,
            allow_insecure=allow_insecure,
            seccomp_profile=seccomp_value,
        )


def sandbox_from_config(config: Mapping[str, object] | None) -> SandboxRunner | None:
    if config is None:
        return None
    cfg = SandboxConfig.from_dict(config)
    if not cfg.enabled:
        return None
    if cfg.engine == "docker":
        return DockerSandboxRunner(cfg)
    if cfg.engine == "process":
        if not cfg.allow_insecure:
            raise PluginExecutionError(
                "process sandbox is disabled; set VORTEX_ALLOW_UNSAFE_SANDBOX=1 to override"
            )
        return ProcessSandboxRunner.from_config(config)
    raise PluginExecutionError(f"Unsupported sandbox engine '{cfg.engine}'")


def _manifest_path(plugin_root: Path) -> Path:
    yaml_path = plugin_root / "manifest.yaml"
    yml_path = plugin_root / "manifest.yml"
    if yaml_path.exists():
        return yaml_path
    if yml_path.exists():
        return yml_path
    raise PluginExecutionError(f"Plugin manifest not found in {plugin_root}")


def _default_pythonpath() -> str:
    return str(Path(__file__).resolve().parents[2])


class ProcessSandboxRunner:
    """Executes a plugin in a separate local process (non-isolated)."""

    engine = "process"

    def __init__(self, python_bin: str | None = None) -> None:
        self._python_bin = python_bin or sys.executable

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None) -> ProcessSandboxRunner:
        data = config or {}
        python_bin = str(data.get("python_bin", sys.executable))
        return cls(python_bin=python_bin)

    def is_available(self) -> bool:
        return shutil.which(self._python_bin) is not None

    async def run(
        self,
        manifest: PluginManifest,
        plugin_root: Path,
        payload: Mapping[str, Any],
        budget_ms: int,
    ) -> SandboxResult:
        plugin_root = plugin_root.resolve()
        if not plugin_root.exists():
            raise PluginExecutionError(f"Plugin directory not found: {plugin_root}")
        payload_str = json.dumps(dict(payload))
        manifest_path = _manifest_path(plugin_root)
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        base = _default_pythonpath()
        env["PYTHONPATH"] = f"{base}:{pythonpath}" if pythonpath else base

        cmd = [
            self._python_bin,
            "-m",
            "vortex.plugins.sandbox_runner",
            "--entrypoint",
            manifest.entrypoint,
            "--plugin-dir",
            str(plugin_root),
            "--manifest",
            str(manifest_path),
            "--payload",
            payload_str,
            "--timeout-ms",
            str(budget_ms),
        ]

        LOG.info(
            "Sandbox exec (process) plugin=%s timeout_ms=%s",
            manifest.name,
            budget_ms,
        )
        if SANDBOX_RUNS is not None:
            SANDBOX_RUNS.labels(engine=self.engine).inc()
        start_time = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=(budget_ms / 1000) + 1
            )
        except TimeoutError as exc:
            proc.kill()
            if SANDBOX_FAILURES is not None:
                SANDBOX_FAILURES.labels(engine=self.engine, reason="timeout").inc()
            raise PluginExecutionError(
                f"Sandbox timeout for plugin '{manifest.name}'"
            ) from exc

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            if SANDBOX_FAILURES is not None:
                SANDBOX_FAILURES.labels(engine=self.engine, reason="exit").inc()
            raise PluginExecutionError(
                f"Sandbox failed for plugin '{manifest.name}': {stderr.strip()}"
            )

        result = _parse_sandbox_output(stdout, stderr, proc.returncode, self.engine)
        if SANDBOX_DURATION is not None:
            SANDBOX_DURATION.labels(engine=self.engine).observe(
                (time.monotonic() - start_time) * 1000
            )
        return result


class DockerSandboxRunner:
    """Executes a plugin inside a Docker container sandbox."""

    engine = "docker"

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._available = self._probe()

    def _probe(self) -> bool:
        if shutil.which(self._config.docker_bin) is None:
            return False
        try:
            subprocess.run(
                [self._config.docker_bin, "version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=2,
            )
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        return self._available

    async def run(
        self,
        manifest: PluginManifest,
        plugin_root: Path,
        payload: Mapping[str, Any],
        budget_ms: int,
    ) -> SandboxResult:
        if not self.is_available():
            raise PluginExecutionError("Docker sandbox is not available")

        plugin_root = plugin_root.resolve()
        if not plugin_root.exists():
            raise PluginExecutionError(f"Plugin directory not found: {plugin_root}")
        payload_str = json.dumps(dict(payload))
        manifest_path = _manifest_path(plugin_root)
        container_name = f"vortex-plugin-{manifest.name}-{uuid.uuid4().hex[:8]}"
        mount_spec = f"type=bind,src={plugin_root},dst=/plugin,readonly"
        tmpfs_spec = f"/tmp:rw,noexec,nosuid,size={self._config.tmpfs_mb}m"

        cmd = [
            self._config.docker_bin,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            self._config.network,
            "--read-only",
            "--pids-limit",
            str(self._config.pids_limit),
            "--memory",
            f"{self._config.memory_mb}m",
            "--cpus",
            str(self._config.cpu_cores),
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--tmpfs",
            tmpfs_spec,
            "--mount",
            mount_spec,
            "-w",
            "/plugin",
        ]

        if self._config.seccomp_profile:
            cmd.extend(["--security-opt", f"seccomp={self._config.seccomp_profile}"])

        if self._config.gpus:
            cmd.extend(["--gpus", self._config.gpus])

        cmd.extend(
            [
                self._config.docker_image,
                "python3",
                "-m",
                "vortex.plugins.sandbox_runner",
                "--entrypoint",
                manifest.entrypoint,
                "--plugin-dir",
                "/plugin",
                "--manifest",
                "/plugin/" + manifest_path.name,
                "--payload",
                payload_str,
                "--timeout-ms",
                str(budget_ms),
            ]
        )

        LOG.info(
            "Sandbox exec (docker) plugin=%s image=%s timeout_ms=%s",
            manifest.name,
            self._config.docker_image,
            budget_ms,
        )
        if SANDBOX_RUNS is not None:
            SANDBOX_RUNS.labels(engine=self.engine).inc()
        start_time = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        timeout = (budget_ms + self._config.timeout_grace_ms) / 1000
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError as exc:
            await self._kill_container(container_name)
            if SANDBOX_FAILURES is not None:
                SANDBOX_FAILURES.labels(engine=self.engine, reason="timeout").inc()
            raise PluginExecutionError(
                f"Sandbox timeout for plugin '{manifest.name}'"
            ) from exc

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            if SANDBOX_FAILURES is not None:
                SANDBOX_FAILURES.labels(engine=self.engine, reason="exit").inc()
            raise PluginExecutionError(
                f"Sandbox failed for plugin '{manifest.name}': {stderr.strip()}"
            )

        result = _parse_sandbox_output(stdout, stderr, proc.returncode, self.engine)
        if SANDBOX_DURATION is not None:
            SANDBOX_DURATION.labels(engine=self.engine).observe(
                (time.monotonic() - start_time) * 1000
            )
        return result

    async def _kill_container(self, name: str) -> None:
        docker = self._config.docker_bin
        for cmd in ([docker, "kill", name], [docker, "rm", "-f", name]):
            proc = await asyncio.create_subprocess_exec(*cmd)
            await proc.wait()


def _parse_sandbox_output(
    stdout: str,
    stderr: str,
    exit_code: int,
    engine: str,
) -> SandboxResult:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise PluginExecutionError("Sandbox returned no output")
    payload = lines[-1]
    try:
        response = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PluginExecutionError("Sandbox returned invalid JSON") from exc

    output = response.get("output")
    duration_ms = response.get("duration_ms", 0.0)
    if not isinstance(output, dict):
        raise PluginExecutionError("Sandbox output missing 'output' object")

    try:
        duration_value = float(duration_ms)
    except (TypeError, ValueError) as exc:
        raise PluginExecutionError("Sandbox returned invalid duration_ms") from exc

    return SandboxResult(
        output=output,
        duration_ms=duration_value,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        engine=engine,
    )
