"""Custom exceptions for Vortex VRAM management and pipeline errors.

This module defines exceptions used throughout the Vortex engine for:
- VRAM memory pressure detection (soft and hard limits)
- Pipeline initialization failures
- Memory leak warnings
"""


class MemoryPressureWarning(UserWarning):
    """Soft VRAM limit exceeded (non-blocking).

    Raised when VRAM usage exceeds the soft limit (default: 11.0GB).
    Generation continues but emergency cleanup may be triggered.

    Attributes:
        current_gb: Current VRAM usage in GB
        soft_limit_gb: Configured soft limit in GB
        delta_gb: Amount over the soft limit

    Example:
        >>> try:
        >>>     check_limits()
        >>> except MemoryPressureWarning as e:
        >>>     logger.warning(f"Soft limit exceeded: {e}")
        >>>     # Continue processing with cleanup
    """

    def __init__(self, current_gb: float, soft_limit_gb: float, delta_gb: float):
        self.current_gb = current_gb
        self.soft_limit_gb = soft_limit_gb
        self.delta_gb = delta_gb
        super().__init__(
            f"VRAM usage {current_gb:.2f}GB exceeds soft limit {soft_limit_gb:.2f}GB "
            f"(+{delta_gb:.2f}GB over)"
        )


class MemoryPressureError(RuntimeError):
    """Hard VRAM limit exceeded (blocking).

    Raised when VRAM usage exceeds the hard limit (default: 11.5GB).
    Generation must be aborted to prevent OOM crash.

    Attributes:
        current_gb: Current VRAM usage in GB
        hard_limit_gb: Configured hard limit in GB
        delta_gb: Amount over the hard limit

    Example:
        >>> try:
        >>>     check_limits()
        >>> except MemoryPressureError as e:
        >>>     logger.error(f"Hard limit exceeded: {e}")
        >>>     abort_generation()
    """

    def __init__(self, current_gb: float, hard_limit_gb: float, delta_gb: float):
        self.current_gb = current_gb
        self.hard_limit_gb = hard_limit_gb
        self.delta_gb = delta_gb
        super().__init__(
            f"VRAM usage {current_gb:.2f}GB exceeds hard limit {hard_limit_gb:.2f}GB "
            f"(+{delta_gb:.2f}GB over) - aborting to prevent OOM"
        )


class VortexInitializationError(RuntimeError):
    """Vortex pipeline pre-flight check failure.

    Raised during VortexPipeline initialization if:
    - Insufficient VRAM (<11.8GB required for all models)
    - Missing required models or dependencies
    - CUDA not available

    Attributes:
        reason: Human-readable description of the failure
        available_gb: Available VRAM in GB (if CUDA available)
        required_gb: Required VRAM in GB for full pipeline

    Example:
        >>> if available_vram < 11.8:
        >>>     raise VortexInitializationError(
        >>>         reason="Insufficient VRAM",
        >>>         available_gb=8.0,
        >>>         required_gb=11.8
        >>>     )
    """

    def __init__(
        self,
        reason: str,
        available_gb: float | None = None,
        required_gb: float | None = None,
    ):
        self.reason = reason
        self.available_gb = available_gb
        self.required_gb = required_gb

        if available_gb is not None and required_gb is not None:
            message = (
                f"{reason}: {available_gb:.1f}GB available, "
                f"{required_gb:.1f}GB required. "
                "Upgrade to RTX 3060 12GB or higher."
            )
        else:
            message = reason

        super().__init__(message)


class MemoryLeakWarning(UserWarning):
    """Potential memory leak detected over time.

    Raised when VRAM usage grows significantly over multiple generations
    (default: >100MB delta after 100 generations).

    Attributes:
        initial_gb: Baseline VRAM usage in GB
        current_gb: Current VRAM usage in GB
        delta_mb: Memory growth in MB
        generations: Number of generations elapsed

    Example:
        >>> if delta_mb > 100:
        >>>     raise MemoryLeakWarning(
        >>>         initial_gb=10.8,
        >>>         current_gb=11.0,
        >>>         delta_mb=200,
        >>>         generations=100
        >>>     )
    """

    def __init__(
        self,
        initial_gb: float,
        current_gb: float,
        delta_mb: float,
        generations: int,
    ):
        self.initial_gb = initial_gb
        self.current_gb = current_gb
        self.delta_mb = delta_mb
        self.generations = generations
        super().__init__(
            f"Potential memory leak: {initial_gb:.2f}GB → {current_gb:.2f}GB "
            f"(+{delta_mb:.1f}MB) over {generations} generations"
        )
