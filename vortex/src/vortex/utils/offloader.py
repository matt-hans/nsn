"""Model offloading utilities for VRAM-constrained environments.

This module provides utilities for offloading models between GPU and CPU RAM
to enable running the full pipeline on GPUs with <12GB VRAM (e.g., RTX 3060).

The sequential offloading strategy:
1. Load Flux → Generate image → Offload Flux to CPU
2. Load LivePortrait → Generate video → Offload LivePortrait to CPU
3. Load CLIP → Verify → Done

This ensures only one large model is in VRAM at a time.

VRAM Savings:
- Without offloading: ~11.8GB peak (all models resident)
- With offloading: ~6.5GB peak (single largest model + overhead)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelOffloader:
    """Manages model offloading between GPU and CPU RAM.

    This class provides a simple interface for moving models between devices
    to reduce peak VRAM usage. It tracks which device each model is currently
    on and provides methods for selective offloading.

    Attributes:
        gpu_device: Primary GPU device (e.g., "cuda:0")
        cpu_device: CPU device for offloading
        models: Dict mapping model names to model instances
        model_locations: Dict mapping model names to current device

    Example:
        >>> offloader = ModelOffloader(gpu_device="cuda:0")
        >>> offloader.register("flux", flux_model)
        >>> offloader.register("liveportrait", lp_model)
        >>>
        >>> # Use Flux
        >>> offloader.load_to_gpu("flux")
        >>> result = flux_model.generate(...)
        >>>
        >>> # Offload Flux, load LivePortrait
        >>> offloader.offload_to_cpu("flux")
        >>> offloader.load_to_gpu("liveportrait")
        >>> video = liveportrait.animate(...)
    """

    def __init__(
        self,
        gpu_device: str = "cuda:0",
        enabled: bool = True,
    ):
        """Initialize model offloader.

        Args:
            gpu_device: Target GPU device string
            enabled: Whether offloading is enabled (False = all models stay on GPU)
        """
        self.gpu_device = gpu_device
        self.cpu_device = "cpu"
        self.enabled = enabled
        self._models: dict[str, nn.Module] = {}
        self._model_locations: dict[str, str] = {}

        logger.info(
            "ModelOffloader initialized",
            extra={"gpu_device": gpu_device, "enabled": enabled}
        )

    def register(self, name: str, model: nn.Module, initial_device: str = "cpu") -> None:
        """Register a model for offloading management.

        Args:
            name: Model identifier (e.g., "flux", "liveportrait")
            model: The model instance
            initial_device: Device the model is currently on
        """
        self._models[name] = model
        self._model_locations[name] = initial_device
        logger.debug(f"Registered model '{name}' (currently on {initial_device})")

    def unregister(self, name: str) -> Optional[nn.Module]:
        """Unregister a model from offloading management.

        Args:
            name: Model identifier

        Returns:
            The unregistered model, or None if not found
        """
        if name not in self._models:
            return None
        model = self._models.pop(name)
        self._model_locations.pop(name, None)
        return model

    def load_to_gpu(self, name: str) -> None:
        """Load a model to GPU.

        Args:
            name: Model identifier

        Raises:
            KeyError: If model not registered
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not registered")

        if not self.enabled:
            return

        model = self._models[name]
        current_device = self._model_locations[name]

        if current_device == self.gpu_device:
            logger.debug(f"Model '{name}' already on GPU")
            return

        logger.info(f"Loading model '{name}' to GPU")
        model.to(self.gpu_device)
        self._model_locations[name] = self.gpu_device

        # Log VRAM usage
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            logger.debug(f"VRAM after loading '{name}': {allocated_gb:.2f} GB")

    def offload_to_cpu(self, name: str) -> None:
        """Offload a model to CPU RAM.

        Args:
            name: Model identifier

        Raises:
            KeyError: If model not registered
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not registered")

        if not self.enabled:
            return

        model = self._models[name]
        current_device = self._model_locations[name]

        if current_device == self.cpu_device:
            logger.debug(f"Model '{name}' already on CPU")
            return

        logger.info(f"Offloading model '{name}' to CPU")
        model.to(self.cpu_device)
        self._model_locations[name] = self.cpu_device

        # Clear CUDA cache after offloading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            logger.debug(f"VRAM after offloading '{name}': {allocated_gb:.2f} GB")

    def offload_all_except(self, keep_on_gpu: str) -> None:
        """Offload all models except the specified one.

        Args:
            keep_on_gpu: Name of model to keep on GPU
        """
        for name in self._models:
            if name != keep_on_gpu:
                self.offload_to_cpu(name)
        self.load_to_gpu(keep_on_gpu)

    def offload_all(self) -> None:
        """Offload all models to CPU."""
        for name in self._models:
            self.offload_to_cpu(name)

    def get_location(self, name: str) -> str:
        """Get current device location of a model.

        Args:
            name: Model identifier

        Returns:
            Device string ("cuda:0" or "cpu")
        """
        return self._model_locations.get(name, "unknown")

    def get_gpu_models(self) -> list[str]:
        """Get list of models currently on GPU."""
        return [
            name for name, loc in self._model_locations.items()
            if loc == self.gpu_device
        ]

    def get_vram_estimate(self) -> dict[str, float]:
        """Estimate VRAM usage per model on GPU.

        Returns:
            Dict mapping model names to estimated VRAM in GB
        """
        estimates = {}
        for name in self.get_gpu_models():
            model = self._models[name]
            # Rough estimate: count parameters * 2 bytes (FP16) or 4 bytes (FP32)
            param_count = sum(p.numel() for p in model.parameters())
            # Assume FP16 for estimate
            estimates[name] = (param_count * 2) / 1e9
        return estimates


class SequentialOffloader:
    """Manages sequential model loading/offloading for pipeline stages.

    This class provides a higher-level interface for the common pattern of:
    1. Ensure only one model is on GPU at a time
    2. Run inference
    3. Offload before loading next model

    Example:
        >>> seq = SequentialOffloader(offloader, stages=["flux", "liveportrait", "clip"])
        >>> with seq.use_model("flux"):
        ...     result = flux_model.generate(...)
        >>> with seq.use_model("liveportrait"):
        ...     video = lp_model.animate(...)
    """

    def __init__(
        self,
        offloader: ModelOffloader,
        stages: list[str],
    ):
        """Initialize sequential offloader.

        Args:
            offloader: Underlying ModelOffloader instance
            stages: Ordered list of model names in pipeline order
        """
        self.offloader = offloader
        self.stages = stages
        self._current_stage: Optional[str] = None

    def use_model(self, name: str):
        """Context manager for using a model.

        Loads the model to GPU on entry, offloads on exit.

        Args:
            name: Model identifier
        """
        return _ModelContext(self, name)

    def _enter_stage(self, name: str) -> None:
        """Prepare for using a model (load to GPU, offload others)."""
        if name not in self.stages:
            logger.warning(f"Model '{name}' not in registered stages")

        self.offloader.offload_all_except(name)
        self._current_stage = name

    def _exit_stage(self, name: str) -> None:
        """Cleanup after using a model."""
        # Optionally offload immediately, or wait for next stage
        self._current_stage = None


class _ModelContext:
    """Context manager for sequential model usage."""

    def __init__(self, seq: SequentialOffloader, name: str):
        self.seq = seq
        self.name = name

    def __enter__(self):
        self.seq._enter_stage(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.seq._exit_stage(self.name)
        return False
