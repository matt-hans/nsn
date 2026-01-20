"""Model loader factory functions for Vortex pipeline.

Provides lazy-loading factory functions for:
- Flux-Schnell (NF4 quantized image generation)
- LivePortrait (FP16 video warping)
- Kokoro-82M (FP32 text-to-speech)
- CLIP-ViT-B-32 (INT8 semantic verification)
- CLIP-ViT-L-14 (INT8 semantic verification)

Each function returns a torch.nn.Module loaded to the specified device with
appropriate precision/quantization.

Note: Actual model implementations will be added in T015-T018.
      This module provides the interface for T014 (core pipeline).
"""

import logging
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ModelName = Literal["flux", "liveportrait", "kokoro", "clip_b", "clip_l"]
Precision = Literal["fp32", "fp16", "int8", "nf4"]


class MockModel(nn.Module):
    """Mock model for testing pipeline without real model weights.

    This is a placeholder used in unit tests and during T014 development.
    Real models will be implemented in T015-T018.
    """

    def __init__(self, name: str, vram_gb: float):
        super().__init__()
        self.name = name
        self.vram_gb = vram_gb
        # Allocate dummy parameter to simulate VRAM usage
        # 1GB = 268435456 float32 params (4 bytes each)
        param_count = int(vram_gb * 268435456)
        self.dummy_weight = nn.Parameter(torch.randn(param_count, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dummy forward pass."""
        return x


def load_flux(
    device: str = "cuda:0",
    precision: Precision = "nf4",
    cache_dir: str | None = None,
    local_only: bool = False,
) -> nn.Module:
    """Load Flux-Schnell image generation model.

    Args:
        device: Target device (e.g., "cuda:0", "cpu")
        precision: Model precision ("nf4" for 4-bit quantization)
            Note: Flux-Schnell only supports NF4. Other precisions will use mock model.

    Returns:
        nn.Module: Loaded Flux model with NF4 quantization

    VRAM Budget:
        ~6.0 GB with NF4 quantization

    Example:
        >>> flux = load_flux(device="cuda:0", precision="nf4")
        >>> image = flux.generate(prompt="a scientist")
    """
    logger.info("Loading Flux-Schnell model", extra={"device": device, "precision": precision})

    if precision != "nf4":
        raise ValueError(
            f"Flux-Schnell only supports NF4 quantization. Got precision={precision}."
        )

    try:
        from vortex.models.flux import load_flux_schnell
        model = load_flux_schnell(
            device=device,
            quantization=precision,
            cache_dir=cache_dir,
            local_only=local_only,
        )
        logger.info("Flux-Schnell loaded successfully")
        return model
    except (ImportError, Exception) as e:
        logger.error(
            "Failed to load Flux-Schnell model",
            extra={"error_type": type(e).__name__, "error": str(e)},
        )
        raise


def load_liveportrait(
    device: str = "cuda:0",
    precision: Precision = "fp16",
    cache_dir: str | None = None,
    local_only: bool = False,
) -> nn.Module:
    """Load LivePortrait video warping model.

    Args:
        device: Target device (e.g., "cuda:0", "cpu")
        precision: Model precision ("fp16" for half precision)

    Returns:
        nn.Module: Loaded LivePortrait model wrapper (LivePortraitModel)

    VRAM Budget:
        ~3.5 GB with FP16

    Example:
        >>> liveportrait = load_liveportrait(device="cuda:0")
        >>> video = liveportrait.animate(actor_img, audio, expression="excited")
    """
    logger.info(
        "Loading LivePortrait model", extra={"device": device, "precision": precision}
    )

    # T016: Real LivePortrait implementation with FP16 precision
    try:
        from vortex.models.liveportrait import load_liveportrait as load_liveportrait_real
        model = load_liveportrait_real(
            device=device,
            precision=precision,
            cache_dir=cache_dir,
            local_only=local_only,
        )
        logger.info("LivePortrait loaded successfully (real implementation)")
        return model
    except (ImportError, Exception) as e:
        logger.error(
            "Failed to load LivePortrait model",
            extra={"error_type": type(e).__name__, "error": str(e)},
        )
        raise


def load_kokoro(
    device: str = "cuda:0",
    precision: Precision = "fp32",
    cache_dir: str | None = None,
    local_only: bool = False,
) -> nn.Module:
    """Load Kokoro-82M text-to-speech model.

    Args:
        device: Target device (e.g., "cuda:0", "cpu")
        precision: Model precision ("fp32" for full precision)

    Returns:
        nn.Module: Loaded Kokoro model wrapper (KokoroWrapper)

    VRAM Budget:
        ~0.4 GB with FP32

    Example:
        >>> kokoro = load_kokoro(device="cuda:0")
        >>> audio = kokoro.synthesize(text="Hello world", voice_id="rick_c137")
    """
    logger.info("Loading Kokoro-82M model", extra={"device": device, "precision": precision})

    # T017: Real Kokoro-82M implementation
    try:
        from vortex.models.kokoro import load_kokoro as load_kokoro_real
        _ = cache_dir
        _ = local_only
        model = load_kokoro_real(device=device)
        logger.info("Kokoro-82M loaded successfully (real implementation)")
        return model
    except (ImportError, Exception) as e:
        logger.error(
            "Failed to load Kokoro model",
            extra={"error_type": type(e).__name__, "error": str(e)},
        )
        raise


def load_clip_b(
    device: str = "cuda:0",
    precision: Precision = "int8",
) -> nn.Module:
    """Load CLIP-ViT-B-32 semantic verification model.

    Args:
        device: Target device (e.g., "cuda:0", "cpu")
        precision: Model precision ("int8" for 8-bit quantization)

    Returns:
        nn.Module: Loaded CLIP-B model with INT8 quantization

    VRAM Budget:
        ~0.3 GB with INT8

    Example:
        >>> clip_b = load_clip_b(device="cuda:0")
        >>> embedding = clip_b.encode_image(image_tensor)
    """
    logger.info("Loading CLIP-ViT-B-32 model", extra={"device": device, "precision": precision})

    # T018: Real CLIP-ViT-B-32 implementation with INT8 quantization
    try:
        import open_clip

        clip_b, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
            device=device,
        )
        clip_b.eval()

        # Apply INT8 quantization
        if precision == "int8":
            clip_b = torch.quantization.quantize_dynamic(
                clip_b, {torch.nn.Linear}, dtype=torch.qint8
            )

        logger.info("CLIP-ViT-B-32 loaded successfully (real implementation)")
        return clip_b
    except (ImportError, Exception) as e:
        logger.error(
            "Failed to load CLIP-B model",
            extra={"error_type": type(e).__name__, "error": str(e)},
        )
        raise


def load_clip_l(
    device: str = "cuda:0",
    precision: Precision = "int8",
) -> nn.Module:
    """Load CLIP-ViT-L-14 semantic verification model.

    Args:
        device: Target device (e.g., "cuda:0", "cpu")
        precision: Model precision ("int8" for 8-bit quantization)

    Returns:
        nn.Module: Loaded CLIP-L model with INT8 quantization

    VRAM Budget:
        ~0.6 GB with INT8

    Example:
        >>> clip_l = load_clip_l(device="cuda:0")
        >>> embedding = clip_l.encode_image(image_tensor)
    """
    logger.info("Loading CLIP-ViT-L-14 model", extra={"device": device, "precision": precision})

    # T018: Real CLIP-ViT-L-14 implementation with INT8 quantization
    try:
        import open_clip

        clip_l, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            device=device,
        )
        clip_l.eval()

        # Apply INT8 quantization
        if precision == "int8":
            clip_l = torch.quantization.quantize_dynamic(
                clip_l, {torch.nn.Linear}, dtype=torch.qint8
            )

        logger.info("CLIP-ViT-L-14 loaded successfully (real implementation)")
        return clip_l
    except (ImportError, Exception) as e:
        logger.error(
            "Failed to load CLIP-L model",
            extra={"error_type": type(e).__name__, "error": str(e)},
        )
        raise


MODEL_LOADERS: dict[ModelName, callable] = {
    "flux": load_flux,
    "liveportrait": load_liveportrait,
    "kokoro": load_kokoro,
    "clip_b": load_clip_b,
    "clip_l": load_clip_l,
}


def load_model(
    name: ModelName,
    device: str = "cuda:0",
    precision: Precision | None = None,
    cache_dir: str | None = None,
    local_only: bool = False,
) -> nn.Module:
    """Load a model by name using the appropriate loader.

    Args:
        name: Model name (flux, liveportrait, kokoro, clip_b, clip_l)
        device: Target device
        precision: Override precision (None = use default per model)

    Returns:
        nn.Module: Loaded model

    Raises:
        ValueError: If model name is invalid

    Example:
        >>> model = load_model("flux", device="cuda:0", precision="nf4")
    """
    if name not in MODEL_LOADERS:
        raise ValueError(f"Unknown model: {name}. Valid: {list(MODEL_LOADERS.keys())}")

    loader = MODEL_LOADERS[name]
    if precision:
        return loader(
            device=device,
            precision=precision,
            cache_dir=cache_dir,
            local_only=local_only,
        )
    return loader(device=device, cache_dir=cache_dir, local_only=local_only)
