"""Flux-Schnell image generation model with NF4 quantization.

This module provides the FluxModel wrapper for generating 512×512 actor images
from text prompts. Flux-Schnell is optimized for fast inference (4 steps) with
NF4 quantization to fit within 6GB VRAM budget.

Key Features:
- NF4 4-bit quantization via bitsandbytes PipelineQuantizationConfig
- Quantizes BOTH transformer AND text_encoder_2 (T5) for full VRAM savings
- 4-step inference (Schnell fast variant)
- Guidance scale 0.0 (unconditional for speed)
- Disabled safety checker (CLIP handles content verification)
- Direct output to pre-allocated buffers (prevents fragmentation)

VRAM Budget: ~6.0 GB (5.5-6.5GB measured)
Latency Target: <12s P99 on RTX 3060
"""

import logging

import torch
from diffusers import FluxPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import T5EncoderModel

logger = logging.getLogger(__name__)


class VortexInitializationError(Exception):
    """Raised when Flux model initialization fails (e.g., CUDA OOM)."""

    pass


class FluxModel:
    """Flux-Schnell wrapper for actor image generation.

    This class wraps the diffusers FluxPipeline with ICN-specific defaults:
    - 4 inference steps (Schnell fast variant)
    - Guidance scale 0.0 (unconditional generation)
    - 512×512 resolution (matches LivePortrait input)
    - Direct output to pre-allocated actor_buffer

    Example:
        >>> model = FluxModel(pipeline, device="cuda:0")
        >>> image = model.generate(
        ...     prompt="a scientist in a laboratory",
        ...     negative_prompt="blurry, low quality",
        ...     output=actor_buffer
        ... )
    """

    # CLIP text encoder token limit
    MAX_PROMPT_TOKENS = 77

    def __init__(self, pipeline: FluxPipeline, device: str):
        """Initialize FluxModel wrapper.

        Args:
            pipeline: Loaded FluxPipeline from diffusers
            device: Target device (e.g., "cuda:0", "cpu")
        """
        self.pipeline = pipeline
        self.device = device
        logger.info("FluxModel initialized", extra={"device": device})

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        output: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate 512×512 actor image from text prompt.

        Args:
            prompt: Text description of the actor/scene
            negative_prompt: Negative prompt for quality control
                (e.g., "blurry, low quality, watermark")
            num_inference_steps: Number of denoising steps (default: 4 for Schnell)
            guidance_scale: Classifier-free guidance scale (default: 0.0 for speed)
            output: Pre-allocated output buffer (shape: [3, 512, 512])
                If None, creates new tensor
            seed: Random seed for deterministic generation (optional)

        Returns:
            torch.Tensor: Generated image tensor, shape [3, 512, 512], range [0, 1]
                If output buffer provided, returns the buffer (in-place write)

        Raises:
            ValueError: If prompt is empty or invalid parameters

        Example:
            >>> actor_buffer = torch.zeros(3, 512, 512, device="cuda")
            >>> image = model.generate(
            ...     prompt="manic scientist, blue spiked hair, white lab coat",
            ...     negative_prompt="blurry, low quality",
            ...     output=actor_buffer,
            ...     seed=42
            ... )
        """
        # Input validation
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Truncate long prompts to CLIP limit (77 tokens)
        # Approximate: 1 token ≈ 4 characters
        approx_tokens = len(prompt.split())
        if approx_tokens > self.MAX_PROMPT_TOKENS:
            logger.warning(
                "Prompt truncated to %d tokens (approx %d tokens provided)",
                self.MAX_PROMPT_TOKENS,
                approx_tokens,
                extra={"original_length": len(prompt)},
            )
            # Truncate to first N words (rough approximation)
            words = prompt.split()
            prompt = " ".join(words[: self.MAX_PROMPT_TOKENS])

        # Set deterministic seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            logger.debug("Set random seed: %d", seed)

        # Generate image
        logger.debug(
            "Generating actor image",
            extra={
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "has_negative_prompt": bool(negative_prompt),
            },
        )

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
            output_type="pt",  # Return PyTorch tensor
        ).images[0]

        # Write to pre-allocated buffer if provided
        if output is not None:
            output.copy_(result)
            logger.debug("Wrote to pre-allocated buffer")
            return output

        return result

    def to(self, device: str) -> "FluxModel":
        """Move pipeline weights to device for offloading support."""
        if hasattr(self.pipeline, "reset_device_map"):
            try:
                self.pipeline.reset_device_map()
            except Exception as exc:
                logger.warning(
                    "FluxPipeline reset_device_map failed: %s",
                    exc,
                )
        self.pipeline.to(device)
        self.device = device
        return self


def load_flux_schnell(
    device: str = "cuda:0",
    quantization: str = "nf4",
    cache_dir: str | None = None,
    local_only: bool = False,
) -> FluxModel:
    """Load Flux-Schnell image generation model with NF4 quantization.

    This function loads the Flux.1-Schnell model from Hugging Face with:
    - NF4 4-bit quantization (via bitsandbytes)
    - bfloat16 compute dtype
    - Safety checker disabled (CLIP handles content verification)
    - Safetensors format for secure loading

    Args:
        device: Target device (e.g., "cuda:0", "cpu")
            Note: "cpu" is only for testing, generation will be extremely slow
        quantization: Quantization type ("nf4" for 4-bit)
        cache_dir: Model cache directory (default: ~/.cache/huggingface/hub)
        local_only: Require cached weights and skip network access

    Returns:
        FluxModel: Initialized Flux model wrapper

    Raises:
        VortexInitializationError: If model loading fails (CUDA OOM, network error)

    VRAM Budget:
        ~6.0 GB with NF4 quantization (measured 5.5-6.5GB)

    Example:
        >>> flux = load_flux_schnell(device="cuda:0")
        >>> image = flux.generate(prompt="a scientist")

    Notes:
        - Weights must be present in cache_dir when local_only=True
        - Requires NVIDIA GPU with CUDA 12.1+ and driver 535+
        - bitsandbytes must be installed for NF4 support
    """
    logger.info(
        "Loading Flux-Schnell model",
        extra={"device": device, "quantization": quantization},
    )

    try:
        if quantization != "nf4":
            raise ValueError(f"Unsupported quantization: {quantization}. Only 'nf4' supported.")

        # CRITICAL: Must load transformer AND text_encoder_2 (T5) separately with quantization
        # FluxPipeline doesn't support PipelineQuantizationConfig directly
        # Without quantizing T5, it uses ~10GB alone!

        # Step 1: Load T5 text encoder with NF4 quantization (transformers config)
        t5_quant_config = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        logger.info("Loading T5 text_encoder_2 with NF4 quantization...")
        # NOTE: For 4-bit quantized models, do NOT specify device_map
        # bitsandbytes handles device placement automatically during quantization
        # Specifying device_map causes accelerate to call .to() which fails
        text_encoder_2 = T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            subfolder="text_encoder_2",
            quantization_config=t5_quant_config,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            local_files_only=local_only,
        )
        logger.info("T5 text_encoder_2 loaded with NF4 quantization")

        # Step 2: Load transformer with NF4 quantization (diffusers config)
        transformer_quant_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        logger.info("Loading FluxTransformer2DModel with NF4 quantization...")
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            subfolder="transformer",
            quantization_config=transformer_quant_config,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=cache_dir,
            local_files_only=local_only,
        )
        logger.info("FluxTransformer2DModel loaded with NF4 quantization")

        # Step 3: Load full pipeline with pre-quantized components
        logger.info("Loading FluxPipeline with quantized components...")
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=cache_dir,
            device_map="cuda",  # Required for quantized models
            local_files_only=local_only,
        )
        logger.info("FluxPipeline loaded with NF4 quantization")

        # Disable safety checker (CLIP semantic verification handles content policy)
        pipeline.safety_checker = None
        logger.info("Disabled safety checker (CLIP handles content verification)")

        # NOTE: Do NOT call .to(device) - device_map already handles placement

        logger.info("Flux-Schnell loaded successfully")

        return FluxModel(pipeline, device)

    except torch.cuda.OutOfMemoryError as e:
        # Get VRAM stats for debugging
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated(device) / 1e9
            total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            error_msg = (
                f"CUDA OOM during Flux-Schnell loading. "
                f"Allocated: {allocated_gb:.2f}GB, Total: {total_gb:.2f}GB. "
                f"Required: ~6.0GB for Flux-Schnell with NF4. "
                f"Remediation: Upgrade to GPU with >=12GB VRAM (RTX 3060 minimum)."
            )
        else:
            error_msg = (
                "CUDA OOM during Flux-Schnell loading. "
                "Required: ~6.0GB VRAM. Remediation: Upgrade to GPU with >=12GB VRAM."
            )

        logger.error(error_msg, exc_info=True)
        raise VortexInitializationError(error_msg) from e

    except Exception as e:
        error_msg = f"Failed to load Flux-Schnell: {e}"
        logger.error(error_msg, exc_info=True)
        raise VortexInitializationError(error_msg) from e
