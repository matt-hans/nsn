"""CogVideoX-5B Image-to-Video model wrapper for Vortex pipeline.

This module provides the CogVideoXModel class that generates video from keyframe
images using CogVideoX-5B with INT8 quantization for memory efficiency on 12GB GPUs.

The CogVideoX model is part of the Narrative Chain pipeline (Phase 3) and:
- Generates video chunks from keyframe images
- Uses INT8 quantization to fit in 12GB VRAM (down from ~26GB)
- Supports CPU offloading for memory efficiency
- Returns video frames as torch tensors for downstream processing

VRAM Budget: ~10-11 GB (INT8 quantized with CPU offload)
Output: 49 frames at 720x480 (native I2V resolution) at 8fps (~6 seconds)

Example:
    >>> model = CogVideoXModel()
    >>> model.load()
    >>> frames = await model.generate_chunk(
    ...     image=keyframe_tensor,
    ...     prompt="A person slowly turning their head",
    ...     seed=42
    ... )
    >>> print(frames.shape)  # [49, 3, 480, 720]
"""

from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class CogVideoXError(Exception):
    """Base exception for CogVideoX errors.

    Raised when model loading fails, generation encounters an error,
    or any other CogVideoX-related failure occurs.
    """

    pass


@dataclass
class VideoGenerationConfig:
    """Configuration for video generation.

    Attributes:
        num_frames: Number of frames to generate (default 49, ~6 seconds at 8fps)
        guidance_scale: Classifier-free guidance scale (default 5.0 for artifact-free output)
        use_dynamic_cfg: Enable dynamic CFG scheduling for better motion (default True)
        num_inference_steps: Denoising steps (more = better quality, slower)
        fps: Output frame rate
        height: Video height in pixels (must be divisible by 16)
        width: Video width in pixels (must be divisible by 16)
    """

    num_frames: int = 49  # CogVideoX default (~6 seconds at 8fps)
    guidance_scale: float = 5.0  # CFG scale (reduced from 6.0 to minimize artifacts)
    use_dynamic_cfg: bool = True  # Enable dynamic CFG scheduling for better motion
    num_inference_steps: int = 50
    fps: int = 8  # Output frame rate
    height: int = 480  # CogVideoX I2V native resolution
    width: int = 720  # CogVideoX I2V native resolution (3:2 aspect)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.height % 16 != 0:
            raise ValueError(f"height must be divisible by 16, got {self.height}")
        if self.width % 16 != 0:
            raise ValueError(f"width must be divisible by 16, got {self.width}")
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {self.num_frames}")
        if self.guidance_scale < 1.0:
            raise ValueError(f"guidance_scale must be >= 1.0, got {self.guidance_scale}")
        if self.num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {self.num_inference_steps}"
            )


@dataclass
class CogVideoXModel:
    """CogVideoX-5B Image-to-Video model wrapper.

    Generates video from a keyframe image using CogVideoX-5B with INT8
    quantization for memory efficiency on 12GB GPUs.

    This model uses:
    - INT8 weight quantization via torchao to reduce VRAM from ~26GB to ~10GB
    - Model CPU offloading for sequential component loading
    - bfloat16 compute dtype for inference quality

    Attributes:
        model_id: HuggingFace model ID for CogVideoX-5B I2V
        device: Target device ("cuda" or "cpu")
        enable_cpu_offload: Whether to use model CPU offload for VRAM efficiency
        cache_dir: Optional cache directory for model weights

    Example:
        >>> model = CogVideoXModel(enable_cpu_offload=True)
        >>> model.load()
        >>> frames = await model.generate_chunk(
        ...     image=keyframe,
        ...     prompt="slow camera pan with gentle motion"
        ... )
        >>> model.unload()
    """

    model_id: str = "THUDM/CogVideoX-5b-I2V"
    device: str = "cuda"
    enable_cpu_offload: bool = True
    cache_dir: str | None = None

    # Internal state (not part of constructor)
    _pipe: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate initialization parameters."""
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device}")

        logger.info(
            "CogVideoXModel initialized",
            extra={
                "model_id": self.model_id,
                "device": self.device,
                "enable_cpu_offload": self.enable_cpu_offload,
                "cache_dir": self.cache_dir,
            },
        )

    def load(self) -> None:
        """Load the model with INT8 quantization.

        Uses torchao INT8 weight-only quantization to reduce VRAM from ~26GB
        to approximately 10GB. When enable_cpu_offload is True, model components
        are loaded to GPU sequentially during inference.

        Raises:
            CogVideoXError: If model loading fails
            ImportError: If required packages are not installed
        """
        if self._pipe is not None:
            logger.debug("CogVideoX already loaded, skipping")
            return

        logger.info(f"Loading CogVideoX from {self.model_id}...")

        try:
            from diffusers import CogVideoXImageToVideoPipeline, PipelineQuantizationConfig, TorchAoConfig
        except ImportError as e:
            logger.error(
                "Failed to import diffusers. Install with: pip install diffusers>=0.30.0",
                exc_info=True,
            )
            raise ImportError(
                "diffusers package not found or outdated. "
                "Install with: pip install diffusers>=0.30.0"
            ) from e

        try:
            # Configure INT8 quantization via torchao for the transformer
            # This reduces VRAM from ~26GB to ~10GB
            # Uses new quant_mapping API (diffusers >= 0.32)
            pipeline_quant_config = PipelineQuantizationConfig(
                quant_mapping={"transformer": TorchAoConfig("int8wo")}
            )

            # Load pipeline with quantization config
            self._pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                self.model_id,
                quantization_config=pipeline_quant_config,
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir,
            )

            if self.enable_cpu_offload:
                # Sequential CPU offload - loads each component to GPU as needed
                self._pipe.enable_model_cpu_offload()
                logger.info("CogVideoX loaded with model CPU offload enabled")
            else:
                # Move entire pipeline to device (requires more VRAM)
                self._pipe = self._pipe.to(self.device)
                logger.info(f"CogVideoX loaded on {self.device}")

            # Enable VAE tiling to reduce VRAM during decode
            # This processes video in spatial tiles instead of all at once
            # Critical for avoiding OOM during VAE decode (44GB -> ~7GB)
            if hasattr(self._pipe, "vae") and hasattr(self._pipe.vae, "enable_tiling"):
                self._pipe.vae.enable_tiling()
                logger.info("CogVideoX VAE tiling enabled")

            logger.info(
                "CogVideoX model loaded successfully",
                extra={
                    "model_id": self.model_id,
                    "quantization": "int8wo",
                    "cpu_offload": self.enable_cpu_offload,
                },
            )

        except Exception as e:
            logger.error(f"Failed to load CogVideoX model: {e}", exc_info=True)
            raise CogVideoXError(f"Failed to load CogVideoX model: {e}") from e

    def unload(self) -> None:
        """Unload the model and free VRAM.

        Releases all GPU memory held by the model and runs garbage collection.
        Safe to call multiple times.
        """
        if self._pipe is not None:
            logger.info("Unloading CogVideoX model...")
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("CogVideoX model unloaded")
        else:
            logger.debug("CogVideoX already unloaded, skipping")

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded.

        Returns:
            True if the model is loaded and ready for inference, False otherwise
        """
        return self._pipe is not None

    async def generate_chunk(
        self,
        image: torch.Tensor | Image.Image,
        prompt: str,
        config: VideoGenerationConfig | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate a video chunk from a keyframe image.

        Takes an input keyframe and generates a video sequence showing motion
        guided by the text prompt. The video starts from the provided image
        and evolves according to the prompt description.

        Args:
            image: Input keyframe as either:
                   - torch.Tensor [3, H, W] or [C, H, W] with values in 0-1 range
                   - PIL.Image.Image
            prompt: Text prompt describing the desired motion/action
            config: Optional generation configuration (uses defaults if None)
            seed: Optional seed for deterministic/reproducible generation

        Returns:
            Video frames tensor of shape [num_frames, channels, height, width]
            with values in 0-1 range (float32)

        Raises:
            CogVideoXError: If generation fails or model not loaded
            ValueError: If image format is invalid

        Example:
            >>> frames = await model.generate_chunk(
            ...     image=keyframe_tensor,  # [3, 480, 720] tensor
            ...     prompt="slowly pan camera to the right",
            ...     config=VideoGenerationConfig(num_frames=49),
            ...     seed=42
            ... )
            >>> print(frames.shape)  # [49, 3, 480, 720]
        """
        # Ensure model is loaded
        if not self.is_loaded:
            self.load()

        if config is None:
            config = VideoGenerationConfig()

        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        # Convert image to PIL if needed
        pil_image = self._to_pil_image(image)

        # Set up generator for deterministic results
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(
            "Generating video chunk",
            extra={
                "prompt_length": len(prompt),
                "num_frames": config.num_frames,
                "guidance_scale": config.guidance_scale,
                "num_inference_steps": config.num_inference_steps,
                "seed": seed,
            },
        )

        try:
            # Run generation in executor to not block async event loop
            # The diffusers pipeline is synchronous
            loop = asyncio.get_event_loop()
            video_frames = await loop.run_in_executor(
                None,
                self._generate_sync,
                pil_image,
                prompt,
                config,
                generator,
            )

            # Convert list of PIL images to tensor [T, C, H, W]
            frames_tensor = self._frames_to_tensor(video_frames)

            logger.info(
                "Video chunk generated successfully",
                extra={
                    "output_shape": list(frames_tensor.shape),
                    "dtype": str(frames_tensor.dtype),
                },
            )

            return frames_tensor

        except Exception as e:
            logger.error(f"Video generation failed: {e}", exc_info=True)
            raise CogVideoXError(f"Video generation failed: {e}") from e

    async def generate_chain(
        self,
        keyframe: torch.Tensor,
        prompt: str,
        target_duration: float,
        config: VideoGenerationConfig | None = None,
        seed: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        """Generate a long video by chaining multiple chunks.

        Uses autoregressive chaining: each chunk's last frame becomes
        the next chunk's keyframe, ensuring visual continuity.

        Args:
            keyframe: Initial keyframe tensor (RGB, 0-1 range) [C, H, W]
            prompt: Text prompt for all chunks
            target_duration: Desired total video duration in seconds
            config: Optional generation config (uses defaults if None)
            seed: Optional seed for deterministic generation
            progress_callback: Optional callback(chunk_num, total_chunks)

        Returns:
            Concatenated video frames tensor [total_frames, C, H, W]

        Example:
            >>> # Generate 12-second video from a keyframe
            >>> video = await model.generate_chain(
            ...     keyframe=image_tensor,
            ...     prompt="A ham-man giving a speech, animated",
            ...     target_duration=12.0,
            ...     seed=42,
            ... )
            >>> print(video.shape)  # [96, 3, 480, 720]  # 96 frames at 8fps = 12s
        """
        if config is None:
            config = VideoGenerationConfig()

        # Calculate chunks needed
        chunk_duration = config.num_frames / config.fps  # ~6 seconds per chunk
        num_chunks = max(1, int(np.ceil(target_duration / chunk_duration)))

        logger.info(
            f"Generating {num_chunks} chunks for {target_duration}s video",
            extra={
                "target_duration": target_duration,
                "chunk_duration": chunk_duration,
                "num_chunks": num_chunks,
                "seed": seed,
            },
        )

        all_frames: list[torch.Tensor] = []
        current_frame = keyframe

        for chunk_idx in range(num_chunks):
            if progress_callback:
                progress_callback(chunk_idx, num_chunks)

            # Vary seed per chunk for diversity while maintaining determinism
            chunk_seed = seed + chunk_idx if seed is not None else None

            logger.info(f"Generating chunk {chunk_idx + 1}/{num_chunks}...")

            # Generate chunk
            chunk_frames = await self.generate_chunk(
                image=current_frame,
                prompt=prompt,
                config=config,
                seed=chunk_seed,
            )

            # Skip first frame of subsequent chunks (overlap with previous)
            if chunk_idx > 0:
                chunk_frames = chunk_frames[1:]

            all_frames.append(chunk_frames)

            # Extract last frame for next iteration
            current_frame = chunk_frames[-1]  # [C, H, W]

            logger.info(
                f"Chunk {chunk_idx + 1} complete: {chunk_frames.shape[0]} frames"
            )

        # Concatenate all chunks
        video = torch.cat(all_frames, dim=0)  # [T, C, H, W]

        # Trim to exact target duration
        target_frames = int(target_duration * config.fps)
        if video.shape[0] > target_frames:
            video = video[:target_frames]

        logger.info(
            f"Video chain complete: {video.shape[0]} frames "
            f"({video.shape[0] / config.fps:.1f}s)"
        )

        if progress_callback:
            progress_callback(num_chunks, num_chunks)

        return video

    async def generate_montage(
        self,
        keyframes: list[torch.Tensor],
        prompts: list[str],
        config: VideoGenerationConfig | None = None,
        seed: int | None = None,
        trim_frames: int = 40,  # Trim each 49-frame clip to 40 frames (~5s)
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        """Generate video montage from multiple independent keyframes.

        Unlike generate_chain(), this method generates each clip independently
        from its own keyframe, avoiding autoregressive degradation.

        Args:
            keyframes: List of keyframe tensors [C, H, W] (one per scene)
            prompts: List of text prompts (one per scene)
            config: Optional generation config
            seed: Optional base seed (scene seeds = seed + scene_idx)
            trim_frames: Frames to keep per clip (default 40 = 5s @ 8fps).
                        Set to 0 to disable trimming and keep all frames.
            progress_callback: Optional callback(scene_num, total_scenes) called
                              before each scene and after completion

        Returns:
            Concatenated video tensor [total_frames, C, H, W]

        Example:
            >>> video = await model.generate_montage(
            ...     keyframes=[img1, img2, img3],
            ...     prompts=["Scene 1...", "Scene 2...", "Scene 3..."],
            ...     seed=42,
            ... )
            >>> print(video.shape)  # [120, 3, 480, 720] for 3x40 frames
        """
        if config is None:
            config = VideoGenerationConfig()

        if len(keyframes) != len(prompts):
            raise ValueError(
                f"keyframes ({len(keyframes)}) and prompts ({len(prompts)}) must match"
            )

        if len(keyframes) == 0:
            raise ValueError("keyframes list cannot be empty")

        num_scenes = len(keyframes)
        logger.info(f"Generating {num_scenes}-scene montage...")

        clips = []
        for i, (keyframe, prompt) in enumerate(zip(keyframes, prompts)):
            if progress_callback:
                progress_callback(i, num_scenes)

            # Derived seed for determinism
            scene_seed = seed + i if seed is not None else None

            logger.info(f"Generating scene {i+1}/{num_scenes}: {prompt[:50]}...")

            # Generate clip from fresh keyframe
            clip = await self.generate_chunk(
                image=keyframe,
                prompt=prompt,
                config=config,
                seed=scene_seed,
            )

            # Trim to target frames (remove potential tail degradation)
            if trim_frames and clip.shape[0] > trim_frames:
                clip = clip[:trim_frames]
                logger.debug(f"Trimmed scene {i+1} to {trim_frames} frames")

            clips.append(clip)
            logger.info(f"Scene {i+1} complete: {clip.shape[0]} frames")

        # Hard cut concatenation
        video = torch.cat(clips, dim=0)

        if progress_callback:
            progress_callback(num_scenes, num_scenes)

        logger.info(
            f"Montage complete: {video.shape[0]} frames "
            f"({video.shape[0] / config.fps:.1f}s @ {config.fps}fps)"
        )

        return video

    def _generate_sync(
        self,
        image: Image.Image,
        prompt: str,
        config: VideoGenerationConfig,
        generator: torch.Generator | None,
    ) -> list:
        """Synchronous video generation (called from executor).

        Args:
            image: PIL image input
            prompt: Text prompt
            config: Generation configuration
            generator: Optional random generator for determinism

        Returns:
            List of PIL image frames
        """
        result = self._pipe(
            prompt=prompt,
            image=image,
            num_frames=config.num_frames,
            height=config.height,
            width=config.width,
            guidance_scale=config.guidance_scale,
            use_dynamic_cfg=config.use_dynamic_cfg,
            num_inference_steps=config.num_inference_steps,
            generator=generator,
        )

        # CogVideoX returns frames[0] as list of PIL images
        return result.frames[0]

    def _to_pil_image(self, image: torch.Tensor | Image.Image) -> Image.Image:
        """Convert input image to PIL format.

        Args:
            image: Input image as tensor [C, H, W] or PIL Image

        Returns:
            PIL.Image.Image in RGB format

        Raises:
            ValueError: If image format is invalid
        """
        # Import PIL here to avoid import at module level
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            # Ensure RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image

        if isinstance(image, torch.Tensor):
            # Validate tensor shape
            if image.dim() not in (3, 4):
                raise ValueError(
                    f"Image tensor must be 3D [C, H, W] or 4D [B, C, H, W], "
                    f"got {image.dim()}D"
                )

            # Handle batch dimension
            if image.dim() == 4:
                if image.shape[0] != 1:
                    raise ValueError(
                        f"Batch size must be 1, got {image.shape[0]}"
                    )
                image = image.squeeze(0)

            # Validate channels
            if image.shape[0] not in (1, 3, 4):
                raise ValueError(
                    f"Image must have 1, 3, or 4 channels, got {image.shape[0]}"
                )

            # Convert to [H, W, C] for PIL
            image_np = image.detach().cpu().numpy()

            # Handle value range - normalize if needed
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)

            # Transpose from [C, H, W] to [H, W, C]
            image_np = np.transpose(image_np, (1, 2, 0))

            # Handle grayscale
            if image_np.shape[2] == 1:
                image_np = np.repeat(image_np, 3, axis=2)
            # Handle RGBA
            elif image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]

            return PILImage.fromarray(image_np, mode="RGB")

        raise ValueError(
            f"Image must be torch.Tensor or PIL.Image, got {type(image).__name__}"
        )

    def _frames_to_tensor(self, frames: list) -> torch.Tensor:
        """Convert list of PIL frames to tensor.

        Args:
            frames: List of PIL Image frames

        Returns:
            Tensor of shape [T, C, H, W] with values in 0-1 range
        """
        frame_tensors = []
        for frame in frames:
            # Convert PIL to numpy
            frame_np = np.array(frame)  # [H, W, C] uint8

            # Convert to tensor and normalize to 0-1
            frame_tensor = torch.from_numpy(frame_np).float() / 255.0

            # Transpose from [H, W, C] to [C, H, W]
            frame_tensor = frame_tensor.permute(2, 0, 1)

            frame_tensors.append(frame_tensor)

        # Stack to [T, C, H, W]
        return torch.stack(frame_tensors, dim=0)


def load_cogvideox(
    device: str = "cuda",
    enable_cpu_offload: bool = True,
    cache_dir: str | None = None,
) -> CogVideoXModel:
    """Factory function to load CogVideoX model.

    Convenience function that creates a CogVideoXModel instance and loads
    the weights. Use this for simple one-shot usage.

    Args:
        device: Target device ("cuda" or "cpu")
        enable_cpu_offload: Use model CPU offload for VRAM efficiency
        cache_dir: Optional cache directory for model weights

    Returns:
        Loaded CogVideoXModel instance ready for inference

    Example:
        >>> model = load_cogvideox(enable_cpu_offload=True)
        >>> frames = await model.generate_chunk(image, "gentle motion")
        >>> model.unload()
    """
    model = CogVideoXModel(
        device=device,
        enable_cpu_offload=enable_cpu_offload,
        cache_dir=cache_dir,
    )
    model.load()
    return model
