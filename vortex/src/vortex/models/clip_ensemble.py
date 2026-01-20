"""Dual CLIP ensemble for semantic verification.

Implements dual CLIP model ensemble (ViT-B-32 + ViT-L-14) with:
- INT8 quantization for VRAM efficiency (0.9GB total)
- Keyframe sampling (5 frames from video)
- Weighted ensemble scoring (0.4 × B + 0.6 × L)
- Self-check thresholds (score_b ≥0.70, score_l ≥0.72)
- Outlier detection for adversarial inputs (score divergence >0.15)
- L2-normalized embeddings for BFT consensus

This module provides both director self-checking (before BFT submission)
and validator verification (during consensus).

VRAM Budget:
- CLIP-ViT-B-32 (INT8): ~0.3 GB
- CLIP-ViT-L-14 (INT8): ~0.6 GB
- Total: ~0.9 GB

Latency Target: <1s P99 for 5-frame verification on RTX 3060
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional

logger = logging.getLogger(__name__)


@dataclass
class DualClipResult:
    """Result from dual CLIP self-verification.

    Attributes:
        embedding: Combined embedding for BFT consensus (512-dim, L2-normalized)
        score_clip_b: CLIP-ViT-B-32 cosine similarity score [0, 1]
        score_clip_l: CLIP-ViT-L-14 cosine similarity score [0, 1]
        ensemble_score: Weighted average (0.4 × score_b + 0.6 × score_l)
        self_check_passed: Whether content passes self-check thresholds
        outlier_detected: Whether scores diverge >0.15 (adversarial indicator)
    """

    embedding: torch.Tensor
    score_clip_b: float
    score_clip_l: float
    ensemble_score: float
    self_check_passed: bool
    outlier_detected: bool


class ClipEnsemble:
    """Dual CLIP ensemble for semantic verification.

    Uses CLIP-ViT-B-32 (weight 0.4) and CLIP-ViT-L-14 (weight 0.6) to verify
    video frames semantically match text prompt.

    Self-check thresholds (v8.0.1):
    - score_b ≥ 0.70
    - score_l ≥ 0.72

    Both thresholds must be met for self-check to pass.
    """

    def __init__(
        self,
        clip_b: torch.nn.Module,
        clip_l: torch.nn.Module,
        preprocess_b: callable,
        preprocess_l: callable,
        tokenizer_b: callable,
        tokenizer_l: callable,
        device: str = "cuda",
    ):
        """Initialize dual CLIP ensemble.

        Args:
            clip_b: CLIP-ViT-B-32 model (INT8 quantized)
            clip_l: CLIP-ViT-L-14 model (INT8 quantized)
            preprocess_b: Image preprocessor for ViT-B-32
            preprocess_l: Image preprocessor for ViT-L-14
            tokenizer_b: Text tokenizer for ViT-B-32
            tokenizer_l: Text tokenizer for ViT-L-14
            device: Target device ("cuda" or "cpu")
        """
        self.clip_b = clip_b.to(device)
        self.clip_l = clip_l.to(device)
        self.preprocess_b = preprocess_b
        self.preprocess_l = preprocess_l
        self.tokenizer_b = tokenizer_b
        self.tokenizer_l = tokenizer_l
        self.device = device

        # Ensemble weights (from PRD §12.2)
        self.weight_b = 0.4
        self.weight_l = 0.6

        # Self-check thresholds (v8.0.1)
        self.threshold_b = 0.70
        self.threshold_l = 0.72

        # Outlier detection threshold
        self.outlier_threshold = 0.15

        logger.info(
            "ClipEnsemble initialized",
            extra={
                "device": device,
                "weight_b": self.weight_b,
                "weight_l": self.weight_l,
                "threshold_b": self.threshold_b,
                "threshold_l": self.threshold_l,
            },
        )

    def _model_dtype(self, model: torch.nn.Module) -> torch.dtype:
        visual = getattr(model, "visual", None)
        conv1 = getattr(visual, "conv1", None)
        if conv1 is not None and hasattr(conv1, "weight"):
            return conv1.weight.dtype
        try:
            return next(model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def to(self, device: str) -> "ClipEnsemble":
        """Move CLIP models to target device for offloading."""
        self.clip_b = self.clip_b.to(device)
        self.clip_l = self.clip_l.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def verify(
        self,
        video_frames: torch.Tensor,
        prompt: str,
        threshold: float | None = None,
        seed: int | None = None,
    ) -> DualClipResult:
        """Verify video semantically matches prompt using dual CLIP ensemble.

        Args:
            video_frames: Video tensor [T, C, H, W] (e.g., 1080 frames @ 512x512)
            prompt: Text prompt to verify against
            threshold: Override ensemble threshold (default uses self-check thresholds)
            seed: Random seed for deterministic results

        Returns:
            DualClipResult with scores, embedding, and self-check status

        Raises:
            ValueError: If video_frames invalid shape or prompt empty
            RuntimeError: If CLIP encoding fails or CUDA OOM
        """
        if seed is not None:
            torch.manual_seed(seed)

        self._validate_inputs(video_frames, prompt)

        logger.debug(
            "Starting CLIP verification",
            extra={
                "video_shape": video_frames.shape,
                "prompt_length": len(prompt),
                "device": self.device,
            },
        )

        # Sample keyframes
        keyframes = self._sample_keyframes(video_frames, num_frames=5)
        logger.debug("Keyframes sampled", extra={"keyframe_count": keyframes.shape[0]})

        # Compute scores with both models
        score_b, score_l = self._compute_scores(keyframes, prompt)

        # Check thresholds and outliers
        ensemble_score = score_b * self.weight_b + score_l * self.weight_l
        self_check_passed = self._check_thresholds(score_b, score_l)
        outlier_detected = self._detect_outlier(score_b, score_l)

        # Generate embedding
        embedding = self._generate_embedding(keyframes, prompt)

        logger.info(
            "CLIP verification complete",
            extra={
                "score_b": score_b,
                "score_l": score_l,
                "ensemble_score": ensemble_score,
                "self_check_passed": self_check_passed,
                "outlier_detected": outlier_detected,
            },
        )

        return DualClipResult(
            embedding=embedding,
            score_clip_b=score_b,
            score_clip_l=score_l,
            ensemble_score=ensemble_score,
            self_check_passed=self_check_passed,
            outlier_detected=outlier_detected,
        )

    def _validate_inputs(self, video_frames: torch.Tensor, prompt: str) -> None:
        """Validate input video and prompt."""
        if video_frames.ndim != 4:
            raise ValueError(
                f"video_frames must be 4D [T,C,H,W], got shape {video_frames.shape}"
            )
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

    def _compute_scores(
        self, keyframes: torch.Tensor, prompt: str
    ) -> tuple[float, float]:
        """Compute CLIP scores from both models."""
        score_b = self._compute_similarity(
            keyframes, prompt, self.clip_b, self.tokenizer_b, self.preprocess_b
        )
        score_l = self._compute_similarity(
            keyframes, prompt, self.clip_l, self.tokenizer_l, self.preprocess_l
        )
        return score_b, score_l

    def _check_thresholds(self, score_b: float, score_l: float) -> bool:
        """Check if scores meet self-check thresholds."""
        passed = score_b >= self.threshold_b and score_l >= self.threshold_l

        if not passed:
            logger.warning(
                "Self-check failed",
                extra={
                    "score_b": score_b,
                    "score_l": score_l,
                    "threshold_b": self.threshold_b,
                    "threshold_l": self.threshold_l,
                },
            )

        return passed

    def _detect_outlier(self, score_b: float, score_l: float) -> bool:
        """Detect outlier (adversarial indicator) via score divergence."""
        score_divergence = abs(score_b - score_l)
        outlier = score_divergence > self.outlier_threshold

        if outlier:
            logger.warning(
                "Score divergence detected (potential adversarial)",
                extra={
                    "score_b": score_b,
                    "score_l": score_l,
                    "divergence": score_divergence,
                    "threshold": self.outlier_threshold,
                },
            )

        return outlier

    def _sample_keyframes(
        self, video: torch.Tensor, num_frames: int = 5
    ) -> torch.Tensor:
        """Sample evenly spaced keyframes from video.

        Args:
            video: Video tensor [T, C, H, W]
            num_frames: Number of keyframes to extract

        Returns:
            Keyframe tensor [num_frames, C, H, W]

        Raises:
            ValueError: If video has 0 frames or num_frames > video length

        Example:
            For 1080-frame video with num_frames=5:
            Indices: [0, 270, 540, 810, 1079]
        """
        num_total_frames = video.shape[0]

        if num_total_frames == 0:
            raise ValueError("Cannot sample keyframes from empty video (0 frames)")

        # If video has fewer frames than requested, sample all available
        actual_num_frames = min(num_frames, num_total_frames)

        if actual_num_frames == num_total_frames:
            logger.warning(
                "Video has fewer frames than requested, sampling all frames",
                extra={
                    "requested_frames": num_frames,
                    "available_frames": num_total_frames,
                },
            )

        indices = torch.linspace(0, num_total_frames - 1, actual_num_frames).long()
        return video[indices]

    def _compute_similarity(
        self,
        keyframes: torch.Tensor,
        prompt: str,
        clip_model: torch.nn.Module,
        tokenizer: callable,
        preprocess: Callable | None = None,
    ) -> float:
        """Compute cosine similarity between keyframes and prompt.

        Args:
            keyframes: Keyframe tensor [N, C, H, W] with values in [0, 1]
            prompt: Text prompt
            clip_model: CLIP model (ViT-B-32 or ViT-L-14)
            tokenizer: Text tokenizer for the model
            preprocess: Optional preprocessing transform for the model

        Returns:
            Average cosine similarity score [0, 1]

        Raises:
            RuntimeError: If CUDA out of memory
        """
        try:
            target_dtype = self._model_dtype(clip_model)

            # Preprocess keyframes for CLIP (resize to 224x224, normalize)
            if preprocess is not None:
                # Convert tensor frames to PIL-like format and apply transform
                # OpenCLIP preprocess expects PIL images or tensors in [0, 255]
                from torchvision.transforms import functional as TF

                processed_frames = []
                for frame in keyframes:
                    # Ensure frame is in [0, 1] range, then convert to PIL
                    frame_uint8 = (frame.clamp(0, 1) * 255).byte()
                    pil_image = TF.to_pil_image(frame_uint8.cpu())
                    processed = preprocess(pil_image)
                    processed_frames.append(processed)
                keyframes = torch.stack(processed_frames).to(self.device, dtype=target_dtype)
            else:
                keyframes = keyframes.to(self.device, dtype=target_dtype)

            # Encode keyframes
            image_features = clip_model.encode_image(keyframes)

            # Encode text
            text_tokens = tokenizer([prompt]).to(self.device)
            text_features = clip_model.encode_text(text_tokens)

            # Normalize features
            image_features = functional.normalize(image_features, dim=-1)
            text_features = functional.normalize(text_features, dim=-1)

            # Average keyframe features for video-level representation
            video_feature = image_features.mean(dim=0, keepdim=True)

            # Cosine similarity
            similarity = (video_feature @ text_features.T).squeeze().item()

            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    "CUDA out of memory during CLIP encoding",
                    extra={
                        "keyframe_shape": keyframes.shape,
                        "prompt_length": len(prompt),
                        "device": self.device,
                    },
                )
                raise RuntimeError(
                    f"CUDA OOM during CLIP encoding. "
                    f"Keyframes: {keyframes.shape}, Device: {self.device}"
                ) from e
            raise

    def _generate_embedding(
        self, keyframes: torch.Tensor, prompt: str
    ) -> torch.Tensor:
        """Generate L2-normalized embedding for BFT consensus.

        Uses CLIP-ViT-B-32 for embedding generation (512-dim output).
        Note: ViT-L-14 produces 768-dim embeddings, so we use only B-32 for
        the consensus embedding while both models contribute to scoring.

        Args:
            keyframes: Keyframe tensor [N, C, H, W]
            prompt: Text prompt

        Returns:
            L2-normalized embedding tensor (512-dim)

        Raises:
            RuntimeError: If CUDA out of memory
        """
        try:
            # Preprocess keyframes for CLIP-B-32 (resize to 224x224, normalize)
            from torchvision.transforms import functional as TF

            processed_frames = []
            for frame in keyframes:
                # Ensure frame is in [0, 1] range, then convert to PIL
                frame_uint8 = (frame.clamp(0, 1) * 255).byte()
                pil_image = TF.to_pil_image(frame_uint8.cpu())
                processed = self.preprocess_b(pil_image)
                processed_frames.append(processed)
            target_dtype = self._model_dtype(self.clip_b)
            processed_keyframes = torch.stack(processed_frames).to(
                self.device,
                dtype=target_dtype,
            )

            # Use CLIP-ViT-B-32 for embedding (native 512-dim output)
            # ViT-L-14 has 768-dim output which can't be directly combined
            img_emb_b = self.clip_b.encode_image(processed_keyframes)

            text_tokens_b = self.tokenizer_b([prompt]).to(self.device)
            txt_emb_b = self.clip_b.encode_text(text_tokens_b)

            # Average keyframe features
            img_emb_b = img_emb_b.mean(dim=0)

            # Combine image and text embeddings (both 512-dim)
            final_embedding = (img_emb_b + txt_emb_b.squeeze()) / 2

            # L2 normalize
            final_embedding = functional.normalize(final_embedding, dim=-1)

            return final_embedding.cpu()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    "CUDA out of memory during embedding generation",
                    extra={
                        "keyframe_shape": keyframes.shape,
                        "prompt_length": len(prompt),
                        "device": self.device,
                    },
                )
                raise RuntimeError(
                    f"CUDA OOM during embedding generation. "
                    f"Keyframes: {keyframes.shape}, Device: {self.device}"
                ) from e
            raise


def load_clip_ensemble(
    device: str = "cuda",
    cache_dir: Path | None = None,
    precision: str = "fp16",
    local_only: bool = False,
) -> ClipEnsemble:
    """Load dual CLIP models with configurable precision.

    Args:
        device: Target device ("cuda" or "cpu")
        cache_dir: Model cache directory (default: ~/.cache/vortex/clip)
        precision: Model precision - "fp16" (default, recommended), "fp32", or "bf16"
        local_only: Require cached weights and skip network access

    Returns:
        ClipEnsemble instance with both models loaded

    Raises:
        ImportError: If open_clip not installed
        RuntimeError: If model loading fails

    VRAM Usage:
        - CLIP-ViT-B-32 (FP16): ~0.2 GB
        - CLIP-ViT-L-14 (FP16): ~0.4 GB
        - Total: ~0.6 GB (FP16) vs ~1.2 GB (FP32)

    Example:
        >>> ensemble = load_clip_ensemble(device="cuda", precision="fp16")
        >>> result = ensemble.verify(video_frames, "a scientist")
        >>> print(f"Ensemble score: {result.ensemble_score:.3f}")
    """
    try:
        import open_clip
    except ImportError as e:
        raise ImportError(
            "open_clip not found. Install with: pip install open-clip-torch==2.23.0"
        ) from e

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "vortex" / "clip"
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Validate and normalize precision
    precision = precision.lower()
    if precision not in ("fp16", "fp32", "bf16"):
        logger.warning(f"Unknown precision '{precision}', defaulting to fp16")
        precision = "fp16"

    # Use FP16 by default to save VRAM (~50% reduction vs FP32)
    # This is the recommended configuration for RTX 3060 12GB
    use_fp16 = precision == "fp16"
    use_bf16 = precision == "bf16"

    logger.info(
        "Loading CLIP ensemble",
        extra={"device": device, "cache_dir": str(cache_dir), "precision": precision},
    )

    if local_only:
        def _assert_cached_weights(model_name: str, pretrained_tag: str) -> None:
            cfg = open_clip.get_pretrained_cfg(model_name, pretrained_tag)
            if not cfg:
                raise RuntimeError(
                    f"Unknown pretrained config for {model_name} ({pretrained_tag})"
                )

            url = cfg.get("url")
            hf_hub = cfg.get("hf_hub")

            # Check HuggingFace hub cache first (preferred - used by download script)
            if hf_hub:
                try:
                    from huggingface_hub import try_to_load_from_cache
                except ImportError as exc:
                    raise RuntimeError(
                        "huggingface_hub is required to verify CLIP cache entries."
                    ) from exc

                model_id, _ = os.path.split(hf_hub)
                # Try common weight file names - safetensors is preferred by modern open_clip
                for filename in ["open_clip_model.safetensors", "open_clip_pytorch_model.bin"]:
                    result = try_to_load_from_cache(
                        repo_id=model_id,
                        filename=filename,
                        cache_dir=str(cache_dir),
                    )
                    if result is not None:
                        return  # Found in HuggingFace cache

            # Fallback to URL-based cache
            if url:
                filename = Path(url).name
                expected = cache_dir / filename
                if expected.is_file():
                    return  # Found in URL cache

            # Neither cache has the weights
            raise RuntimeError(
                f"Local-only mode missing CLIP weights for {model_name} ({pretrained_tag}). "
                f"Run: python scripts/download_and_quantize_clip.py"
            )

        _assert_cached_weights("ViT-B-32", "openai")
        _assert_cached_weights("ViT-L-14", "openai")

    # Determine torch dtype for model loading
    if use_fp16:
        torch_precision = "fp16"
    elif use_bf16:
        torch_precision = "bf16"
    else:
        torch_precision = "fp32"

    # Load ViT-B-32 with specified precision
    logger.info(f"Loading CLIP-ViT-B-32 ({precision.upper()})")
    clip_b, _, preprocess_b = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device=device,
        precision=torch_precision,
        cache_dir=str(cache_dir),
        require_pretrained=True,
    )
    clip_b.eval()
    tokenizer_b = open_clip.get_tokenizer("ViT-B-32")

    # Load ViT-L-14 with specified precision
    logger.info(f"Loading CLIP-ViT-L-14 ({precision.upper()})")
    clip_l, _, preprocess_l = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        device=device,
        precision=torch_precision,
        cache_dir=str(cache_dir),
        require_pretrained=True,
    )
    clip_l.eval()
    tokenizer_l = open_clip.get_tokenizer("ViT-L-14")

    # Log VRAM usage after loading
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        logger.info(
            "CLIP ensemble loaded successfully",
            extra={"vram_allocated_gb": allocated_gb, "precision": precision}
        )
    else:
        logger.info("CLIP ensemble loaded successfully")

    return ClipEnsemble(
        clip_b=clip_b,
        clip_l=clip_l,
        preprocess_b=preprocess_b,
        preprocess_l=preprocess_l,
        tokenizer_b=tokenizer_b,
        tokenizer_l=tokenizer_l,
        device=device,
    )
