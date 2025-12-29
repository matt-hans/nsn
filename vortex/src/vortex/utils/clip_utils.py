"""Utility functions for CLIP ensemble operations.

Provides helper functions for:
- Keyframe sampling from videos
- Embedding normalization
- Cosine similarity computation
- Frame preprocessing
"""

import logging

import torch
import torch.nn.functional as functional
from PIL import Image

logger = logging.getLogger(__name__)


def sample_keyframes(
    video: torch.Tensor,
    num_frames: int = 5,
    method: str = "evenly_spaced",
) -> torch.Tensor:
    """Sample keyframes from video tensor.

    Args:
        video: Video tensor [T, C, H, W]
        num_frames: Number of keyframes to extract
        method: Sampling method ("evenly_spaced", "random", "saliency")

    Returns:
        Keyframe tensor [num_frames, C, H, W]

    Raises:
        ValueError: If num_frames > video length or invalid method

    Example:
        >>> video = torch.randn(1080, 3, 512, 512)  # 45s @ 24fps
        >>> keyframes = sample_keyframes(video, num_frames=5)
        >>> keyframes.shape
        torch.Size([5, 3, 512, 512])
    """
    num_total_frames = video.shape[0]

    if num_frames > num_total_frames:
        raise ValueError(f"num_frames ({num_frames}) exceeds video length ({num_total_frames})")

    if method == "evenly_spaced":
        # Evenly spaced indices across video
        indices = torch.linspace(0, num_total_frames - 1, num_frames).long()
    elif method == "random":
        # Random sampling without replacement
        indices = torch.randperm(num_total_frames)[:num_frames].sort()[0]
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    logger.debug(
        "Keyframes sampled",
        extra={
            "method": method,
            "num_frames": num_frames,
            "total_frames": num_total_frames,
            "indices": indices.tolist(),
        },
    )

    return video[indices]


def normalize_embedding(embedding: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """L-p normalize embedding tensor.

    Args:
        embedding: Embedding tensor (any shape)
        p: Norm order (2.0 for L2 norm)

    Returns:
        Normalized embedding with ||embedding||_p = 1.0

    Example:
        >>> emb = torch.tensor([3.0, 4.0])
        >>> normalized = normalize_embedding(emb)
        >>> torch.linalg.norm(normalized).item()
        1.0
    """
    return functional.normalize(embedding, p=p, dim=-1)


def compute_cosine_similarity(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
) -> float:
    """Compute cosine similarity between two embeddings.

    Args:
        emb1: First embedding tensor
        emb2: Second embedding tensor

    Returns:
        Cosine similarity in range [-1, 1]

    Example:
        >>> emb1 = torch.tensor([1.0, 0.0, 0.0])
        >>> emb2 = torch.tensor([1.0, 0.0, 0.0])
        >>> compute_cosine_similarity(emb1, emb2)
        1.0
    """
    # Normalize embeddings
    emb1_norm = functional.normalize(emb1, dim=-1)
    emb2_norm = functional.normalize(emb2, dim=-1)

    # Cosine similarity
    similarity = (emb1_norm * emb2_norm).sum().item()

    return similarity


def preprocess_frames(
    frames: list[Image.Image],
    target_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Preprocess PIL images for CLIP input.

    Args:
        frames: List of PIL images
        target_size: Target (H, W) for resizing

    Returns:
        Tensor [N, C, H, W] in range [0, 1]

    Example:
        >>> from PIL import Image
        >>> frames = [Image.new('RGB', (512, 512)) for _ in range(5)]
        >>> tensor = preprocess_frames(frames, target_size=(224, 224))
        >>> tensor.shape
        torch.Size([5, 3, 224, 224])
    """
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    tensors = [transform(frame) for frame in frames]
    return torch.stack(tensors)


def detect_outliers(
    scores: list[float],
    threshold: float = 0.15,
) -> bool:
    """Detect outliers in CLIP score list.

    Args:
        scores: List of CLIP scores
        threshold: Maximum allowed pairwise difference

    Returns:
        True if any pairwise difference exceeds threshold

    Example:
        >>> scores = [0.82, 0.85]  # Normal case
        >>> detect_outliers(scores, threshold=0.15)
        False
        >>> scores = [0.45, 0.75]  # Adversarial case
        >>> detect_outliers(scores, threshold=0.15)
        True
    """
    if len(scores) < 2:
        return False

    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if abs(scores[i] - scores[j]) > threshold:
                logger.warning(
                    "Outlier detected",
                    extra={
                        "score_i": scores[i],
                        "score_j": scores[j],
                        "difference": abs(scores[i] - scores[j]),
                        "threshold": threshold,
                    },
                )
                return True

    return False


def compute_ensemble_score(
    scores: list[float],
    weights: list[float],
) -> float:
    """Compute weighted ensemble score.

    Args:
        scores: List of individual model scores
        weights: List of model weights (must sum to 1.0)

    Returns:
        Weighted average score

    Raises:
        ValueError: If weights don't sum to 1.0 or length mismatch

    Example:
        >>> scores = [0.82, 0.85]
        >>> weights = [0.4, 0.6]
        >>> compute_ensemble_score(scores, weights)
        0.838
    """
    if len(scores) != len(weights):
        raise ValueError(
            f"Length mismatch: {len(scores)} scores vs {len(weights)} weights"
        )

    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

    ensemble = sum(s * w for s, w in zip(scores, weights))
    return ensemble


def verify_self_check(
    scores: list[float],
    thresholds: list[float],
) -> bool:
    """Verify all scores meet self-check thresholds.

    Args:
        scores: List of individual model scores
        thresholds: List of minimum thresholds

    Returns:
        True if all scores >= their thresholds

    Example:
        >>> scores = [0.75, 0.80]
        >>> thresholds = [0.70, 0.72]
        >>> verify_self_check(scores, thresholds)
        True
        >>> scores = [0.65, 0.80]
        >>> verify_self_check(scores, thresholds)
        False
    """
    if len(scores) != len(thresholds):
        raise ValueError("Length mismatch between scores and thresholds")

    return all(score >= threshold for score, threshold in zip(scores, thresholds))
