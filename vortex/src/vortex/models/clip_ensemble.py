"""CLIP ensemble placeholder for semantic verification.

NOTE: This is a placeholder implementation. Real CLIP verification
will be implemented in Phase 4.4. Currently returns mock passing scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class DualClipResult:
    """Result from dual CLIP verification (ViT-B/32 + ViT-L/14).

    Attributes:
        score_clip_b: ViT-B/32 similarity score (0-1)
        score_clip_l: ViT-L/14 similarity score (0-1)
        ensemble_score: Weighted average of both scores
        self_check_passed: True if score exceeds threshold
        outlier_detected: True if scores diverge significantly
        embedding: Combined 512-dim embedding for BFT consensus
        is_placeholder: True if this result came from placeholder implementation
    """

    score_clip_b: float = 0.0
    score_clip_l: float = 0.0
    ensemble_score: float = 0.0
    self_check_passed: bool = True
    outlier_detected: bool = False
    embedding: torch.Tensor = field(default_factory=lambda: torch.zeros(512))
    is_placeholder: bool = False


class ClipEnsemble(nn.Module):
    """Placeholder CLIP ensemble - returns mock verification scores.

    TODO(Phase 4.4): Implement real dual-CLIP verification using:
    - OpenAI CLIP ViT-L/14
    - SigLIP ViT-SO400M
    """

    def __init__(self, device: str = "cuda", precision: str = "fp16"):
        super().__init__()
        self.device = device
        self.precision = precision
        self._warned = False

    def verify(
        self,
        video_frames: torch.Tensor,
        prompt: str,
        threshold: float = 0.70,
        seed: int | None = None,
    ) -> DualClipResult:
        """Return mock verification result.

        Args:
            video_frames: Video tensor [T, C, H, W] (not analyzed in placeholder)
            prompt: Text prompt to verify against (logged but not used)
            threshold: Minimum score to pass (ignored in placeholder)
            seed: Optional seed (unused in placeholder)

        Returns:
            DualClipResult with mock passing values and is_placeholder=True
        """
        if not self._warned:
            logger.warning(
                "Using CLIP placeholder - verification scores are mock values. "
                "Real implementation pending Phase 4.4."
            )
            self._warned = True

        return DualClipResult(
            score_clip_b=0.78,
            score_clip_l=0.76,
            ensemble_score=0.77,
            self_check_passed=True,
            outlier_detected=False,
            embedding=torch.randn(512, device=self.device),
            is_placeholder=True,
        )

    def get_embedding(self, image: torch.Tensor) -> torch.Tensor:
        """Return random embedding (placeholder).

        Args:
            image: Image tensor (not analyzed in placeholder)

        Returns:
            Random 512-dim embedding tensor
        """
        return torch.randn(512, device=self.device)


def load_clip_ensemble(
    device: str = "cuda",
    precision: str = "fp16",
    local_only: bool = False,
) -> ClipEnsemble:
    """Load CLIP ensemble placeholder.

    Args:
        device: Target device ("cuda" or "cpu")
        precision: Compute precision ("fp16" or "fp32")
        local_only: Only use local model cache (ignored in placeholder)

    Returns:
        ClipEnsemble placeholder instance
    """
    logger.info("Loading CLIP ensemble placeholder (real implementation pending Phase 4.4)")
    return ClipEnsemble(device=device, precision=precision)
