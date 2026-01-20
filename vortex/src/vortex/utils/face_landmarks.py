"""Face landmark detection and lip mask generation for inpainting.

This module provides utilities to detect facial landmarks and create lip masks
for diffusion inpainting. The goal is to create a "mouth void" in sealed-lip
images before LivePortrait animation, preventing the "rubber mask" artifact.

Uses InsightFace for landmark detection (already a LivePortrait dependency).
"""

import logging
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Cache for loaded detector
_DETECTOR_CACHE: dict[str, "FaceLandmarkDetector"] = {}


class FaceLandmarkDetector:
    """Face landmark detector using InsightFace.

    Provides 68-point landmark detection compatible with dlib format,
    specifically optimized for lip region extraction.
    """

    def __init__(self, device: str = "cuda"):
        """Initialize detector.

        Args:
            device: Torch device (used to determine ONNX provider)
        """
        self.device = device
        self._app = None

    def _ensure_loaded(self) -> None:
        """Lazy-load InsightFace face analysis."""
        if self._app is not None:
            return

        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "insightface is required for face landmark detection. "
                "Install with: pip install insightface"
            ) from exc

        logger.info("Loading InsightFace face analysis...")
        providers = ['CUDAExecutionProvider'] if 'cuda' in self.device else ['CPUExecutionProvider']
        self._app = FaceAnalysis(
            name='buffalo_l',
            providers=providers,
        )
        self._app.prepare(ctx_id=0 if 'cuda' in self.device else -1)
        logger.info("InsightFace loaded successfully")

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect facial landmarks from image.

        Args:
            image: BGR image array (H, W, 3) uint8

        Returns:
            68-point landmarks array (68, 2) or None if no face detected.
            Lip landmarks are indices 48-67.
        """
        self._ensure_loaded()

        if self._app is None:
            return None

        faces = self._app.get(image)
        if not faces:
            logger.warning("No face detected in image")
            return None

        # Take the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # InsightFace provides 106-point landmarks
        # Convert to 68-point format (dlib-compatible)
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            lm106 = face.landmark_2d_106
            # Extract approximate 68-point subset
            # Lip region mapping from 106 to 68-point format
            landmarks_68 = self._convert_106_to_68(lm106)
            return landmarks_68
        elif hasattr(face, 'kps') and face.kps is not None:
            # 5-point landmarks only - insufficient for lip mask
            logger.warning("Only 5-point landmarks available, lip mask will be approximate")
            return None

        return None

    def _convert_106_to_68(self, lm106: np.ndarray) -> np.ndarray:
        """Convert 106-point landmarks to 68-point format.

        This extracts the lip region landmarks (indices 48-67 in 68-point format).
        The mapping is approximate but sufficient for lip mask generation.
        """
        # InsightFace 106-point to 68-point mapping (approximate)
        # Lip region mapping (indices in 106-point format)
        # Outer lip: 52-63 in 106 -> 48-59 in 68
        # Inner lip: 64-75 in 106 -> 60-67 in 68

        # Full 68-point mapping (simplified - focus on lip region)
        idx_68_from_106 = [
            # Jaw line (0-16)
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
            # Eyebrows (17-26)
            33, 34, 35, 36, 37, 42, 43, 44, 45, 46,
            # Nose (27-35)
            51, 52, 53, 54, 55, 56, 57, 58, 59,
            # Eyes (36-47)
            66, 67, 68, 69, 70, 71, 75, 76, 77, 78, 79, 80,
            # Outer lip (48-59) from 106-point indices 84-95
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
            # Inner lip (60-67) from 106-point indices 96-103
            96, 97, 98, 99, 100, 101, 102, 103,
        ]

        landmarks_68 = np.zeros((68, 2), dtype=np.float32)
        for i, idx in enumerate(idx_68_from_106):
            if idx < len(lm106):
                landmarks_68[i] = lm106[idx]

        return landmarks_68


def detect_lip_landmarks(
    image: torch.Tensor,
    device: str = "cuda"
) -> Optional[np.ndarray]:
    """Detect facial landmarks and return lip region.

    Args:
        image: Image tensor [C, H, W] or [H, W, C] in range [0, 1]
        device: Torch device for detector

    Returns:
        Lip landmarks array (20, 2) for indices 48-67, or None if failed.
    """
    # Get or create cached detector
    if device not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[device] = FaceLandmarkDetector(device)
    detector = _DETECTOR_CACHE[device]

    # Convert tensor to numpy BGR
    if image.dim() == 3:
        if image.shape[0] == 3:  # [C, H, W]
            image = image.permute(1, 2, 0)  # [H, W, C]
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        # Convert RGB to BGR for InsightFace
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Expected 3D tensor, got shape {image.shape}")

    landmarks = detector.detect(image_np)
    if landmarks is None:
        return None

    # Return lip region (indices 48-67)
    return landmarks[48:68]


def create_lip_mask(
    landmarks: np.ndarray,
    image_size: tuple[int, int],
    dilation_px: int = 8,
) -> torch.Tensor:
    """Create dilated binary mask from lip landmarks.

    Args:
        landmarks: Lip landmarks array (20, 2) from indices 48-67
        image_size: Target size (H, W)
        dilation_px: Pixels to dilate the mask (softens edges)

    Returns:
        Binary mask tensor [1, H, W] with lip region as 1.0
    """
    h, w = image_size

    # Outer lip contour (indices 0-11 in lip region = 48-59 in full)
    outer_lip = landmarks[:12].astype(np.int32)

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [outer_lip], 255)

    # Dilate to create soft boundary for inpainting
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_px * 2, dilation_px * 2)
        )
        mask = cv2.dilate(mask, kernel)

    # Convert to tensor
    mask_tensor = torch.from_numpy(mask / 255.0).float().unsqueeze(0)

    return mask_tensor


def create_mouth_void_mask(
    image: torch.Tensor,
    device: str = "cuda",
    dilation_px: int = 8,
) -> Optional[torch.Tensor]:
    """Create inpainting mask for mouth region.

    This is a convenience function that combines landmark detection
    and mask creation.

    Args:
        image: Image tensor [C, H, W] in range [0, 1]
        device: Torch device
        dilation_px: Dilation for soft mask edges

    Returns:
        Binary mask tensor [1, H, W] or None if face detection failed
    """
    lip_landmarks = detect_lip_landmarks(image, device)
    if lip_landmarks is None:
        return None

    h, w = image.shape[-2], image.shape[-1]
    return create_lip_mask(lip_landmarks, (h, w), dilation_px)
