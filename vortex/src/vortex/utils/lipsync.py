"""Lip-sync utilities for audio-to-viseme conversion.

This module provides utilities for converting audio waveforms into per-frame
viseme (mouth shape) parameters to achieve realistic lip-sync in LivePortrait
video generation.

Key Functions:
- audio_to_visemes: Convert audio waveform to per-frame viseme parameters
- phoneme_to_viseme: Map phonemes to mouth shape parameters
- smooth_viseme_sequence: Smooth viseme transitions for natural motion

Viseme Format:
    Each viseme is a 3-element tensor: [jaw_open, lip_width, lip_rounding]
    - jaw_open: 0.0 (closed) to 1.0 (wide open)
    - lip_width: 0.0 (narrow) to 1.0 (wide smile)
    - lip_rounding: 0.0 (flat) to 1.0 (fully rounded)

Target Accuracy:
    ±2 frames (~83ms at 24fps) audio-visual alignment
"""

import logging

import torch

logger = logging.getLogger(__name__)


# Phoneme-to-viseme mapping based on ARPAbet phonemes
# Each phoneme maps to [jaw_open, lip_width, lip_rounding]
PHONEME_TO_VISEME = {
    # Vowels
    "AA": [0.8, 0.6, 0.3],  # father - wide open jaw
    "AH": [0.7, 0.5, 0.3],  # cut - medium open
    "AO": [0.8, 0.4, 0.6],  # bought - open + rounded
    "AE": [0.6, 0.7, 0.2],  # cat - half open, wide
    "EH": [0.5, 0.6, 0.2],  # bed - half open
    "IH": [0.3, 0.8, 0.1],  # bit - small opening, wide
    "IY": [0.3, 0.9, 0.1],  # beet - small opening, very wide
    "UH": [0.4, 0.4, 0.7],  # book - medium, rounded
    "UW": [0.4, 0.3, 0.9],  # boot - medium, very rounded
    "ER": [0.4, 0.5, 0.4],  # bird - neutral
    # Diphthongs
    "AW": [0.7, 0.5, 0.5],  # how - open to rounded
    "AY": [0.6, 0.7, 0.2],  # my - open to wide
    "EY": [0.4, 0.7, 0.2],  # say - medium to wide
    "OW": [0.5, 0.4, 0.7],  # go - medium rounded
    "OY": [0.5, 0.5, 0.5],  # boy - medium
    # Bilabials (lip closure)
    "B": [0.1, 0.3, 0.9],  # bat - lips together, rounded
    "P": [0.1, 0.3, 0.9],  # pat - lips together, rounded
    "M": [0.1, 0.3, 0.9],  # mat - lips together, rounded
    # Labiodentals (lip-teeth contact)
    "F": [0.2, 0.5, 0.3],  # fat - small opening
    "V": [0.2, 0.5, 0.3],  # vat - small opening
    # Dentals (tongue visible)
    "TH": [0.3, 0.6, 0.2],  # thin - tongue between teeth
    "DH": [0.3, 0.6, 0.2],  # this - tongue between teeth
    # Alveolars (tongue to ridge)
    "T": [0.3, 0.5, 0.3],  # tap - neutral
    "D": [0.3, 0.5, 0.3],  # dap - neutral
    "N": [0.3, 0.5, 0.3],  # nap - neutral
    "L": [0.3, 0.5, 0.3],  # lap - neutral
    "S": [0.2, 0.6, 0.2],  # sap - narrow opening
    "Z": [0.2, 0.6, 0.2],  # zap - narrow opening
    # Palatals/affricates
    "CH": [0.3, 0.4, 0.4],  # chat - slight rounding
    "JH": [0.3, 0.4, 0.4],  # jump - slight rounding
    "SH": [0.3, 0.4, 0.5],  # ship - rounded
    "ZH": [0.3, 0.4, 0.5],  # measure - rounded
    # Velars (back of mouth)
    "K": [0.4, 0.5, 0.3],  # cat - neutral open
    "G": [0.4, 0.5, 0.3],  # gap - neutral open
    "NG": [0.4, 0.5, 0.3],  # sing - neutral open
    # Glides
    "R": [0.4, 0.4, 0.5],  # rat - rounded
    "W": [0.3, 0.4, 0.8],  # wat - very rounded
    "Y": [0.3, 0.7, 0.3],  # yap - wide
    # Glottal
    "H": [0.4, 0.5, 0.3],  # hat - neutral open
    # Silence
    "SIL": [0.2, 0.5, 0.3],  # silence - relaxed neutral
}


def audio_to_visemes(
    audio: torch.Tensor,
    fps: int,
    sample_rate: int = 24000,
    smoothing_window: int = 3,
) -> list[torch.Tensor]:
    """Convert audio waveform to per-frame viseme parameters.

    This function analyzes the audio waveform and generates viseme (mouth shape)
    parameters for each video frame to achieve realistic lip-sync.

    Current implementation uses energy-based heuristics. For production, this
    should be enhanced with:
    1. Wav2Vec2 or Whisper for phoneme detection
    2. Phoneme-to-viseme mapping from PHONEME_TO_VISEME table
    3. Temporal smoothing for natural transitions

    Args:
        audio: Audio waveform, shape [num_samples], mono
        fps: Output video frame rate
        sample_rate: Audio sample rate (default: 24000 Hz)
        smoothing_window: Number of frames for smoothing (default: 3)

    Returns:
        List of viseme tensors, one per frame
        Each viseme has shape [3]: [jaw_open, lip_width, lip_rounding]

    Example:
        >>> audio = torch.randn(24000)  # 1 second @ 24kHz
        >>> visemes = audio_to_visemes(audio, fps=24)
        >>> len(visemes)
        24
        >>> visemes[0].shape
        torch.Size([3])
    """
    # Calculate number of frames
    duration_sec = len(audio) / sample_rate
    num_frames = int(fps * duration_sec)

    # Generate raw visemes based on audio energy
    raw_visemes = []
    for i in range(num_frames):
        # Extract audio segment for this frame
        start_sample = int(i * sample_rate / fps)
        end_sample = int((i + 1) * sample_rate / fps)
        frame_audio = audio[start_sample:end_sample]

        # Compute viseme from audio features
        viseme = _audio_segment_to_viseme(frame_audio)
        raw_visemes.append(viseme)

    # Smooth viseme sequence
    smoothed_visemes = smooth_viseme_sequence(raw_visemes, window_size=smoothing_window)

    return smoothed_visemes


def _audio_segment_to_viseme(audio_segment: torch.Tensor) -> torch.Tensor:
    """Convert audio segment to viseme parameters using energy heuristics.

    This is a simplified implementation. Production version should use:
    - Wav2Vec2 for phoneme detection
    - Phoneme-to-viseme mapping
    - Context-aware adjustments

    Args:
        audio_segment: Audio samples for one frame

    Returns:
        Viseme tensor [jaw_open, lip_width, lip_rounding] on same device as input
    """
    device = audio_segment.device if len(audio_segment) > 0 else "cpu"

    if len(audio_segment) == 0:
        # Silence - neutral viseme
        return torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32, device=device)

    # Compute audio features
    energy = audio_segment.abs().mean().item()
    spectral_centroid = _compute_spectral_centroid(audio_segment)

    # Heuristic mapping (simplified):
    # - High energy → wide jaw opening
    # - High spectral centroid → wide lips (bright vowels)
    # - Low spectral centroid → rounded lips (dark vowels)

    jaw_open = min(energy * 2.0, 1.0)  # Energy → jaw opening
    lip_width = min(spectral_centroid * 0.8, 1.0)  # Brightness → width
    lip_rounding = max(0.3, 1.0 - spectral_centroid)  # Darkness → rounding

    return torch.tensor([jaw_open, lip_width, lip_rounding], dtype=torch.float32, device=device)


def _compute_spectral_centroid(audio: torch.Tensor) -> float:
    """Compute normalized spectral centroid (brightness).

    Args:
        audio: Audio segment

    Returns:
        float: Spectral centroid in [0, 1] range
    """
    if len(audio) < 2:
        return 0.5

    # Simplified spectral centroid using FFT magnitude
    fft = torch.fft.rfft(audio)
    magnitude = torch.abs(fft)
    # Create freqs on the same device as the audio tensor
    freqs = torch.arange(len(magnitude), dtype=torch.float32, device=audio.device)

    # Weighted average of frequencies
    if magnitude.sum() > 1e-8:
        centroid = (freqs * magnitude).sum() / magnitude.sum()
        # Normalize to [0, 1]
        normalized = centroid / len(magnitude)
        return float(normalized.clamp(0.0, 1.0))
    else:
        return 0.5  # Neutral for silence


def phoneme_to_viseme(phoneme: str) -> torch.Tensor:
    """Map phoneme to viseme parameters.

    Args:
        phoneme: ARPAbet phoneme (e.g., "AA", "B", "IY")

    Returns:
        Viseme tensor [jaw_open, lip_width, lip_rounding]

    Example:
        >>> viseme = phoneme_to_viseme("AA")
        >>> viseme
        tensor([0.8000, 0.6000, 0.3000])
    """
    if phoneme not in PHONEME_TO_VISEME:
        logger.warning(
            f"Unknown phoneme: {phoneme}, using neutral viseme",
            extra={"available_phonemes": list(PHONEME_TO_VISEME.keys())[:10]},
        )
        return torch.tensor([0.4, 0.5, 0.3], dtype=torch.float32)

    params = PHONEME_TO_VISEME[phoneme]
    return torch.tensor(params, dtype=torch.float32)


def smooth_viseme_sequence(
    visemes: list[torch.Tensor], window_size: int = 3
) -> list[torch.Tensor]:
    """Smooth viseme sequence using moving average for natural transitions.

    Args:
        visemes: List of viseme tensors
        window_size: Smoothing window size (odd number recommended)

    Returns:
        List of smoothed viseme tensors

    Example:
        >>> raw_visemes = [torch.tensor([0.8, 0.6, 0.3]) for _ in range(10)]
        >>> smoothed = smooth_viseme_sequence(raw_visemes, window_size=3)
        >>> len(smoothed) == len(raw_visemes)
        True
    """
    if window_size <= 1 or len(visemes) < window_size:
        return visemes

    smoothed = []
    half_window = window_size // 2

    for i in range(len(visemes)):
        # Determine window bounds
        start = max(0, i - half_window)
        end = min(len(visemes), i + half_window + 1)

        # Compute average viseme in window
        window_visemes = torch.stack(visemes[start:end])
        avg_viseme = window_visemes.mean(dim=0)

        smoothed.append(avg_viseme)

    return smoothed


def interpolate_visemes(
    viseme1: torch.Tensor, viseme2: torch.Tensor, t: float
) -> torch.Tensor:
    """Interpolate between two visemes using cubic smoothstep.

    Args:
        viseme1: Starting viseme [jaw_open, lip_width, lip_rounding]
        viseme2: Ending viseme [jaw_open, lip_width, lip_rounding]
        t: Interpolation factor [0, 1]

    Returns:
        Interpolated viseme

    Example:
        >>> v1 = torch.tensor([0.2, 0.5, 0.3])
        >>> v2 = torch.tensor([0.8, 0.6, 0.4])
        >>> mid = interpolate_visemes(v1, v2, t=0.5)
        >>> mid.shape
        torch.Size([3])
    """
    # Cubic smoothstep for smooth transitions
    t_smooth = 3 * t**2 - 2 * t**3

    return viseme1 + t_smooth * (viseme2 - viseme1)


def validate_viseme_sequence(
    visemes: list[torch.Tensor], fps: int, audio_duration: float
) -> bool:
    """Validate that viseme sequence has correct length and format.

    Args:
        visemes: List of viseme tensors
        fps: Expected frame rate
        audio_duration: Expected audio duration (seconds)

    Returns:
        bool: True if valid, False otherwise

    Example:
        >>> visemes = [torch.tensor([0.5, 0.5, 0.3]) for _ in range(24)]
        >>> validate_viseme_sequence(visemes, fps=24, audio_duration=1.0)
        True
    """
    expected_frames = int(fps * audio_duration)

    # Check length
    if len(visemes) != expected_frames:
        logger.error(
            f"Viseme sequence length mismatch: expected {expected_frames}, got {len(visemes)}",
            extra={"fps": fps, "audio_duration": audio_duration},
        )
        return False

    # Check viseme format
    for i, viseme in enumerate(visemes):
        if viseme.shape != (3,):
            logger.error(
                f"Invalid viseme shape at index {i}: {viseme.shape}, expected (3,)"
            )
            return False

        if not (viseme >= 0.0).all() or not (viseme <= 1.0).all():
            logger.error(
                f"Viseme values out of range [0, 1] at index {i}: {viseme}"
            )
            return False

    return True


def measure_lipsync_accuracy(
    visemes: list[torch.Tensor],
    reference_phonemes: list[tuple[str, float]],
    fps: int,
    tolerance_frames: int = 2,
) -> float:
    """Measure lip-sync accuracy by comparing visemes to reference phoneme timings.

    Args:
        visemes: Generated viseme sequence
        reference_phonemes: List of (phoneme, timestamp_sec) tuples
        fps: Frame rate
        tolerance_frames: Acceptable alignment error in frames (default: 2)

    Returns:
        float: Accuracy percentage (0.0 to 1.0)

    Example:
        >>> visemes = [torch.tensor([0.8, 0.6, 0.3]) for _ in range(100)]
        >>> ref_phonemes = [("AA", 0.5), ("B", 1.0), ("IY", 1.5)]
        >>> accuracy = measure_lipsync_accuracy(visemes, ref_phonemes, fps=24)
        >>> 0.0 <= accuracy <= 1.0
        True
    """
    if not reference_phonemes:
        logger.warning("No reference phonemes provided for accuracy measurement")
        return 1.0

    correct_alignments = 0
    total_phonemes = len(reference_phonemes)

    for phoneme, timestamp in reference_phonemes:
        # Convert timestamp to frame index
        expected_frame = int(timestamp * fps)

        if expected_frame >= len(visemes):
            continue

        # Get expected viseme for this phoneme
        expected_viseme = phoneme_to_viseme(phoneme)

        # Check visemes in tolerance window
        start_frame = max(0, expected_frame - tolerance_frames)
        end_frame = min(len(visemes), expected_frame + tolerance_frames + 1)

        # Find closest viseme match in window
        min_distance = float("inf")
        for frame_idx in range(start_frame, end_frame):
            distance = torch.norm(visemes[frame_idx] - expected_viseme).item()
            min_distance = min(min_distance, distance)

        # Threshold for "correct" alignment (Euclidean distance < 0.3)
        if min_distance < 0.3:
            correct_alignments += 1

    accuracy = correct_alignments / total_phonemes if total_phonemes > 0 else 0.0
    return accuracy
