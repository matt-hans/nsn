"""Kokoro-82M TTS model wrapper for Vortex pipeline.

This module provides the KokoroWrapper class that integrates Kokoro-82M
text-to-speech model into the Vortex pipeline with:
- Voice selection (rick_c137, morty, summer mapped to Kokoro voice IDs)
- Speed control (0.8-1.2×)
- Emotion modulation (neutral, excited, sad, angry, manic)
- Pre-allocated buffer output (no VRAM fragmentation)
- Deterministic generation with seed control

VRAM Budget: ~0.4 GB (FP32 precision)
Output: 24kHz mono audio, up to 45 seconds per slot
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


class KokoroWrapper(nn.Module):
    """Wrapper for Kokoro-82M TTS model with ICN-specific voice/emotion mapping.

    This wrapper provides a consistent interface for the Vortex pipeline while
    handling voice ID mapping, emotion modulation, and output buffer management.

    Attributes:
        model: Underlying Kokoro TTS model
        voice_config: Mapping of ICN voice IDs to Kokoro voice IDs
        emotion_config: Emotion name to synthesis parameter mapping
        device: Target device (cuda or cpu)
        sample_rate: Output sample rate (24000 Hz for ICN)
        max_duration_sec: Maximum audio duration (45 seconds)

    Example:
        >>> wrapper = KokoroWrapper(model, voice_config, emotion_config)
        >>> audio = wrapper.synthesize(
        ...     text="Wubba lubba dub dub!",
        ...     voice_id="rick_c137",
        ...     speed=1.1,
        ...     emotion="manic"
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        voice_config: dict[str, str],
        emotion_config: dict[str, dict],
        device: str = "cuda",
        sample_rate: int = 24000,
        max_duration_sec: float = 45.0,
    ):
        """Initialize Kokoro wrapper.

        Args:
            model: Kokoro TTS model instance
            voice_config: ICN voice ID → Kokoro voice ID mapping
            emotion_config: Emotion name → synthesis parameters
            device: Target device
            sample_rate: Output sample rate (Hz)
            max_duration_sec: Maximum audio duration (seconds)
        """
        super().__init__()
        self.model = model
        self.voice_config = voice_config
        self.emotion_config = emotion_config
        self.device = device
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec

        logger.info(
            "KokoroWrapper initialized",
            extra={
                "voices": list(voice_config.keys()),
                "emotions": list(emotion_config.keys()),
                "sample_rate": sample_rate,
                "max_duration_sec": max_duration_sec,
            },
        )

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        voice_id: str = "rick_c137",
        speed: float = 1.0,
        emotion: str = "neutral",
        output: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate 24kHz mono audio from text.

        Args:
            text: Input text to synthesize
            voice_id: ICN voice ID (rick_c137, morty, summer)
            speed: Speech speed multiplier (0.8-1.2)
            emotion: Emotion name (neutral, excited, sad, angry, manic)
            output: Pre-allocated output buffer (optional)
            seed: Random seed for deterministic generation (optional)

        Returns:
            torch.Tensor: Audio waveform of shape (num_samples,)

        Raises:
            ValueError: If voice_id is unknown or text is empty

        Example:
            >>> audio = wrapper.synthesize(
            ...     text="Hello world",
            ...     voice_id="rick_c137",
            ...     speed=1.0,
            ...     emotion="neutral"
            ... )
            >>> print(audio.shape)  # (num_samples,)
        """
        # Validate and prepare inputs
        self._validate_synthesis_inputs(text, voice_id)
        if seed is not None:
            torch.manual_seed(seed)

        kokoro_voice = self.voice_config[voice_id]
        emotion_params = self._get_emotion_params(emotion)
        text = self._truncate_text_if_needed(text, speed)
        effective_speed = emotion_params["tempo"] * speed

        # Generate and process waveform
        waveform = self._generate_audio(text, kokoro_voice, effective_speed)
        waveform = self._process_waveform(waveform, emotion_params)

        # Write to output buffer if provided
        return self._write_to_buffer(waveform, output)

    def _validate_synthesis_inputs(self, text: str, voice_id: str) -> None:
        """Validate synthesis input parameters.

        Args:
            text: Input text
            voice_id: Voice identifier

        Raises:
            ValueError: If validation fails
        """
        if not text or text.strip() == "":
            raise ValueError("Text cannot be empty")

        if voice_id not in self.voice_config:
            raise ValueError(
                f"Unknown voice_id: {voice_id}. "
                f"Available: {list(self.voice_config.keys())}"
            )

    def _generate_audio(
        self, text: str, voice: str, speed: float
    ) -> torch.Tensor:
        """Generate audio using Kokoro model.

        Args:
            text: Input text
            voice: Kokoro voice ID
            speed: Effective speed multiplier

        Returns:
            torch.Tensor: Generated audio waveform

        Raises:
            Exception: If generation fails
        """
        try:
            # KPipeline returns generator yielding (graphemes, phonemes, audio)
            # We only need the audio from the final output
            generator = self.model(text, voice=voice, speed=speed)

            # Collect all audio chunks from generator
            audio_chunks = []
            for _, _, audio_chunk in generator:
                audio_chunks.append(audio_chunk)

            # Concatenate all chunks if multiple
            if len(audio_chunks) == 0:
                raise RuntimeError("No audio generated from Kokoro pipeline")
            elif len(audio_chunks) == 1:
                waveform = audio_chunks[0]
            else:
                # Stack and flatten if multiple chunks
                import numpy as np
                waveform = np.concatenate(audio_chunks, axis=0)

            return waveform
        except Exception as e:
            logger.error(f"Kokoro generation failed: {e}", exc_info=True)
            raise

    def _process_waveform(
        self, waveform: torch.Tensor, emotion_params: dict
    ) -> torch.Tensor:
        """Process raw waveform: convert to tensor, normalize, apply effects.

        Args:
            waveform: Raw waveform from Kokoro
            emotion_params: Emotion parameters

        Returns:
            torch.Tensor: Processed waveform
        """
        # Ensure tensor on correct device
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32, device=self.device)
        else:
            waveform = waveform.to(self.device)

        # Ensure mono (squeeze if needed)
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        # Normalize to [-1, 1]
        waveform = self._normalize_audio(waveform)

        # Apply emotion modulation if not natively supported
        waveform = self._apply_emotion_modulation(waveform, emotion_params)

        return waveform

    def _write_to_buffer(
        self, waveform: torch.Tensor, output: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Write waveform to pre-allocated buffer or return directly.

        Args:
            waveform: Processed waveform
            output: Optional pre-allocated buffer

        Returns:
            torch.Tensor: Waveform or buffer slice
        """
        if output is not None:
            num_samples = waveform.shape[0]
            output[:num_samples].copy_(waveform)
            return output[:num_samples]
        return waveform

    def _get_emotion_params(self, emotion: str) -> dict:
        """Retrieve emotion synthesis parameters.

        Args:
            emotion: Emotion name

        Returns:
            dict: Emotion parameters (pitch_shift, tempo, energy)
        """
        if emotion not in self.emotion_config:
            logger.warning(
                f"Unknown emotion '{emotion}', falling back to 'neutral'",
                extra={"available_emotions": list(self.emotion_config.keys())},
            )
            emotion = "neutral"

        return self.emotion_config[emotion]

    def _truncate_text_if_needed(self, text: str, speed: float) -> str:
        """Truncate text to fit max_duration_sec if needed.

        Uses heuristic: ~80ms per character for English speech.

        Args:
            text: Input text
            speed: Speech speed multiplier

        Returns:
            str: Possibly truncated text
        """
        # Estimate duration: 80ms per character, adjusted for speed
        char_duration_sec = 0.08 / speed
        estimated_duration = len(text) * char_duration_sec

        if estimated_duration > self.max_duration_sec:
            # Calculate max characters
            max_chars = int(self.max_duration_sec / char_duration_sec)
            original_len = len(text)
            text = text[:max_chars]

            warnings.warn(
                f"Script truncated to fit {self.max_duration_sec}s duration "
                f"(original: {original_len} chars, truncated: {max_chars} chars)",
                UserWarning,
            )
            logger.warning(
                "Script truncated",
                extra={
                    "original_length": original_len,
                    "truncated_length": max_chars,
                    "max_duration_sec": self.max_duration_sec,
                },
            )

        return text

    def _normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio waveform to [-1, 1] range.

        Args:
            waveform: Input waveform

        Returns:
            torch.Tensor: Normalized waveform
        """
        max_val = waveform.abs().max()
        if max_val > 1e-8:  # Avoid division by zero
            waveform = waveform / max_val
        return waveform

    def _apply_emotion_modulation(
        self, waveform: torch.Tensor, emotion_params: dict
    ) -> torch.Tensor:
        """Apply emotion-based modulation to waveform.

        This is a post-processing step if Kokoro doesn't natively support
        emotion parameters. For now, we rely on speed/tempo modulation
        which is handled in synthesis.

        Future: Could add pitch shifting, energy envelope modulation, etc.

        Args:
            waveform: Input waveform
            emotion_params: Emotion parameters

        Returns:
            torch.Tensor: Modulated waveform
        """
        # For now, emotion is primarily handled via speed/tempo
        # Future enhancement: Add pitch shifting using librosa or torch.stft

        # Placeholder for future emotion modulation
        # pitch_shift = emotion_params.get("pitch_shift", 0)
        # energy = emotion_params.get("energy", 1.0)

        return waveform

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass for nn.Module compatibility.

        Delegates to synthesize() method.
        """
        return self.synthesize(*args, **kwargs)


def load_kokoro(
    device: str = "cuda",
    voices_config_path: Optional[str] = None,
    emotions_config_path: Optional[str] = None,
) -> KokoroWrapper:
    """Load Kokoro-82M model with ICN voice and emotion configurations.

    This factory function:
    1. Loads the Kokoro TTS model (KPipeline)
    2. Loads voice ID mappings from config
    3. Loads emotion parameter mappings from config
    4. Returns a wrapped KokoroWrapper instance

    Args:
        device: Target device (cuda or cpu)
        voices_config_path: Path to voices config YAML (optional)
        emotions_config_path: Path to emotions config YAML (optional)

    Returns:
        KokoroWrapper: Initialized wrapper ready for synthesis

    Raises:
        ImportError: If kokoro package is not installed
        FileNotFoundError: If config files are missing

    Example:
        >>> kokoro = load_kokoro(device="cuda:0")
        >>> audio = kokoro.synthesize(text="Hello", voice_id="rick_c137")

    VRAM Budget:
        ~0.4 GB with FP32 precision
    """
    logger.info(f"Loading Kokoro-82M model on device: {device}")

    # Import and initialize model
    model = _import_and_create_kokoro_model()

    # Load configuration files
    voice_config, emotion_config = _load_configs(
        voices_config_path, emotions_config_path
    )

    # Create and return wrapper
    return _create_wrapper(model, voice_config, emotion_config, device)


def _import_and_create_kokoro_model():
    """Import Kokoro package and create KPipeline instance.

    Returns:
        KPipeline instance

    Raises:
        ImportError: If kokoro package is not installed
    """
    try:
        from kokoro import KPipeline
    except ImportError as e:
        logger.error(
            "Failed to import 'kokoro' package. "
            "Install with: pip install kokoro soundfile",
            exc_info=True,
        )
        raise ImportError(
            "Kokoro package not found. Install with: pip install kokoro soundfile"
        ) from e

    # Create KPipeline with American English (lang_code='a')
    # KPipeline handles model loading internally
    try:
        pipeline = KPipeline(lang_code='a')
        logger.info("KPipeline initialized for American English")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to create KPipeline: {e}", exc_info=True)
        raise


def _load_configs(
    voices_config_path: Optional[str], emotions_config_path: Optional[str]
) -> tuple[dict, dict]:
    """Load voice and emotion configuration files.

    Args:
        voices_config_path: Path to voices config YAML (optional)
        emotions_config_path: Path to emotions config YAML (optional)

    Returns:
        tuple: (voice_config dict, emotion_config dict)

    Raises:
        FileNotFoundError: If config files are missing
    """
    # Default config paths
    if voices_config_path is None:
        voices_config_path = str(
            Path(__file__).parent / "configs" / "kokoro_voices.yaml"
        )

    if emotions_config_path is None:
        emotions_config_path = str(
            Path(__file__).parent / "configs" / "kokoro_emotions.yaml"
        )

    # Load voice config
    try:
        with open(voices_config_path) as f:
            voice_config = yaml.safe_load(f)
        logger.info(f"Loaded voice config from {voices_config_path}")
    except FileNotFoundError:
        logger.error(f"Voice config not found: {voices_config_path}")
        raise

    # Load emotion config
    try:
        with open(emotions_config_path) as f:
            emotion_config = yaml.safe_load(f)
        logger.info(f"Loaded emotion config from {emotions_config_path}")
    except FileNotFoundError:
        logger.error(f"Emotion config not found: {emotions_config_path}")
        raise

    return voice_config, emotion_config


def _create_wrapper(
    model, voice_config: dict, emotion_config: dict, device: str
) -> KokoroWrapper:
    """Create KokoroWrapper instance with loaded model and configs.

    Args:
        model: KPipeline instance
        voice_config: Voice ID mapping
        emotion_config: Emotion parameter mapping
        device: Target device

    Returns:
        KokoroWrapper: Initialized wrapper
    """
    wrapper = KokoroWrapper(
        model=model,
        voice_config=voice_config,
        emotion_config=emotion_config,
        device=device,
    )

    logger.info("Kokoro wrapper initialized successfully")
    return wrapper
