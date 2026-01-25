"""Bark TTS model wrapper for Vortex pipeline.

This module provides the BarkVoiceEngine class that integrates Bark TTS
into the Vortex pipeline with:
- Voice selection (rick_c137, morty, summer mapped to Bark speaker presets)
- Emotion modulation via temperature control (coarse_temp, fine_temp)
- Pre-allocated buffer output (no VRAM fragmentation)
- Deterministic generation with seed control
- 3-retry logic with fallback audio

VRAM Budget: ~3.5 GB (FP32 precision, larger than Kokoro)
Output: 24kHz mono audio

Bark is a transformer-based text-to-audio model from Suno AI that can
generate realistic speech with emotional variation through temperature
parameters.
"""

import logging
import re
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import yaml


def _patch_torch_load_for_bark() -> None:
    """Patch torch.load to use weights_only=False for Bark compatibility.

    PyTorch 2.6+ changed torch.load default to weights_only=True.
    Bark's checkpoints contain numpy objects that fail with weights_only=True.
    This patch makes torch.load default to weights_only=False to maintain
    compatibility with Bark's model loading.

    This is safe because we trust Bark's official model checkpoints.
    """
    original_load = torch.load

    @wraps(original_load)
    def patched_load(*args, **kwargs):
        # Default to weights_only=False for Bark compatibility
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load


# Apply patch before importing bark
_patch_torch_load_for_bark()

# Import bark functions - these will be mocked in tests
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
except ImportError:
    # Allow module to be imported without bark installed (for testing)
    preload_models = None
    generate_audio = None
    SAMPLE_RATE = 24000

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences for Bark processing.

    Bark has ~13 second max duration per generation, so long text
    must be split into sentences and generated separately.

    Args:
        text: Input text to split

    Returns:
        List of sentences (preserves Bark tokens like [laughs])
    """
    # Split on sentence boundaries while preserving Bark tokens
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


# Official Bark tokens that actually work (per suno-ai/bark README)
VALID_BARK_TOKENS = frozenset({
    "[laughter]", "[laughs]", "[sighs]", "[music]",
    "[gasps]", "[clears throat]", "—", "..."
})

# Regex patterns to convert unbracketed stage directions to Bark tokens
# Must run BEFORE protection step so newly created tokens get protected
# Uses negative lookbehind (?<!\[) to avoid matching words already in brackets
STAGE_DIRECTION_PATTERNS = [
    (r'(?<!\[)\b[Ss]ighs?\b(?!\])', '[sighs]'),       # Sigh, sigh, Sighs, sighs
    (r'(?<!\[)\b[Ll]aughs?\b(?!\])', '[laughs]'),     # Laugh, laugh, Laughs, laughs
    (r'(?<!\[)\b[Ll]aughter\b(?!\])', '[laughs]'),    # Laughter -> [laughs]
    (r'(?<!\[)\b[Gg]asps?\b(?!\])', '[gasps]'),       # Gasp, gasp, Gasps, gasps
]


def _clean_text_for_bark(text: str) -> str:
    """Sanitize text for Bark TTS using strict token whitelist.

    Bark will try to pronounce anything in brackets or asterisks literally.
    This function:
    0. Converts unbracketed stage directions to valid Bark tokens
    1. Protects valid Bark tokens (e.g., [laughs], [gasps])
    2. Strips all other bracketed content (e.g., [excited], [fast])
    3. Strips asterisk stage directions (e.g., *looks around*)
    4. Removes file extensions, URLs, paths, special characters

    Args:
        text: Raw text from script

    Returns:
        Cleaned text with only valid Bark tokens preserved
    """
    # 0. Convert unbracketed stage direction words to valid Bark tokens
    # Must run BEFORE protection step so these newly created tokens get protected
    for pattern, replacement in STAGE_DIRECTION_PATTERNS:
        text = re.sub(pattern, replacement, text)

    # 1. Protect valid tokens by temporarily replacing them
    # Use alphanumeric placeholders to survive the special character stripping
    protected_map = {}
    for i, token in enumerate(VALID_BARK_TOKENS):
        placeholder = f"BARKPLACEHOLDER{i}ENDTOKEN"
        if token in text:
            text = text.replace(token, placeholder)
            protected_map[placeholder] = token

    # 2. Remove ALL other bracketed content (invalid tokens like [excited], [fast])
    text = re.sub(r'\[.*?\]', '', text)

    # 3. Remove asterisk stage directions (e.g., *looks around*, *gasps*)
    text = re.sub(r'\*[^*]+\*', '', text)

    # 4. Remove file extensions (causes "dot S-S-D" stuttering)
    text = re.sub(r'\.\w{2,4}\b', '', text)

    # 5. Remove URLs and paths
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'/[\w/.-]+', '', text)

    # 6. Replace remaining dots with spaces (except ellipsis already protected)
    text = re.sub(r'(?<!\.)\.(?!\.)', ' ', text)

    # 7. Remove remaining special characters (but NOT brackets - already handled)
    text = re.sub(r'[*_~`#@$%^&+=|\\<>{}]', '', text)

    # 8. Restore valid tokens
    for placeholder, token in protected_map.items():
        text = text.replace(placeholder, token)

    # 9. Normalize whitespace
    text = ' '.join(text.split())

    return text.strip()


class BarkVoiceEngine:
    """Wrapper for Bark TTS model with ICN-specific voice/emotion mapping.

    This wrapper provides a consistent interface for the Vortex pipeline while
    handling voice ID mapping, emotion modulation via temperature settings,
    and output buffer management.

    Attributes:
        voice_profiles: Mapping of ICN voice IDs to Bark speaker presets
        emotion_config: Emotion name to temperature parameter mapping
        device: Target device (cuda or cpu)
        sample_rate: Output sample rate (24000 Hz for Bark)
        max_retries: Maximum retry attempts before using fallback (3)

    Example:
        >>> engine = BarkVoiceEngine(device="cuda")
        >>> audio = engine.synthesize(
        ...     text="Wubba lubba dub dub!",
        ...     voice_id="rick_c137",
        ...     emotion="manic"
        ... )
    """

    def __init__(
        self,
        device: str = "cuda",
        voice_config_path: str | None = None,
        emotion_config_path: str | None = None,
    ):
        """Initialize Bark voice engine.

        Args:
            device: Target device (cuda or cpu)
            voice_config_path: Path to voice_profiles.yaml (optional)
            emotion_config_path: Path to bark_emotions.yaml (optional)
        """
        self.device = device
        self.sample_rate = 24000
        self.max_retries = 3

        # Load configuration files
        self.voice_profiles = self._load_voice_config(voice_config_path)
        self.emotion_config = self._load_emotion_config(emotion_config_path)

        # Load fallback audio
        self.fallback_audio = self._load_fallback_audio()

        # Preload Bark models
        self._preload_models()

        logger.info(
            "BarkVoiceEngine initialized",
            extra={
                "device": device,
                "voices": list(self.voice_profiles.keys()),
                "emotions": list(self.emotion_config.keys()),
                "sample_rate": self.sample_rate,
            },
        )

    def _load_voice_config(self, config_path: str | None) -> dict:
        """Load voice profile configuration.

        Args:
            config_path: Optional path to voice config YAML

        Returns:
            dict: Voice profiles mapping
        """
        if config_path is None:
            config_path = str(
                Path(__file__).parent / "configs" / "voice_profiles.yaml"
            )

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded voice profiles from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(
                f"Voice config not found at {config_path}, using defaults"
            )
            return {
                "rick_c137": {"bark_speaker": "v2/en_speaker_6"},
                "morty": {"bark_speaker": "v2/en_speaker_9"},
                "summer": {"bark_speaker": "v2/en_speaker_5"},
                "default": {"bark_speaker": "v2/en_speaker_0"},
            }

    def _load_emotion_config(self, config_path: str | None) -> dict:
        """Load emotion configuration.

        Args:
            config_path: Optional path to emotion config YAML

        Returns:
            dict: Emotion settings mapping
        """
        if config_path is None:
            config_path = str(
                Path(__file__).parent / "configs" / "bark_emotions.yaml"
            )

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded emotion config from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(
                f"Emotion config not found at {config_path}, using defaults"
            )
            return {
                "neutral": {"coarse_temp": 0.7, "fine_temp": 0.5},
                "manic": {"coarse_temp": 0.95, "fine_temp": 0.7},
                "excited": {"coarse_temp": 0.85, "fine_temp": 0.55},
                "sad": {"coarse_temp": 0.5, "fine_temp": 0.4},
                "angry": {"coarse_temp": 0.85, "fine_temp": 0.6},
            }

    def _load_fallback_audio(self) -> np.ndarray:
        """Load fallback audio for when generation fails.

        Returns:
            np.ndarray: Fallback audio waveform
        """
        fallback_path = (
            Path(__file__).parent.parent.parent.parent
            / "assets"
            / "voices"
            / "fallback_audio.wav"
        )

        try:
            import soundfile as sf
            audio, sr = sf.read(fallback_path)
            if sr != self.sample_rate:
                # Resample if needed (simple decimation/interpolation)
                import scipy.signal as signal
                audio = signal.resample(
                    audio, int(len(audio) * self.sample_rate / sr)
                )
            logger.info(f"Loaded fallback audio from {fallback_path}")
            return audio.astype(np.float32)
        except (FileNotFoundError, ImportError) as e:
            logger.warning(
                f"Could not load fallback audio: {e}, using silence"
            )
            # Return 1 second of silence as ultimate fallback
            return np.zeros(self.sample_rate, dtype=np.float32)

    def _preload_models(self) -> None:
        """Preload Bark models to VRAM."""
        if preload_models is None:
            logger.warning("Bark not installed, skipping model preload")
            return

        # Validate device - only cuda and cpu are supported by Bark
        valid_devices = ("cuda", "cpu")
        if not any(self.device.startswith(d) for d in valid_devices):
            logger.warning(
                f"Invalid device '{self.device}' specified, defaulting to CPU. "
                f"Supported devices: {valid_devices}"
            )
            self.device = "cpu"

        try:
            use_gpu = self.device.startswith("cuda")
            preload_models(
                text_use_gpu=use_gpu,
                coarse_use_gpu=use_gpu,
                fine_use_gpu=use_gpu,
                codec_use_gpu=use_gpu,
            )
            logger.info(
                "Bark models preloaded",
                extra={"device": self.device, "gpu": use_gpu},
            )
        except Exception as e:
            logger.error(f"Failed to preload Bark models: {e}", exc_info=True)
            raise

    def _get_bark_speaker(self, voice_id: str) -> str:
        """Get Bark speaker preset for voice ID.

        Args:
            voice_id: ICN voice ID (e.g., "rick_c137")

        Returns:
            str: Bark speaker preset (e.g., "v2/en_speaker_6")
        """
        if voice_id in self.voice_profiles:
            profile = self.voice_profiles[voice_id]
            if isinstance(profile, dict):
                return profile.get("bark_speaker", "v2/en_speaker_0")
            return profile

        # Unknown voice, use default
        logger.warning(
            f"Unknown voice ID '{voice_id}', using default speaker",
            extra={"voice_id": voice_id},
        )
        default = self.voice_profiles.get("default", {})
        if isinstance(default, dict):
            return default.get("bark_speaker", "v2/en_speaker_0")
        return "v2/en_speaker_0"

    def _get_emotion_params(self, emotion: str) -> dict:
        """Get temperature parameters for emotion.

        Args:
            emotion: Emotion name (e.g., "manic", "neutral")

        Returns:
            dict: Temperature settings (coarse_temp, fine_temp)
        """
        if emotion in self.emotion_config:
            return self.emotion_config[emotion]

        # Unknown emotion, use neutral
        logger.warning(
            f"Unknown emotion '{emotion}', using neutral settings",
            extra={"emotion": emotion},
        )
        return self.emotion_config.get(
            "neutral", {"coarse_temp": 0.7, "fine_temp": 0.5}
        )

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        voice_id: str = "rick_c137",
        emotion: str = "neutral",
        output: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate 24kHz mono audio from text.

        Uses 3-retry logic: attempts generation up to 3 times before
        falling back to pre-loaded fallback audio.

        Args:
            text: Input text to synthesize
            voice_id: ICN voice ID (rick_c137, morty, summer, etc.)
            emotion: Emotion name (neutral, excited, sad, angry, manic)
            output: Pre-allocated output buffer (optional)
            seed: Random seed for deterministic generation (optional)

        Returns:
            torch.Tensor: Audio waveform of shape (num_samples,)

        Raises:
            ValueError: If text is empty

        Example:
            >>> audio = engine.synthesize(
            ...     text="Hello world",
            ...     voice_id="rick_c137",
            ...     emotion="manic"
            ... )
            >>> print(audio.shape)  # (num_samples,)
        """
        # Validate inputs
        if not text or text.strip() == "":
            raise ValueError("Text cannot be empty")

        # Split into sentences BEFORE cleaning (cleaning removes periods)
        sentences = split_into_sentences(text)

        # Clean each sentence individually and filter out empty ones
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = _clean_text_for_bark(sentence)
            if cleaned:  # Only keep non-empty sentences
                cleaned_sentences.append(cleaned)

        # Check if all sentences normalized to empty
        if not cleaned_sentences:
            raise ValueError("Text is empty after normalization")

        # Set seed for determinism
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Get voice and emotion parameters
        speaker = self._get_bark_speaker(voice_id)
        emotion_params = self._get_emotion_params(emotion)

        if len(cleaned_sentences) <= 1:
            # Short text - generate directly with retry logic
            audio = self._generate_with_retry(
                cleaned_sentences[0], speaker, emotion_params, seed
            )
        else:
            # Long text - generate each sentence and concatenate
            audio_segments = []
            for i, sentence in enumerate(cleaned_sentences):
                segment_seed = seed + i if seed is not None else None
                if segment_seed is not None:
                    np.random.seed(segment_seed)
                    torch.manual_seed(segment_seed)
                segment = self._generate_with_retry(
                    sentence, speaker, emotion_params, segment_seed
                )
                audio_segments.append(segment)
            audio = np.concatenate(audio_segments)

        # Convert to tensor and normalize
        waveform = self._process_waveform(audio)

        # Write to output buffer if provided
        return self._write_to_buffer(waveform, output)

    def _generate_with_retry(
        self,
        text: str,
        speaker: str,
        emotion_params: dict,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate audio with retry logic and fallback.

        Attempts generation up to max_retries times before falling back
        to pre-loaded fallback audio.

        Args:
            text: Input text to synthesize
            speaker: Bark speaker preset
            emotion_params: Temperature settings
            seed: Random seed for determinism (optional)

        Returns:
            np.ndarray: Generated audio waveform
        """
        audio = None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Reset seed for each retry attempt to ensure reproducibility
                if seed is not None:
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                audio = self._generate_bark(text, speaker, emotion_params)
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Bark generation attempt {attempt + 1}/{self.max_retries} failed",
                    extra={"error": str(e), "attempt": attempt + 1},
                )

        # Use fallback if all attempts failed
        if audio is None:
            logger.error(
                "All Bark generation attempts failed, using fallback audio",
                extra={"last_error": str(last_error)},
            )
            audio = self.fallback_audio.copy()

        return audio

    def _generate_bark(
        self, text: str, speaker: str, emotion_params: dict
    ) -> np.ndarray:
        """Generate audio using Bark.

        Args:
            text: Input text
            speaker: Bark speaker preset
            emotion_params: Temperature settings

        Returns:
            np.ndarray: Generated audio waveform
        """
        if generate_audio is None:
            raise RuntimeError(
                "Bark TTS is not installed. Install with: pip install suno-bark"
            )

        # Map our emotion parameters to Bark's API:
        # - text_temp (coarse_temp): Controls semantic token generation variability
        # - waveform_temp (fine_temp): Controls acoustic detail generation variability
        # - min_eos_p: Minimum probability threshold for end-of-sentence detection
        #              Prevents semantic drift/gibberish by helping Bark recognize
        #              when to stop generating (e.g., "laves are allegations" nonsense)
        audio = generate_audio(
            text,
            history_prompt=speaker,
            text_temp=emotion_params.get("coarse_temp", 0.7),
            waveform_temp=emotion_params.get("fine_temp", 0.5),
            min_eos_p=0.05,
        )

        return audio

    def _process_waveform(self, audio: np.ndarray) -> torch.Tensor:
        """Process raw waveform: convert to tensor and normalize.

        Args:
            audio: Raw waveform from Bark

        Returns:
            torch.Tensor: Processed waveform
        """
        # Convert to tensor
        if isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio.copy()).float()
        else:
            waveform = torch.tensor(audio, dtype=torch.float32)

        # Ensure mono
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        # Normalize to [-1, 1]
        waveform = self._normalize_audio(waveform)

        return waveform

    def _normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio waveform to [-1, 1] range.

        Args:
            waveform: Input waveform

        Returns:
            torch.Tensor: Normalized waveform
        """
        max_val = waveform.abs().max()

        if max_val > 1e-4:
            # Normalize to 0.9 max to avoid clipping
            waveform = waveform * (0.9 / max_val)
        else:
            logger.warning(f"Audio appears silent (max_val={max_val:.2e})")

        return waveform

    def _write_to_buffer(
        self, waveform: torch.Tensor, output: torch.Tensor | None
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
            buffer_size = output.shape[0]

            # Handle overflow: truncate waveform if larger than buffer
            if num_samples > buffer_size:
                logger.warning(
                    f"Audio waveform ({num_samples} samples) exceeds buffer "
                    f"({buffer_size} samples), truncating"
                )
                waveform = waveform[:buffer_size]
                num_samples = buffer_size

            output[:num_samples].copy_(waveform)
            return output[:num_samples]
        return waveform

    def unload(self) -> None:
        """Free VRAM by clearing CUDA cache.

        Call this when the engine is no longer needed to free GPU memory.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("BarkVoiceEngine VRAM cleared")


def load_bark(
    device: str = "cuda",
    voice_config_path: str | None = None,
    emotion_config_path: str | None = None,
) -> BarkVoiceEngine:
    """Load Bark TTS model with ICN voice and emotion configurations.

    This factory function:
    1. Creates BarkVoiceEngine instance
    2. Loads voice ID mappings from config
    3. Loads emotion parameter mappings from config
    4. Preloads Bark models to GPU

    Args:
        device: Target device (cuda or cpu)
        voice_config_path: Path to voice_profiles.yaml (optional)
        emotion_config_path: Path to bark_emotions.yaml (optional)

    Returns:
        BarkVoiceEngine: Initialized engine ready for synthesis

    Raises:
        ImportError: If bark package is not installed
        FileNotFoundError: If config files are missing

    Example:
        >>> bark = load_bark(device="cuda:0")
        >>> audio = bark.synthesize(text="Hello", voice_id="rick_c137")

    VRAM Budget:
        ~3.5 GB with FP32 precision
    """
    logger.info(f"Loading Bark TTS model on device: {device}")

    engine = BarkVoiceEngine(
        device=device,
        voice_config_path=voice_config_path,
        emotion_config_path=emotion_config_path,
    )

    logger.info("BarkVoiceEngine loaded successfully")
    return engine
