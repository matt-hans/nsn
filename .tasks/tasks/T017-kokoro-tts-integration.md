---
id: T017
title: Kokoro-82M TTS Integration - High-Quality Voice Synthesis
status: pending
priority: 1
agent: ai-ml
dependencies: [T014]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [vortex, ai-ml, python, gpu, tts, audio, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - prd.md#section-12.1-static-resident-vram-layout
  - prd.md#section-12.2-generation-pipeline
  - architecture.md#section-5.3-ai-ml-pipeline

est_tokens: 10000
actual_tokens: null
---

## Description

Integrate Kokoro-82M text-to-speech model to generate high-quality voice audio from Recipe scripts. The audio output drives LivePortrait video animation and provides the final audio track for video slots.

**Critical Requirements**:
- VRAM budget: 0.4 GB (FP32 precision - model is small enough for full precision)
- Output: 24kHz mono audio (matches LivePortrait input requirements)
- Duration: Up to 45 seconds per slot
- Voice IDs: Support multiple character voices (e.g., "rick_c137", "morty", "summer")
- Speed control: 0.8-1.2× playback speed
- Emotion parameters: "neutral", "excited", "sad", "angry", "manic"
- Latency target: <2s P99 for 45-second script on RTX 3060

**Integration Points**:
- Loaded by VortexPipeline._load_kokoro() at startup
- Called by VortexPipeline._generate_audio() in parallel with Flux image generation
- Outputs to pre-allocated audio_buffer (45s × 24kHz = 1,080,000 samples)

## Business Context

**User Story**: As a Director node, I want to generate realistic character voices from text scripts in <2 seconds, so that I can run audio generation in parallel with image generation and complete my 12-second generation deadline.

**Why This Matters**:
- Audio quality directly impacts lip-sync accuracy in LivePortrait (T016)
- Voice consistency is critical for ICN's character-driven content format
- Fast audio generation (<2s) enables parallelization with Flux (12s), saving 2s overall
- Poor audio quality leads to slot rejections and reputation penalties

**What It Unblocks**:
- T016 (LivePortrait) - depends on driving audio
- T020 (Slot timing orchestration) - audio runs parallel to Flux
- Full end-to-end Vortex pipeline

**Priority Justification**: Priority 1 (Critical Path) - Blocks video animation (T016). Without audio, LivePortrait cannot generate lip-synced videos, breaking the pipeline.

## Acceptance Criteria

- [ ] Kokoro-82M model loads with FP32 precision at VortexPipeline initialization
- [ ] VRAM usage for Kokoro is 0.3-0.5 GB (measured via torch.cuda.memory_allocated())
- [ ] synthesize() method accepts: text, voice_id, speed (0.8-1.2), emotion, output=audio_buffer
- [ ] Output is mono 24kHz waveform (torch.Tensor of shape [samples])
- [ ] Generation time is <2s P99 for 45-second output on RTX 3060
- [ ] Voice IDs supported: minimum 3 distinct voices (e.g., "rick_c137", "morty", "summer")
- [ ] Speed control: 0.8× (slower), 1.0× (normal), 1.2× (faster)
- [ ] Emotion parameters: "neutral", "excited", "sad", "angry", "manic"
- [ ] Audio quality: Natural prosody, no robotic artifacts, clear pronunciation
- [ ] Output length matches input text duration (±5% tolerance)
- [ ] Outputs are written to pre-allocated audio_buffer (no new allocations)
- [ ] Model determinism: same text + voice + seed = identical waveform
- [ ] Error handling for: long scripts (>500 words), invalid voice IDs, CUDA errors

## Test Scenarios

**Test Case 1: Standard TTS Generation**
- Given: VortexPipeline with Kokoro loaded
- When: synthesize(text="Wubba lubba dub dub!", voice_id="rick_c137", speed=1.1, emotion="manic")
- Then: Output is mono 24kHz waveform
  And VRAM usage increases by <100MB during synthesis
  And generation completes in <2 seconds
  And audio_buffer contains valid audio (samples in [-1, 1])
  And output duration matches expected (text length / speech rate)

**Test Case 2: Voice ID Consistency**
- Given: Same text script
- When: synthesize(text="Hello", voice_id="rick_c137") vs voice_id="morty"
- Then: "rick_c137" output has deep, raspy voice
  And "morty" output has higher-pitched, anxious voice
  And both outputs are intelligible and natural

**Test Case 3: Speed Control**
- Given: Text "This is a test sentence"
- When: synthesize() with speed=0.8, 1.0, 1.2
- Then: 0.8× output is 25% longer duration
  And 1.2× output is 17% shorter duration
  And all outputs maintain natural prosody (no chipmunk/slow-mo artifacts)

**Test Case 4: Emotion Parameters**
- Given: Same text "I am very happy today"
- When: synthesize() with emotion="neutral" vs emotion="excited"
- Then: "excited" output has higher pitch variance and faster tempo
  And "neutral" output has flat intonation
  And both are intelligible

**Test Case 5: VRAM Budget Compliance**
- Given: Fresh Python process with only Kokoro loaded
- When: Model initialization completes
- Then: torch.cuda.memory_allocated() shows 0.3-0.5 GB
  And model dtype is torch.float32 (FP32)

**Test Case 6: Long Script Handling**
- Given: Script with 1000 words (far exceeds 45s duration)
- When: synthesize() is called
- Then: Script is truncated to fit 45s duration
  And warning logged: "Script truncated to fit 45s duration (original: 1000 words)"
  And generation succeeds with ≤45s audio

**Test Case 7: Deterministic Output**
- Given: Kokoro model with torch.manual_seed(42)
- When: synthesize(text="Test", voice_id="rick_c137", seed=42) called twice
- Then: Both outputs are bit-identical (torch.equal() returns True)

## Technical Implementation

**Required Components**:

1. **vortex/models/kokoro.py** (Model wrapper)
   - `load_kokoro(device: str = "cuda") -> KokoroModel`
   - `KokoroModel.synthesize(text, voice_id, speed, emotion, output, seed)`
   - Voice embedding loader
   - Emotion parameter mapping

2. **vortex/models/configs/kokoro_voices.yaml** (Voice definitions)
   - Voice ID → embedding file mapping
   - `rick_c137: ~/.cache/vortex/voices/rick_c137.pt`
   - `morty: ~/.cache/vortex/voices/morty.pt`
   - `summer: ~/.cache/vortex/voices/summer.pt`

3. **vortex/models/configs/kokoro_emotions.yaml** (Emotion parameters)
   - Emotion → pitch/tempo/energy mapping
   - `neutral: {pitch_shift: 0, tempo: 1.0, energy: 1.0}`
   - `excited: {pitch_shift: +50Hz, tempo: 1.15, energy: 1.3}`
   - `manic: {pitch_shift: +100Hz, tempo: 1.25, energy: 1.5}`

4. **vortex/tests/integration/test_kokoro_synthesis.py** (Integration tests)
   - Test cases 1-7 from above
   - Audio quality validation (MOS score estimation)
   - Intelligibility test (ASR roundtrip)

5. **vortex/benchmarks/kokoro_latency.py** (Performance benchmark)
   - Measure synthesis time over 50 iterations
   - Plot latency vs. script length
   - Identify bottlenecks

**Validation Commands**:
```bash
# Install Kokoro dependencies
pip install kokoro-tts==0.1.0  # hypothetical package
pip install librosa==0.10.0 soundfile==0.12.0

# Download model weights + voice embeddings
python vortex/scripts/download_kokoro.py

# Unit tests (mocked Kokoro)
pytest vortex/tests/unit/test_kokoro.py -v

# Integration test (real model, requires GPU)
pytest vortex/tests/integration/test_kokoro_synthesis.py --gpu -v

# VRAM profiling
python vortex/benchmarks/kokoro_vram_profile.py

# Latency benchmark
python vortex/benchmarks/kokoro_latency.py --iterations 50

# Audio quality check (manual listening)
python vortex/scripts/audio_check_kokoro.py --text "Test script" --voice rick_c137 --output /tmp/kokoro_test.wav

# Intelligibility test (ASR roundtrip)
python vortex/tests/quality/kokoro_intelligibility.py
```

**Code Patterns**:
```python
# From vortex/models/kokoro.py
import torch
import torch.nn as nn
from typing import Optional

class KokoroModel:
    def __init__(self, model: nn.Module, voice_embeddings: dict, device: str = "cuda"):
        self.model = model.float().to(device)  # FP32
        self.voice_embeddings = voice_embeddings
        self.device = device

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        voice_id: str = "rick_c137",
        speed: float = 1.0,
        emotion: str = "neutral",
        output: Optional[torch.Tensor] = None,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """Generate 24kHz mono audio from text."""
        if seed is not None:
            torch.manual_seed(seed)

        # Get voice embedding
        if voice_id not in self.voice_embeddings:
            raise ValueError(f"Unknown voice_id: {voice_id}. Available: {list(self.voice_embeddings.keys())}")
        voice_embedding = self.voice_embeddings[voice_id]

        # Get emotion parameters
        emotion_params = self._get_emotion_params(emotion)

        # Estimate output length (characters → duration heuristic)
        estimated_duration = len(text) * 0.08 * (1.0 / speed)  # ~80ms per character
        max_duration = 45.0  # seconds
        if estimated_duration > max_duration:
            # Truncate text to fit
            max_chars = int(max_duration / (0.08 / speed))
            text = text[:max_chars]
            logger.warning(f"Script truncated to fit 45s duration (original: {len(text)} chars)")

        # Tokenize text
        tokens = self.model.tokenize(text)

        # Synthesize with voice + emotion
        waveform = self.model.infer(
            tokens=tokens,
            voice_embedding=voice_embedding,
            pitch_shift=emotion_params["pitch_shift"],
            tempo=emotion_params["tempo"] * speed,
            energy=emotion_params["energy"],
            sample_rate=24000
        )

        # Normalize to [-1, 1]
        waveform = waveform / waveform.abs().max().clamp(min=1e-8)

        # Write to pre-allocated buffer
        if output is not None:
            output[:waveform.shape[0]].copy_(waveform)
            return output[:waveform.shape[0]]
        return waveform

    def _get_emotion_params(self, emotion: str) -> dict:
        """Map emotion name to synthesis parameters."""
        emotion_map = {
            "neutral": {"pitch_shift": 0, "tempo": 1.0, "energy": 1.0},
            "excited": {"pitch_shift": 50, "tempo": 1.15, "energy": 1.3},
            "sad": {"pitch_shift": -30, "tempo": 0.9, "energy": 0.7},
            "angry": {"pitch_shift": 20, "tempo": 1.1, "energy": 1.4},
            "manic": {"pitch_shift": 100, "tempo": 1.25, "energy": 1.5},
        }
        return emotion_map.get(emotion, emotion_map["neutral"])

def load_kokoro(device: str = "cuda") -> KokoroModel:
    """Load Kokoro-82M model with voice embeddings."""
    from kokoro import KokoroTTS  # hypothetical import

    model = KokoroTTS.from_pretrained("kokoro-82m", cache_dir="~/.cache/vortex/")
    model = model.to(device).float()

    # Load voice embeddings
    voice_embeddings = {
        "rick_c137": torch.load("~/.cache/vortex/voices/rick_c137.pt"),
        "morty": torch.load("~/.cache/vortex/voices/morty.pt"),
        "summer": torch.load("~/.cache/vortex/voices/summer.pt"),
    }

    return KokoroModel(model, voice_embeddings, device)
```

## Dependencies

**Hard Dependencies** (must be complete first):
- [T014] Vortex Core Pipeline - provides ModelRegistry, VRAMMonitor, audio_buffer

**Soft Dependencies** (nice to have):
- None

**External Dependencies**:
- Python 3.11
- PyTorch 2.1+ with CUDA 12.1+
- kokoro-tts library (hypothetical, may need custom build or GitHub repo)
- librosa 0.10.0+ (for audio utilities)
- soundfile 0.12.0+ (for WAV export)

## Design Decisions

**Decision 1: FP32 Precision (No Quantization)**
- **Rationale**: Kokoro-82M is only 0.4GB in FP32. Quantization saves minimal VRAM (~0.2GB) but adds complexity and potential quality loss.
- **Alternatives**:
  - FP16 (rejected: minimal VRAM savings, potential audio artifacts)
  - INT8 (rejected: significant quality loss for TTS)
- **Trade-offs**: (+) Simple, best quality. (-) Slight VRAM overhead (acceptable given budget).

**Decision 2: 24kHz Sample Rate**
- **Rationale**: Matches LivePortrait input requirements. Higher sample rates (48kHz) double file size and VRAM with minimal perceptual gain for speech.
- **Alternatives**:
  - 16kHz (rejected: noticeably lower quality)
  - 48kHz (rejected: unnecessary for speech)
- **Trade-offs**: (+) Good quality, efficient. (-) Not as high as music-quality (48kHz).

**Decision 3: Voice Embeddings (Not Fine-Tuned Models)**
- **Rationale**: Voice embeddings allow multiple voices from a single base model. Fine-tuning separate models would multiply VRAM budget.
- **Alternatives**:
  - Separate fine-tuned models per voice (rejected: 3 voices = 3× VRAM)
- **Trade-offs**: (+) Efficient, flexible. (-) Slightly lower voice distinctiveness than fine-tuned models.

**Decision 4: Emotion via Parameter Modulation (Not Separate Models)**
- **Rationale**: Pitch/tempo/energy modulation is fast and expressive. Training emotion-specific models is unnecessary.
- **Alternatives**:
  - Emotion-specific models (rejected: multiplies VRAM and complexity)
- **Trade-offs**: (+) Simple, fast. (-) Less nuanced than emotion-trained models.

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Voice embeddings unavailable | High (no character voices) | Low (cacheable) | Provide download script, bundle embeddings with package, fallback to default voice |
| Audio quality insufficient | Medium (poor lip-sync) | Low (Kokoro is proven) | A/B test with human evaluators, provide quality settings (fast vs. high-quality mode) |
| Long scripts exceed 45s | Medium (truncation) | Medium (user error) | Estimate duration from text length, warn in Recipe validation, truncate gracefully |
| Kokoro model unavailable | High (pipeline broken) | Low (cache weights) | Local weight caching, provide download script, document model source |
| VRAM usage exceeds 0.5GB | Low (within budget) | Low (small model) | Monitor VRAM, log spikes, verify FP32 precision |

## Context7 Enrichment

> **Source**: Context7 `/nikkoxgonzales/streaming-tts` - Kokoro-based TTS

### streaming-tts TTSStream API

**Basic Configuration**:
```python
from streaming_tts import TTSStream, TTSConfig

config = TTSConfig(
    voice="af_heart",           # Voice name or blend formula
    speed=1.0,                  # Speech speed (1.0 = normal)
    device=None,                # "cuda", "mps", "cpu", or None (auto-detect)
    trim_silence=True,          # Trim leading/trailing silence
    silence_threshold=0.005,
    memory_threshold_gb=2.0,    # GPU memory threshold for auto-clearing
)

stream = TTSStream(config=config)
```

### Voice Blending Syntax

**Equal Blend** (recommended syntax):
```python
config = TTSConfig(voice="af_sarah+af_jessica")
```

**Weighted Blend**:
```python
# 30% sarah, 70% jessica
config = TTSConfig(voice="af_sarah(0.3)+af_jessica(0.7)")
```

**Multi-Voice Blend**:
```python
config = TTSConfig(voice="af_sarah(0.4)+af_jessica(0.4)+af_daniel(0.2)")
```

**Dynamic Voice Setting**:
```python
stream = TTS()
stream.set_voice("af_heart+am_adam")
stream.feed("Dynamic voice text.").play(output_path="output.wav")
```

### Concurrent Synthesis with asyncio

```python
import asyncio
from streaming_tts import TTSStream, TTSConfig

async def synthesize_text(text: str, voice: str):
    config = TTSConfig(voice=voice)
    stream = TTSStream(config=config)
    stream.feed(text)
    
    chunks = []
    async for chunk in stream.stream_async():
        chunks.append(chunk)
    
    stream.shutdown()
    return voice, len(chunks)

async def generate_multiple():
    tasks = [
        synthesize_text("Hello from voice one", "af_heart"),
        synthesize_text("Greetings from voice two", "am_adam"),
    ]
    results = await asyncio.gather(*tasks)
    return results
```

## Progress Log

### [2025-12-24T00:00:00Z] - Task Created

**Created By:** task-creator agent
**Reason:** User request to create comprehensive Vortex Engine tasks per PRD sections 12.1-12.3
**Dependencies:** T014 (Vortex Core Pipeline)
**Estimated Complexity:** Standard (10,000 tokens estimated)

**Notes**: Kokoro-82M provides high-quality TTS for ICN character voices. FP32 precision is acceptable given small model size (0.4GB). Fast synthesis (<2s) enables parallelization with Flux.

## Completion Checklist

**Code Complete**:
- [ ] vortex/models/kokoro.py implemented with load_kokoro(), KokoroModel.synthesize()
- [ ] FP32 precision config verified
- [ ] Voice embeddings loader (rick_c137, morty, summer)
- [ ] Emotion parameter mapping
- [ ] Integration with VortexPipeline._generate_audio()
- [ ] Output to pre-allocated audio_buffer

**Testing**:
- [ ] Unit tests pass (mocked Kokoro)
- [ ] Integration test generates 24kHz audio
- [ ] VRAM profiling shows 0.3-0.5GB usage
- [ ] Latency benchmark P99 <2s on RTX 3060
- [ ] Voice ID consistency test passes
- [ ] Speed control test (0.8×, 1.0×, 1.2×)
- [ ] Emotion parameter test shows audible differences
- [ ] Deterministic output test passes

**Documentation**:
- [ ] Docstrings for load_kokoro(), synthesize()
- [ ] vortex/models/README.md updated with Kokoro usage
- [ ] VRAM budget documented (0.4GB target)
- [ ] Voice ID definitions documented
- [ ] Emotion parameter effects explained

**Performance**:
- [ ] P99 synthesis latency <2s for 45s script
- [ ] VRAM usage stable at ~0.4GB across 100 syntheses
- [ ] No memory leaks (VRAM delta <20MB after 100 calls)
- [ ] Audio quality acceptable (MOS >4.0 on test dataset)

**Definition of Done:**
Task is complete when ALL acceptance criteria met, ALL validations pass, and Kokoro integration produces high-quality 24kHz audio within 0.4GB VRAM budget and <2s P99 latency on RTX 3060.
