# T050: Integrate CLIP Verification for Lane 0

## Priority: P1 (Critical Path)
## Complexity: 1 week
## Status: Pending
## Depends On: T019 (VRAM Manager)

---

## Objective

Integrate the dual CLIP ensemble (ViT-B-32 + ViT-L-14) as the semantic verification layer for Lane 0 video generation, replacing the current placeholder that accepts all outputs.

## Background

Current verification in `vortex/src/vortex/pipeline/verify.py` accepts everything:

```python
def verify_output(self, output: GeneratedOutput) -> bool:
    # TODO: Implement CLIP verification
    return True
```

This means Directors can submit arbitrary content without semantic compliance checks.

## Implementation

### Step 1: Dual CLIP Model Loading

```python
import torch
import open_clip
from dataclasses import dataclass

@dataclass
class CLIPConfig:
    model_b32: str = "ViT-B-32"
    model_l14: str = "ViT-L-14"
    pretrained_b32: str = "laion2b_s34b_b79k"
    pretrained_l14: str = "laion2b_s34b_b88k"
    weight_b32: float = 0.4
    weight_l14: float = 0.6
    threshold_b32: float = 0.70
    threshold_l14: float = 0.72
    device: str = "cuda"
    dtype: torch.dtype = torch.float16  # INT8 for production


class DualCLIPVerifier:
    def __init__(self, config: CLIPConfig):
        self.config = config

        # Load ViT-B-32
        self.model_b32, _, self.preprocess_b32 = open_clip.create_model_and_transforms(
            config.model_b32,
            pretrained=config.pretrained_b32,
            device=config.device,
        )
        self.model_b32.eval()

        # Load ViT-L-14
        self.model_l14, _, self.preprocess_l14 = open_clip.create_model_and_transforms(
            config.model_l14,
            pretrained=config.pretrained_l14,
            device=config.device,
        )
        self.model_l14.eval()

        # Tokenizers
        self.tokenizer_b32 = open_clip.get_tokenizer(config.model_b32)
        self.tokenizer_l14 = open_clip.get_tokenizer(config.model_l14)
```

### Step 2: Semantic Verification

```python
from PIL import Image
from typing import Tuple, Optional

class DualCLIPVerifier:
    # ... (init from above)

    @torch.no_grad()
    def verify(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
    ) -> Tuple[bool, VerificationResult]:
        """
        Verify image matches prompt using dual CLIP ensemble.

        Returns:
            Tuple of (passes, VerificationResult with scores)
        """
        # Preprocess image for both models
        img_b32 = self.preprocess_b32(image).unsqueeze(0).to(self.config.device)
        img_l14 = self.preprocess_l14(image).unsqueeze(0).to(self.config.device)

        # Tokenize prompts
        text_b32 = self.tokenizer_b32([prompt]).to(self.config.device)
        text_l14 = self.tokenizer_l14([prompt]).to(self.config.device)

        # Encode with both models
        with torch.cuda.amp.autocast():
            img_features_b32 = self.model_b32.encode_image(img_b32)
            img_features_l14 = self.model_l14.encode_image(img_l14)
            text_features_b32 = self.model_b32.encode_text(text_b32)
            text_features_l14 = self.model_l14.encode_text(text_l14)

        # Normalize
        img_features_b32 = img_features_b32 / img_features_b32.norm(dim=-1, keepdim=True)
        img_features_l14 = img_features_l14 / img_features_l14.norm(dim=-1, keepdim=True)
        text_features_b32 = text_features_b32 / text_features_b32.norm(dim=-1, keepdim=True)
        text_features_l14 = text_features_l14 / text_features_l14.norm(dim=-1, keepdim=True)

        # Compute similarities
        sim_b32 = (img_features_b32 @ text_features_b32.T).item()
        sim_l14 = (img_features_l14 @ text_features_l14.T).item()

        # Check against thresholds
        passes_b32 = sim_b32 >= self.config.threshold_b32
        passes_l14 = sim_l14 >= self.config.threshold_l14

        # Compute weighted combined score
        combined_score = (
            self.config.weight_b32 * sim_b32 +
            self.config.weight_l14 * sim_l14
        )

        # Both models must pass
        passes = passes_b32 and passes_l14

        result = VerificationResult(
            passes=passes,
            score_b32=sim_b32,
            score_l14=sim_l14,
            combined_score=combined_score,
            threshold_b32=self.config.threshold_b32,
            threshold_l14=self.config.threshold_l14,
        )

        return passes, result

    def compute_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Compute combined embedding for BFT exchange.
        """
        img_b32 = self.preprocess_b32(image).unsqueeze(0).to(self.config.device)
        img_l14 = self.preprocess_l14(image).unsqueeze(0).to(self.config.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            emb_b32 = self.model_b32.encode_image(img_b32)
            emb_l14 = self.model_l14.encode_image(img_l14)

        # Weighted combination for BFT
        combined = (
            self.config.weight_b32 * emb_b32 +
            self.config.weight_l14 * emb_l14
        )

        return combined / combined.norm(dim=-1, keepdim=True)
```

### Step 3: Negative Prompt Filtering

```python
class DualCLIPVerifier:
    # ... (previous methods)

    def check_banned_concepts(
        self,
        image: Image.Image,
        banned_concepts: List[str],
        threshold: float = 0.5,
    ) -> Tuple[bool, List[str]]:
        """
        Check if image contains banned concepts.

        Returns:
            Tuple of (safe, list of detected banned concepts)
        """
        detected = []

        for concept in banned_concepts:
            passes, result = self.verify(image, concept)
            if result.combined_score > threshold:
                detected.append(concept)

        return len(detected) == 0, detected
```

### Step 4: Integration with Vortex Pipeline

```python
class VortexPipeline:
    def __init__(self, config: VortexConfig):
        # ... existing init
        self.verifier = DualCLIPVerifier(config.clip)

    async def generate_and_verify(self, recipe: Recipe) -> GenerationResult:
        # Generate content
        output = await self.generate(recipe)

        # Verify semantic compliance
        passes, verification = self.verifier.verify(
            output.frame,
            recipe.visual_track.prompt,
        )

        if not passes:
            logger.warning(
                f"Verification failed: B32={verification.score_b32:.3f}, "
                f"L14={verification.score_l14:.3f}"
            )
            return GenerationResult(
                success=False,
                error="Semantic verification failed",
                verification=verification,
            )

        # Check banned concepts
        safe, banned = self.verifier.check_banned_concepts(
            output.frame,
            recipe.semantic_constraints.banned_concepts,
        )

        if not safe:
            logger.warning(f"Detected banned concepts: {banned}")
            return GenerationResult(
                success=False,
                error=f"Banned concepts detected: {banned}",
            )

        # Compute embedding for BFT
        embedding = self.verifier.compute_embedding(output.frame)

        return GenerationResult(
            success=True,
            output=output,
            verification=verification,
            embedding=embedding,
        )
```

## VRAM Budget

| Component | VRAM |
|-----------|------|
| CLIP-ViT-B-32 (INT8) | ~0.3 GB |
| CLIP-ViT-L-14 (INT8) | ~0.6 GB |
| **Total CLIP** | **~0.9 GB** |

This fits within the 11.8 GB total budget.

## Acceptance Criteria

- [ ] Dual CLIP models load and remain resident in VRAM
- [ ] Verification passes for semantically matching content
- [ ] Verification fails for non-matching content
- [ ] Banned concept detection works
- [ ] Embeddings computed for BFT exchange
- [ ] Integration with Vortex pipeline
- [ ] Performance: verification < 100ms
- [ ] Unit tests for verification logic
- [ ] Integration tests with real images

## Testing

```python
def test_verification_passes_matching():
    verifier = DualCLIPVerifier(CLIPConfig())

    # Load test image of a cat
    image = Image.open("tests/fixtures/cat.jpg")

    passes, result = verifier.verify(image, "a photograph of a cat")
    assert passes
    assert result.score_b32 > 0.7
    assert result.score_l14 > 0.7


def test_verification_fails_non_matching():
    verifier = DualCLIPVerifier(CLIPConfig())

    # Load test image of a cat
    image = Image.open("tests/fixtures/cat.jpg")

    passes, result = verifier.verify(image, "a photograph of a car")
    assert not passes
    assert result.score_b32 < 0.5


def test_banned_concept_detection():
    verifier = DualCLIPVerifier(CLIPConfig())

    image = Image.open("tests/fixtures/violence.jpg")

    safe, detected = verifier.check_banned_concepts(
        image,
        ["violence", "blood", "weapons"],
    )
    assert not safe
    assert len(detected) > 0
```

## Deliverables

1. `vortex/src/vortex/verification/clip.py` - Dual CLIP verifier
2. `vortex/src/vortex/verification/config.py` - Configuration
3. Integration with pipeline
4. Unit and integration tests
5. Documentation

---

**This task is critical for Lane 0 content quality.**
