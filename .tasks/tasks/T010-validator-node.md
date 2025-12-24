---
id: T010
title: Validator Node Implementation
status: pending
priority: 2
agent: backend
dependencies: [T001, T002, T003, T004, T009]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [off-chain, validator, rust, clip, verification, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - docs/prd.md#section-4.1
  - docs/architecture.md#section-8

est_tokens: 10000
actual_tokens: null
---

## Description

Implement the Validator Node, ICN's independent verification layer for director-generated content. Validators run CLIP (Contrastive Language-Image Pretraining) models to semantically verify that generated video matches the recipe prompt, ensuring content quality and policy compliance without trusting directors.

Validators perform three core functions:
1. **Content Verification**: Run CLIP-ViT-B-32 + CLIP-ViT-L-14 dual ensemble on video frames to compute semantic similarity scores
2. **Attestation Generation**: Sign and timestamp verification results with Ed25519 keypair
3. **Challenge Participation**: Provide attestations during BFT challenge disputes (50-block challenge period, see ADR v8.0.1)

Unlike directors (5 elected per slot), validators operate continuously and permissionlessly with minimum 10 ICN stake. They earn reputation (+5 per correct validation, -10 per incorrect) and are called upon during challenge resolution to adjudicate disputes.

**Technical Approach:**
- Rust 1.75+ with Tokio for async runtime
- CLIP models via ONNX Runtime (optimized inference, no PyTorch dependency)
- rust-libp2p for GossipSub subscription to `/icn/attestations/1.0.0` and `/icn/challenges/1.0.0`
- subxt for on-chain attestation submission and challenge monitoring
- Image processing via image crate (decode video frames from chunks)

**Integration Points:**
- Subscribes to GossipSub `/icn/video/1.0.0` topic for video chunk broadcast
- Publishes to `/icn/attestations/1.0.0` topic with signed CLIP scores
- Monitors `pallet-icn-director::PendingChallenges` storage for active disputes
- Submits attestations via `pallet-icn-director::resolve_challenge` extrinsic

## Business Context

**User Story:** As a Validator Node operator, I want an automated verification service that runs CLIP inference on director outputs and submits attestations, so that I can earn reputation rewards while ensuring content quality without needing expensive GPU hardware.

**Why This Matters:** Validators are ICN's trust-minimization layer. Directors could collude to submit low-quality or policy-violating content. Validators provide economic deterrence (stake slashing during challenges) and cryptographic proof of semantic compliance.

**What It Unblocks:**
- Challenge mechanism in pallet-icn-director (validators provide evidence)
- Community-driven content moderation (anyone with 10 ICN can validate)
- Reputation-weighted validator selection (high-reputation validators prioritized for challenges)
- Reduced BFT failure rate (validators catch issues before finalization)

**Priority Justification:** Priority 2 (Important) - Required for mainnet launch but not critical for ICN Testnet MVP. Directors can initially operate without challenge mechanism enabled. Validators add security layer before significant economic value at stake.

## Acceptance Criteria

- [ ] Binary compiles with `cargo build --release -p icn-validator`
- [ ] CLIP models (ViT-B-32, ViT-L-14) loaded from ONNX format at startup (<10s load time)
- [ ] GossipSub subscription receives video chunks from directors
- [ ] Video frame extraction from chunks successful (AV1/VP9 decode)
- [ ] CLIP inference completes on 5 keyframes in <3 seconds
- [ ] Dual CLIP ensemble computes weighted score (0.4 × B-32 + 0.6 × L-14)
- [ ] Attestation struct signed with Ed25519 keypair and timestamped
- [ ] Attestations broadcast to `/icn/attestations/1.0.0` GossipSub topic
- [ ] Challenge monitor detects active challenges from on-chain storage
- [ ] Attestation submission to `resolve_challenge` extrinsic succeeds
- [ ] Prometheus metrics exposed on port 9101 (validation count, CLIP scores, attestation submissions)
- [ ] CPU-only inference supported (no GPU required, uses ONNX Runtime CPU backend)
- [ ] Graceful shutdown on SIGTERM with P2P cleanup
- [ ] Configuration loaded from TOML (chain endpoint, keypair, CLIP model paths, thresholds)
- [ ] Unit tests for CLIP inference, attestation signing, score aggregation
- [ ] Integration test: receives mock video chunk, computes CLIP score, verifies signature

## Test Scenarios

**Test Case 1: Video Chunk Reception and Verification (Success)**
- Given: Validator subscribed to `/icn/video/1.0.0` GossipSub topic
  And: Director publishes video chunk for slot 100 with recipe prompt "scientist in lab coat"
- When: Validator receives chunk and extracts 5 keyframes
  And: CLIP inference computes: CLIP-B = 0.82, CLIP-L = 0.85
- Then: Ensemble score = 0.4 × 0.82 + 0.6 × 0.85 = 0.838
  And: Attestation generated: `{ slot: 100, score: 0.838, passed: true, timestamp: <unix>, signature: <ed25519> }`
  And: Attestation broadcast to `/icn/attestations/1.0.0`

**Test Case 2: Semantic Mismatch Detection (Failure)**
- Given: Recipe prompt is "peaceful garden scene"
  And: Director submits video of "explosion and fire"
- When: CLIP inference runs
- Then: CLIP-B = 0.15, CLIP-L = 0.12 (low similarity)
  And: Ensemble score = 0.132 (below 0.75 threshold)
  And: Attestation: `{ ..., passed: false, reason: "semantic_mismatch" }`
  And: Attestation broadcast with negative validation

**Test Case 3: Challenge Participation**
- Given: Slot 50 has pending challenge (fraudulent BFT result suspected)
  And: Validator monitored `PendingChallenges[50]` and sees challenge active
- When: Validator retrieves video chunk for slot 50 from DHT
  And: Re-runs CLIP inference independently
  And: Computes score = 0.68 (below threshold, confirming fraud)
- Then: Validator submits attestation to `resolve_challenge(slot=50, [(validator_id, agrees_with_challenge=true, embedding_hash)])`
  And: If >50% validators agree, challenge is upheld and directors slashed

**Test Case 4: ONNX Model Load Failure**
- Given: CLIP model file corrupted or missing
- When: Validator startup attempts to load ONNX model
- Then: Error logged: "Failed to load CLIP model from <path>"
  And: Process exits with code 1 (fail-fast, no degraded mode)
  And: Alerting triggered for operator intervention

**Test Case 5: Keyframe Extraction from Corrupted Chunk**
- Given: Video chunk received with incomplete data (network corruption)
- When: Frame decoder attempts to extract keyframes
- Then: Decoder returns error
  And: Validation skipped for this chunk
  And: Warning logged: "Failed to decode video chunk for slot X"
  And: No attestation submitted (avoid false negatives)

**Test Case 6: Attestation Signature Verification**
- Given: Validator generates attestation with Ed25519 signature
- When: Peer validator receives attestation via GossipSub
- Then: Signature verified against validator's public key (from on-chain stake record)
  And: Timestamp checked (must be within 5 minutes of current time)
  And: If invalid signature or timestamp, attestation rejected and peer downscored

## Technical Implementation

**Required Components:**
- `icn-validator/src/main.rs` - Binary entrypoint with CLI args (--config, --chain-endpoint, --keypair, --models-dir)
- `icn-validator/src/config.rs` - Configuration struct with TOML deserialization
- `icn-validator/src/chain_client.rs` - subxt integration for challenge monitoring and attestation submission
- `icn-validator/src/clip_engine.rs` - ONNX Runtime integration, dual model inference
- `icn-validator/src/video_decoder.rs` - Frame extraction from video chunks (using ffmpeg-next or image crate)
- `icn-validator/src/attestation.rs` - Attestation struct, Ed25519 signing, serialization
- `icn-validator/src/p2p_service.rs` - libp2p GossipSub for chunk reception and attestation broadcast
- `icn-validator/src/challenge_monitor.rs` - Polling `PendingChallenges` storage, trigger re-validation
- `icn-validator/src/metrics.rs` - Prometheus metrics (validation_count, clip_score_histogram, attestation_submissions)

**Validation Commands:**
```bash
# Build
cargo build --release -p icn-validator

# Run unit tests
cargo test -p icn-validator --lib

# Run integration tests
cargo test -p icn-validator --features integration-tests

# Clippy
cargo clippy -p icn-validator -- -D warnings

# Format check
cargo fmt -p icn-validator -- --check

# Run validator (requires CLIP models downloaded)
./target/release/icn-validator \
  --config config/validator.toml \
  --chain-endpoint ws://localhost:9944 \
  --keypair keys/validator.json \
  --models-dir models/clip/

# Check metrics
curl http://localhost:9101/metrics | grep icn_validator_
```

**Code Patterns:**
```rust
// CLIP inference with ONNX Runtime
use ort::{Environment, Session, SessionBuilder, Value};

pub struct ClipEngine {
    session_b32: Session,
    session_l14: Session,
}

impl ClipEngine {
    pub fn new(env: &Environment, models_dir: &Path) -> Result<Self> {
        let session_b32 = SessionBuilder::new(env)?
            .with_model_from_file(models_dir.join("clip-vit-b-32.onnx"))?;
        let session_l14 = SessionBuilder::new(env)?
            .with_model_from_file(models_dir.join("clip-vit-l-14.onnx"))?;

        Ok(Self { session_b32, session_l14 })
    }

    pub fn compute_score(&self, frames: &[Image], prompt: &str) -> Result<f32> {
        // Preprocess frames to [N, 3, 224, 224] tensor
        let image_tensor = preprocess_images(frames)?;
        let text_tensor = tokenize(prompt)?;

        // Run inference
        let image_emb_b32 = self.session_b32.run(vec![image_tensor.clone()])?.remove(0);
        let text_emb_b32 = self.session_b32.run(vec![text_tensor.clone()])?.remove(1);
        let score_b32 = cosine_similarity(&image_emb_b32, &text_emb_b32);

        let image_emb_l14 = self.session_l14.run(vec![image_tensor])?.remove(0);
        let text_emb_l14 = self.session_l14.run(vec![text_tensor])?.remove(1);
        let score_l14 = cosine_similarity(&image_emb_l14, &text_emb_l14);

        // Weighted ensemble
        Ok(score_b32 * 0.4 + score_l14 * 0.6)
    }
}

// Attestation signing
use ed25519_dalek::{Keypair, Signature, Signer};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Attestation {
    pub slot: u64,
    pub validator_id: String,
    pub clip_score: f32,
    pub passed: bool,
    pub timestamp: u64,
    pub signature: Vec<u8>,
}

impl Attestation {
    pub fn sign(mut self, keypair: &Keypair) -> Self {
        let message = format!("{}:{}:{}", self.slot, self.clip_score, self.timestamp);
        let signature: Signature = keypair.sign(message.as_bytes());
        self.signature = signature.to_bytes().to_vec();
        self
    }

    pub fn verify(&self, public_key: &PublicKey) -> bool {
        let message = format!("{}:{}:{}", self.slot, self.clip_score, self.timestamp);
        let signature = Signature::from_bytes(&self.signature).ok()?;
        public_key.verify(message.as_bytes(), &signature).is_ok()
    }
}
```

## Dependencies

**Hard Dependencies** (must be complete first):
- [T001] ICN Chain Repository Fork - Need chain metadata for subxt
- [T002] pallet-icn-stake - Validators must stake minimum 10 ICN
- [T003] pallet-icn-reputation - Validators earn reputation for correct attestations
- [T004] pallet-icn-director - Need `PendingChallenges` storage and `resolve_challenge` extrinsic
- [T009] Director Node Core Runtime - Validators verify director outputs

**Soft Dependencies** (nice to have):
- [T011] Super-Node - Could cache video chunks for validators to retrieve, but not critical
- Content policy definitions - For now, only semantic matching; future could add banned concepts

**External Dependencies:**
- CLIP ONNX models (ViT-B-32, ViT-L-14) - download from Hugging Face or convert from PyTorch
- ONNX Runtime 1.16+ library (CPU backend sufficient)
- Video codec libraries (libav/ffmpeg for frame extraction)

## Design Decisions

**Decision 1: ONNX Runtime instead of PyTorch for CLIP**
- **Rationale:** Validators don't need training, only inference. ONNX Runtime is optimized for inference, supports CPU-only execution, and has smaller binary size. Avoids Python dependency.
- **Alternatives:**
  - PyTorch: Requires Python, heavier runtime, slower CPU inference
  - TensorFlow Lite: Less mature Rust bindings
- **Trade-offs:** (+) 3-5× faster CPU inference, single binary deployment. (-) Requires ONNX model conversion (one-time cost).

**Decision 2: Dual CLIP ensemble (B-32 + L-14) instead of single model**
- **Rationale:** Architecture diversity reduces adversarial attack success. ViT-B-32 is fast (baseline), ViT-L-14 is accurate (catch edge cases). Weighted ensemble (0.4 + 0.6) balances speed and quality.
- **Alternatives:**
  - Single CLIP model: Faster but vulnerable to model-specific adversarial examples
  - Triple ensemble (add RN50): Higher latency, diminishing returns
- **Trade-offs:** (+) ~40% reduction in disputes per PRD. (-) 2× inference time (still <3s target).

**Decision 3: GossipSub for attestation broadcast instead of on-chain submission**
- **Rationale:** Broadcasting all attestations on-chain would exceed 50 TPS limit. GossipSub allows epidemic dissemination with eventual on-chain aggregation during challenges only.
- **Alternatives:**
  - On-chain for every validation: TPS bottleneck, expensive gas
  - Direct P2P to directors: Doesn't create audit trail
- **Trade-offs:** (+) Scalable, low cost. (-) Attestations not immediately verifiable on-chain (only during challenges).

**Decision 4: Ed25519 signatures instead of on-chain account signatures**
- **Rationale:** Validators use same keypair for P2P (PeerId) and attestations. Ed25519 is fast, standard, and libp2p-compatible.
- **Alternatives:**
  - Sr25519 (Substrate native): Would require separate keypair, more complex
  - ECDSA (EVM): Slower verification
- **Trade-offs:** (+) Single keypair for both P2P and attestations. (-) Requires mapping PeerId ↔ AccountId on-chain.

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| ONNX model inference too slow (>5s) | Medium (validation bottleneck) | Medium | Profile on target hardware (ARM, x86). If too slow, reduce keyframe count from 5 to 3. Consider GPU acceleration for validators (RTX 3050 minimum). |
| Video codec incompatibility (can't decode AV1) | High (validators can't verify) | Low | Standardize on VP9 codec (wider support). Bundle ffmpeg libraries in Docker image. Test on multiple platforms (Linux, macOS ARM). |
| Attestation flood attack (spam GossipSub) | Medium (network congestion) | Medium | Implement rate limiting (max 10 attestations/minute per validator). Require proof-of-stake (verify validator staked on-chain before accepting attestations). |
| Challenge deadline missed (validator too slow) | Low (challenge proceeds without attestation) | Low | Set internal deadline buffer (respond within 40 of 50 blocks). Pre-cache video chunks for active challenges. Alert if validation takes >10s. |
| False positive (flagging valid content) | Medium (director reputation penalty) | Low | Conservative thresholds (0.75 for failure, not 0.50). Log borderline cases (0.70-0.75) for manual review. Allow directors to dispute with human escalation. |
| Validator collusion with directors | High (defeats verification) | Low | Economic disincentive (validators lose stake if caught). Require diverse validator set (geographic, operator diversity). Statistical anomaly detection for validators always agreeing. |

## Progress Log

### [2025-12-24] - Task Created

**Created By:** task-creator agent
**Reason:** User request to create comprehensive off-chain node tasks for ICN project
**Dependencies:** T001 (ICN Chain fork), T002 (stake pallet), T003 (reputation pallet), T004 (director pallet), T009 (director node)
**Estimated Complexity:** Standard (10,000 tokens) - Focused verification service with CLIP inference and attestation

## Completion Checklist

**Code Complete:**
- [ ] All acceptance criteria met and verified
- [ ] Unit tests pass with >80% coverage
- [ ] Integration tests pass (mock video chunks, CLIP inference)
- [ ] Clippy warnings resolved
- [ ] Code formatted with rustfmt
- [ ] Documentation comments complete

**Integration Ready:**
- [ ] CLIP inference verified on sample video chunks (5 keyframes in <3s)
- [ ] Attestations successfully broadcast to GossipSub
- [ ] Challenge participation tested (submit to `resolve_challenge`)
- [ ] Metrics verified in Prometheus
- [ ] CPU-only inference confirmed (no GPU required)

**Production Ready:**
- [ ] ONNX models bundled or download script provided
- [ ] Docker image builds with ffmpeg libraries
- [ ] Resource limits tested (max 1GB RAM, 50% CPU)
- [ ] Logs structured and parseable
- [ ] Error paths tested (model load failure, decode error)
- [ ] Monitoring alerts configured (validation failures, attestation rejections)
- [ ] Deployment guide written (systemd service, Docker Compose)

**Definition of Done:**
Task is complete when validator node runs for 24 hours on ICN Testnet testnet, successfully validates 100+ video chunks with <2% false positive rate, participates in 5+ challenge resolutions, and all CLIP inference completes within 3-second budget.
