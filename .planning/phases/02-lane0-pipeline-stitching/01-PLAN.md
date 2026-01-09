# Phase 2, Plan 1: Lane 0 Pipeline Stitching

## Objective

Wire the Lane 0 video generation flow from prompt to playback by implementing the `lane0` crate orchestration layer. This connects existing components: scheduler epoch tracking → sidecar/vortex execution → P2P video chunk delivery → on-chain BFT consensus storage.

## Execution Context

**Files to read:**
- `node-core/crates/lane0/src/lib.rs` (currently empty placeholder)
- `node-core/crates/scheduler/src/epoch.rs` (epoch tracker, On-Deck events)
- `node-core/crates/scheduler/src/state_machine.rs` (lane mode management)
- `node-core/crates/p2p/src/topics.rs` (GossipSub topic definitions)
- `node-core/crates/p2p/src/video.rs` (video chunk encoding/signing)
- `node-core/sidecar/src/service.rs` (gRPC sidecar interface)
- `node-core/sidecar/src/client.rs` (gRPC client for calling sidecar)
- `node-core/crates/types/src/lib.rs` (Recipe, VideoChunk, NodeCapability types)
- `node-core/crates/chain-client/src/lib.rs` (chain submission)
- `vortex/src/vortex/pipeline.py` (VortexPipeline.generate_slot interface)
- `vortex/src/vortex/plugins/runner.py` (plugin CLI entry point)

**Build commands:**
```bash
cd node-core && cargo build --release -p lane0
cd node-core && cargo test -p lane0
cd node-core && cargo clippy -p lane0 -- -D warnings
```

## Context

**Current State:**
- `lane0` crate is an empty placeholder (6 lines, just a comment)
- Scheduler has epoch tracking with On-Deck notification (120s before epoch)
- P2P layer has complete GossipSub topics and video chunk infrastructure
- Sidecar has complete gRPC service for calling Vortex plugins
- Vortex has VortexPipeline.generate_slot() returning GenerationResult with CLIP embedding
- On-chain pallets validated in Phase 1 (stake→reputation→director→bft chain works)

**Architecture Summary:**

```
Scheduler                    Lane0 Crate (TO IMPLEMENT)
┌─────────────────┐          ┌──────────────────────────────┐
│ EpochTracker    │──OnDeck──▶│ DirectorService              │
│ SchedulerState  │          │   ├── RecipeProcessor        │
└─────────────────┘          │   ├── VortexClient           │
                             │   ├── BftParticipant         │
                             │   └── ChunkPublisher         │
                             └──────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
     Sidecar (gRPC)              P2P Layer               Chain Client
     ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
     │ ExecuteTask │          │ VideoChunks │          │ submit_bft  │
     │ (Vortex)    │          │ BftSignals  │          │ attestation │
     └─────────────┘          └─────────────┘          └─────────────┘
```

**What Already Exists (no changes needed):**
- `p2p/topics.rs`: VideoChunks, BftSignals, Attestations topics
- `p2p/video.rs`: build_video_chunks(), verify_video_chunk(), publish_video_chunks()
- `sidecar/client.rs`: SidecarClient::execute_task()
- `types/lib.rs`: Recipe, VideoChunk, VideoChunkHeader, NodeCapability, LaneMode
- `scheduler/epoch.rs`: EpochTracker, EpochEvent::OnDeck

## Tasks

### Task 1: Create Lane 0 Crate Module Structure

Set up the `lane0` crate with proper module organization and shared types.

**Files to create/modify:**
- `node-core/crates/lane0/src/lib.rs` - Module exports and crate documentation
- `node-core/crates/lane0/src/director.rs` - DirectorService struct and state
- `node-core/crates/lane0/src/recipe.rs` - Recipe processing and validation
- `node-core/crates/lane0/src/vortex_client.rs` - Sidecar gRPC wrapper for Vortex
- `node-core/crates/lane0/src/bft.rs` - BFT participant logic
- `node-core/crates/lane0/src/publisher.rs` - Video chunk publisher
- `node-core/crates/lane0/src/error.rs` - Error types with thiserror
- `node-core/crates/lane0/Cargo.toml` - Add required dependencies

**Acceptance criteria:**
- [ ] Module compiles with `cargo build -p lane0`
- [ ] All modules have proper documentation comments
- [ ] Error types defined for each failure mode
- [ ] DirectorService struct defined with state management

**Checkpoint:** `cargo build -p lane0` succeeds

### Task 2: Implement Director Service Core

Implement DirectorService that manages director lifecycle: epoch notifications, mode transitions, and slot generation coordination.

**Implementation:**
```rust
pub struct DirectorService {
    state: DirectorState,
    config: DirectorConfig,
    epoch_rx: mpsc::Receiver<EpochEvent>,
    vortex_client: VortexClient,
    bft: BftParticipant,
    publisher: ChunkPublisher,
    p2p: P2pHandle,
    chain_client: ChainClient,
}

pub enum DirectorState {
    Standby,              // Not elected, waiting
    OnDeck { epoch: u32 },// 120s before epoch, pre-warming
    Active { epoch: u32, slot: u32 }, // Currently generating
    Draining,             // Finishing last slot
}
```

**Key methods:**
- `new()` - Initialize with dependencies
- `run()` - Main event loop (epoch events, recipe events)
- `on_epoch_start()` - Transition to Active state
- `on_epoch_end()` - Transition to Draining, then Standby
- `process_slot()` - Orchestrate single slot generation

**Files to modify:**
- `node-core/crates/lane0/src/director.rs`

**Acceptance criteria:**
- [ ] DirectorService compiles and can be instantiated
- [ ] State machine transitions correctly (Standby → OnDeck → Active → Draining → Standby)
- [ ] Epoch event subscription works
- [ ] Unit tests for state transitions

**Checkpoint:** `cargo test -p lane0 -- director` passes

### Task 3: Implement Recipe Processor

Handle incoming recipes from P2P GossipSub and validate before generation.

**Implementation:**
- Subscribe to `/nsn/recipes/1.0.0` topic
- Parse and validate Recipe structure
- Queue recipes for slot generation
- Handle recipe conflicts (same slot, different content)

**Key types (from nsn-types):**
```rust
pub struct Recipe {
    pub recipe_id: String,
    pub slot_params: SlotParams,
    pub audio_track: AudioTrack,
    pub visual_track: VisualTrack,
    pub semantic_constraints: SemanticConstraints,
    pub security: SecurityMetadata,
}
```

**Files to modify:**
- `node-core/crates/lane0/src/recipe.rs`

**Acceptance criteria:**
- [ ] Recipe deserialization from CBOR/JSON
- [ ] Validation of required fields
- [ ] Slot parameter bounds checking
- [ ] Queue management for pending recipes

**Checkpoint:** `cargo test -p lane0 -- recipe` passes

### Task 4: Implement Vortex Client

Wrapper around sidecar gRPC to call Vortex pipeline for video generation.

**Implementation:**
```rust
pub struct VortexClient {
    sidecar: SidecarClient,
    timeout_ms: u64,
}

impl VortexClient {
    pub async fn generate_slot(&self, recipe: &Recipe) -> Result<GenerationOutput, VortexError> {
        let payload = serde_json::to_string(recipe)?;
        let response = self.sidecar.execute_task(ExecuteTaskRequest {
            task_id: uuid::Uuid::new_v4().to_string(),
            plugin_name: "vortex-lane0".to_string(),
            input_cid: "".to_string(), // Recipe in payload
            parameters: payload,
            lane: 0,
            timeout_ms: self.timeout_ms,
            ..Default::default()
        }).await?;

        Ok(GenerationOutput::from_response(response)?)
    }
}

pub struct GenerationOutput {
    pub video_chunks: Vec<Vec<u8>>,
    pub audio_waveform: Vec<u8>,
    pub clip_embedding: Vec<f32>, // 512-dim from dual CLIP
    pub generation_time_ms: u64,
    pub slot_id: u64,
}
```

**Files to modify:**
- `node-core/crates/lane0/src/vortex_client.rs`

**Acceptance criteria:**
- [ ] Calls sidecar ExecuteTask with correct parameters
- [ ] Parses GenerationOutput from JSON response
- [ ] Handles timeout and error cases
- [ ] CLIP embedding extracted for BFT consensus

**Checkpoint:** Integration test with mock sidecar passes

### Task 5: Implement BFT Participant

Handle BFT consensus for video verification using CLIP embeddings.

**Implementation:**
```rust
pub struct BftParticipant {
    p2p: P2pHandle,
    my_keypair: identity::Keypair,
    threshold: u8, // 3 of 5
}

impl BftParticipant {
    pub async fn run_consensus(
        &self,
        slot: u64,
        my_embedding: Vec<f32>,
        timeout_ms: u64,
    ) -> Result<BftResult, BftError> {
        // 1. Publish my embedding to /nsn/bft/1.0.0
        let msg = BftSignal::Embedding {
            slot,
            embedding: my_embedding.clone(),
            signer: self.my_keypair.public().to_peer_id(),
        };
        self.p2p.publish(Topic::BftSignals, msg).await?;

        // 2. Collect embeddings from other directors
        let embeddings = self.collect_embeddings(slot, timeout_ms).await?;

        // 3. Verify consensus (cosine similarity threshold)
        let consensus = self.verify_consensus(&embeddings)?;

        Ok(consensus)
    }
}

pub struct BftResult {
    pub slot: u64,
    pub canonical_hash: [u8; 32], // Blake3 of agreed embedding
    pub signers: Vec<PeerId>,
    pub success: bool,
}
```

**Files to modify:**
- `node-core/crates/lane0/src/bft.rs`

**Acceptance criteria:**
- [ ] Publishes CLIP embedding to BftSignals topic
- [ ] Collects embeddings from other directors with timeout
- [ ] Computes cosine similarity for consensus
- [ ] Returns BftResult with canonical hash for chain submission

**Checkpoint:** `cargo test -p lane0 -- bft` passes

### Task 6: Implement Video Chunk Publisher

Batch video output into signed chunks and publish to P2P network.

**Implementation:**
```rust
pub struct ChunkPublisher {
    p2p: P2pHandle,
    keypair: identity::Keypair,
    chunk_size: usize, // 1 MiB default
}

impl ChunkPublisher {
    pub async fn publish_video(
        &self,
        slot: u64,
        content_id: &str,
        video_data: &[u8],
    ) -> Result<Vec<VideoChunkHeader>, PublishError> {
        // Use existing p2p::video::build_video_chunks()
        let chunks = video::build_video_chunks(
            &self.keypair,
            slot,
            content_id,
            video_data,
            self.chunk_size,
        )?;

        // Publish each chunk
        for chunk in &chunks {
            self.p2p.publish(Topic::VideoChunks, chunk).await?;
        }

        Ok(chunks.iter().map(|c| c.header.clone()).collect())
    }
}
```

**Files to modify:**
- `node-core/crates/lane0/src/publisher.rs`

**Acceptance criteria:**
- [ ] Chunks video data into 1 MiB segments
- [ ] Signs each chunk with Ed25519 keypair
- [ ] Publishes to VideoChunks topic
- [ ] Returns headers for tracking/verification

**Checkpoint:** `cargo test -p lane0 -- publisher` passes

### Task 7: Integrate Slot Generation Pipeline

Wire all components together in DirectorService.process_slot() method.

**Implementation:**
```rust
impl DirectorService {
    async fn process_slot(&mut self, recipe: Recipe) -> Result<SlotResult, SlotError> {
        let slot_id = recipe.slot_params.slot_id;
        tracing::info!(slot = slot_id, "Starting slot generation");

        // 1. Call Vortex for generation
        let output = self.vortex_client.generate_slot(&recipe).await?;
        tracing::info!(slot = slot_id, ms = output.generation_time_ms, "Generation complete");

        // 2. Run BFT consensus with other directors
        let bft_result = self.bft.run_consensus(
            slot_id,
            output.clip_embedding,
            self.config.bft_timeout_ms,
        ).await?;

        if !bft_result.success {
            return Err(SlotError::BftFailed(bft_result));
        }

        // 3. Publish video chunks to P2P
        let headers = self.publisher.publish_video(
            slot_id,
            &output.content_id,
            &output.video_data,
        ).await?;

        // 4. Submit BFT result to chain
        self.chain_client.submit_bft_result(
            slot_id,
            bft_result.canonical_hash,
            bft_result.signers,
        ).await?;

        Ok(SlotResult {
            slot_id,
            chunk_count: headers.len(),
            bft_hash: bft_result.canonical_hash,
            generation_ms: output.generation_time_ms,
        })
    }
}
```

**Files to modify:**
- `node-core/crates/lane0/src/director.rs`

**Acceptance criteria:**
- [ ] Full slot pipeline executes in sequence
- [ ] Errors propagate correctly with context
- [ ] Metrics emitted for observability
- [ ] Chain submission completes after P2P publish

**Checkpoint:** Integration test with mock dependencies passes

### Task 8: End-to-End Director Integration Test

Create integration test that exercises full director lifecycle with mock P2P and sidecar.

**Test scenario:**
1. Setup: Create DirectorService with mock dependencies
2. Epoch notification: Receive OnDeck event, transition to Active
3. Recipe arrival: Receive recipe from mock P2P
4. Generation: Mock sidecar returns GenerationOutput
5. BFT: Mock other directors sending embeddings
6. Publish: Verify video chunks published to mock P2P
7. Chain: Verify BFT result submitted to mock chain client
8. Epoch end: Transition back to Standby

**Files to create:**
- `node-core/crates/lane0/tests/director_integration.rs`

**Acceptance criteria:**
- [ ] Full lifecycle test passes
- [ ] All mock interactions verified
- [ ] State transitions logged correctly
- [ ] Error cases tested (timeout, BFT failure, sidecar error)

**Checkpoint:** `cargo test -p lane0 --test director_integration` passes

## Verification

**Build and test:**
```bash
cd node-core && cargo build --release -p lane0
cd node-core && cargo test -p lane0
cd node-core && cargo clippy -p lane0 -- -D warnings
```

**Integration verification:**
```bash
# Run with mock dependencies
cd node-core && cargo test -p lane0 --test director_integration -- --nocapture
```

**Expected output:**
- All unit tests pass for each module
- Integration test completes full lifecycle
- No clippy warnings
- Compiles in release mode

## Success Criteria

- [ ] Lane 0 crate compiles with all modules
- [ ] DirectorService manages lifecycle correctly
- [ ] Recipe processing validates and queues
- [ ] Vortex client calls sidecar successfully
- [ ] BFT participant runs consensus with timeout
- [ ] Video chunks published to P2P
- [ ] Chain submission works via chain-client
- [ ] Integration test passes with mock dependencies
- [ ] No clippy warnings, cargo fmt clean

## Output

**Artifacts:**
- Fully implemented `node-core/crates/lane0/` crate
- DirectorService with state machine
- Integration tests for director lifecycle
- Module documentation

**Dependencies for next phases:**
- Phase 4 (Viewer Web Extraction) needs P2P subscription to receive published chunks
- Phase 5 (Multi-Node E2E Simulation) will test multiple directors running BFT

**Open questions for user clarification:**
1. Should BFT timeout be configurable or hardcoded (suggest 5000ms)?
2. How many retry attempts for sidecar failures before slot abort (suggest 1)?
3. Should we emit Prometheus metrics in this phase or defer to Phase 5?
