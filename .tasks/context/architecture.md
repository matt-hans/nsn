# NSN Architecture Context

## Tech Stack

### On-Chain Layer (NSN Chain)
- **Framework**: Polkadot SDK (polkadot-stable2409)
- **Deployment**: NSN Solochain → Parachain (Cumulus)
- **Language**: Rust 1.75+
- **Optional EVM**: Frontier for ERC-20 compatibility (Phase D)
- **Pallets**: 8 custom pallets (stake, reputation, director, bft, storage, treasury, task-market, model-registry)

### Off-Chain Layer (P2P Mesh)
- **Runtime**: Rust 1.75+ with Tokio 1.35+ async
- **P2P**: rust-libp2p 0.53.0 (GossipSub, Kademlia DHT, QUIC transport)
- **Chain Client**: subxt 0.34+ for type-safe NSN Chain RPC
- **AI Bridge**: PyO3 0.20+ for Rust ↔ Python FFI

### Dual-Lane Architecture
- **Lane 0 (Video Generation)**: Epoch-based elections, 5 Directors, BFT consensus, Vortex pipeline
- **Lane 1 (General Compute)**: Task marketplace, arbitrary AI workloads, node-core orchestration

### AI/ML Pipeline (Vortex Engine)
- **Runtime**: Python 3.11
- **Framework**: PyTorch 2.1+
- **Models**:
  - Image Gen: Flux-Schnell (NF4 quantized, 6.0 GB VRAM)
  - Video: LivePortrait (FP16, 3.5 GB VRAM)
  - TTS: Kokoro-82M (FP32, 0.4 GB VRAM)
  - Verify: CLIP-ViT-B-32 + CLIP-ViT-L-14 (INT8 dual ensemble, 0.9 GB VRAM)
- **Quantization**: bitsandbytes 0.41+ for NF4 support
- **Total VRAM**: ~11.8 GB (RTX 3060 12GB minimum)

### Viewer Client
- **Framework**: Tauri 2.0 (Rust backend, React 18 UI)
- **Video**: WebCodecs (hardware accelerated)
- **State**: Zustand 4.x
- **P2P**: libp2p-js 1.x

### DevOps & Infrastructure
- **CI/CD**: GitHub Actions (Rust/Python matrix testing)
- **Monitoring**: Prometheus + Grafana
- **Tracing**: Jaeger (OpenTelemetry)
- **Logging**: Vector + Loki
- **Secrets**: SOPS + age
- **Container**: Docker (--gpus all for Directors)

## System Architecture

### Hybrid On-Chain/Off-Chain Design
- **On-Chain (NSN Chain)**: State changes, stake/slashing, reputation events, BFT result storage, challenge disputes, task marketplace
- **Off-Chain (P2P)**: AI generation, BFT coordination, video distribution, erasure coding, Lane 1 task execution

### Hierarchical Swarm (4 Tiers)
1. **Tier 0 - Directors**: 5 elected per epoch (100 blocks), generate video via Vortex, coordinate BFT (100+ NSN stake)
2. **Tier 1 - Super-Nodes**: 7 regions × 2 replicas, store erasure shards, run CLIP validation (50+ NSN stake)
3. **Tier 2 - Regional Relays**: City-level distribution, minimal storage (10+ NSN stake)
4. **Tier 3 - Viewers**: End users, optional seeding (no stake required)

### Epoch-Based Elections
- **Epoch Duration**: 100 blocks (~10 minutes)
- **On-Deck Set**: 20 Director candidates prepared for next epoch
- **Election**: 5 Directors elected from On-Deck set per epoch

### Key Data Models

**pallet-nsn-stake**:
```rust
StakeInfo { amount, locked_until, role, region, delegated_to_me }
NodeRole: Director | SuperNode | Validator | Relay | None
```

**pallet-nsn-reputation**:
```rust
ReputationScore { director_score, validator_score, seeder_score, last_activity }
ReputationEventType: DirectorSlotAccepted (+100) | DirectorSlotRejected (-200) | ...
```

**pallet-nsn-director**:
```rust
BftConsensusResult { slot, success, canonical_hash, attestations }
BftChallenge { slot, challenger, deadline, evidence_hash, resolved }
```

**pallet-nsn-task-market** (Lane 1):
```rust
Task { id, creator, model_id, input_cid, max_price, deadline, status }
TaskStatus: Pending | Assigned | Completed | Disputed | Expired
```

**pallet-nsn-model-registry**:
```rust
ModelInfo { id, name, capabilities, vram_requirement, version }
ModelCapability: TextGeneration | ImageGeneration | VideoGeneration | Embedding
```

### Critical Design Decisions (ADRs)

1. **NSN Chain over Moonbeam Extension**: Full sovereignty, no governance dependency, staged deployment
2. **Dual-Lane Architecture**: Lane 0 (video) + Lane 1 (general compute) with unified stake/reputation
3. **Epoch-Based Elections**: On-Deck set (20 candidates), 5 elected per epoch (100 blocks)
4. **Staged Deployment**: Solochain MVP → Parachain → Coretime scaling
5. **Static VRAM Residency**: All AI models remain GPU-resident (no swapping, predictable latency)
6. **Dual CLIP Ensemble**: ViT-B-32 + ViT-L-14 weighted ensemble reduces disputes by ~40%
7. **BFT Challenge Period**: 50-block (~5 min) window for disputes, 25 NSN challenger bond, 100 NSN director slashing
8. **NSN Chain Randomness**: Runtime `T::Randomness` source for cryptographically unpredictable director selection
9. **Reed-Solomon Erasure Coding**: 10+4 scheme with 5× geographic replication
10. **Optional Frontier EVM**: Ethereum compatibility available when needed (Phase D)

## Validation Strategy

### On-Chain
- **Staking**: `deposit_stake`, `delegate`, `slash` via pallet-nsn-stake
- **Reputation**: Merkle-tree batched events, governance-adjustable retention period
- **BFT**: Challenge mechanism with validator attestations
- **Audits**: Stake-weighted probability for pinning audits
- **Task Market**: Task creation, assignment, completion, disputes via pallet-nsn-task-market

### Off-Chain
- **P2P Tests**: libp2p network integration tests
- **Vortex Pipeline**: GPU memory profiling, generation benchmarks (Lane 0)
- **node-core**: Scheduler and sidecar tests (Lane 1)
- **NAT Traversal**: STUN → UPnP → Circuit Relay → TURN fallback stack

### Commands
```bash
# NSN Chain build
cargo build --release --all-features
cargo test --all-features
cargo clippy --all-features -- -D warnings

# Runtime WASM
cargo build --release --target wasm32-unknown-unknown -p nsn-runtime

# Run local node
./target/release/nsn-node --dev --alice --rpc-port 9944

# Vortex engine (Lane 0)
pytest vortex/tests/unit --cov=vortex
python vortex/benchmarks/slot_generation.py --slots 5 --max-time 15

# node-core (Lane 1)
cargo test -p scheduler -p sidecar

# Integration tests
cd nsn-chain/test && pnpm test
```

## Critical Paths

1. **Pallet Development**: stake → reputation → director → bft → storage → treasury → task-market → model-registry
2. **Lane 0 (Video)**: Director (Vortex) → Super-Node (storage) → Viewer (Tauri)
3. **Lane 1 (Compute)**: node-core scheduler → sidecar → task-market integration
4. **Deployment**: NSN Testnet → Security audit → Validator onboarding → NSN Mainnet → Parachain (optional)
