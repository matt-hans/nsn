# ICN Architecture Context

## Tech Stack

### On-Chain Layer (ICN Chain)
- **Framework**: Polkadot SDK (polkadot-stable2409)
- **Deployment**: ICN Solochain → Parachain (Cumulus)
- **Language**: Rust 1.75+
- **Optional EVM**: Frontier for ERC-20 compatibility (Phase D)
- **Pallets**: 6 custom pallets (stake, reputation, director, bft, pinning, treasury)

### Off-Chain Layer (P2P Mesh)
- **Runtime**: Rust 1.75+ with Tokio 1.35+ async
- **P2P**: rust-libp2p 0.53.0 (GossipSub, Kademlia DHT, QUIC transport)
- **Chain Client**: subxt 0.34+ for type-safe ICN Chain RPC
- **AI Bridge**: PyO3 0.20+ for Rust ↔ Python FFI

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
- **On-Chain (ICN Chain)**: State changes, stake/slashing, reputation events, BFT result storage, challenge disputes
- **Off-Chain (P2P)**: AI generation, BFT coordination, video distribution, erasure coding

### Hierarchical Swarm (4 Tiers)
1. **Tier 0 - Directors**: 5 elected per slot, generate video via Vortex, coordinate BFT (100+ ICN stake)
2. **Tier 1 - Super-Nodes**: 7 regions × 2 replicas, store erasure shards, run CLIP validation (50+ ICN stake)
3. **Tier 2 - Regional Relays**: City-level distribution, minimal storage (10+ ICN stake)
4. **Tier 3 - Viewers**: End users, optional seeding (no stake required)

### Key Data Models

**pallet-icn-stake**:
```rust
StakeInfo { amount, locked_until, role, region, delegated_to_me }
NodeRole: Director | SuperNode | Validator | Relay | None
```

**pallet-icn-reputation**:
```rust
ReputationScore { director_score, validator_score, seeder_score, last_activity }
ReputationEventType: DirectorSlotAccepted (+100) | DirectorSlotRejected (-200) | ...
```

**pallet-icn-director**:
```rust
BftConsensusResult { slot, success, canonical_hash, attestations }
BftChallenge { slot, challenger, deadline, evidence_hash, resolved }
```

### Critical Design Decisions (ADRs)

1. **ICN Chain over Moonbeam Extension**: Full sovereignty, no governance dependency, staged deployment
2. **Staged Deployment**: Solochain MVP → Parachain → Coretime scaling
3. **Static VRAM Residency**: All AI models remain GPU-resident (no swapping, predictable latency)
4. **Dual CLIP Ensemble**: ViT-B-32 + ViT-L-14 weighted ensemble reduces disputes by ~40%
5. **BFT Challenge Period**: 50-block (~5 min) window for disputes, 25 ICN challenger bond, 100 ICN director slashing
6. **ICN Chain Randomness**: Runtime `T::Randomness` source for cryptographically unpredictable director selection
7. **Reed-Solomon Erasure Coding**: 10+4 scheme with 5× geographic replication
8. **Optional Frontier EVM**: Ethereum compatibility available when needed (Phase D)

## Validation Strategy

### On-Chain
- **Staking**: `deposit_stake`, `delegate`, `slash` via pallet-icn-stake
- **Reputation**: Merkle-tree batched events, governance-adjustable retention period
- **BFT**: Challenge mechanism with validator attestations
- **Audits**: Stake-weighted probability for pinning audits

### Off-Chain
- **P2P Tests**: libp2p network integration tests
- **Vortex Pipeline**: GPU memory profiling, generation benchmarks
- **NAT Traversal**: STUN → UPnP → Circuit Relay → TURN fallback stack

### Commands
```bash
# ICN Chain build
cargo build --release --all-features
cargo test --all-features
cargo clippy --all-features -- -D warnings

# Runtime WASM
cargo build --release --target wasm32-unknown-unknown -p icn-runtime

# Run local node
./target/release/icn-node --dev --alice --rpc-port 9944

# Vortex engine
pytest vortex/tests/unit --cov=vortex
python vortex/benchmarks/slot_generation.py --slots 5 --max-time 15

# Integration tests
cd icn-chain/test && pnpm test
```

## Critical Paths

1. **Pallet Development**: stake → reputation → director → bft → pinning → treasury
2. **Off-Chain Nodes**: Director (Vortex) → Super-Node (storage) → Viewer (Tauri)
3. **Deployment**: ICN Testnet → Security audit → Validator onboarding → ICN Mainnet → Parachain (optional)
