# Neural Sovereign Network (NSN)
# Technical Architecture Document v2.0

**Version:** 2.0
**Date:** 2025-12-29
**Status:** Approved - NSN Dual-Lane Architecture
**PRD Reference:** NSN_PRD_v10.0

---

## 1. Overview & Context

### 1.1 Purpose

This Technical Architecture Document (TAD) translates the NSN PRD v10.0 into concrete technical architecture and implementation guidance. It serves as reference for:

- **Engineering teams** implementing the system
- **DevOps/SRE** deploying infrastructure
- **Security teams** auditing the system
- **External auditors** reviewing Substrate pallets

### 1.2 Scope

**In Scope:** NSN Chain runtime (FRAME pallets including task-market and model-registry), off-chain node architecture, P2P network topology, dual-lane architecture (Lane 0 video + Lane 1 general compute), AI inference pipeline (Vortex), node-core orchestration, data storage, security architecture, deployment procedures, observability, optional Frontier EVM compatibility.

**Out of Scope:** UI/UX design, marketing strategy, legal/regulatory specifics, detailed tokenomics modeling.

### 1.3 Key References

| Document | Version | Description |
|----------|---------|-------------|
| NSN_PRD_v10.0 | 10.0.0 | Product Requirements Document |
| Polkadot SDK | polkadot-stable2409 | Runtime framework |
| Cumulus Documentation | Latest | Parachain support (Phase B) |
| Frontier Documentation | Latest | Optional EVM compatibility |
| libp2p Specification | 0.53.0 | P2P networking protocols |

### 1.4 System Context (C4 Level 1)

The NSN system consists of four primary domains:

1. **On-Chain Layer (NSN Chain):** NSN's own Polkadot SDK chain with custom Substrate pallets for staking, reputation, epoch-based elections, BFT result storage, pinning deals, treasury, task marketplace, and model registry. Solo phase uses controlled validators; parachain phase inherits Polkadot relay chain security (~$20B economic security).

2. **Off-Chain Layer (P2P Mesh):** Hierarchical network of nodes with epoch-based elections and On-Deck protocol - Directors generate AI output, Validators verify semantic compliance, Super-Nodes store erasure-coded shards, Regional Relays distribute content, Viewers consume streams.

3. **Lane 0 (Video Generation):** GPU-resident Vortex pipeline running Flux-Schnell, LivePortrait, Kokoro-82M, and dual CLIP ensemble for deterministic video generation.

4. **Lane 1 (General Compute):** node-core orchestration system with scheduler and sidecar for arbitrary AI workloads via open task marketplace.

---

## 2. Architectural Drivers & Constraints

### 2.1 Functional Requirements Summary

| Requirement | PRD Section | Technical Implication |
|-------------|-------------|----------------------|
| Multi-Director BFT | §3.3 | 5 directors, 3-of-5 consensus, off-chain coordination |
| Semantic Verification | §10.2 | Dual CLIP ensemble (ViT-B-32, ViT-L-14) |
| Static VRAM Residency | §10.1 | All models loaded at startup, no swapping |
| Hierarchical Swarm | §13.2 | 4-tier architecture |
| Erasure Coding | §3.4 | Reed-Solomon (10+4), 5x replication |
| NAT Traversal | §13.1 | STUN → UPnP → Circuit Relay → TURN fallback |
| BFT Challenge Period | §3.3 | 50-block dispute window with stake slashing |
| VRF Elections | §3.3 | NSN Chain randomness source |

### 2.2 Non-Functional Requirements

| Attribute | Target | Priority |
|-----------|--------|----------|
| **Availability** | 99.5% playback continuity | High |
| **Latency** | <45s glass-to-glass | High |
| **Throughput** | 100+ TPS (own chain) | Medium |
| **Scalability** | 500+ mainnet nodes | Medium |
| **Security** | 8-layer model | Critical |
| **Cost** | $80k-$200k MVP | High |

### 2.3 Constraints

| Constraint | Type | Impact |
|------------|------|--------|
| NSN Chain validator set (solo phase) | Technical | Initial security depends on trusted operators |
| RTX 3060 12GB floor | Technical | VRAM budget = 11.8GB max |
| 3-6 month timeline | Business | Limited scope for MVP |
| $80k-$200k budget | Business | 2-3 developers max |
| Cumulus compatibility | Technical | Pallets must work with parachain migration |

### 2.4 Quality Attributes Priority

1. **Security** - On-chain funds at risk
2. **Availability** - Continuous playback required
3. **Latency** - 45s glass-to-glass is tight
4. **Scalability** - 500+ nodes at mainnet
5. **Cost** - Startup budget constraints
6. **Maintainability** - Long-term evolution

---

## 3. Architectural Decisions (ADRs)

### ADR-001: NSN Chain over Moonbeam Extension

**Decision:** Build NSN as its own Polkadot SDK chain with custom pallets instead of deploying pallets to Moonbeam's runtime.

**Rationale:**
- Eliminates Moonbeam governance dependency (referendum approval was HIGH risk)
- Full control over runtime upgrades and chain parameters
- "Deployable by anyone" - open-source chain artifacts
- Enables staged security model (solo → parachain → coretime)
- No external approval needed for protocol evolution

**Trade-offs:**
- (+) Full sovereignty over chain operations
- (+) No governance risk or external dependency
- (+) Faster iteration on chain logic
- (-) More operational responsibility (validators)
- (-) Shared security only after parachain migration (Phase B)
- (-) Initial security relies on trusted validator set

### ADR-012: Dual-Lane Architecture (Lane 0 + Lane 1)

**Decision:** Split NSN into two execution lanes: Lane 0 (deterministic video generation with epochs/BFT) and Lane 1 (open compute marketplace with task-market pallet).

**Rationale:**
- Lane 0 preserves proven epoch-based elections with On-Deck protocol for video generation
- Lane 1 unlocks arbitrary AI workloads (inference, training, fine-tuning) via marketplace
- Separates concerns: deterministic consensus (Lane 0) vs. flexible marketplace (Lane 1)
- Unified reputation and stake system across both lanes
- Enables gradual expansion from video to general AI compute

**Trade-offs:**
- (+) Flexibility: arbitrary tasks beyond video generation
- (+) Market-driven pricing for Lane 1 tasks
- (+) Reuses NSN infrastructure (stake, reputation, P2P)
- (-) Increased complexity with two execution paths
- (-) Requires node-core orchestration layer for Lane 1

### ADR-013: Epoch-Based Elections with On-Deck Protocol

**Decision:** Replace per-slot elections with epoch-based elections (100 blocks) and On-Deck set (20 Directors), electing 5 Directors per epoch from On-Deck pool.

**Rationale:**
- Reduces on-chain election overhead from every slot to every 100 blocks
- On-Deck set provides predictable Director pipeline (20 candidates → 5 elected)
- Enables Directors to prepare models and VRAM ahead of time
- Smoother transition between epochs
- Better UX for node operators (predictable scheduling)

**Trade-offs:**
- (+) Lower computational overhead (election every 100 blocks vs. every slot)
- (+) Predictable Director pipeline for warm-up
- (+) Fairer distribution (all On-Deck members get turns)
- (-) Less dynamic than per-slot elections
- (-) Requires On-Deck set maintenance logic

### ADR-002: Hybrid On-Chain/Off-Chain Architecture

**Decision:** On-chain = state changes (stake, reputation, BFT results). Off-chain = computation (AI generation, BFT negotiation, video distribution).

**Rationale:** Cost-effective, low latency.

**Trade-offs:** Off-chain nodes must be trusted for BFT result accuracy; mitigated by challenge period and slashing.

### ADR-003: libp2p over Custom P2P Stack

**Decision:** Use libp2p (rust-libp2p 0.53.0) with GossipSub, Kademlia DHT, and QUIC transport.

**Rationale:** Battle-tested, NAT traversal built-in, Substrate ecosystem familiarity.

### ADR-004: Dual CLIP Ensemble for Semantic Verification

**Decision:** CLIP-ViT-B-32 (0.4 weight) + CLIP-ViT-L-14 (0.6 weight) ensemble with independent thresholds.

**Rationale:** ~40% reduction in disputes, robust to adversarial inputs.

### ADR-005: Static VRAM Residency

**Decision:** All models (Flux, LivePortrait, Kokoro, CLIP×2) remain resident in VRAM at all times.

**Rationale:** Predictable latency, no thrashing. RTX 3060 12GB is hard minimum.

### ADR-006: BFT Challenge Period

**Decision:** 50-block (~5 minute) challenge period before finalization. Challengers stake 25 NSN, successful challenges slash 100 NSN per fraudulent director.

**Rationale:** Economic security against collusion.

### ADR-007: NSN Chain Randomness

**Decision:** Use NSN Chain's runtime randomness source (`T::Randomness` trait) for cryptographically secure On-Deck and epoch-based director selection.

**Rationale:** Cryptographically unpredictable, verifiable on-chain. In solo mode, uses BABE-like randomness; in parachain mode, integrates with relay chain randomness.

### ADR-008: Reed-Solomon Erasure Coding (10+4)

**Decision:** 10+4 Reed-Solomon coding, 5× replication across regions.

**Rationale:** 1.4× overhead vs 3× for replication at same durability.

### ADR-009: Optional Frontier EVM on NSN Chain

**Decision:** Ethereum compatibility is optional, provided via Frontier (pallet-evm + pallet-ethereum) on NSN Chain when needed. Alternative: Snowbridge for Ethereum mainnet bridging post-parachain.

**Rationale:**
- Core ICN functionality does not require EVM
- ICN token is native (not ERC-20)
- Frontier available for dApp developer convenience
- Snowbridge provides Ethereum mainnet reach when ICN becomes parachain

### ADR-010: Hierarchical Swarm Topology

**Decision:** 4-tier hierarchy: Directors → Super-Nodes → Regional Relays → Edge Viewers.

**Rationale:** O(log N) propagation, regional locality.

### ADR-011: Staged Deployment Model

**Decision:** Deploy ICN in stages: Solochain MVP → Parachain (Cumulus) → Coretime Scaling.

**Rationale:**
- Solochain enables fast MVP with controlled validator set
- Parachain migration provides Polkadot shared security when adoption justifies cost
- Coretime enables elastic scaling as demand grows
- Each phase has clear entry/exit criteria

**Trade-offs:**
- (+) Lower initial operational complexity
- (+) No slot auction required for MVP
- (+) Gradual security improvement with adoption
- (-) Solo phase has weaker security guarantees
- (-) Parachain migration requires engineering effort

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ON-CHAIN LAYER (NSN Chain)                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │pallet-nsn-  │ │pallet-nsn-  │ │pallet-nsn-  │ │pallet-nsn-  │   │
│  │   stake     │ │ reputation  │ │   epochs    │ │   pinning   │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │pallet-nsn-  │ │pallet-nsn-  │ │pallet-nsn-  │ │pallet-nsn-  │   │
│  │  treasury   │ │     bft     │ │task-market  │ │model-registry│  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
│  ┌──────────────────────────────┐                                   │
│  │  OPTIONAL: FRONTIER EVM      │                                   │
│  │  (pallet-evm + pallet-eth)   │                                   │
│  └──────────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │ subxt RPC / Events
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OFF-CHAIN LAYER (P2P Mesh)                       │
│  LANE 0 (Video):                                                    │
│    - Directors (GPU + Vortex Engine, Epoch-elected)                 │
│    - On-Deck Set (20 candidates, 5 elected per epoch)               │
│    - Validators (CLIP verification)                                 │
│                                                                     │
│  LANE 1 (General Compute):                                          │
│    - node-core Scheduler (task matching + orchestration)            │
│    - Sidecar Runtime (arbitrary AI workload execution)              │
│    - Model Registry (capability discovery)                          │
│                                                                     │
│  SHARED:                                                            │
│    - Super-Nodes (7 Regions, Erasure Storage)                       │
│    - Regional Relays (Cache + Distribution)                         │
│    - Edge Viewers (Tauri App, WebCodecs)                            │
│                                                                     │
│  Transport: libp2p + QUIC + GossipSub + Kademlia DHT                │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Director Node Components

- **Core Runtime (Tokio):** Election Monitor, Slot Scheduler, BFT Coordinator
- **Vortex Pipeline (Python):** Recipe → Audio Gen (Kokoro) → Actor Gen (Flux) → Video Warp (LivePortrait) → CLIP Verify
- **VRAM Manager (CUDA):** Model registry (resident), output buffer pool (pre-allocated), memory pressure monitor
- **P2P Network Service:** GossipSub, Kademlia DHT, QUIC transport, Reputation Oracle (cached)
- **Chain Client (subxt):** Block subscriptions, BFT result submission, reputation queries
- **Observability:** Prometheus (port 9100), OpenTelemetry (port 4317), JSON structured logging

### 4.3 Pallet Interaction Flows

**Director Election Flow:**
1. `on_initialize(block)` triggers election
2. `pallet-nsn-director` queries `pallet-nsn-stake` for stakes/roles
3. Queries `pallet-nsn-reputation` for scores
4. Emits `Event::DirectorsElected(slot, directors)`

**BFT Result Submission Flow:**
1. `submit_bft_result(slot, directors, hash)` called
2. Verifies submitter is elected director
3. Stores pending result with 50-block challenge period
4. After challenge period (no challenge), calls `finalize_slot()`
5. Records reputation events via `pallet-nsn-reputation`

**Challenge Resolution Flow:**
1. `challenge_bft_result()` reserves 25 NSN bond
2. `resolve_challenge()` counts validator attestations
3. If upheld: slash 100 NSN per director, refund challenger + 10 NSN reward
4. If rejected: slash challenger bond, finalize original result

### 4.4 Data Architecture

**On-Chain Data:**
- `StakeInfo`: amount, locked_until, role, region, delegated_to_me
- `ReputationScore`: director_score, validator_score, seeder_score, last_activity
- `BftConsensusResult`: slot, success, canonical_hash, attestations
- `PinningDeal`: deal_id, creator, shards, expires_at, total_reward, status

**Off-Chain Data:**

| Data Type | Storage | Retention | Replication |
|-----------|---------|-----------|-------------|
| Video Chunks | Super-Nodes | 7 days | 5× (erasure coded) |
| Recipes | GossipSub + DHT | 24 hours | Epidemic |
| CLIP Embeddings | P2P Exchange | Per slot | 5× (directors) |
| Merkle Proofs | On-chain | 6 months | Blockchain |

### 4.5 API & Integration

| API | Type | Purpose | Auth |
|-----|------|---------|------|
| NSN Chain RPC | JSON-RPC 2.0 | Chain queries | Public |
| NSN Chain WS | WebSocket | Block subscriptions | Public |
| Director gRPC | gRPC | BFT coordination | mTLS + PeerId |
| P2P GossipSub | libp2p | Recipe/video broadcast | Ed25519 signing |

**GossipSub Topics:**
- `/icn/recipes/1.0.0` - Recipe JSON
- `/icn/video/1.0.0` - Video Chunks
- `/icn/bft/1.0.0` - CLIP Embeddings
- `/icn/attestations/1.0.0` - Validator attestations
- `/icn/challenges/1.0.0` - Challenges

---

## 5. Technology Stack

### 5.1 Frontend (Viewer Client)
- **Tauri 2.0** + React 18.x + WebCodecs + Zustand + libp2p-js

### 5.2 Backend / Off-Chain
- **Rust 1.75+** + Tokio + rust-libp2p 0.53.0 + subxt + PyO3

### 5.3 AI/ML Pipeline
- **Python 3.11** + PyTorch 2.1+ + Flux-Schnell (NF4) + LivePortrait (FP16) + Kokoro-82M (FP32) + CLIP (INT8)

### 5.4 On-Chain
- **Polkadot SDK** (polkadot-stable2409) + ICN custom pallets + Optional Frontier EVM

### 5.5 Infrastructure
- Bare metal/Cloud GPU (RTX 3060+) + Docker + Kubernetes (optional) + Coturn (STUN/TURN)

### 5.6 DevOps
- GitHub Actions + Prometheus + Grafana + Jaeger + Vector + Loki + AlertManager

### 5.7 Security Tools
- cargo-audit, cargo-deny, SOPS + age, rustls

---

## 6. Deployment & Operations

### 6.1 Deployment Model

**NSN Chain Node:** Linux (Ubuntu 22.04+), 4+ CPU cores, 16GB RAM, 500GB SSD, `nsn-node` binary

**Director Node:** Bare metal/Cloud GPU (RTX 3060+), Ubuntu 22.04, NVIDIA 535+, Docker with `--gpus all`, nsn-director container (Rust + Python + model weights ~15GB)

**Super-Node:** Kubernetes cluster, 2 replicas per region (7 regions), 10TB PersistentVolumeClaims

### 6.2 Environments

| Environment | Purpose | Chain |
|-------------|---------|-------|
| Local Dev | Developer testing | Local ICN solochain |
| NSN Testnet | Integration testing | Public ICN testnet |
| NSN Mainnet | Production | ICN mainnet (solo → parachain) |

### 6.3 Scalability & Resilience

**Horizontal Scaling:**
- Validators: 3-5 (solo phase), 50+ (parachain phase)
- Directors: 5 elected per slot (fixed)
- Super-Nodes: 7 regions × 2 replicas = 14 minimum
- Regional Relays: Auto-scale based on viewer count
- Viewers: Unlimited (P2P assisted)

**Resilience Patterns:**
- Validator Failure → 2/3 consensus still possible (solo), relay chain handles (parachain)
- Director Failure → 3-of-5 still reaches consensus
- Super-Node Failure → 2 replicas + erasure coding
- Relay Failure → Viewer falls back to Super-Node
- NSN Chain Outage → Off-chain continues, BFT results queued

### 6.4 Key Observability Metrics

- `icn_vortex_generation_time_seconds` (P99 < 15s)
- `icn_bft_round_duration_seconds` (P99 < 10s)
- `icn_p2p_connected_peers` (> 10)
- `icn_total_staked_tokens`
- `icn_slashing_events_total`
- `icn_chain_block_height`
- `icn_chain_finalized_height`

**Critical Alerts:**
- DirectorSlotMissed, VortexOOM, ChainDisconnected
- StakeConcentration (region > 25%), BftLatencyHigh
- ValidatorOffline, ConsensusStalled

### 6.5 Backup & DR

| Component | Backup Strategy | RPO | RTO |
|-----------|-----------------|-----|-----|
| On-Chain State | Blockchain | 0 | 0 |
| Video Content | Erasure coding (5×) | 1 slot | 1 min |
| Model Weights | Object storage | 1 day | 1 hour |
| Node Config | Git + encrypted secrets | Realtime | 10 min |
| Chain Database | Periodic snapshots | 6 hours | 30 min |

---

## 7. Security & Compliance

### 7.1 Threat Model

| Asset | Threats | Controls |
|-------|---------|----------|
| Staked Tokens | Key theft, rug pull, contract bug | HSM, multisig, audits |
| Validator Keys | Compromise, insider threat | HSM, key rotation, geographic distribution |
| Reputation Scores | Score manipulation, Sybil, collusion | Merkle proofs, stake + PoI, challenge period |
| Video Content | CSAM, copyright, adversarial inputs | CLIP filter, content policy, ensemble |
| P2P Network | Eclipse, DDoS, Sybil nodes | Peer diversity, rate limiting, stake gating |
| Director Nodes | RCE via model, side-channel, key extraction | Sandboxed exec, isolated VRAM, secure enclave |

**Attacker Profiles:**
1. Script Kiddie → PoW on recipes, rate limiting
2. Competitor → Geographic diversity, reputation
3. Nation State → E2E encryption, decentralization
4. Malicious Insider → Multisig, code review, audits
5. Validator Collusion → 2/3 threshold, slashing, monitoring

### 7.2 Authentication & Authorization

| Layer | Mechanism |
|-------|-----------|
| P2P Identity | Ed25519 (PeerId keypair) |
| Chain Transactions | Sr25519 (Substrate) |
| Validator Consensus | Session keys (GRANDPA, BABE/Aura) |
| TLS (P2P) | Noise XX (ephemeral) |

| Action | Required Role | Min Stake |
|--------|---------------|-----------|
| submit_bft_result | Director (elected) | 100 NSN |
| challenge_bft_result | Any staker | 25 NSN bond |
| create_deal | Any | Payment amount |

### 7.3 Data Protection

| Data Type | At Rest | In Transit |
|-----------|---------|------------|
| Private Keys | Encrypted (SOPS/age) | N/A |
| Validator Keys | HSM or encrypted | N/A |
| Video Content | Plaintext (erasure coded) | AES-256-GCM |
| CLIP Embeddings | Plaintext | Noise encrypted |
| Recipes | Plaintext | Ed25519 signed |

---

## 8. Risks, Assumptions, Dependencies

### 8.1 Technical Risks

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Solo chain security insufficient | Medium | Conservative slashing, low initial TVL, trusted operators |
| Parachain migration complexity | Medium | Design Cumulus-compatible from start, test on Rococo |
| CLIP adversarial bypass | Medium | Ensemble models, outlier detection |
| libp2p performance at scale | Medium | Hierarchical topology, load testing |

### 8.2 Assumptions

| Assumption | Validation |
|------------|------------|
| NSN Chain TPS sufficient (100+) | Benchmark during testnet |
| 12GB VRAM fits all models | Extensive memory testing |
| 45s achievable glass-to-glass | Benchmark on target hardware |
| Trusted validators available | Recruit operators pre-launch |

### 8.3 External Dependencies

| Dependency | Criticality | Fallback |
|------------|-------------|----------|
| Polkadot SDK | Critical | Pin stable version |
| Polkadot Relay Chain | Medium (parachain phase) | Remain solochain |
| Hugging Face Models | High | Self-hosted cache |
| STUN/TURN Servers | Medium | Community-run, circuit relay |

### 8.4 Key Mitigations

**Solo Chain Security:** Start with low TVL, use trusted validator set (known operators), implement conservative slashing, maintain backup validators, 24/7 monitoring.

**Parachain Migration:** Use Cumulus-compatible pallet design from day 1, test on Rococo, document migration runbook.

**CLIP Adversarial:** Dual CLIP ensemble, outlier detection, prompt sanitization, human escalation.

**Director Collusion:** Multi-region requirement (max 2 per region), challenge period with slashing, statistical anomaly detection.

---

## 9. Roadmap & Evolution

### 9.1 Phased Implementation

**Phase A: ICN Solochain MVP (Weeks 1-8)**
- Bootstrap NSN Chain from Polkadot SDK template
- Implement pallet-nsn-stake, pallet-nsn-reputation
- Implement pallet-nsn-director with VRF election
- Deploy ICN Public Testnet with 10+ nodes

**Phase B: NSN Mainnet (Weeks 9-16)**
- Security audit (Oak Security / SRLabs)
- Validator onboarding and genesis configuration
- ICN token launch
- Mainnet launch with 3-5 validators

**Phase C: Parachain Migration (Post-adoption)**
- Add Cumulus integration
- Test on Rococo/Westend
- Acquire parachain slot
- Migrate to Polkadot shared security

**Phase D: Scaling & Ethereum (Ongoing)**
- Enable optional Frontier EVM
- Coretime acquisition (on-demand → bulk)
- Snowbridge integration for Ethereum bridging
- Additional CLIP models (RN50)

### 9.2 Extensibility Points

| Extension Point | Mechanism |
|-----------------|-----------|
| New AI Models | Vortex plugin system |
| Runtime Upgrades | Forkless via `set_code` |
| Content Policies | CLIP prompt filters |
| Token Standards | Optional ERC-721/1155 via Frontier |
| Cross-Chain | XCM (parachain phase) |

---

## 10. Appendices

### 10.1 Glossary

| Term | Definition |
|------|------------|
| **BFT** | Byzantine Fault Tolerance – 3-of-5 consensus |
| **CLIP** | Contrastive Language-Image Pretraining |
| **Coretime** | Polkadot's execution time allocation model |
| **Cumulus** | Framework for building Polkadot parachains |
| **Director** | Node elected to generate video for a slot |
| **Erasure Coding** | Reed-Solomon (10+4) redundancy |
| **FRAME** | Substrate's modular runtime framework |
| **Frontier** | Substrate EVM compatibility layer |
| **GossipSub** | libp2p pubsub protocol |
| **Epoch** | 100-block period for Director elections |
| **Lane 0** | Deterministic video generation lane |
| **Lane 1** | Open AI compute marketplace lane |
| **NSN Chain** | NSN's Polkadot SDK-based blockchain |
| **On-Deck Set** | 20 Director candidates for next epoch |
| **Pallet** | Substrate runtime module |
| **Recipe** | JSON instruction set for AI generation |
| **Slot** | 45-90 second content generation window |
| **Snowbridge** | Trustless Polkadot ↔ Ethereum bridge |
| **Solochain** | Standalone blockchain (not parachain) |
| **Super-Node** | High-stake storage and relay node |
| **Vortex** | NSN's GPU-resident AI generation engine (Lane 0) |
| **VRF** | Verifiable Random Function |
| **node-core** | Universal compute orchestration (Lane 1) |

### 10.2 Key References

- Polkadot SDK: https://github.com/paritytech/polkadot-sdk
- Substrate FRAME: https://docs.substrate.io/reference/frame-pallets/
- Cumulus: https://paritytech.github.io/polkadot-sdk/master/polkadot_sdk_docs/polkadot_sdk/cumulus/
- Frontier: https://polkadot-evm.github.io/frontier/
- Coretime: https://docs.polkadot.com/polkadot-protocol/architecture/system-chains/coretime/
- libp2p Spec: https://github.com/libp2p/specs
- CLIP Paper: https://arxiv.org/abs/2103.00020

---

**Document Status:** Approved - NSN Dual-Lane Architecture
**Last Updated:** 2025-12-29
**Next Review:** After Phase A completion

*End of Technical Architecture Document*
