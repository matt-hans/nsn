# Interdimensional Cable Network (ICN)
# Technical Architecture Document v1.0

**Version:** 1.0  
**Date:** 2025-12-24  
**Status:** Draft for Review  
**Classification:** Internal - Engineering  
**PRD Reference:** ICN_PRD_v8.0.1

---

## Table of Contents

1. [Overview & Context](#1-overview--context)
2. [Architectural Drivers & Constraints](#2-architectural-drivers--constraints)
3. [Architectural Decisions (ADRs)](#3-architectural-decisions-adrs)
4. [System Architecture](#4-system-architecture)
5. [Technology Stack & Rationale](#5-technology-stack--rationale)
6. [Deployment & Operations](#6-deployment--operations)
7. [Security & Compliance](#7-security--compliance)
8. [Risks, Assumptions, Dependencies & Mitigations](#8-risks-assumptions-dependencies--mitigations)
9. [Roadmap & Evolution](#9-roadmap--evolution)
10. [Appendices](#10-appendices)

---

## 1. Overview & Context

### 1.1 Purpose of This Document

This Technical Architecture Document (TAD) translates the ICN Product Requirements Document (PRD v8.0.1) into concrete technical architecture, design decisions, and implementation guidance. It serves as the authoritative reference for:

- **Engineering teams** implementing the system
- **DevOps/SRE** deploying and operating the infrastructure
- **Security teams** auditing the system
- **External auditors** reviewing the Substrate pallets
- **Future maintainers** understanding architectural rationale

This document does NOT replace the PRD; it complements it by focusing on the "how" rather than the "what."

### 1.2 Scope

**In Scope:**
- Moonbeam pallet architecture and inter-pallet communication
- Off-chain node architecture (Director, Validator, Super-Node, Relay, Viewer)
- P2P network topology and protocols
- AI inference pipeline (Vortex Engine)
- Data storage and replication strategies
- Security architecture across all layers
- Deployment and operational procedures
- Observability and monitoring infrastructure

**Out of Scope:**
- Business logic details (covered in PRD)
- UI/UX design specifications
- Marketing and go-to-market strategy
- Legal and regulatory compliance specifics (covered separately)
- Detailed tokenomics modeling

### 1.3 References

| Document | Version | Description |
|----------|---------|-------------|
| ICN_PRD_v8.0.1 | 8.0.1 | Product Requirements Document |
| Moonbeam Documentation | v0.35.0 | Runtime extension guide |
| Substrate FRAME | polkadot-v1.0.0 | Pallet development reference |
| libp2p Specification | 0.53.0 | P2P networking protocols |
| CLIP Paper | OpenAI 2021 | Semantic verification model |

### 1.4 Key Stakeholders & Intended Audience

| Stakeholder | Role | Primary Interest |
|-------------|------|------------------|
| Core Engineering | Rust/Substrate Developers | Pallet implementation details |
| AI/ML Team | Python/PyTorch Engineers | Vortex pipeline architecture |
| Infrastructure | DevOps/SRE | Deployment, monitoring, scaling |
| Security | Security Engineers | Threat model, cryptographic choices |
| Product | Product Managers | Technical feasibility, timelines |
| External Auditors | Security Auditors | Pallet security, economic attacks |

### 1.5 High-Level System Context (C4 Level 1)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM CONTEXT                                      │
│                                                                                  │
│  ┌──────────────┐                                         ┌──────────────────┐  │
│  │              │                                         │                  │  │
│  │   Viewers    │◄──────── Video Streams ─────────────────│   ICN Network    │  │
│  │  (Consumers) │                                         │                  │  │
│  │              │                                         │  ┌────────────┐  │  │
│  └──────────────┘                                         │  │ Moonbeam   │  │  │
│                                                           │  │ (On-Chain) │  │  │
│  ┌──────────────┐                                         │  └────────────┘  │  │
│  │              │──────── Stake/Reputation ───────────────│                  │  │
│  │  Directors   │                                         │  ┌────────────┐  │  │
│  │ (Generators) │◄─────── Election/Rewards ───────────────│  │  P2P Mesh  │  │  │
│  │              │                                         │  │ (Off-Chain)│  │  │
│  └──────────────┘                                         │  └────────────┘  │  │
│                                                           │                  │  │
│  ┌──────────────┐                                         │  ┌────────────┐  │  │
│  │              │──────── CLIP Attestations ──────────────│  │   Vortex   │  │  │
│  │  Validators  │                                         │  │  (AI Gen)  │  │  │
│  │  (Verifiers) │◄─────── Reputation Rewards ─────────────│  └────────────┘  │  │
│  │              │                                         │                  │  │
│  └──────────────┘                                         └──────────────────┘  │
│                                                                                  │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────────────────┐ │
│  │   Polkadot   │         │   Moonbeam   │         │    External Services     │ │
│  │ Relay Chain  │◄────────│  Collators   │         │  • STUN/TURN servers     │ │
│  │ (Security)   │         │              │         │  • DNS seeds             │ │
│  └──────────────┘         └──────────────┘         │  • Bootstrap nodes       │ │
│                                                    └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Context Description:**

The ICN system consists of three primary domains:

1. **On-Chain Layer (Moonbeam):** Custom Substrate pallets deployed on Moonbeam parachain handle staking, reputation, director election, BFT result storage, pinning deals, and treasury operations. This layer inherits security from Polkadot's relay chain (~$20B economic security).

2. **Off-Chain Layer (P2P Mesh):** A hierarchical network of nodes running libp2p-based software. Directors generate AI video content, Validators verify semantic compliance, Super-Nodes store erasure-coded shards, Regional Relays distribute content, and Viewers consume streams.

3. **AI Generation Layer (Vortex):** A GPU-resident pipeline running Flux-Schnell (image generation), LivePortrait (video warping), Kokoro-82M (TTS), and CLIP (semantic verification).

---

## 2. Architectural Drivers & Constraints

### 2.1 Functional Requirements Summary

| Requirement | PRD Section | Technical Implication |
|-------------|-------------|----------------------|
| Multi-Director BFT | §3.3 | 5 directors, 3-of-5 consensus, off-chain coordination |
| Semantic Verification | §12.2 | Dual CLIP model ensemble (ViT-B-32, ViT-L-14) |
| Static VRAM Residency | §12.1 | All models loaded at startup, no swapping |
| Hierarchical Swarm | §17.2 | 4-tier architecture (Directors → Super-Nodes → Relays → Viewers) |
| Erasure Coding | §3.4 | Reed-Solomon (10+4), 5x replication |
| NAT Traversal | §17.1 | STUN → UPnP → Circuit Relay → TURN fallback |
| BFT Challenge Period | §3.3 | 50-block dispute window with stake slashing |
| VRF Elections | §3.3 | Moonbeam's BABE-based VRF for director selection |

### 2.2 Non-Functional Requirements

| Attribute | Target | Measurement | Priority |
|-----------|--------|-------------|----------|
| **Availability** | 99.5% | Playback continuity | High |
| **Latency** | <45s glass-to-glass | Generation → Playback | High |
| **Throughput** | 50 TPS (on-chain) | Moonbeam limit | Medium |
| **Scalability** | 500+ mainnet nodes | Node count | Medium |
| **Security** | 8-layer model | Attack surface reduction | Critical |
| **Cost** | $80k-$200k MVP | Development budget | High |
| **Maintainability** | Modular pallets | Separation of concerns | Medium |
| **Developer Experience** | EVM + Substrate | Dual interface | Medium |

### 2.3 Business & Technical Constraints

| Constraint | Type | Impact | Mitigation |
|------------|------|--------|------------|
| Moonbeam governance | Business | Runtime upgrades require GLMR vote | Progressive pallet deployment |
| RTX 3060 12GB floor | Technical | VRAM budget = 11.8GB max | Aggressive quantization |
| 3-6 month timeline | Business | Limited scope for MVP | Phased feature rollout |
| $80k-$200k budget | Business | 2-3 developers max | Focus on critical path |
| Substrate version lock | Technical | Must match Moonbeam runtime | Pin to polkadot-v1.0.0 |
| Moonriver testnet | Technical | No real value at stake | Extensive testing before mainnet |

### 2.4 Quality Attributes Prioritization

| Rank | Attribute | Rationale |
|------|-----------|-----------|
| 1 | **Security** | On-chain funds at risk; slashing must be correct |
| 2 | **Availability** | Streaming requires continuous playback |
| 3 | **Latency** | 45s glass-to-glass is tight for AI generation |
| 4 | **Scalability** | Must support 500+ nodes at mainnet |
| 5 | **Cost** | Startup budget constraints |
| 6 | **Maintainability** | Long-term evolution via governance |

---

## 3. Architectural Decisions (ADRs)

### ADR-001: Moonbeam Pallets over Custom Parachain

| Field | Content |
|-------|---------|
| **Status** | Accepted |
| **Context** | ICN needs on-chain staking, reputation, and consensus signals. Building a custom parachain requires 9-18 months and $500k-$1M+. |
| **Decision** | Deploy custom FRAME pallets on Moonbeam's existing runtime via governance upgrades. |
| **Alternatives** | (1) Custom Polkadot parachain, (2) Ethereum L2 rollup, (3) Cosmos appchain |
| **Consequences** | (+) 3-6× faster, 5-10× cheaper, shared security. (-) Limited to ~50 TPS, governance dependency. |
| **Risks** | Governance rejection. **Mitigation:** Start with treasury pallet, build community support. |

### ADR-002: Hybrid On-Chain/Off-Chain Architecture

| Field | Content |
|-------|---------|
| **Status** | Accepted |
| **Context** | Full BFT consensus on-chain is prohibitively expensive (~$1/transaction) and slow (~12s finality). |
| **Decision** | On-chain = state changes (stake, reputation, BFT results). Off-chain = computation (AI generation, BFT negotiation, video distribution). |
| **Alternatives** | (1) Fully on-chain, (2) Fully off-chain with anchoring, (3) zk-rollup |
| **Consequences** | (+) Cost-effective, low latency. (-) Off-chain nodes must be trusted for BFT result accuracy. |
| **Risks** | Director collusion. **Mitigation:** Challenge period, validator attestations, stake slashing. |

### ADR-003: libp2p over Custom P2P Stack

| Field | Content |
|-------|---------|
| **Status** | Accepted |
| **Context** | Need reliable P2P networking with NAT traversal, peer discovery, and pub/sub messaging. |
| **Decision** | Use libp2p (rust-libp2p 0.53.0) with GossipSub, Kademlia DHT, and QUIC transport. |
| **Alternatives** | (1) Custom UDP/TCP stack, (2) WebRTC only, (3) gRPC mesh |
| **Consequences** | (+) Battle-tested, NAT traversal built-in, Substrate ecosystem familiarity. (-) Larger binary size, learning curve. |
| **Risks** | GossipSub performance at scale. **Mitigation:** Tune mesh parameters, implement hierarchical topics. |

### ADR-004: Dual CLIP Ensemble for Semantic Verification

| Field | Content |
|-------|---------|
| **Status** | Accepted |
| **Context** | Single CLIP model is vulnerable to adversarial attacks and has edge-case failures. |
| **Decision** | Use CLIP-ViT-B-32 (0.4 weight) + CLIP-ViT-L-14 (0.6 weight) ensemble with independent thresholds. |
| **Alternatives** | (1) Single CLIP model, (2) CLIP + BLIP ensemble, (3) Fine-tuned domain model |
| **Consequences** | (+) ~40% reduction in disputes, robust to adversarial inputs. (-) Additional 0.3GB VRAM, 1.5× inference time. |
| **Risks** | Both models fooled by same adversarial. **Mitigation:** Add RN50 third model in future. |

### ADR-005: Static VRAM Residency (No Model Swapping)

| Field | Content |
|-------|---------|
| **Status** | Accepted |
| **Context** | PCIe model swapping takes 1-2s, which exceeds our 45s slot budget. |
| **Decision** | All models (Flux, LivePortrait, Kokoro, CLIP×2) remain resident in VRAM at all times. |
| **Alternatives** | (1) Dynamic loading, (2) CPU offload with pinned memory, (3) Model distillation |
| **Consequences** | (+) Predictable latency, no thrashing. (-) RTX 3060 12GB is hard minimum. |
| **Risks** | OOM on edge cases. **Mitigation:** Pre-allocated output buffers, memory monitoring. |

### ADR-006: BFT Challenge Period for Off-Chain Trust

| Field | Content |
|-------|---------|
| **Status** | Accepted |
| **Context** | Directors could collude to submit fraudulent BFT results. |
| **Decision** | 50-block (~5 minute) challenge period before finalization. Challengers stake 25 ICN, successful challenges slash 100 ICN from each fraudulent director. |
| **Alternatives** | (1) Immediate finalization with validator votes, (2) Optimistic rollup style, (3) zkSNARK proofs |
| **Consequences** | (+) Economic security against collusion. (-) 5-minute delay before reputation updates. |
| **Risks** | False challenges as griefing. **Mitigation:** Challenger stake is slashed if challenge fails. |

### ADR-007: VRF-Based Director Election

| Field | Content |
|-------|---------|
| **Status** | Accepted |
| **Context** | Director election must be unpredictable to prevent pre-computation attacks. |
| **Decision** | Use Moonbeam's BABE VRF via `T::Randomness` trait for cryptographically secure director selection. |
| **Alternatives** | (1) Block hash randomness, (2) External oracle (Chainlink VRF), (3) Commit-reveal scheme |
| **Consequences** | (+) Cryptographically unpredictable, verifiable on-chain. (-) Dependent on Moonbeam's VRF implementation. |
| **Risks** | VRF bias attacks. **Mitigation:** Multi-region requirement reduces single-actor impact. |

### ADR-008: Reed-Solomon Erasure Coding (10+4)

| Field | Content |
|-------|---------|
| **Status** | Accepted |
| **Context** | Video content must remain available even if pinners go offline. |
| **Decision** | 10+4 Reed-Solomon coding (can recover from any 4 shard losses), 5× replication across regions. |
| **Alternatives** | (1) Simple replication (3×), (2) Fountain codes, (3) LDPC codes |
| **Consequences** | (+) 1.4× overhead vs 3× for replication at same durability. (-) Compute cost for encoding/decoding. |
| **Risks** | Correlated failures. **Mitigation:** Geographic distribution requirement for pinners. |

### ADR-009: EVM + Substrate Dual Interface

| Field | Content |
|-------|---------|
| **Status** | Accepted |
| **Context** | Users expect MetaMask/ethers.js UX, but Substrate pallets offer more flexibility. |
| **Decision** | ICN token as ERC-20 on Frontier EVM, with precompiles bridging to Substrate pallets. |
| **Alternatives** | (1) Pure Substrate, (2) Pure EVM, (3) Ink! smart contracts |
| **Consequences** | (+) Best of both worlds, familiar UX. (-) Precompile development complexity. |
| **Risks** | Precompile security bugs. **Mitigation:** Phased rollout, extensive auditing. |

### ADR-010: Hierarchical Swarm Topology

| Field | Content |
|-------|---------|
| **Status** | Accepted |
| **Context** | Flat mesh doesn't scale beyond ~100 nodes for real-time video distribution. |
| **Decision** | 4-tier hierarchy: Directors → Super-Nodes → Regional Relays → Edge Viewers. |
| **Alternatives** | (1) Flat GossipSub, (2) CDN hybrid, (3) Tree-based multicast |
| **Consequences** | (+) O(log N) propagation, regional locality. (-) Super-Node dependency, potential bottlenecks. |
| **Risks** | Super-Node failures. **Mitigation:** 7 regions minimum, automatic failover. |

---

## 4. System Architecture

### 4.1 High-Level Architecture Overview (C4 Level 2: Containers)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 ICN SYSTEM ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                           ON-CHAIN LAYER (Moonbeam)                              │    │
│  │                                                                                   │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐     │    │
│  │  │ pallet-icn-   │  │ pallet-icn-   │  │ pallet-icn-   │  │ pallet-icn-   │     │    │
│  │  │    stake      │◄─│  reputation   │◄─│   director    │◄─│   pinning     │     │    │
│  │  │               │  │               │  │               │  │               │     │    │
│  │  │ • deposit     │  │ • record      │  │ • elect       │  │ • create_deal │     │    │
│  │  │ • delegate    │  │ • prune       │  │ • submit_bft  │  │ • audit       │     │    │
│  │  │ • slash       │  │ • checkpoint  │  │ • challenge   │  │ • distribute  │     │    │
│  │  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘     │    │
│  │                            │                  │                                  │    │
│  │  ┌───────────────┐        │                  │         ┌───────────────┐        │    │
│  │  │ pallet-icn-   │        │                  │         │ pallet-icn-   │        │    │
│  │  │   treasury    │◄───────┴──────────────────┴────────►│     bft       │        │    │
│  │  │               │                                     │               │        │    │
│  │  │ • distribute  │                                     │ • store_hash  │        │    │
│  │  │ • fund        │                                     │ • finalize    │        │    │
│  │  └───────────────┘                                     └───────────────┘        │    │
│  │                                                                                   │    │
│  │  ┌───────────────────────────────────────────────────────────────────────────┐  │    │
│  │  │                    FRONTIER EVM (Ethereum Layer)                           │  │    │
│  │  │  • ICN Token (ERC-20)  • Staking Precompile  • Reputation Precompile       │  │    │
│  │  └───────────────────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                         │                                                │
│                              subxt RPC  │  Events                                        │
│                                         ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                          OFF-CHAIN LAYER (P2P Mesh)                              │    │
│  │                                                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐    │    │
│  │  │                        TIER 0: DIRECTOR NODES                            │    │    │
│  │  │                                                                          │    │    │
│  │  │  ┌───────────────────────────────────────────────────────────────────┐  │    │    │
│  │  │  │                     VORTEX ENGINE (GPU)                            │  │    │    │
│  │  │  │                                                                    │  │    │    │
│  │  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │  │    │    │
│  │  │  │  │  Flux-   │  │  Live-   │  │ Kokoro-  │  │ CLIP Ensemble    │   │  │    │    │
│  │  │  │  │ Schnell  │  │ Portrait │  │  82M     │  │ (B-32 + L-14)    │   │  │    │    │
│  │  │  │  │ (NF4)    │  │ (FP16)   │  │ (FP32)   │  │ (INT8)           │   │  │    │    │
│  │  │  │  │ 6.0 GB   │  │ 3.5 GB   │  │ 0.4 GB   │  │ 0.9 GB           │   │  │    │    │
│  │  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │  │    │    │
│  │  │  └───────────────────────────────────────────────────────────────────┘  │    │    │
│  │  │                                                                          │    │    │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │    │    │
│  │  │  │ Chain Client │  │ BFT Coord    │  │ P2P Service  │                   │    │    │
│  │  │  │ (subxt)      │  │ (gRPC)       │  │ (libp2p)     │                   │    │    │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘                   │    │    │
│  │  └─────────────────────────────────────────────────────────────────────────┘    │    │
│  │                                         │                                        │    │
│  │                              GossipSub  │  Video Chunks                          │    │
│  │                                         ▼                                        │    │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐    │    │
│  │  │                      TIER 1: SUPER-NODES (7 Regions)                     │    │    │
│  │  │                                                                          │    │    │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │    │    │
│  │  │  │ Erasure      │  │ CLIP         │  │ P2P Relay    │                   │    │    │
│  │  │  │ Storage      │  │ Validator    │  │ Service      │                   │    │    │
│  │  │  │ (10TB+)      │  │              │  │              │                   │    │    │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘                   │    │    │
│  │  └─────────────────────────────────────────────────────────────────────────┘    │    │
│  │                                         │                                        │    │
│  │                              QUIC       │  Shards                                │    │
│  │                                         ▼                                        │    │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐    │    │
│  │  │                     TIER 2: REGIONAL RELAYS                              │    │    │
│  │  │                                                                          │    │    │
│  │  │  ┌──────────────┐  ┌──────────────┐                                     │    │    │
│  │  │  │ Cache        │  │ P2P Relay    │                                     │    │    │
│  │  │  │ (1TB)        │  │ Service      │                                     │    │    │
│  │  │  └──────────────┘  └──────────────┘                                     │    │    │
│  │  └─────────────────────────────────────────────────────────────────────────┘    │    │
│  │                                         │                                        │    │
│  │                              QUIC       │  Streams                               │    │
│  │                                         ▼                                        │    │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐    │    │
│  │  │                      TIER 3: EDGE VIEWERS                                │    │    │
│  │  │                                                                          │    │    │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │    │    │
│  │  │  │ Tauri App    │  │ Video Player │  │ Optional     │                   │    │    │
│  │  │  │ (React)      │  │ (WebCodecs)  │  │ Seeding      │                   │    │    │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘                   │    │    │
│  │  └─────────────────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component View (C4 Level 3)

#### 4.2.1 Director Node Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DIRECTOR NODE INTERNALS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         CORE RUNTIME (Tokio)                         │    │
│  │                                                                       │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │ Election     │  │ Slot         │  │ BFT          │               │    │
│  │  │ Monitor      │──│ Scheduler    │──│ Coordinator  │               │    │
│  │  │              │  │              │  │              │               │    │
│  │  │ • subscribe  │  │ • pipeline   │  │ • exchange   │               │    │
│  │  │   elections  │  │   lookahead  │  │   embeddings │               │    │
│  │  │ • verify     │  │ • deadline   │  │ • compute    │               │    │
│  │  │   my_role    │  │   tracking   │  │   agreement  │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  │         │                  │                  │                      │    │
│  │         ▼                  ▼                  ▼                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │                    VORTEX PIPELINE (Python)                   │   │    │
│  │  │                                                               │   │    │
│  │  │  Recipe ──► Audio Gen ──► Actor Gen ──► Video Warp ──► CLIP  │   │    │
│  │  │              (Kokoro)     (Flux)      (LivePortrait)   Verify │   │    │
│  │  │                                                               │   │    │
│  │  │  ┌────────────────────────────────────────────────────────┐  │   │    │
│  │  │  │                 VRAM MANAGER (CUDA)                     │  │   │    │
│  │  │  │  • Model registry (resident)                            │  │   │    │
│  │  │  │  • Output buffer pool (pre-allocated)                   │  │   │    │
│  │  │  │  • Memory pressure monitor                              │  │   │    │
│  │  │  └────────────────────────────────────────────────────────┘  │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │         │                                                            │    │
│  │         ▼                                                            │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │                    P2P NETWORK SERVICE                        │   │    │
│  │  │                                                               │   │    │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐              │   │    │
│  │  │  │ GossipSub  │  │ Kademlia   │  │ QUIC       │              │   │    │
│  │  │  │ (Topics)   │  │ (DHT)      │  │ (Transport)│              │   │    │
│  │  │  └────────────┘  └────────────┘  └────────────┘              │   │    │
│  │  │                                                               │   │    │
│  │  │  ┌────────────────────────────────────────────────────────┐  │   │    │
│  │  │  │               REPUTATION ORACLE (Cached)                │  │   │    │
│  │  │  │  • sync_loop (60s)  • peer_score_integration            │  │   │    │
│  │  │  └────────────────────────────────────────────────────────┘  │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │         │                                                            │    │
│  │         ▼                                                            │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │                    CHAIN CLIENT (subxt)                       │   │    │
│  │  │                                                               │   │    │
│  │  │  • subscribe_finalized_blocks()                               │   │    │
│  │  │  • submit_bft_result()                                        │   │    │
│  │  │  • query_reputation()                                         │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         OBSERVABILITY                                │    │
│  │  • Prometheus metrics (port 9100)                                    │    │
│  │  • OpenTelemetry traces (OTLP port 4317)                             │    │
│  │  • Structured logging (JSON to stdout)                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.2.2 Pallet Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PALLET INTERACTION FLOWS                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  FLOW 1: Director Election                                               │
│  ─────────────────────────────                                           │
│                                                                          │
│  on_initialize(block)                                                    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌────────────────┐    query    ┌────────────────┐                      │
│  │ pallet-icn-    │───────────►│ pallet-icn-    │                      │
│  │   director     │             │    stake       │                      │
│  │                │◄────────────│                │                      │
│  │ elect_         │   Stakes    │ Stakes::iter() │                      │
│  │   directors()  │   + Roles   │                │                      │
│  └────────────────┘             └────────────────┘                      │
│       │                                                                  │
│       │ query                                                            │
│       ▼                                                                  │
│  ┌────────────────┐                                                     │
│  │ pallet-icn-    │                                                     │
│  │  reputation    │                                                     │
│  │                │                                                     │
│  │ reputation_    │                                                     │
│  │   scores()     │                                                     │
│  └────────────────┘                                                     │
│       │                                                                  │
│       │ emit                                                             │
│       ▼                                                                  │
│  Event::DirectorsElected(slot, directors)                               │
│                                                                          │
│                                                                          │
│  FLOW 2: BFT Result Submission                                          │
│  ─────────────────────────────────                                       │
│                                                                          │
│  submit_bft_result(slot, directors, hash)                               │
│       │                                                                  │
│       ▼                                                                  │
│  ┌────────────────┐             ┌────────────────┐                      │
│  │ pallet-icn-    │   verify    │ pallet-icn-    │                      │
│  │   director     │────────────►│    stake       │                      │
│  │                │             │                │                      │
│  │ - check        │ role check  │ Stakes::get()  │                      │
│  │   elected      │◄────────────│                │                      │
│  │ - store        │             └────────────────┘                      │
│  │   pending      │                                                     │
│  └────────────────┘                                                     │
│       │                                                                  │
│       │ after 50 blocks (no challenge)                                   │
│       ▼                                                                  │
│  ┌────────────────┐             ┌────────────────┐                      │
│  │ pallet-icn-    │   record    │ pallet-icn-    │                      │
│  │   director     │────────────►│  reputation    │                      │
│  │                │             │                │                      │
│  │ finalize_      │             │ record_event() │                      │
│  │   slot()       │             │                │                      │
│  └────────────────┘             └────────────────┘                      │
│                                                                          │
│                                                                          │
│  FLOW 3: Challenge Resolution                                           │
│  ────────────────────────────                                            │
│                                                                          │
│  challenge_bft_result(slot, evidence)                                   │
│       │                                                                  │
│       ▼                                                                  │
│  ┌────────────────┐             ┌────────────────┐                      │
│  │ pallet-icn-    │   reserve   │ T::Currency    │                      │
│  │   director     │────────────►│ (Balances)     │                      │
│  │                │             │                │                      │
│  │ - verify not   │  25 ICN     │ reserve()      │                      │
│  │   finalized    │             │                │                      │
│  │ - store        │             └────────────────┘                      │
│  │   challenge    │                                                     │
│  └────────────────┘                                                     │
│       │                                                                  │
│       ▼                                                                  │
│  resolve_challenge(slot, attestations)                                  │
│       │                                                                  │
│       ├─── if upheld ───┐                                               │
│       │                 ▼                                                │
│       │         ┌────────────────┐                                      │
│       │         │ pallet-icn-    │                                      │
│       │         │    stake       │                                      │
│       │         │                │                                      │
│       │         │ slash() 100    │                                      │
│       │         │ ICN per dir    │                                      │
│       │         └────────────────┘                                      │
│       │                                                                  │
│       └─── if rejected ─┐                                               │
│                         ▼                                                │
│                 ┌────────────────┐                                      │
│                 │ T::Currency    │                                      │
│                 │ (Balances)     │                                      │
│                 │                │                                      │
│                 │ slash_reserved │                                      │
│                 │ 25 ICN bond    │                                      │
│                 └────────────────┘                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Data Architecture

#### 4.3.1 On-Chain Data Models

```rust
// Key on-chain data structures

// pallet-icn-stake
pub struct StakeInfo<T: Config> {
    pub amount: BalanceOf<T>,           // Staked tokens
    pub locked_until: T::BlockNumber,   // Lock expiry
    pub role: NodeRole,                 // Director/SuperNode/Validator/Relay
    pub region: Region,                 // Geographic region (7 options)
    pub delegated_to_me: BalanceOf<T>,  // Delegated stake
}

// pallet-icn-reputation
pub struct ReputationScore {
    pub director_score: u64,    // Director performance
    pub validator_score: u64,   // Validator accuracy
    pub seeder_score: u64,      // Pinning reliability
    pub last_activity: u64,     // Last active block
}

// pallet-icn-director
pub struct BftConsensusResult<T: Config> {
    pub slot: u64,
    pub success: bool,
    pub canonical_hash: T::Hash,        // CLIP embedding hash
    pub attestations: Vec<(T::AccountId, bool)>,
}

pub struct BftChallenge<T: Config> {
    pub slot: u64,
    pub challenger: T::AccountId,
    pub deadline: T::BlockNumber,
    pub evidence_hash: T::Hash,
    pub resolved: bool,
}

// pallet-icn-pinning
pub struct PinningDeal<T: Config> {
    pub deal_id: [u8; 32],
    pub creator: T::AccountId,
    pub shards: Vec<[u8; 32]>,          // Shard hashes
    pub expires_at: T::BlockNumber,
    pub total_reward: BalanceOf<T>,
    pub status: DealStatus,
}
```

#### 4.3.2 Off-Chain Data Storage

| Data Type | Storage | Format | Retention | Replication |
|-----------|---------|--------|-----------|-------------|
| Video Chunks | Super-Nodes | Binary (AV1/VP9) | 7 days default | 5× (erasure coded) |
| Recipes | GossipSub + DHT | JSON | 24 hours | Epidemic |
| CLIP Embeddings | P2P Exchange | Float32[512] | Per slot | 5× (directors) |
| Merkle Proofs | On-chain | Bytes | 6 months | Blockchain |
| Audit Challenges | On-chain | Struct | Until resolved | Blockchain |

#### 4.3.3 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW: SLOT GENERATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. RECIPE CREATION                                                         │
│  ──────────────────                                                          │
│                                                                              │
│  Content Creator ──► Recipe JSON ──► Director (GossipSub)                   │
│                          │                                                   │
│                          │ ~100KB                                            │
│                          ▼                                                   │
│                   ┌─────────────┐                                            │
│                   │   Recipe    │                                            │
│                   │   Topic     │                                            │
│                   │  /icn/      │                                            │
│                   │  recipes/   │                                            │
│                   └─────────────┘                                            │
│                                                                              │
│  2. AI GENERATION                                                           │
│  ────────────────                                                            │
│                                                                              │
│  Recipe ──► Vortex Pipeline ──► Video Frames + Audio + CLIP Embedding       │
│                    │                                                         │
│                    │ GPU Memory Flow                                         │
│                    ▼                                                         │
│        ┌─────────────────────────────────────────────────┐                  │
│        │  Flux (6GB) → LivePortrait (3.5GB) → CLIP (0.9GB) │                │
│        │           ↑           ↑                          │                  │
│        │           └─ Kokoro (0.4GB) ──┘                  │                  │
│        └─────────────────────────────────────────────────┘                  │
│                                                                              │
│  3. BFT CONSENSUS                                                           │
│  ───────────────                                                             │
│                                                                              │
│  Director 1 ───┐                                                             │
│  Director 2 ───┼──► Exchange Embeddings ──► Agreement Matrix ──► Result     │
│  Director 3 ───┤     (gRPC mesh)               (cosine sim)                  │
│  Director 4 ───┤                                                             │
│  Director 5 ───┘                                                             │
│                          │                                                   │
│                          │ 3-of-5 agree                                      │
│                          ▼                                                   │
│                   ┌─────────────┐                                            │
│                   │  submit_bft │                                            │
│                   │  _result()  │                                            │
│                   │  (on-chain) │                                            │
│                   └─────────────┘                                            │
│                                                                              │
│  4. VIDEO DISTRIBUTION                                                      │
│  ─────────────────────                                                       │
│                                                                              │
│  Canonical Director ──► Super-Nodes ──► Regional Relays ──► Viewers         │
│           │                  │                  │                │           │
│           │                  │                  │                │           │
│           ▼                  ▼                  ▼                ▼           │
│       [Video]           [Shards]           [Cache]          [Stream]        │
│        ~50MB             10+4 RS            ~1TB             ~5Mbps         │
│                                                                              │
│  5. ERASURE STORAGE                                                         │
│  ──────────────────                                                          │
│                                                                              │
│  Video Chunk ──► Reed-Solomon(10,4) ──► 14 Shards ──► 5× Replication        │
│       │                                      │                               │
│       │ 50MB                                 │ 7MB each                      │
│       ▼                                      ▼                               │
│  create_deal() ◄──────────────────── ShardAssignments                       │
│  (on-chain)                          (off-chain DHT)                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Integration Architecture

#### 4.4.1 API Specifications

| API | Type | Purpose | Authentication |
|-----|------|---------|----------------|
| Moonbeam RPC | JSON-RPC 2.0 | Chain queries, extrinsics | None (public) |
| Substrate WS | WebSocket | Block subscriptions | None (public) |
| Director gRPC | gRPC | BFT coordination | mTLS + PeerId |
| P2P GossipSub | libp2p | Recipe/video broadcast | Ed25519 signing |
| EVM Precompiles | Solidity ABI | EVM ↔ Substrate bridge | EVM signature |

#### 4.4.2 GossipSub Topics

| Topic | Message Type | Publishers | Subscribers |
|-------|--------------|------------|-------------|
| `/icn/recipes/1.0.0` | Recipe JSON | Content Creators | Directors |
| `/icn/video/1.0.0` | Video Chunk | Directors | Super-Nodes |
| `/icn/bft/1.0.0` | CLIP Embedding | Directors | Directors |
| `/icn/attestations/1.0.0` | Attestation | Validators | All |
| `/icn/challenges/1.0.0` | Challenge | Any staker | Validators |

#### 4.4.3 Authentication Flows

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AUTHENTICATION ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. P2P IDENTITY                                                            │
│  ───────────────                                                             │
│                                                                              │
│  Ed25519 Keypair ──► PeerId (libp2p) ◄──► AccountId (Substrate)             │
│        │                   │                      │                          │
│        │                   │                      │                          │
│        ▼                   ▼                      ▼                          │
│  ┌──────────┐       ┌──────────┐          ┌──────────────┐                  │
│  │  Noise   │       │  Signed  │          │ On-Chain     │                  │
│  │Protocol  │       │ Messages │          │ Stake/Rep    │                  │
│  │  (XX)    │       │          │          │              │                  │
│  └──────────┘       └──────────┘          └──────────────┘                  │
│                                                                              │
│  2. CHAIN AUTHENTICATION                                                    │
│  ───────────────────────                                                     │
│                                                                              │
│  EVM Wallet ──► ICN Token ──► stakeForRole() ──► pallet-icn-stake           │
│       │              │              │                    │                   │
│       │ sign tx      │ burn ERC20   │ precompile         │ reserve          │
│       ▼              ▼              ▼                    ▼                   │
│  MetaMask       Frontier EVM   IcnStakePrecompile   Substrate Balances      │
│                                                                              │
│  3. BFT AUTHENTICATION                                                      │
│  ─────────────────────                                                       │
│                                                                              │
│  Director ──► Elected? ──► sign(slot, embedding) ──► broadcast              │
│       │           │                │                      │                  │
│       │           │ query          │ Ed25519              │ verify           │
│       ▼           ▼                ▼                      ▼                  │
│  pallet-icn-   ElectedDirectors   Signature            Other Directors      │
│   director     storage             bytes                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Technology Stack & Rationale

### 5.1 Frontend (Viewer Client)

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Framework | Tauri | 2.0 | Native performance, Rust backend, small binary |
| UI | React | 18.x | Component model, large ecosystem |
| Video | WebCodecs | - | Hardware acceleration, low latency |
| State | Zustand | 4.x | Lightweight, TypeScript-first |
| P2P | libp2p-js | 1.x | Browser-compatible P2P |

**Alternative Considered:** Electron. Rejected due to 100MB+ binary size and higher memory usage.

### 5.2 Backend / Off-Chain Services

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Runtime | Rust | 1.75+ | Memory safety, Substrate compatibility |
| Async | Tokio | 1.35+ | Mature async runtime |
| P2P | rust-libp2p | 0.53.0 | Substrate ecosystem, QUIC support |
| Chain Client | subxt | 0.34+ | Type-safe Substrate RPC |
| AI Bridge | PyO3 | 0.20+ | Rust ↔ Python FFI |

### 5.3 AI / ML Pipeline

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Runtime | Python | 3.11 | ML ecosystem, PyTorch support |
| Framework | PyTorch | 2.1+ | CUDA support, dynamic graphs |
| Image Gen | Flux-Schnell | NF4 | Fast inference, 4-bit quantized |
| Video | LivePortrait | FP16 | TensorRT optimized |
| TTS | Kokoro-82M | FP32 | Quality voice synthesis |
| Verify | CLIP | ViT-B-32, ViT-L-14 | Semantic verification |
| Quantization | bitsandbytes | 0.41+ | NF4 support |

### 5.4 On-Chain / Blockchain

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Framework | Substrate FRAME | polkadot-v1.0.0 | Moonbeam compatibility |
| Parachain | Moonbeam | v0.35.0 | EVM + Substrate, shared security |
| EVM | Frontier | - | Full EVM compatibility |
| Token | OpenZeppelin | 5.x | Audited ERC-20 base |

### 5.5 Infrastructure / Cloud

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Compute | Bare metal / Cloud GPU | RTX 3060+ required for Directors |
| Container | Docker | Standardized deployment |
| Orchestration | Kubernetes (optional) | Scaling for Super-Nodes |
| DNS | Cloudflare | DDoS protection, DNS seeds |
| STUN/TURN | Coturn | NAT traversal |

### 5.6 DevOps & Tooling

| Component | Technology | Rationale |
|-----------|------------|-----------|
| CI/CD | GitHub Actions | Rust/Python matrix testing |
| Container Registry | GitHub Packages | Integrated with repo |
| Monitoring | Prometheus + Grafana | Industry standard |
| Tracing | Jaeger | OpenTelemetry compatible |
| Alerting | AlertManager | Prometheus integration |
| Logging | Vector + Loki | Structured log aggregation |

### 5.7 Security & Compliance Tools

| Component | Technology | Rationale |
|-----------|------------|-----------|
| SAST | cargo-audit, cargo-deny | Rust vulnerability scanning |
| Secrets | SOPS + age | Encrypted secrets in git |
| TLS | rustls | Pure Rust TLS, no OpenSSL |
| Audit | Trail of Bits, Oak Security | Third-party pallet audit |

---

## 6. Deployment & Operations

### 6.1 Deployment Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    DIRECTOR NODE DEPLOYMENT                            │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                   Bare Metal / Cloud GPU                         │  │  │
│  │  │                   (RTX 3060+ / A10 / L4)                         │  │  │
│  │  │                                                                  │  │  │
│  │  │  ┌─────────────────┐  ┌─────────────────┐                       │  │  │
│  │  │  │ Host OS         │  │ NVIDIA Driver   │                       │  │  │
│  │  │  │ (Ubuntu 22.04)  │  │ 535+            │                       │  │  │
│  │  │  └─────────────────┘  └─────────────────┘                       │  │  │
│  │  │           │                    │                                 │  │  │
│  │  │           ▼                    ▼                                 │  │  │
│  │  │  ┌─────────────────────────────────────────────────────────┐    │  │  │
│  │  │  │              Docker (--gpus all)                         │    │  │  │
│  │  │  │                                                          │    │  │  │
│  │  │  │  ┌─────────────────────────────────────────────────┐    │    │  │  │
│  │  │  │  │           icn-director:latest                    │    │    │  │  │
│  │  │  │  │                                                  │    │    │  │  │
│  │  │  │  │  • Rust binary (core runtime)                    │    │    │  │  │
│  │  │  │  │  • Python sidecar (Vortex)                       │    │    │  │  │
│  │  │  │  │  • Model weights (~15GB volume)                  │    │    │  │  │
│  │  │  │  │                                                  │    │    │  │  │
│  │  │  │  │  Ports: 9000 (P2P), 9100 (metrics), 50051 (gRPC) │    │    │  │  │
│  │  │  │  └─────────────────────────────────────────────────┘    │    │  │  │
│  │  │  └─────────────────────────────────────────────────────────┘    │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                   SUPER-NODE DEPLOYMENT (Kubernetes)                   │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      Kubernetes Cluster                          │  │  │
│  │  │                                                                  │  │  │
│  │  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │  │  │
│  │  │  │ super-node    │  │ super-node    │  │ super-node    │       │  │  │
│  │  │  │ (NA-WEST)     │  │ (EU-WEST)     │  │ (APAC)        │       │  │  │
│  │  │  │               │  │               │  │               │       │  │  │
│  │  │  │ Replicas: 2   │  │ Replicas: 2   │  │ Replicas: 2   │       │  │  │
│  │  │  └───────────────┘  └───────────────┘  └───────────────┘       │  │  │
│  │  │                                                                  │  │  │
│  │  │  ┌───────────────────────────────────────────────────────────┐  │  │  │
│  │  │  │              PersistentVolumeClaims (10TB each)            │  │  │  │
│  │  │  └───────────────────────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Environments

| Environment | Purpose | Infrastructure | On-Chain |
|-------------|---------|----------------|----------|
| **Local Dev** | Developer testing | Docker Compose | Local Substrate node |
| **Moonriver (Testnet)** | Integration testing | Cloud GPU (2-3 nodes) | Moonriver parachain |
| **Moonbeam (Mainnet)** | Production | Distributed (50+ nodes) | Moonbeam parachain |

### 6.3 Scalability & Resilience Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SCALABILITY ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HORIZONTAL SCALING                                                         │
│  ──────────────────                                                          │
│                                                                              │
│  Directors:      5 elected per slot (fixed by protocol)                     │
│  Super-Nodes:    7 regions × 2 replicas = 14 minimum                        │
│  Regional Relays: Auto-scale based on viewer count                          │
│  Viewers:        Unlimited (P2P assisted)                                    │
│                                                                              │
│  VERTICAL SCALING                                                           │
│  ────────────────                                                            │
│                                                                              │
│  Directors:      GPU upgrade (3060 → 4090) = 3× faster generation           │
│  Super-Nodes:    Storage expansion (10TB → 100TB)                           │
│  Regional Relays: Bandwidth upgrade (100Mbps → 1Gbps)                       │
│                                                                              │
│  RESILIENCE PATTERNS                                                        │
│  ───────────────────                                                         │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │ Director Failure│──► 3-of-5 still reaches consensus                      │
│  └─────────────────┘                                                        │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │ Super-Node      │──► 2 replicas per region + erasure coding              │
│  │ Failure         │                                                        │
│  └─────────────────┘                                                        │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │ Relay Failure   │──► Viewer falls back to Super-Node directly            │
│  └─────────────────┘                                                        │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │ Moonbeam Outage │──► Off-chain continues, BFT results queued             │
│  └─────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Observability

```yaml
# Observability Stack Configuration

metrics:
  prometheus:
    scrape_interval: 15s
    retention: 30d
    targets:
      - icn-director:9100
      - icn-super-node:9100
      - moonbeam-node:9615
  
  key_metrics:
    - icn_vortex_generation_time_seconds   # P99 < 15s
    - icn_bft_round_duration_seconds       # P99 < 10s
    - icn_p2p_connected_peers              # > 10
    - icn_total_staked_tokens              # Monitor centralization
    - icn_slashing_events_total            # Alert on any

logging:
  format: json
  level: info  # debug for dev
  fields:
    - timestamp
    - level
    - target  # Rust module path
    - slot    # Current slot number
    - peer_id # P2P identity
  
  aggregation:
    tool: vector
    destination: loki
    retention: 7d

tracing:
  protocol: otlp
  endpoint: jaeger:4317
  sampling: 0.1  # 10% of requests
  
  span_names:
    - vortex.generate_slot
    - bft.exchange_embeddings
    - p2p.publish_video

alerting:
  critical:
    - DirectorSlotMissed: "BFT failure in last 5 minutes"
    - VortexOOM: "VRAM usage > 11.5GB"
    - ChainDisconnected: "No new blocks in 60s"
  
  warning:
    - StakeConcentration: "Region > 25% of total stake"
    - BftLatencyHigh: "BFT round > 10s for 5 minutes"
    - PeerCountLow: "Connected peers < 5"
```

### 6.5 Backup, Disaster Recovery, and SLAs

| Component | Backup Strategy | RPO | RTO | SLA |
|-----------|-----------------|-----|-----|-----|
| On-Chain State | Blockchain (inherent) | 0 | 0 | 99.99% (Polkadot) |
| Video Content | Erasure coding (5×) | 1 slot | 1 minute | 99.5% |
| Model Weights | Object storage | 1 day | 1 hour | 99.9% |
| Node Config | Git + encrypted secrets | Realtime | 10 minutes | N/A |

**Disaster Recovery Procedures:**

1. **Director Key Compromise:** Rotate keypair, slash own stake, re-stake with new identity
2. **Super-Node Data Loss:** Reconstruct from erasure shards on other Super-Nodes
3. **Region Outage:** Viewers automatically failover to next-closest region
4. **Moonbeam Outage:** Off-chain continues, BFT results submitted when chain recovers

---

## 7. Security & Compliance

### 7.1 Threat Model Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            THREAT MODEL                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ASSET                          THREATS                    CONTROLS          │
│  ─────                          ───────                    ────────          │
│                                                                              │
│  Staked Tokens ($)              • Key theft                • HSM storage     │
│                                 • Rug pull                 • Multisig        │
│                                 • Smart contract bug       • Audits          │
│                                                                              │
│  Reputation Scores              • Score manipulation       • Merkle proofs   │
│                                 • Sybil attack             • Stake + PoI     │
│                                 • Collusion                • Challenge period│
│                                                                              │
│  Video Content                  • CSAM/illegal content     • CLIP filter     │
│                                 • Copyright infringement   • Content policy  │
│                                 • Adversarial inputs       • Ensemble CLIP   │
│                                                                              │
│  P2P Network                    • Eclipse attack           • Peer diversity  │
│                                 • DDoS                     • Rate limiting   │
│                                 • Sybil nodes              • Stake gating    │
│                                                                              │
│  Director Nodes                 • RCE via model            • Sandboxed exec  │
│                                 • Side-channel GPU         • Isolated VRAM   │
│                                 • Key extraction           • Secure enclave  │
│                                                                              │
│  ATTACKER PROFILES                                                          │
│  ─────────────────                                                           │
│                                                                              │
│  1. Script Kiddie:     DDoS, spam recipes                                   │
│     Motivation:        Disruption                                            │
│     Capability:        Low (botnets)                                         │
│     Mitigation:        PoW on recipes, rate limiting                         │
│                                                                              │
│  2. Competitor:        Eclipse attacks, content sabotage                     │
│     Motivation:        Market disruption                                     │
│     Capability:        Medium (funded)                                       │
│     Mitigation:        Geographic diversity, reputation                      │
│                                                                              │
│  3. Nation State:      Censorship, surveillance                              │
│     Motivation:        Control                                               │
│     Capability:        High (infrastructure)                                 │
│     Mitigation:        E2E encryption, decentralization                      │
│                                                                              │
│  4. Malicious Insider: Key theft, backdoor                                   │
│     Motivation:        Financial gain                                        │
│     Capability:        High (access)                                         │
│     Mitigation:        Multisig, code review, audits                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Authentication & Authorization

| Layer | Mechanism | Keys | Rotation |
|-------|-----------|------|----------|
| P2P Identity | Ed25519 | PeerId keypair | Manual (requires re-stake) |
| Chain Transactions | Sr25519 | Substrate keypair | Manual (requires migration) |
| EVM Transactions | ECDSA secp256k1 | Ethereum keypair | Via MetaMask |
| TLS (P2P) | Noise XX | Per-connection ephemeral | Automatic |
| BFT Signing | Ed25519 | Same as P2P | Inherent |

**Authorization Matrix:**

| Action | Required Role | Minimum Stake | Additional Checks |
|--------|---------------|---------------|-------------------|
| submit_bft_result | Director | 100 ICN | Elected for slot |
| challenge_bft_result | Any staker | 25 ICN (bond) | Result not finalized |
| initiate_audit | Root (pallet) | N/A | Random selection |
| create_deal | Any | Payment amount | N/A |

### 7.3 Data Protection

| Data Type | At Rest | In Transit | Access Control |
|-----------|---------|------------|----------------|
| Private Keys | Encrypted (SOPS/age) | N/A | File permissions |
| Video Content | Plaintext (erasure coded) | AES-256-GCM | P2P authenticated |
| CLIP Embeddings | Plaintext | Noise encrypted | Directors only |
| Recipes | Plaintext | Ed25519 signed | Public |
| Reputation | On-chain (public) | N/A | Pallet origin |

### 7.4 Compliance Considerations

| Requirement | Approach | Status |
|-------------|----------|--------|
| GDPR | No PII stored; pseudonymous PeerIds | Compliant |
| DMCA | Content policy enforcement via CLIP | Planned |
| AML/KYC | Not required (utility token) | N/A |
| SOC 2 | Not applicable (decentralized) | N/A |

---

## 8. Risks, Assumptions, Dependencies & Mitigations

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Moonbeam governance rejection | Medium | High | Progressive deployment, community engagement |
| CLIP adversarial bypass | Medium | High | Ensemble models, outlier detection |
| RTX 3060 availability | Low | Medium | Support for cloud GPUs (A10, L4) |
| libp2p performance at scale | Medium | Medium | Hierarchical topology, load testing |
| Substrate version incompatibility | Low | High | Pin to Moonbeam's exact version |

### 8.2 Assumptions

| Assumption | Impact if Invalid | Validation |
|------------|-------------------|------------|
| Moonbeam TPS sufficient (~50) | Must migrate to parachain | Monitor during testnet |
| 12GB VRAM fits all models | Exclude low-end GPUs | Extensive memory testing |
| CLIP quality is sufficient | Content quality issues | A/B testing with human eval |
| VRF is unbiased | Election manipulation | Audit Moonbeam VRF impl |
| 45s is achievable glass-to-glass | Poor UX | Benchmark on target hardware |

### 8.3 External Dependencies

| Dependency | Type | Criticality | Fallback |
|------------|------|-------------|----------|
| Moonbeam Network | Infrastructure | Critical | Astar migration |
| Polkadot Relay Chain | Security | Critical | None (fundamental) |
| Hugging Face Models | AI Assets | High | Self-hosted model cache |
| STUN/TURN Servers | NAT Traversal | Medium | Community-run, circuit relay |
| DNS Seeds | Bootstrap | Low | Hardcoded peers |

### 8.4 Mitigation Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RISK MITIGATION MATRIX                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RISK: Governance Rejection                                                 │
│  ─────────────────────────────                                               │
│                                                                              │
│  Phase 1: Deploy treasury pallet (low risk, community benefit)              │
│  Phase 2: Add reputation pallet (read-only for validators)                  │
│  Phase 3: Full stake + director pallets                                     │
│  Fallback: Astar network (similar capabilities)                             │
│                                                                              │
│  RISK: CLIP Adversarial Attacks                                             │
│  ──────────────────────────────                                              │
│                                                                              │
│  Layer 1: Dual CLIP ensemble (ViT-B + ViT-L)                                │
│  Layer 2: Outlier detection on embedding variance                           │
│  Layer 3: Prompt sanitization (remove injection patterns)                   │
│  Layer 4: Human escalation for borderline cases                             │
│  Future:  Add RN50 third model for architecture diversity                   │
│                                                                              │
│  RISK: Director Collusion                                                   │
│  ────────────────────────────                                                │
│                                                                              │
│  Control 1: Multi-region requirement (max 2 directors per region)           │
│  Control 2: Challenge period with stake slashing                            │
│  Control 3: Statistical anomaly detection on voting patterns                │
│  Control 4: Jitter in election to prevent coordination                      │
│                                                                              │
│  RISK: Network Partition                                                    │
│  ──────────────────────────                                                  │
│                                                                              │
│  Detection: Peer count monitoring, cross-region probes                      │
│  Mitigation: Bootstrap from multiple sources (DNS, HTTP, DHT)               │
│  Recovery: Automatic reconnection with exponential backoff                  │
│  Viewer UX: "Soft fork" – continue current stream until natural break       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Roadmap & Evolution

### 9.1 Phased Implementation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMPLEMENTATION ROADMAP                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: MOONRIVER TESTNET (Weeks 1-8)                                     │
│  ──────────────────────────────────────                                      │
│                                                                              │
│  Week 1-2: Foundation                                                       │
│  ├── Fork Moonbeam repository                                               │
│  ├── Set up dev environment (local node)                                    │
│  └── Implement pallet-icn-stake (deposit, slash)                            │
│                                                                              │
│  Week 3-4: Reputation                                                       │
│  ├── Implement pallet-icn-reputation                                        │
│  ├── Merkle tree for event batching                                         │
│  └── Unit tests for decay and scoring                                       │
│                                                                              │
│  Week 5-6: Director Logic                                                   │
│  ├── Implement pallet-icn-director                                          │
│  ├── VRF-based election                                                     │
│  ├── Challenge mechanism                                                    │
│  └── Integration tests                                                      │
│                                                                              │
│  Week 7-8: Moonriver Deployment                                             │
│  ├── Build runtime WASM                                                     │
│  ├── Deploy to Moonriver testnet                                            │
│  └── End-to-end testing with 10+ nodes                                      │
│                                                                              │
│  EXIT CRITERIA:                                                             │
│  ☐ All pallets compile and pass unit tests                                  │
│  ☐ Runtime upgrade deployed to Moonriver                                    │
│  ☐ Staking → Election → Reputation flow works                               │
│  ☐ 10+ test nodes participating                                             │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  PHASE 2: MOONBEAM MAINNET (Weeks 9-16)                                     │
│  ────────────────────────────────────────                                    │
│                                                                              │
│  Week 9-10: Security Audit                                                  │
│  ├── Engage Oak Security / SRLabs                                           │
│  ├── Address critical findings                                              │
│  └── Re-audit patched pallets                                               │
│                                                                              │
│  Week 11-12: Governance Proposal                                            │
│  ├── Prepare proposal documentation                                         │
│  ├── Community engagement (forums, calls)                                   │
│  └── Submit to Moonbeam OpenGov                                             │
│                                                                              │
│  Week 13-14: Token Launch                                                   │
│  ├── Deploy ICN ERC-20 contract                                             │
│  ├── Bootstrap initial liquidity (DEX)                                      │
│  └── Airdrop to testnet participants                                        │
│                                                                              │
│  Week 15-16: Production Launch                                              │
│  ├── Mainnet runtime upgrade (if approved)                                  │
│  ├── Onboard 50+ nodes                                                      │
│  └── Public announcement                                                    │
│                                                                              │
│  EXIT CRITERIA:                                                             │
│  ☐ Security audit passed                                                    │
│  ☐ Governance proposal approved                                             │
│  ☐ ICN token live on Moonbeam                                               │
│  ☐ 50+ mainnet nodes                                                        │
│  ☐ 500+ community members                                                   │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  PHASE 3: SCALE & ITERATE (Ongoing)                                        │
│  ───────────────────────────────────                                         │
│                                                                              │
│  Quarter 1 Post-Launch:                                                     │
│  ├── EVM precompiles for staking/reputation                                 │
│  ├── Mobile viewer app (iOS/Android)                                        │
│  └── Additional CLIP models (RN50)                                          │
│                                                                              │
│  Quarter 2:                                                                 │
│  ├── Cross-chain messaging (XCM)                                            │
│  ├── Advanced content policies                                              │
│  └── Director hardware acceleration (TensorRT)                              │
│                                                                              │
│  Future:                                                                    │
│  ├── Dedicated parachain (if TPS > 50)                                      │
│  ├── ZK proofs for privacy                                                  │
│  └── Decentralized governance (DAO)                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Extensibility Points

| Extension Point | Mechanism | Use Case |
|-----------------|-----------|----------|
| New AI Models | Vortex plugin system | Add new generators (SD3, etc.) |
| Custom Pallets | Moonbeam governance | New protocol features |
| Content Policies | CLIP prompt filters | Regional compliance |
| Token Standards | ERC-721/1155 | NFT integration |
| Cross-Chain | XCM messages | Multi-chain reputation |

---

## 10. Appendices

### 10.1 Glossary

| Term | Definition |
|------|------------|
| **BFT** | Byzantine Fault Tolerance – consensus requiring 3-of-5 agreement |
| **CLIP** | Contrastive Language-Image Pretraining – OpenAI model for semantic matching |
| **Director** | Node elected to generate video content for a slot |
| **Erasure Coding** | Reed-Solomon (10+4) data redundancy scheme |
| **FRAME** | Substrate's modular runtime framework |
| **GossipSub** | libp2p pubsub protocol for message propagation |
| **Moonbeam** | Polkadot parachain with EVM compatibility |
| **Pallet** | Substrate runtime module (analogous to smart contract) |
| **Recipe** | JSON instruction set for AI content generation |
| **Slot** | 45-90 second time window for content generation |
| **Super-Node** | High-stake node providing storage and relay services |
| **Vortex** | ICN's GPU-resident AI generation engine |
| **VRF** | Verifiable Random Function for unpredictable director election |

### 10.2 References

| Resource | URL |
|----------|-----|
| Moonbeam Docs | https://docs.moonbeam.network/ |
| Substrate FRAME | https://docs.substrate.io/reference/frame-pallets/ |
| libp2p Spec | https://github.com/libp2p/specs |
| CLIP Paper | https://arxiv.org/abs/2103.00020 |
| Reed-Solomon | https://en.wikipedia.org/wiki/Reed–Solomon_error_correction |
| Polkadot Wiki | https://wiki.polkadot.network/ |

### 10.3 Diagram Tools

| Diagram Type | Recommended Tool |
|--------------|------------------|
| Architecture | Excalidraw, draw.io |
| Sequence | Mermaid, PlantUML |
| Data Flow | Lucidchart |
| Component | C4 Model (Structurizr) |

---

**Document Status:** Draft for Review  
**Author:** Senior Architecture Team  
**Reviewers:** Core Engineering, Security, DevOps  
**Last Updated:** 2025-12-24  
**Next Review:** After Phase 1 completion

---

*End of Technical Architecture Document*

