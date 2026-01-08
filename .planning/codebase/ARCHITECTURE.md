# Architecture

**Analysis Date:** 2026-01-08

## Pattern Overview

**Overall:** Decentralized Multi-Component System with Dual-Lane Architecture

**Key Characteristics:**
- Sovereign blockchain (Polkadot SDK solochain) for coordination and consensus
- Off-chain P2P mesh network for compute and video delivery
- Dual-lane architecture: Lane 0 (verified video) + Lane 1 (general AI marketplace)
- Epoch-based director elections with On-Deck protocol
- Desktop client for end-user streaming

## Layers

**On-Chain Layer (nsn-chain):**
- Purpose: Coordination, staking, reputation, elections, consensus finalization
- Contains: FRAME pallets for stake, reputation, director elections, BFT, storage deals, task market
- Location: `nsn-chain/pallets/`, `nsn-chain/runtime/`
- Depends on: Polkadot SDK, Substrate runtime
- Used by: node-core via subxt chain client

**Off-Chain Layer (node-core):**
- Purpose: P2P networking, task scheduling, compute orchestration
- Contains: libp2p networking, GossipSub topics, scheduler, lane orchestrators
- Location: `node-core/crates/`, `node-core/bin/nsn-node/`
- Depends on: nsn-chain (via subxt), libp2p, tokio
- Used by: AI workers, validators, directors

**AI Layer (vortex):**
- Purpose: GPU-resident video generation for Lane 0
- Contains: Model loaders (Flux, LivePortrait, Kokoro, CLIP), generation pipeline
- Location: `vortex/src/vortex/`
- Depends on: PyTorch, diffusers, transformers
- Used by: node-core sidecar for Lane 0 generation tasks

**Compute Layer (sidecar):**
- Purpose: Isolated execution environment for AI tasks
- Contains: Task execution runtime, resource management
- Location: `node-core/sidecar/`
- Depends on: vortex (for Lane 0), container runtime
- Used by: scheduler via gRPC (tonic)

**Client Layer (viewer):**
- Purpose: Desktop streaming client for end users
- Contains: React UI, video playback, P2P stream reception
- Location: `viewer/src/`, `viewer/src-tauri/`
- Depends on: Tauri, React, P2P client
- Used by: End users watching streams

## Data Flow

**Lane 0 - Video Generation Flow:**

1. Epoch begins, directors elected from On-Deck set (on-chain)
2. Director receives prompt via P2P GossipSub
3. Director distributes work to validators
4. Vortex pipeline generates video chunks (GPU)
5. Validators verify via CLIP semantic ensemble
6. BFT consensus finalizes chunk (3-of-5 threshold)
7. Video chunk published to P2P mesh
8. Viewer client receives and renders stream

**Lane 1 - Task Marketplace Flow:**

1. User submits task to nsn-task-market pallet (on-chain)
2. Task announced via P2P GossipSub
3. Scheduler matches task to available compute
4. Sidecar executes task in isolated environment
5. Result submitted and verified
6. Payment released via treasury pallet

**State Management:**
- On-chain: Substrate storage (StorageValue, StorageMap, StorageDoubleMap)
- Off-chain: In-memory state with P2P gossip synchronization
- Client: Zustand store for React state management

## Key Abstractions

**Pallet:**
- Purpose: On-chain business logic module
- Examples: `pallet-nsn-stake`, `pallet-nsn-reputation`, `pallet-nsn-director`
- Location: `nsn-chain/pallets/*/src/lib.rs`
- Pattern: FRAME pallet macro with Config, Storage, Events, Errors, Call

**Crate:**
- Purpose: Off-chain Rust module for specific functionality
- Examples: `nsn-p2p`, `nsn-scheduler`, `nsn-lane0`
- Location: `node-core/crates/*/`
- Pattern: Workspace member with lib.rs entry point

**Pipeline:**
- Purpose: AI processing pipeline for video generation
- Examples: FluxPipeline, LivePortraitPipeline, CLIPEnsemble
- Location: `vortex/src/vortex/models/`, `vortex/src/vortex/pipeline/`
- Pattern: Model loading + inference methods

**Topic:**
- Purpose: P2P message routing channel
- Examples: epoch announcements, task broadcasts, video chunks
- Pattern: GossipSub topic with subscription management

## Entry Points

**nsn-chain/node:**
- Location: `nsn-chain/node/src/main.rs`
- Triggers: Binary execution
- Responsibilities: Start blockchain node, RPC server, networking

**nsn-node (off-chain):**
- Location: `node-core/bin/nsn-node/src/main.rs`
- Triggers: Binary execution with mode flag
- Responsibilities: P2P networking, scheduler, mode-specific logic (director/validator/storage)

**Vortex Pipeline:**
- Location: `vortex/src/vortex/pipeline/`
- Triggers: gRPC call from sidecar
- Responsibilities: Load models, run inference, return results

**Viewer App:**
- Location: `viewer/src/main.tsx`, `viewer/src-tauri/src/main.rs`
- Triggers: Application launch
- Responsibilities: Render UI, handle P2P streams, display video

## Error Handling

**Strategy:** Typed errors at boundaries, Result types internally

**Patterns:**
- Rust: `thiserror` for custom error types, `anyhow` for application errors
- Python: Exception classes with structured error info
- TypeScript: Error boundaries in React, typed errors in services
- On-chain: Pallet errors via `#[pallet::error]` macro

## Cross-Cutting Concerns

**Logging:**
- Rust: `tracing` with structured spans and events
- Python: Standard logging with structured output
- TypeScript: Console logging (Tauri handles persistence)

**Metrics:**
- Prometheus metrics via `prometheus-client` (Python) and `prometheus-endpoint` (Rust)
- Key metrics: VRAM usage, inference latency, P2P message rates

**Configuration:**
- Environment variables for secrets
- TOML/YAML files for runtime configuration
- CLI flags via clap for node operation modes

**Consensus:**
- On-chain: Substrate consensus (Aura/GRANDPA for solo, Cumulus for parachain)
- Off-chain: BFT for Lane 0 video chunks (3-of-5 director threshold)

---

*Architecture analysis: 2026-01-08*
*Update when major patterns change*
