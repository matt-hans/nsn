# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Neural Sovereign Network (NSN)** is a decentralized AI compute marketplace built as its own Polkadot SDK chain with dual-lane architecture:

- **Lane 0 (Video Generation)**: AI-powered video streaming with Directors, Validators, and BFT consensus
- **Lane 1 (General AI Compute)**: Open marketplace for arbitrary AI tasks via task-market pallet

The system combines:

- **On-chain layer**: NSN Chain - a Polkadot SDK solochain with custom FRAME pallets for staking, reputation, epochs, task marketplace, and model registry
- **Off-chain layer**: P2P mesh network using libp2p with epoch-based elections and On-Deck protocol
- **AI layer (Vortex)**: GPU-resident pipeline for Lane 0 video generation using Flux-Schnell, LivePortrait, Kokoro TTS, and dual CLIP ensemble
- **Compute layer (node-core)**: Universal compute orchestration for Lane 1 with scheduler and sidecar architecture

The project uses the Polkadot SDK to build NSN as a sovereign chain with staged deployment: Solochain MVP → Parachain (optional) → Coretime scaling.

## Repository Structure

```
interdim-cable/
├── nsn-chain/               # NSN Chain (Polkadot SDK runtime + pallets)
│   ├── pallets/
│   │   ├── nsn-stake/       # Token staking, slashing, role eligibility
│   │   ├── nsn-reputation/  # Reputation scoring with Merkle proofs
│   │   ├── nsn-director/    # Epoch-based elections with On-Deck protocol
│   │   ├── nsn-bft/         # BFT consensus storage and finalization
│   │   ├── nsn-storage/     # Erasure coding deals and audits
│   │   ├── nsn-treasury/    # Reward distribution and emissions
│   │   ├── nsn-task-market/ # Lane 1 task marketplace
│   │   └── nsn-model-registry/ # Model capability registry
│   ├── runtime/             # NSN runtime configuration
│   ├── node/                # NSN node client implementation
│   ├── precompiles/         # Optional EVM precompiles (Frontier)
│   └── test/                # Integration tests
├── node-core/               # Off-chain node core (Rust)
│   ├── bin/nsn-node/         # Unified node binary (modes for director/validator/storage)
│   ├── crates/p2p/           # P2P networking (GossipSub + reputation)
│   ├── crates/scheduler/     # Task scheduler with On-Deck protocol
│   ├── crates/lane0/         # Lane 0 orchestration logic
│   ├── crates/lane1/         # Lane 1 orchestration logic
│   └── sidecar/              # Compute execution runtime
├── vortex/                 # Lane 0: AI generation engine - Python
│   └── src/vortex/
│       ├── models/         # Flux, LivePortrait, Kokoro, CLIP loaders
│       ├── pipeline/       # Generation orchestration
│       └── utils/          # VRAM management
├── viewer/                 # Desktop app - Tauri + React
│   ├── src/                # React frontend
│   └── src-tauri/          # Tauri Rust backend
├── .claude/rules/          # Project architecture and PRD documents
│   ├── architecture.md     # Technical Architecture Document (TAD v2.0)
│   └── prd.md              # Product Requirements Document (PRD v10.0)
└── .tasks/                 # Task management system
    ├── manifest.json       # All tasks with dependencies
    └── tasks/              # Individual task specifications
```

## Build Commands

### On-Chain (nsn-chain/)

```bash
cd nsn-chain

# Build the NSN node (optimized release)
cargo build --release

# Build specific pallet
cargo build --release -p pallet-nsn-stake

# Rust linting and formatting
cargo clippy --release --workspace
cargo fmt -- --check
```

### Off-Chain Node Core (node-core/)

```bash
cd node-core

# Build all components
cargo build --release

# Build the unified node binary
cargo build --release -p nsn-node
```

### Lane 1 Compute (node-core/)

```bash
cd node-core

# Build scheduler and sidecar
cargo build --release

# Build specific component
cargo build --release -p scheduler
cargo build --release -p sidecar

# Rust linting and formatting
cargo clippy --release --workspace
cargo fmt -- --check
```

### Vortex AI Engine (vortex/)

```bash
cd vortex

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Lint
ruff check src/
```

### Viewer App (viewer/)

```bash
cd viewer

# Install dependencies
pnpm install

# Development mode
pnpm tauri:dev

# Production build
pnpm tauri:build
```

## Testing

```bash
# Run all Rust tests
cargo test

# Run specific pallet tests
cargo test -p pallet-nsn-stake
cargo test -p pallet-nsn-reputation
cargo test -p pallet-nsn-director
cargo test -p pallet-nsn-task-market
cargo test -p pallet-nsn-model-registry

# Integration tests
cd nsn-chain/test
pnpm test
```

## Running Development Node

```bash
# Using built binary - local NSN solochain
./target/release/nsn-node --dev --alice --rpc-port 9944

# Multi-node local testnet
./target/release/nsn-node --chain=local --alice --port 30333 --rpc-port 9944
./target/release/nsn-node --chain=local --bob --port 30334 --rpc-port 9945 --bootnodes /ip4/127.0.0.1/tcp/30333/p2p/<ALICE_PEER_ID>
```

## NSN Pallet Architecture

The NSN pallets follow a dependency hierarchy:

```
pallet-nsn-stake (foundation)
    ↓
pallet-nsn-reputation (depends on stake)
    ↓
pallet-nsn-epochs (depends on stake + reputation)
    ↓
pallet-nsn-bft (depends on epochs)

pallet-nsn-pinning (depends on stake)
pallet-nsn-treasury (depends on stake)
pallet-nsn-task-market (depends on stake + reputation)
pallet-nsn-model-registry (depends on stake)
```

**Key design patterns:**
- All pallets use `#![cfg_attr(not(feature = "std"), no_std)]` for WASM compatibility
- Storage items use Substrate's `StorageValue`, `StorageMap`, `StorageDoubleMap`
- Events follow `#[pallet::generate_deposit(pub(super) fn deposit_event)]` pattern
- Extrinsics use `#[pallet::call_index(N)]` for stable call indices
- Pallets designed for Cumulus compatibility (future parachain migration)

## Key Technical Specifications

| Component | Specification |
|-----------|---------------|
| Epoch duration | 100 blocks (~10 minutes) |
| On-Deck set size | 20 Directors |
| Elected Directors per epoch | 5 (3-of-5 BFT threshold) |
| Challenge period | 50 blocks (~5 minutes) |
| Min director stake | 100 NSN |
| VRAM requirement (Lane 0) | 11.8 GB (RTX 3060 12GB minimum) |
| Glass-to-glass latency (Lane 0) | 45 seconds target |
| Polkadot SDK version | polkadot-stable2409 |

## Dual-Lane Architecture

### Lane 0: Video Generation (Deterministic)
- Epoch-based elections with On-Deck protocol
- 5 Directors elected per epoch
- BFT consensus (3-of-5 threshold)
- Vortex AI pipeline (Flux, LivePortrait, Kokoro, CLIP)
- P2P video distribution

### Lane 1: General AI Compute (Marketplace)
- Open task marketplace via nsn-task-market pallet
- Arbitrary AI workloads (inference, training, fine-tuning)
- Model registry for capability discovery
- node-core scheduler with sidecar execution runtime
- Stake-based task acceptance and reputation-driven matching

## Task Management

The project uses a structured task system in `.tasks/`:
- `manifest.json` contains all tasks with dependency graph
- Critical path: T001 → T002 → T003 → T004 → T005 → T009 → T034 → T035 → T037
- Phase A (NSN Solo): Weeks 1-8, Phase B (NSN Mainnet): Weeks 9-16

To check next actionable task: Look at `.tasks/manifest.json` for tasks with status "pending" and all dependencies complete.

## Network Configuration

| Network | Purpose | Deployment |
|---------|---------|------------|
| NSN Dev | Local development | Single-node --dev |
| NSN Local | Multi-node testing | Local testnet |
| NSN Testnet | Integration testing | Public testnet |
| NSN Mainnet | Production | Mainnet (solo → parachain) |

## Staged Deployment Model

1. **Phase A: NSN Solochain** - Controlled validator set, fast iteration
2. **Phase B: NSN Mainnet** - Public validators, production deployment
3. **Phase C: Parachain** - Cumulus integration, Polkadot shared security
4. **Phase D: Coretime** - Elastic scaling via Polkadot coretime

## Claude Code Hooks

### Rust Compile Check Hook

A PostToolUse hook automatically runs after Edit, Write, or MultiEdit operations on Rust files:

**Location:** `.claude/hooks/rust-compile-check.sh`

**Behavior:**
- Triggers only when `*.rs`, `Cargo.toml`, or `Cargo.lock` files are modified
- Runs `cargo check` and `cargo clippy -- -D warnings`
- If checks fail, displays full error output for Claude to fix
- Silent on success (no output clutter)
- Silent when non-Rust files are edited

**Configuration:** Registered in `.claude/settings.json` under `hooks.PostToolUse`

**Note:** If you see compilation errors after editing Rust code, Claude will automatically see them and can fix the issues.

## Architecture Documents

For detailed specifications, refer to:
- `.claude/rules/architecture.md` - Technical Architecture Document (TAD v2.0: dual-lane architecture, epochs, On-Deck protocol)
- `.claude/rules/prd.md` - Product Requirements Document (PRD v10.0: NSN dual-lane architecture, task marketplace, model registry)
