# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Interdimensional Cable Network (ICN)** is a decentralized AI-powered video streaming platform built as its own Polkadot SDK chain. The system combines:

- **On-chain layer**: ICN Chain - a Polkadot SDK solochain with custom FRAME pallets for staking, reputation, BFT consensus, and content pinning
- **Off-chain layer**: P2P mesh network (Directors, Validators, Super-Nodes, Relays, Viewers) using libp2p
- **AI layer (Vortex)**: GPU-resident pipeline for video generation using Flux-Schnell, LivePortrait, Kokoro TTS, and dual CLIP ensemble

The project uses the Polkadot SDK to build ICN as a sovereign chain with staged deployment: Solochain MVP → Parachain (optional) → Coretime scaling.

## Repository Structure

```
interdim-cable/
├── icn-chain/              # ICN Chain (Polkadot SDK runtime + pallets)
│   ├── pallets/
│   │   ├── icn-stake/      # Token staking, slashing, role eligibility
│   │   ├── icn-reputation/ # Reputation scoring with Merkle proofs
│   │   ├── icn-director/   # VRF-based director election, BFT coordination
│   │   ├── icn-bft/        # BFT consensus storage and finalization
│   │   ├── icn-pinning/    # Erasure coding deals and audits
│   │   └── icn-treasury/   # Reward distribution and emissions
│   ├── runtime/            # ICN runtime configuration
│   ├── node/               # ICN node client implementation
│   ├── precompiles/        # Optional EVM precompiles (Frontier)
│   └── test/               # Integration tests
├── icn-nodes/              # Off-chain node implementations (T009-T012, T021-T027)
│   ├── common/             # Shared P2P, chain client, types
│   ├── director/           # GPU video generation + BFT coordination
│   ├── validator/          # CLIP semantic verification
│   ├── super-node/         # Tier 1 erasure-coded storage
│   └── relay/              # Tier 2 regional distribution
├── vortex/                 # AI generation engine - Python (T014-T020)
│   └── src/vortex/
│       ├── models/         # Flux, LivePortrait, Kokoro, CLIP loaders
│       ├── pipeline/       # Generation orchestration
│       └── utils/          # VRAM management
├── viewer/                 # Desktop app - Tauri + React (T013)
│   ├── src/                # React frontend
│   └── src-tauri/          # Tauri Rust backend
├── .claude/rules/          # Project architecture and PRD documents
│   ├── architecture.md     # Technical Architecture Document (TAD v1.1)
│   └── prd.md              # Product Requirements Document (PRD v9.0)
└── .tasks/                 # Task management system
    ├── manifest.json       # All tasks with dependencies
    └── tasks/              # Individual task specifications
```

## Build Commands

### On-Chain (icn-chain/)

```bash
cd icn-chain

# Build the ICN node (optimized release)
cargo build --release

# Build specific pallet
cargo build --release -p pallet-icn-stake

# Rust linting and formatting
cargo clippy --release --workspace
cargo fmt -- --check
```

### Off-Chain Nodes (icn-nodes/)

```bash
cd icn-nodes

# Build all nodes
cargo build --release

# Build specific node
cargo build --release -p icn-director
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
cargo test -p pallet-icn-stake
cargo test -p pallet-icn-reputation
cargo test -p pallet-icn-director

# Integration tests
cd icn-chain/test
pnpm test
```

## Running Development Node

```bash
# Using built binary - local ICN solochain
./target/release/icn-node --dev --alice --rpc-port 9944

# Multi-node local testnet
./target/release/icn-node --chain=local --alice --port 30333 --rpc-port 9944
./target/release/icn-node --chain=local --bob --port 30334 --rpc-port 9945 --bootnodes /ip4/127.0.0.1/tcp/30333/p2p/<ALICE_PEER_ID>
```

## ICN Pallet Architecture

The ICN pallets follow a dependency hierarchy:

```
pallet-icn-stake (foundation)
    ↓
pallet-icn-reputation (depends on stake)
    ↓
pallet-icn-director (depends on stake + reputation)
    ↓
pallet-icn-bft (depends on director)

pallet-icn-pinning (depends on stake)
pallet-icn-treasury (depends on stake)
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
| Directors per slot | 5 (3-of-5 BFT threshold) |
| Challenge period | 50 blocks (~5 minutes) |
| Min director stake | 100 ICN |
| VRAM requirement | 11.8 GB (RTX 3060 12GB minimum) |
| Glass-to-glass latency | 45 seconds target |
| Polkadot SDK version | polkadot-stable2409 |

## Task Management

The project uses a structured task system in `.tasks/`:
- `manifest.json` contains all tasks with dependency graph
- Critical path: T001 → T002 → T003 → T004 → T005 → T009 → T034 → T035 → T037
- Phase A (ICN Solo): Weeks 1-8, Phase B (ICN Mainnet): Weeks 9-16

To check next actionable task: Look at `.tasks/manifest.json` for tasks with status "pending" and all dependencies complete.

## Network Configuration

| Network | Purpose | Deployment |
|---------|---------|------------|
| ICN Dev | Local development | Single-node --dev |
| ICN Local | Multi-node testing | Local testnet |
| ICN Testnet | Integration testing | Public testnet |
| ICN Mainnet | Production | Mainnet (solo → parachain) |

## Staged Deployment Model

1. **Phase A: ICN Solochain** - Controlled validator set, fast iteration
2. **Phase B: ICN Mainnet** - Public validators, production deployment
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
- `.claude/rules/architecture.md` - Technical Architecture Document (system design, ADRs, deployment)
- `.claude/rules/prd.md` - Product Requirements Document (features, tokenomics, security model)
