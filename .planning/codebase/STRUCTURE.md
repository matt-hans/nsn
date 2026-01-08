# Codebase Structure

**Analysis Date:** 2026-01-08

## Directory Layout

```
interdim-cable/
├── nsn-chain/              # Polkadot SDK blockchain (on-chain)
│   ├── node/               # NSN node client
│   ├── runtime/            # WASM runtime
│   └── pallets/            # FRAME pallets
├── node-core/              # Off-chain node (Rust)
│   ├── bin/nsn-node/       # Unified node binary
│   ├── crates/             # Workspace crates
│   └── sidecar/            # Compute execution runtime
├── vortex/                 # AI engine (Python)
│   ├── src/vortex/         # Source code
│   └── tests/              # Test files
├── viewer/                 # Desktop client (Tauri + React)
│   ├── src/                # React frontend
│   ├── src-tauri/          # Tauri Rust backend
│   └── e2e/                # Playwright E2E tests
├── docker/                 # Docker configurations
├── scripts/                # Utility scripts
├── docs/                   # Documentation
├── .tasks/                 # Task management system
└── .claude/                # Claude Code configuration
```

## Directory Purposes

**nsn-chain/**
- Purpose: Polkadot SDK blockchain runtime and node
- Contains: FRAME pallets, runtime configuration, node binary
- Key files: `Cargo.toml` (workspace), `runtime/src/lib.rs`, `node/src/main.rs`
- Subdirectories:
  - `pallets/nsn-stake/` - Token staking and slashing
  - `pallets/nsn-reputation/` - Reputation scoring with Merkle proofs
  - `pallets/nsn-director/` - Epoch-based director elections
  - `pallets/nsn-bft/` - BFT consensus storage
  - `pallets/nsn-bootstrap/` - Chain bootstrap configuration
  - `pallets/nsn-storage/` - Erasure coding storage deals
  - `pallets/nsn-treasury/` - Reward distribution
  - `pallets/nsn-task-market/` - Lane 1 task marketplace
  - `pallets/nsn-model-registry/` - AI model capability registry

**node-core/**
- Purpose: Off-chain P2P node and compute orchestration
- Contains: Rust crates for networking, scheduling, lane orchestration
- Key files: `Cargo.toml` (workspace), `bin/nsn-node/src/main.rs`
- Subdirectories:
  - `crates/primitives/` - Shared primitive types
  - `crates/types/` - Domain types and structures
  - `crates/chain-client/` - Substrate client via subxt
  - `crates/p2p/` - libp2p networking with GossipSub
  - `crates/scheduler/` - Task scheduler with On-Deck protocol
  - `crates/lane0/` - Lane 0 video orchestration
  - `crates/lane1/` - Lane 1 compute orchestration
  - `crates/storage/` - Storage backend
  - `crates/validator/` - Validation logic
  - `sidecar/` - Compute execution runtime

**vortex/**
- Purpose: GPU-resident AI video generation pipeline
- Contains: Python modules for model loading and inference
- Key files: `pyproject.toml`, `src/vortex/__init__.py`
- Subdirectories:
  - `src/vortex/models/` - Model loaders (flux, liveportrait, kokoro, clip)
  - `src/vortex/pipeline/` - Generation orchestration
  - `src/vortex/utils/` - VRAM management, utilities
  - `tests/unit/` - Unit tests
  - `tests/integration/` - Integration tests

**viewer/**
- Purpose: Desktop streaming client for end users
- Contains: React UI components, Tauri backend, services
- Key files: `package.json`, `vite.config.ts`, `src/main.tsx`
- Subdirectories:
  - `src/components/` - React UI components
  - `src/hooks/` - Custom React hooks
  - `src/services/` - Business logic services
  - `src/store/` - Zustand state stores
  - `src-tauri/` - Tauri Rust backend
  - `e2e/` - Playwright E2E tests

**docker/**
- Purpose: Container configurations for development and deployment
- Contains: Dockerfiles, compose fragments
- Key files: `Dockerfile.*`, service-specific configs

**.tasks/**
- Purpose: Task management and tracking system
- Contains: Task specifications, manifest, updates
- Key files: `manifest.json` (task inventory), `tasks/*.md`

## Key File Locations

**Entry Points:**
- `nsn-chain/node/src/main.rs` - Blockchain node entry
- `node-core/bin/nsn-node/src/main.rs` - Off-chain node entry
- `vortex/src/vortex/__init__.py` - Python package entry
- `viewer/src/main.tsx` - React app entry
- `viewer/src-tauri/src/main.rs` - Tauri backend entry

**Configuration:**
- `nsn-chain/Cargo.toml` - On-chain workspace
- `node-core/Cargo.toml` - Off-chain workspace
- `vortex/pyproject.toml` - Python project config
- `viewer/package.json` - TypeScript dependencies
- `viewer/tsconfig.json` - TypeScript config
- `viewer/vite.config.ts` - Build config
- `.env.example` - Environment template
- `docker-compose.yml` - Service orchestration

**Core Logic:**
- `nsn-chain/pallets/*/src/lib.rs` - Pallet implementations
- `nsn-chain/runtime/src/lib.rs` - Runtime configuration
- `node-core/crates/*/src/lib.rs` - Crate implementations
- `vortex/src/vortex/pipeline/` - AI pipeline logic
- `viewer/src/services/` - Client business logic

**Testing:**
- `nsn-chain/pallets/*/src/tests.rs` - Pallet tests
- `node-core/crates/*/src/tests/` - Rust crate tests
- `vortex/tests/` - Python tests
- `viewer/src/**/*.test.ts` - Component tests (co-located)
- `viewer/e2e/` - Playwright E2E tests

**Documentation:**
- `CLAUDE.md` - Claude Code instructions
- `docs/` - Project documentation
- `.claude/rules/` - Development rules and architecture docs

## Naming Conventions

**Files:**
- `snake_case.rs` - Rust source files
- `snake_case.py` - Python source files
- `kebab-case.tsx` - React components
- `kebab-case.ts` - TypeScript modules
- `*.test.ts` - TypeScript test files (co-located)
- `test_*.py` or `*_test.py` - Python test files

**Directories:**
- `snake_case/` - Rust crates and modules
- `kebab-case/` - TypeScript/React directories
- Plural for collections: `pallets/`, `crates/`, `components/`

**Special Patterns:**
- `lib.rs` - Crate/pallet entry point
- `mod.rs` - Rust module directory marker
- `__init__.py` - Python package marker
- `index.ts` - TypeScript barrel exports
- `src/` - Source code directory

## Where to Add New Code

**New Pallet:**
- Implementation: `nsn-chain/pallets/{pallet-name}/`
- Register in: `nsn-chain/Cargo.toml` workspace members
- Wire up in: `nsn-chain/runtime/src/lib.rs`
- Tests: `nsn-chain/pallets/{pallet-name}/src/tests.rs`

**New Off-Chain Crate:**
- Implementation: `node-core/crates/{crate-name}/`
- Register in: `node-core/Cargo.toml` workspace members
- Tests: `node-core/crates/{crate-name}/src/tests/`

**New AI Model/Pipeline:**
- Model loader: `vortex/src/vortex/models/{model_name}.py`
- Pipeline: `vortex/src/vortex/pipeline/{pipeline_name}.py`
- Tests: `vortex/tests/unit/test_{name}.py`

**New UI Component:**
- Component: `viewer/src/components/{ComponentName}.tsx`
- Tests: `viewer/src/components/{ComponentName}.test.tsx`
- Hooks: `viewer/src/hooks/use{HookName}.ts`

**New Service:**
- Viewer: `viewer/src/services/{service-name}.ts`
- Tests: `viewer/src/services/{service-name}.test.ts`

## Special Directories

**.tasks/**
- Purpose: Task management system for project tracking
- Source: Maintained by task management agents
- Committed: Yes (tracks project progress)

**.planning/**
- Purpose: GSD (Get Shit Done) planning documents
- Source: Generated by planning workflows
- Committed: Yes (tracks project plans)

**.claude/**
- Purpose: Claude Code configuration and rules
- Source: Project-specific Claude instructions
- Committed: Yes (project guidance)

**target/** (gitignored)
- Purpose: Rust build artifacts
- Source: Generated by cargo build
- Committed: No

**.venv/, .venv311/** (gitignored)
- Purpose: Python virtual environments
- Source: Created by venv
- Committed: No

---

*Structure analysis: 2026-01-08*
*Update when directory structure changes*
