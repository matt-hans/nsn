# Technology Stack

**Analysis Date:** 2026-01-08

## Languages

**Primary:**
- Rust 2021 Edition - nsn-chain (Polkadot SDK blockchain) and node-core (P2P/scheduler)
- Python 3.11+ - vortex (AI/ML video generation pipeline)
- TypeScript 5.6+ - viewer (Tauri desktop client)

**Secondary:**
- JavaScript - Build scripts, configuration files
- TOML - Rust workspace and crate configuration
- YAML - Docker, CI/CD configuration

## Runtime

**Rust Components:**
- Rust stable (2021 edition)
- Tokio 1.35 async runtime
- WASM target for nsn-chain runtime

**Python Components:**
- Python 3.11+ with virtual environment
- PyTorch 2.1+ (CUDA-enabled for GPU)
- Package Manager: pip with pyproject.toml (hatchling build)

**TypeScript Components:**
- Node.js 20.x (LTS)
- Package Manager: pnpm 8.0+
- Lockfile: pnpm-lock.yaml

## Frameworks

**Core:**
- Polkadot SDK (polkadot-sdk 2512.0.0) - NSN blockchain runtime and pallets
- FRAME - Substrate pallet framework for on-chain logic
- libp2p 0.53 - P2P networking with GossipSub, Kademlia DHT
- Tauri 2.0 - Desktop application framework
- React 18.3 - Frontend UI framework

**AI/ML:**
- PyTorch 2.1+ - Deep learning framework
- Diffusers 0.25+ - Stable diffusion pipelines (Flux-Schnell)
- Transformers 4.36+ - Model loading and inference
- open-clip-torch 2.23+ - CLIP ensemble for semantic verification
- Kokoro 0.7+ - Text-to-speech synthesis

**Testing:**
- Vitest 4.0 - TypeScript unit testing
- Playwright 1.49 - E2E testing
- pytest 7.4+ - Python testing
- Rust built-in test framework

**Build/Dev:**
- Vite 6.0 - TypeScript bundling and dev server
- Biome 1.9 - TypeScript linting and formatting
- Ruff 0.1+ - Python linting
- Cargo - Rust build system
- substrate-wasm-builder - WASM runtime compilation

## Key Dependencies

**Critical (Rust):**
- subxt 0.34 - Substrate client for chain interaction
- parity-scale-codec 3.7 - SCALE encoding/decoding
- sp-core, sp-runtime - Substrate primitives
- tonic 0.10 - gRPC framework
- blake3 - Fast cryptographic hashing

**Critical (Python):**
- torch 2.1+ - Neural network operations
- safetensors 0.4+ - Safe model serialization
- diffusers 0.25+ - Image generation pipelines
- accelerate 0.25+ - Model optimization and distribution
- bitsandbytes 0.41+ - NF4 quantization for VRAM efficiency

**Critical (TypeScript):**
- @tauri-apps/api 2.0 - Tauri IPC and system access
- zustand 4.5 - State management
- @testing-library/react 16.0 - React testing utilities

**Infrastructure:**
- prometheus-client - Metrics collection
- tracing - Structured logging (Rust)
- clap 4.4/4.5 - CLI argument parsing
- serde/serde_json - JSON serialization

## Configuration

**Environment:**
- `.env` files for sensitive configuration (gitignored)
- `.env.example` provides template with required variables
- Rust: config crate with TOML/JSON support
- Python: yaml configuration files

**Build:**
- `nsn-chain/Cargo.toml` - Workspace configuration
- `node-core/Cargo.toml` - Off-chain workspace
- `vortex/pyproject.toml` - Python project config
- `viewer/package.json` - TypeScript dependencies
- `viewer/vite.config.ts` - Build configuration
- `viewer/tsconfig.json` - TypeScript compiler options

## Platform Requirements

**Development:**
- macOS/Linux (Windows with WSL2 for Rust/Python)
- CUDA-capable GPU for vortex (RTX 3060 12GB minimum for Lane 0)
- Docker for local development services
- Rust toolchain with wasm32-unknown-unknown target

**Production:**
- Linux server (Ubuntu 22.04+ recommended)
- NVIDIA GPU with CUDA 11.8+ for AI inference
- 32GB+ RAM for full node operation
- SSD storage for blockchain data

**Deployment:**
- Docker containers for all components
- docker-compose.yml for local orchestration
- Substrate-compatible hosting for nsn-chain nodes

---

*Stack analysis: 2026-01-08*
*Update after major dependency changes*
