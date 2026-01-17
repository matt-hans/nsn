# Neural Sovereign Network (NSN)

> **Decentralized AI Compute Marketplace** built on Polkadot SDK

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![Rust](https://img.shields.io/badge/rust-stable--2024--09--05-orange.svg)](https://www.rust-lang.org/)
[![Polkadot SDK](https://img.shields.io/badge/polkadot--sdk-stable2409-E6007A.svg)](https://github.com/paritytech/polkadot-sdk)

---

## Important Notice

> **This project is currently under active development and is NOT ready for production use.**
>
> - All code, APIs, and protocols are subject to breaking changes without notice
> - Security audits have not been completed
> - Testnet tokens have no monetary value
> - Use at your own risk - data loss may occur
>
> For production deployment timelines, see [Roadmap](#roadmap).

---

## Overview

Neural Sovereign Network (NSN) is a decentralized AI compute marketplace that combines blockchain consensus with GPU-accelerated AI inference. The network operates as a Polkadot SDK-based chain with **dual-lane architecture**:

| Lane | Purpose | Mechanism |
|------|---------|-----------|
| **Lane 0** | AI-powered video generation | Epoch-based elections, BFT consensus (3-of-5 threshold) |
| **Lane 1** | General AI compute marketplace | Open task marketplace, reputation-driven matching |

### Key Features

- **Decentralized AI Inference**: GPU-resident models with deterministic generation
- **BFT Consensus**: Byzantine fault-tolerant validation of AI outputs
- **Dual CLIP Verification**: Semantic verification using ensemble CLIP embeddings
- **Epoch-Based Elections**: On-Deck protocol for fair director selection
- **Stake-Weighted Reputation**: Anti-centralization with regional caps and delegation limits
- **P2P Mesh Network**: libp2p-based peer discovery with GossipSub messaging

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NSN Network Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         On-Chain Layer                               │   │
│  │                                                                      │   │
│  │   NSN Chain (Polkadot SDK)                                          │   │
│  │   ├── pallet-nsn-stake      → Token staking, slashing               │   │
│  │   ├── pallet-nsn-reputation → Reputation scoring, Merkle proofs     │   │
│  │   ├── pallet-nsn-director   → Epoch elections, On-Deck protocol     │   │
│  │   ├── pallet-nsn-bft        → BFT consensus storage                 │   │
│  │   ├── pallet-nsn-task-market→ Lane 1 task marketplace              │   │
│  │   └── pallet-nsn-treasury   → Reward distribution                   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Off-Chain Layer                               │   │
│  │                                                                      │   │
│  │   node-core (Rust)                                                   │   │
│  │   ├── Director Node  → Epoch coordination, BFT leadership           │   │
│  │   ├── Validator Node → Vote on AI outputs, earn reputation          │   │
│  │   ├── Scheduler      → Task routing, On-Deck management             │   │
│  │   └── Sidecar        → Container execution, GPU orchestration       │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          AI Layer                                    │   │
│  │                                                                      │   │
│  │   Vortex (Python + CUDA)                    VRAM: ~11.8 GB          │   │
│  │   ├── Flux-Schnell (NF4)   → Actor generation       ~6.0 GB         │   │
│  │   ├── LivePortrait (FP16)  → Video warping          ~3.5 GB         │   │
│  │   ├── Kokoro-82M (FP32)    → Text-to-speech         ~0.4 GB         │   │
│  │   └── Dual CLIP (INT8)     → Semantic verification  ~0.9 GB         │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Ubuntu 22.04+ / macOS 13+ | Ubuntu 24.04 LTS |
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 100 GB SSD | 500 GB NVMe |
| **GPU** | RTX 3060 12GB | RTX 4080 16GB |

### Software Dependencies

```bash
# Rust (stable-2024-09-05 via rust-toolchain.toml)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Python 3.11+ (for Vortex)
python3 --version  # Should be 3.11+

# Docker & Docker Compose
docker --version   # Should be 24.0+
docker compose version

# NVIDIA Container Toolkit (for GPU support)
nvidia-smi         # Verify GPU is accessible

# Zombienet (for testnet orchestration)
curl -L https://github.com/paritytech/zombienet/releases/latest/download/zombienet-linux-x64 \
  -o ~/.local/bin/zombienet
chmod +x ~/.local/bin/zombienet

# Polkadot binary (for relay chain)
curl -L https://github.com/paritytech/polkadot-sdk/releases/download/polkadot-stable2409/polkadot \
  -o ~/.local/bin/polkadot
chmod +x ~/.local/bin/polkadot
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/neural-sovereign-network/nsn.git
cd nsn
```

### 2. Build All Components

```bash
# Build NSN Chain (on-chain layer)
cd nsn-chain
cargo build --release

# Build node-core (off-chain layer)
cd ../node-core
cargo build --release

# Install Vortex (AI layer) - requires GPU
cd ../vortex
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Start Observability Stack

```bash
# From repository root
docker compose up -d prometheus grafana jaeger
```

### 4. Launch Testnet

```bash
# Start zombienet (relay chain + NSN parachain)
cd nsn-chain
zombienet -p native spawn zombienet.toml
```

### 5. Start Off-Chain Nodes

```bash
# In a new terminal - start director node
cd node-core
./target/release/nsn-node director-only \
    --rpc-url=ws://127.0.0.1:9944 \
    --p2p-listen-port=9001 \
    --p2p-metrics-port=9101

# In additional terminals - start validator nodes
./target/release/nsn-node validator-only \
    --rpc-url=ws://127.0.0.1:9944 \
    --p2p-listen-port=9002 \
    --p2p-metrics-port=9102
```

### 6. Verify Deployment

```bash
# Check chain is producing blocks
curl -s -H "Content-Type: application/json" \
    -d '{"id":1,"jsonrpc":"2.0","method":"chain_getHeader"}' \
    http://127.0.0.1:9944 | jq '.result.number'

# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | \
    jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Access dashboards
# Grafana: http://localhost:3001 (admin/admin)
# Jaeger:  http://localhost:16686
```

---

## Repository Structure

```
nsn/
├── nsn-chain/               # On-chain layer (Polkadot SDK)
│   ├── pallets/             # NSN custom FRAME pallets
│   │   ├── nsn-stake/       # Token staking, slashing
│   │   ├── nsn-reputation/  # Reputation scoring
│   │   ├── nsn-director/    # Epoch elections
│   │   ├── nsn-bft/         # BFT consensus storage
│   │   ├── nsn-storage/     # Erasure coding deals
│   │   ├── nsn-treasury/    # Reward distribution
│   │   ├── nsn-task-market/ # Lane 1 marketplace
│   │   └── nsn-model-registry/ # Model capabilities
│   ├── runtime/             # NSN runtime configuration
│   ├── node/                # NSN node client
│   └── zombienet.toml       # Testnet configuration
│
├── node-core/               # Off-chain layer (Rust)
│   ├── bin/nsn-node/        # Unified node binary
│   ├── crates/p2p/          # libp2p networking
│   ├── crates/scheduler/    # Task scheduler
│   ├── crates/lane0/        # Lane 0 orchestration
│   ├── crates/lane1/        # Lane 1 orchestration
│   └── sidecar/             # Container execution
│
├── vortex/                  # AI layer (Python + CUDA)
│   ├── src/vortex/
│   │   ├── models/          # Model loaders
│   │   ├── pipeline/        # Generation orchestration
│   │   └── utils/           # VRAM management
│   └── config.yaml          # Pipeline configuration
│
├── viewer/                  # Desktop app (Tauri + React)
│   ├── src/                 # React frontend
│   └── src-tauri/           # Tauri backend
│
├── docker/                  # Docker configuration
│   ├── prometheus.yml       # Prometheus scrape config
│   └── grafana/             # Grafana dashboards
│
├── docker-compose.yml       # Local development stack
└── CLAUDE.md                # AI assistant guidelines
```

---

## Development Commands

### NSN Chain (On-Chain)

```bash
cd nsn-chain

# Build
cargo build --release

# Run tests
cargo test --all

# Lint
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt -- --check

# Run single dev node
./target/release/nsn-node --dev
```

### Node Core (Off-Chain)

```bash
cd node-core

# Build
cargo build --release

# Run tests
cargo test --all

# Director mode
./target/release/nsn-node director-only --help

# Validator mode
./target/release/nsn-node validator-only --help
```

### Vortex (AI Layer)

```bash
cd vortex
source .venv/bin/activate

# Run tests
pytest tests/unit/ -v
pytest tests/integration/ -v

# Lint
ruff check src/

# Download models
python scripts/download_flux.py
python scripts/download_kokoro.py --test-synthesis
```

### Docker Stack

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f [service-name]

# Stop all services
docker compose down

# Clean up volumes
docker compose down -v
docker system prune -a
```

---

## Testing

### Unit Tests

```bash
# All Rust tests
cargo test --all

# Specific pallet
cargo test -p pallet-nsn-stake

# Vortex unit tests
cd vortex && pytest tests/unit/ -v
```

### Integration Tests

```bash
# NSN Chain integration tests
cd nsn-chain/test
pnpm install
pnpm test

# Vortex integration tests (requires GPU)
cd vortex && pytest tests/integration/ -v
```

### Coverage

```bash
# Rust coverage
cargo tarpaulin --workspace --out Html

# Python coverage
pytest tests/ --cov=vortex --cov-report=html
```

---

## Observability

### Prometheus Metrics

| Endpoint | Port | Description |
|----------|------|-------------|
| NSN Chain | 9615 | Substrate metrics |
| Director | 9101 | P2P and BFT metrics |
| Validators | 9102-9104 | Validator node metrics |
| Vortex | 9100 | GPU and generation metrics |

Access Prometheus UI: http://localhost:9090

### Grafana Dashboards

Pre-configured dashboards available at http://localhost:3001:

- **NSN Overview**: Block height, peer count, VRAM usage
- **P2P Network**: Message rates, peer connections
- **AI Pipeline**: Generation latency, CLIP verification scores

Default credentials: `admin` / `admin`

### Jaeger Tracing

Distributed tracing for debugging request flows: http://localhost:16686

---

## Network Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Block time | 6 seconds | Target block production interval |
| Epoch duration | 100 blocks | ~10 minutes between elections |
| Directors per epoch | 5 | Elected from On-Deck set |
| BFT threshold | 3-of-5 | Required signatures for consensus |
| Challenge period | 50 blocks | ~5 minutes for disputes |
| Min director stake | 100 NSN | Minimum to enter On-Deck |
| Max stake per node | 1,000 NSN | Anti-whale protection |
| Max region share | 20% | Geographic decentralization |

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase A: NSN Solochain** | In Progress | Local testnet, core pallets, MVP |
| **Phase B: NSN Mainnet** | Planned | Public validators, security audit |
| **Phase C: Parachain** | Future | Polkadot shared security, XCM |
| **Phase D: Coretime** | Future | Elastic scaling via coretime |

---

## Contributing

We welcome contributions! Please read our contributing guidelines before submitting PRs.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`cargo test --all`)
5. Run linting (`cargo clippy && cargo fmt`)
6. Commit with conventional commits (`git commit -m 'feat: add amazing feature'`)
7. Push to your fork (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Standards

- All code must pass CI checks (tests, clippy, fmt)
- Coverage target: >85% for new code
- Security-sensitive changes require review
- Document all public APIs

---

## Security

### Reporting Vulnerabilities

For security issues, please email security@nsn.network rather than opening a public issue.

### Known Limitations

- Development mode exposes unsafe RPC methods
- Default credentials used in local environment
- No TLS in local development stack
- Security audit pending

---

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## Resources

- [Technical Architecture Document](.claude/rules/architecture.md)
- [Product Requirements Document](.claude/rules/prd.md)
- [NSN Chain Documentation](nsn-chain/README.md)
- [Vortex Documentation](vortex/README.md)
- [Docker Configuration](docker/README.md)
- [Polkadot SDK Documentation](https://docs.substrate.io/)

---

## Contact

- **Website**: https://nsn.network (coming soon)
- **Email**: 
- **Discord**: (coming soon)
- **Twitter**: @NSN_Network (coming soon)

---

<p align="center">
  <strong>Neural Sovereign Network</strong><br>
  <em>Decentralized AI for a Sovereign Future</em>
</p>
