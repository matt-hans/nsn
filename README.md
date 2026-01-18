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

### Testnet WebRTC Setup (Viewer Connectivity)

For browser connections to the mesh, each node must advertise a routable WebRTC address and keep its certificate stable.

```bash
nsn-node \
  --p2p-enable-webrtc \
  --p2p-webrtc-port 9003 \
  --p2p-external-address /ip4/192.168.1.10/udp/9003/webrtc-direct \
  --data-dir /var/lib/nsn
```

Notes:
- Use the node's LAN/public IP in `--p2p-external-address` (do not use `0.0.0.0`).
- Keep `--data-dir` stable so the WebRTC certificate (certhash) does not change between restarts.

---

## Video Generation Pipeline

**This is NSN's core innovation**: a decentralized, deterministic video generation pipeline with Byzantine fault-tolerant consensus. Unlike traditional AI services, NSN guarantees that multiple independent nodes produce identical outputs from the same inputs, enabling trustless verification without centralized control.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NSN Video Generation Flow (45-second slot)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. RECIPE SUBMISSION                                                      │
│      User submits generation recipe with script, voice, and visual prompt   │
│                              │                                              │
│                              ▼                                              │
│   2. PARALLEL GENERATION (0-12s)                                            │
│      ┌─────────────────┬─────────────────┐                                  │
│      │  Kokoro TTS     │  Flux-Schnell   │                                  │
│      │  "Hello world"  │  "scientist"    │                                  │
│      │     ↓           │       ↓         │                                  │
│      │  Audio.wav      │  Actor.png      │                                  │
│      └────────┬────────┴────────┬────────┘                                  │
│               │                 │                                           │
│               └────────┬────────┘                                           │
│                        ▼                                                    │
│   3. VIDEO SYNTHESIS (12-15s)                                               │
│      LivePortrait warps actor image to match audio lip-sync                 │
│                        │                                                    │
│                        ▼                                                    │
│   4. SEMANTIC VERIFICATION (15-17s)                                         │
│      Dual CLIP ensemble computes 512-dim embedding + self-check             │
│                        │                                                    │
│                        ▼                                                    │
│   5. BFT CONSENSUS (17-30s)                                                 │
│      5 validators compare embeddings, 3-of-5 must match (cosine > 0.99)     │
│                        │                                                    │
│                        ▼                                                    │
│   6. ON-CHAIN FINALIZATION                                                  │
│      Determinism proof + CLIP embedding stored on-chain                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Deterministic Generation Matters

Traditional AI models are non-deterministic: the same prompt produces different outputs each time. NSN solves this through:

1. **Seeded Random State**: All random number generators use deterministic seeds derived from on-chain entropy
2. **Fixed Precision**: Models run in consistent precision (FP16/FP32) to avoid floating-point divergence
3. **GPU-Resident Models**: All 5 models stay loaded in VRAM, eliminating initialization variance
4. **Cryptographic Proofs**: Each generation produces a SHA-256 hash proving deterministic execution

This enables **trustless verification**: any validator can reproduce the exact same video from the same recipe+seed, then compare CLIP embeddings to detect tampering.

### Starting the Vortex Server

The Vortex server provides an HTTP API for video generation:

```bash
# Navigate to vortex directory
cd vortex
source .venv/bin/activate

# Start server (models load lazily on first request)
python -m vortex.server \
    --host 0.0.0.0 \
    --port 50051 \
    --device cuda:0 \
    --models-path /path/to/models \
    --output-path /path/to/output

# Or with eager model loading (recommended for production)
python -m vortex.server \
    --host 0.0.0.0 \
    --port 50051 \
    --device cuda:0 \
    --eager-init
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate video from recipe |
| `/health` | GET | Server health and pipeline status |
| `/vram` | GET | GPU memory usage statistics |
| `/metrics` | GET | Prometheus metrics for monitoring |

### Generating Video

#### Recipe Format

```json
{
  "recipe": {
    "slot_params": {
      "slot_id": 12345,
      "duration_sec": 45,
      "fps": 24
    },
    "audio_track": {
      "script": "Welcome to the Neural Sovereign Network!",
      "voice_id": "rick_c137",
      "speed": 1.0,
      "emotion": "excited"
    },
    "visual_track": {
      "prompt": "a scientist in a futuristic laboratory explaining quantum mechanics",
      "negative_prompt": "blurry, low quality, watermark, text"
    },
    "semantic_constraints": {
      "clip_threshold": 0.70
    }
  },
  "slot_id": 12345,
  "seed": 42
}
```

#### Recipe Fields

| Field | Type | Description |
|-------|------|-------------|
| `slot_params.slot_id` | int | Unique slot identifier (on-chain reference) |
| `slot_params.duration_sec` | int | Video duration (max 45 seconds) |
| `slot_params.fps` | int | Frame rate (default: 24) |
| `audio_track.script` | string | Text to convert to speech |
| `audio_track.voice_id` | string | Voice character: `rick_c137`, `morty`, `summer`, `jerry`, `beth` |
| `audio_track.speed` | float | Speech rate: 0.8-1.2 (default: 1.0) |
| `audio_track.emotion` | string | Emotion: `neutral`, `excited`, `sad`, `angry`, `manic` |
| `visual_track.prompt` | string | Text description of the actor/scene |
| `visual_track.negative_prompt` | string | Elements to avoid in generation |
| `semantic_constraints.clip_threshold` | float | Minimum CLIP similarity score (0.0-1.0) |
| `seed` | int | Deterministic seed for reproducibility |

#### Example: Generate Video via cURL

```bash
# Generate a 45-second video slot
curl -X POST http://localhost:50051/generate \
    -H "Content-Type: application/json" \
    -d '{
        "recipe": {
            "slot_params": {"slot_id": 1001, "duration_sec": 45, "fps": 24},
            "audio_track": {
                "script": "Morty, we need to go on an adventure!",
                "voice_id": "rick_c137",
                "emotion": "excited"
            },
            "visual_track": {
                "prompt": "a mad scientist with wild grey hair in a garage laboratory",
                "negative_prompt": "blurry, cartoon, anime"
            },
            "semantic_constraints": {"clip_threshold": 0.70}
        },
        "slot_id": 1001,
        "seed": 42
    }'
```

#### Response Format

```json
{
  "success": true,
  "slot_id": 1001,
  "video_path": "/output/slot_1001_video.npy",
  "audio_path": "/output/slot_1001_audio.npy",
  "video_shape": [1080, 3, 512, 512],
  "audio_samples": 1080000,
  "clip_embedding_shape": [512],
  "determinism_proof": "a3f2b1c4d5e6f7...",
  "generation_time_ms": 15234.5
}
```

#### Understanding the Output

| Output | Format | Description |
|--------|--------|-------------|
| `video_path` | NumPy `.npy` | Video frames `[T, C, H, W]` - 1080 frames × 3 channels × 512×512 |
| `audio_path` | NumPy `.npy` | Audio waveform at 24kHz mono (1,080,000 samples for 45s) |
| `clip_embedding_shape` | 512-dim vector | Semantic embedding for BFT consensus comparison |
| `determinism_proof` | SHA-256 hex | Cryptographic proof of deterministic execution |

#### Loading Generated Output

```python
import numpy as np
import torch

# Load video frames
video = np.load("/output/slot_1001_video.npy")
print(f"Video shape: {video.shape}")  # (1080, 3, 512, 512)

# Load audio
audio = np.load("/output/slot_1001_audio.npy")
print(f"Audio samples: {len(audio)}")  # 1080000 (45s @ 24kHz)

# Convert to playable format
from scipy.io.wavfile import write
write("/output/slot_1001.wav", 24000, (audio * 32767).astype(np.int16))

# Convert video to MP4 (requires ffmpeg)
import subprocess
# Reshape to (T, H, W, C) for video encoding
video_hwc = np.transpose(video, (0, 2, 3, 1))
# ... use imageio or cv2 to save as MP4
```

### Monitoring Generation

#### Check Server Health

```bash
curl http://localhost:50051/health | jq
```

```json
{
  "status": "healthy",
  "device": "cuda:0",
  "initialized": true,
  "renderer": "default",
  "renderer_version": "0.1.0"
}
```

#### Monitor VRAM Usage

```bash
curl http://localhost:50051/vram | jq
```

```json
{
  "available": true,
  "device": "cuda:0",
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "total_gb": 24.0,
  "allocated_gb": 22.5,
  "reserved_gb": 23.1,
  "free_gb": 0.9
}
```

#### Prometheus Metrics

```bash
curl http://localhost:50051/metrics
```

```
# HELP nvidia_gpu_memory_used_bytes GPU memory currently used in bytes
# TYPE nvidia_gpu_memory_used_bytes gauge
nvidia_gpu_memory_used_bytes{gpu="0",name="NVIDIA GeForce RTX 3090"} 24159191040
nvidia_gpu_temperature_celsius{gpu="0",name="NVIDIA GeForce RTX 3090"} 45
vortex_pipeline_initialized{device="cuda:0"} 1
```

### Network Integration

In production, video generation is orchestrated by the NSN network:

1. **Epoch Election**: 5 Directors are elected from the On-Deck set each epoch (100 blocks)
2. **Recipe Distribution**: Director broadcasts recipe to elected validators via P2P
3. **Parallel Generation**: All 5 validators generate video independently with same seed
4. **Embedding Exchange**: Validators exchange 512-dim CLIP embeddings via GossipSub
5. **BFT Consensus**: 3-of-5 matching embeddings (cosine similarity > 0.99) required
6. **On-Chain Proof**: Determinism proof and consensus result stored on NSN chain
7. **P2P Streaming**: Finalized video distributed to viewers via mesh network

This ensures no single node can manipulate video content - Byzantine fault tolerance guarantees integrity even if 2-of-5 validators are malicious.

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
| Vortex | 50051 | GPU and generation metrics (via `/metrics`) |

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
