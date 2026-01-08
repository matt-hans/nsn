# External Integrations

**Analysis Date:** 2026-01-08

## APIs & External Services

**Blockchain RPC:**
- NSN Chain JSON-RPC - On-chain data and transaction submission
  - Client: subxt 0.34 in `node-core/crates/chain-client/`
  - Auth: None (public RPC) or API key for hosted nodes
  - Endpoints: Standard Substrate JSON-RPC, custom NSN RPC methods

**AI Model Registries:**
- Hugging Face Hub - Model downloads for Flux, LivePortrait, CLIP
  - Client: transformers, diffusers Python packages
  - Auth: `HF_TOKEN` env var for gated models
  - Cache: `HF_HOME` directory for model weights

**No External Payment Processing** - NSN token native to chain

**No External Email/SMS** - Not applicable for decentralized system

## Data Storage

**Blockchain Storage (nsn-chain):**
- Type: Substrate storage (RocksDB backend)
- Connection: Local node or RPC endpoint
- Client: subxt for off-chain, FRAME storage for on-chain
- Location: `--base-path` flag on node startup

**P2P State:**
- Type: In-memory with libp2p peer store
- Connection: Kademlia DHT for peer discovery
- Client: libp2p 0.53
- Persistence: Bootstrap nodes, peer exchange

**File Storage:**
- Local: Video chunks stored temporarily during processing
- No external cloud storage (fully decentralized)
- Location: Configured via environment variables

**Caching:**
- GPU Model Cache: `vortex/` models cached in memory/VRAM
- No external Redis/Memcached

## Authentication & Identity

**Chain Identity:**
- Type: Substrate cryptographic accounts (sr25519, ed25519)
- Implementation: sp-core keyring
- Key storage: Substrate keystore or external signer

**P2P Identity:**
- Type: libp2p PeerId (Ed25519 keypair)
- Implementation: libp2p identity module
- Persistence: Key file on disk

**Viewer Auth:**
- Type: Wallet connection (future)
- Current: Local development mode only
- Token storage: Not yet implemented

## Monitoring & Observability

**Metrics:**
- Prometheus - Metrics collection and exposition
  - Rust: `prometheus-endpoint` (Substrate ecosystem)
  - Python: `prometheus-client`
  - Endpoints: `/metrics` on configured port
  - Key metrics: VRAM usage, inference latency, P2P peers, block height

**Logging:**
- Rust: `tracing` with `tracing-subscriber`
  - Output: stdout/stderr, optionally to file
  - Filter: `RUST_LOG` environment variable
- Python: Standard logging module
  - Output: stdout/stderr
  - Filter: `LOG_LEVEL` environment variable

**Error Tracking:**
- None (logs only, no Sentry integration)
- Consider: Sentry integration for production

**Analytics:**
- None (decentralized, no centralized analytics)

## CI/CD & Deployment

**CI Pipeline:**
- GitHub Actions
  - Workflows: `.github/workflows/`
  - Tests: Rust clippy/test, Python ruff/pytest, TypeScript lint/test
  - Secrets: Stored in GitHub repository secrets

**Container Registry:**
- Docker Hub or GitHub Container Registry (for Docker images)
- Images: nsn-node, vortex (if containerized)

**Hosting:**
- Self-hosted: Nodes run by validators, directors, storage providers
- No centralized hosting platform
- Docker containers for deployment

**Deployment:**
- Manual: Binary releases, Docker images
- No automatic deployment pipeline yet

## Environment Configuration

**Development:**
- Required env vars:
  - `RUST_LOG` - Logging filter
  - `DATABASE_URL` - SQLite/Postgres for dev tooling (if used)
  - `HF_TOKEN` - Hugging Face token for gated models
- Secrets location: `.env.local` (gitignored)
- Mock services: Local chain with `--dev` flag

**Staging:**
- NSN Testnet for integration testing
- Separate node configuration
- Test tokens from faucet

**Production:**
- Secrets management: Environment variables, secure key storage
- Key management: Hardware security modules recommended
- Node operation: Dedicated servers with GPU (Lane 0)

## Network Communication

**P2P Protocol:**
- libp2p with multiple transports:
  - TCP for reliable connections
  - QUIC for performance
  - Noise protocol for encryption
- GossipSub 1.1 for topic-based messaging
- Kademlia DHT for peer discovery

**Topics:**
- Epoch announcements
- Task broadcasts (Lane 1)
- Video chunks (Lane 0)
- Validator attestations

**Substrate Networking:**
- Grandpa finality
- Block propagation
- Transaction gossip

## gRPC Services

**Sidecar Communication:**
- Service: tonic 0.10 gRPC
  - Location: `node-core/sidecar/`
  - Proto files: Define task execution interface
  - Auth: Local only (same machine)

**Internal Services:**
- Scheduler <-> Sidecar: Task dispatch and results
- Lane0 orchestrator <-> Vortex: Generation requests

## Webhooks & Callbacks

**Incoming:**
- None (decentralized, no external webhooks)

**Outgoing:**
- None

## GPU/Hardware Integration

**CUDA:**
- Required: CUDA 11.8+ for PyTorch
- Detection: pynvml for VRAM monitoring
- Config: `CUDA_VISIBLE_DEVICES` for GPU selection

**VRAM Management:**
- Vortex utils: `vortex/src/vortex/utils/vram.py`
- Monitoring: Prometheus metrics for VRAM usage
- Target: 11.8GB for full Lane 0 pipeline

---

*Integration audit: 2026-01-08*
*Update when adding/removing external services*
