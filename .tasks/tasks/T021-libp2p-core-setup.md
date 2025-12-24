---
id: T021
title: libp2p Core Setup and Transport Layer
status: pending
priority: 1
agent: backend
dependencies: [T001]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [p2p, libp2p, networking, off-chain, phase1, critical-path]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - ../docs/prd.md (§17.1, §17.2, §4.4)
  - ../docs/architecture.md (§4.4, ADR-003)

est_tokens: 10000
actual_tokens: null
---

## Description

Implement the foundational libp2p networking layer for ICN off-chain nodes (Directors, Super-Nodes, Validators, Relays, Viewers). This establishes the core P2P infrastructure including transport, encryption, multiplexing, and connection management.

**Technical Approach:**
- Use rust-libp2p 0.53.0 as the P2P networking foundation
- Configure QUIC transport for UDP-based low-latency connections
- Implement Noise XX protocol for end-to-end encryption
- Use yamux for stream multiplexing over connections
- Generate Ed25519 keypair for PeerId (aligned with Substrate AccountId)
- Enforce connection limits and DoS protection

**Integration Points:**
- Substrate AccountId derives from the same Ed25519 keypair (cross-layer identity)
- GossipSub, Kademlia DHT, and other behaviors build on this foundation
- Chain client (subxt) uses this identity for on-chain transactions

## Business Context

**User Story:** As an ICN node operator, I want secure, efficient P2P communication, so that I can participate in BFT consensus, video distribution, and content verification without centralized infrastructure.

**Why This Matters:**
- Foundation for all off-chain coordination (BFT, video distribution, reputation sync)
- Enables permissionless participation (no central servers)
- Security model relies on Ed25519 cryptographic identity

**What It Unblocks:**
- T022 (GossipSub configuration)
- T023 (NAT traversal)
- T024 (Kademlia DHT)
- All off-chain node implementations (Directors, Super-Nodes, Viewers)

**Priority Justification:** Critical path for Phase 1. No off-chain functionality possible without P2P networking layer.

## Acceptance Criteria

- [ ] **Keypair Generation**: Ed25519 keypair generated and PeerId derived correctly
- [ ] **PeerId to AccountId**: PeerId can be converted to Substrate AccountId32 format
- [ ] **QUIC Transport**: QUIC transport configured and listening on configurable port (default 9000)
- [ ] **Noise Encryption**: Noise XX handshake completes successfully between peers
- [ ] **Yamux Multiplexing**: Multiple streams can be opened over a single connection
- [ ] **Connection Limits**: Maximum 256 total connections, maximum 2 connections per peer enforced
- [ ] **Connection Timeout**: Connections timeout after 30 seconds of inactivity
- [ ] **Graceful Shutdown**: P2P service shuts down cleanly, closing all connections
- [ ] **Error Handling**: Network errors are logged with context (peer, transport, error type)
- [ ] **Configuration**: Port, connection limits, timeout configurable via environment/config file
- [ ] **Integration Test**: Two nodes connect, authenticate, and maintain connection
- [ ] **Metrics Exposed**: Active connections, peer count, bytes sent/received on Prometheus endpoint (port 9100)

## Test Scenarios

**Test Case 1: Successful Peer Connection**
- Given: Two libp2p nodes (A and B) on the same network
- When: Node A dials Node B's multiaddr
- Then:
  - Noise XX handshake completes
  - Connection established
  - Both nodes see peer count = 1
  - Metrics show 1 active connection

**Test Case 2: Connection Limit Enforcement**
- Given: Node A with max_connections = 2
- When: 3 different peers attempt to connect to Node A
- Then:
  - First 2 connections succeed
  - Third connection is rejected
  - Error logged: "Connection limit reached (2/2)"

**Test Case 3: Per-Peer Connection Limit**
- Given: Node A with max_connections_per_peer = 2
- When: Peer B attempts to open 3 connections to Node A
- Then:
  - First 2 connections succeed
  - Third connection is rejected
  - Error logged: "Per-peer connection limit reached"

**Test Case 4: Connection Timeout**
- Given: Two connected peers (A and B)
- When: No traffic for 35 seconds (> 30s timeout)
- Then:
  - Connection is closed
  - Both nodes log "Connection timeout"
  - Peer count decrements to 0

**Test Case 5: PeerId to AccountId Conversion**
- Given: Ed25519 keypair with known public key
- When: PeerId is generated and converted to AccountId32
- Then:
  - AccountId32 matches expected Substrate format
  - Same keypair can sign Substrate extrinsics

**Test Case 6: Graceful Shutdown**
- Given: Node with 5 active peer connections
- When: SIGTERM signal received
- Then:
  - All connections closed gracefully
  - All streams flushed
  - Process exits within 5 seconds

**Test Case 7: QUIC Transport Performance**
- Given: Two nodes connected via QUIC
- When: 100 MB of data sent from A to B
- Then:
  - Transmission completes
  - Latency < 10ms per round-trip (local network)
  - No packet loss

**Test Case 8: Noise Encryption Verification**
- Given: Two nodes connected via Noise XX
- When: Intercepting network traffic between nodes
- Then:
  - All application data is encrypted
  - Only handshake messages are partially cleartext
  - Cannot decrypt stream content without keypair

## Technical Implementation

**Required Components:**

```
off-chain/
├── Cargo.toml
│   └── [dependencies]
│       ├── libp2p = { version = "0.53.0", features = ["quic", "noise", "yamux"] }
│       ├── libp2p-identity = "0.53.0"
│       ├── sp-core = "28.0.0"  # For AccountId32
│       ├── tokio = { version = "1.35", features = ["full"] }
│       ├── tracing = "0.1"
│       ├── serde = { version = "1.0", features = ["derive"] }
│       └── prometheus = "0.13"
│
├── src/
│   ├── p2p/
│   │   ├── mod.rs              # Public API
│   │   ├── config.rs           # P2P configuration struct
│   │   ├── identity.rs         # Ed25519 keypair generation, PeerId ↔ AccountId
│   │   ├── transport.rs        # QUIC + Noise + Yamux setup
│   │   ├── limits.rs           # Connection limit enforcement
│   │   ├── metrics.rs          # Prometheus metrics
│   │   └── tests.rs            # Integration tests
│   │
│   └── main.rs                 # Example node entrypoint
│
└── tests/
    └── integration_p2p.rs      # Multi-node integration tests
```

**Key Rust Modules:**

```rust
// src/p2p/config.rs
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2pConfig {
    pub listen_port: u16,
    pub max_connections: usize,
    pub max_connections_per_peer: usize,
    pub connection_timeout: Duration,
    pub keypair_path: Option<PathBuf>,
}

impl Default for P2pConfig {
    fn default() -> Self {
        Self {
            listen_port: 9000,
            max_connections: 256,
            max_connections_per_peer: 2,
            connection_timeout: Duration::from_secs(30),
            keypair_path: None,
        }
    }
}

// src/p2p/identity.rs
use libp2p_identity::{Keypair, PeerId};
use sp_core::crypto::AccountId32;

pub fn generate_keypair() -> Keypair {
    Keypair::generate_ed25519()
}

pub fn peer_id_to_account_id(peer_id: &PeerId) -> AccountId32 {
    let bytes = peer_id.to_bytes();
    AccountId32::from_slice(&bytes[..32]).expect("PeerId to AccountId32")
}

pub fn save_keypair(keypair: &Keypair, path: &Path) -> Result<(), Error> {
    // Save keypair to file (encrypted)
}

pub fn load_keypair(path: &Path) -> Result<Keypair, Error> {
    // Load keypair from file
}

// src/p2p/transport.rs
use libp2p::{
    quic, noise, yamux,
    core::upgrade,
    Transport,
};

pub fn build_transport(keypair: &Keypair) -> Result<Boxed<(PeerId, StreamMuxerBox)>, Error> {
    let quic_config = quic::Config::new(keypair);

    quic::tokio::Transport::new(quic_config)
        .upgrade(upgrade::Version::V1)
        .authenticate(noise::Config::new(keypair).expect("Noise config"))
        .multiplex(yamux::Config::default())
        .timeout(Duration::from_secs(30))
        .boxed()
}

// src/p2p/mod.rs
use libp2p::Swarm;

pub struct P2pService {
    swarm: Swarm<Behaviour>,
    config: P2pConfig,
    metrics: Arc<Metrics>,
}

impl P2pService {
    pub async fn new(config: P2pConfig) -> Result<Self, Error> {
        let keypair = if let Some(path) = &config.keypair_path {
            load_keypair(path)?
        } else {
            generate_keypair()
        };

        let peer_id = PeerId::from(keypair.public());
        tracing::info!("Local PeerId: {}", peer_id);

        let transport = build_transport(&keypair)?;

        let behaviour = Behaviour::new();

        let swarm = SwarmBuilder::with_tokio_executor(
            transport,
            behaviour,
            peer_id,
        ).build();

        Ok(Self {
            swarm,
            config,
            metrics: Arc::new(Metrics::new()),
        })
    }

    pub async fn start(&mut self) -> Result<(), Error> {
        let listen_addr = format!("/ip4/0.0.0.0/udp/{}/quic-v1", self.config.listen_port);
        self.swarm.listen_on(listen_addr.parse()?)?;

        loop {
            tokio::select! {
                event = self.swarm.select_next_some() => {
                    self.handle_event(event).await?;
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Shutting down...");
                    break;
                }
            }
        }

        Ok(())
    }

    async fn handle_event(&mut self, event: SwarmEvent) -> Result<(), Error> {
        match event {
            SwarmEvent::NewListenAddr { address, .. } => {
                tracing::info!("Listening on {}", address);
            }
            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                tracing::info!("Connected to {}", peer_id);
                self.metrics.connections.inc();
            }
            SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                tracing::info!("Disconnected from {}: {:?}", peer_id, cause);
                self.metrics.connections.dec();
            }
            _ => {}
        }
        Ok(())
    }
}
```

**Validation Commands:**

```bash
# Build P2P module
cargo build --release -p icn-off-chain

# Run unit tests
cargo test -p icn-off-chain p2p::

# Run integration tests (requires 2 nodes)
cargo test --test integration_p2p -- --nocapture

# Check dependencies
cargo audit
cargo deny check

# Start test node
RUST_LOG=info cargo run --release -- --port 9000

# Check metrics
curl http://localhost:9100/metrics | grep icn_p2p
```

**Code Patterns:**
- Use `tracing` for structured logging (JSON output)
- Prometheus metrics for observability (port 9100)
- Graceful shutdown on SIGTERM/SIGINT
- Configuration via environment variables + config file (TOML)

## Dependencies

**Hard Dependencies** (must be complete first):
- [T001] ICN Chain Fork and Development Environment - provides Rust toolchain and Substrate types

**Soft Dependencies:**
- None (foundation layer)

**External Dependencies:**
- rust-libp2p 0.53.0
- Substrate sp-core 28.0.0 (for AccountId32)
- QUIC library (via libp2p)

## Design Decisions

**Decision 1: QUIC over TCP**
- **Rationale:** QUIC provides faster connection establishment (0-RTT resumption), built-in encryption, better congestion control, and multiplexing without head-of-line blocking
- **Alternatives:**
  - TCP + TLS: More mature but slower handshake, head-of-line blocking
  - WebRTC: Browser compatibility but higher overhead
- **Trade-offs:** QUIC requires UDP, which may be blocked by some firewalls (mitigated by NAT traversal stack in T023)

**Decision 2: Noise XX Protocol**
- **Rationale:** Noise XX provides mutual authentication, forward secrecy, and is battle-tested in libp2p ecosystem
- **Alternatives:**
  - TLS 1.3: More standardized but heavier, requires certificate management
  - Custom encryption: Not recommended (cryptographic complexity)
- **Trade-offs:** Noise XX adds ~1 RTT to handshake vs plaintext, but this is acceptable for security

**Decision 3: Ed25519 for PeerId**
- **Rationale:** Ed25519 is Substrate's default signing algorithm, enabling cross-layer identity (same keypair for P2P and on-chain transactions)
- **Alternatives:**
  - secp256k1: EVM compatibility but not Substrate's default
  - RSA: Slower, larger keys
- **Trade-offs:** Ed25519 signatures are not directly compatible with Ethereum EVM (but this is handled by precompiles in T007)

**Decision 4: Connection Limits**
- **Rationale:** Prevent resource exhaustion attacks (DoS) and maintain stable peer graph
- **Alternatives:**
  - Unlimited connections: Simple but vulnerable to attacks
  - Dynamic limits based on resources: Complex, hard to tune
- **Trade-offs:** 256 max connections may be insufficient for very large networks (can be tuned via config)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| QUIC blocked by firewall | High | Medium | Implement TCP fallback in T023 (NAT traversal), advertise both QUIC and TCP multiaddrs |
| Connection limit too low | Medium | Low | Make connection limits configurable, monitor metrics in production, increase if needed |
| Keypair loss/corruption | High | Low | Implement keypair backup/recovery mechanism, encrypted storage, clear documentation for operators |
| libp2p version incompatibility | Medium | Low | Pin to exact version 0.53.0, test upgrades in ICN Testnet before ICN Chain mainnet |
| Noise handshake failure | Medium | Medium | Implement retry logic with exponential backoff, log detailed error context, fallback to unencrypted transport in dev mode only |
| Yamux stream deadlock | Low | Low | Implement stream timeouts, monitor stream count metrics, use libp2p's tested yamux implementation |

## Progress Log

### [2025-12-24T00:00:00Z] - Task Created

**Created By:** task-creator agent
**Reason:** User request for P2P networking layer tasks (Phase 1)
**Dependencies:** T001 (ICN Chain fork and dev environment)
**Estimated Complexity:** Standard (core infrastructure, well-defined scope)

## Completion Checklist

**Code Complete:**
- [ ] All acceptance criteria met
- [ ] All test scenarios pass
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] Clippy/linting passes
- [ ] Formatting applied
- [ ] No regression in existing tests

**Deployment Ready:**
- [ ] Integration tests pass on testnet
- [ ] Metrics verified in Grafana
- [ ] Logs structured and parseable
- [ ] Error paths tested
- [ ] Resource usage within limits
- [ ] Monitoring alerts configured

**Definition of Done:**
Task is complete when ALL acceptance criteria met, ALL validations pass, integration tests demonstrate two nodes connecting and maintaining stable connection, and production-ready with metrics and graceful shutdown.
