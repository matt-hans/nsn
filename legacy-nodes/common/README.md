# ICN Common Library

Shared libraries for ICN off-chain nodes (Directors, Validators, Super-Nodes, Relays, Viewers).

## Features

### P2P Networking (`icn_common::p2p`)

Foundational libp2p-based P2P networking layer with:

- **QUIC Transport**: UDP-based low-latency connections with 0-RTT resumption
- **Noise XX Encryption**: End-to-end encrypted connections with forward secrecy
- **Ed25519 Identity**: Cross-layer identity compatible with Substrate AccountId32
- **Connection Management**: Configurable limits (256 total, 2 per-peer) with automatic enforcement
- **Prometheus Metrics**: Active connections, peer count, bytes sent/received exposed on port 9100
- **Graceful Shutdown**: Clean connection closure on SIGTERM/SIGINT

#### Quick Start

```rust
use icn_common::p2p::{P2pConfig, P2pService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = P2pConfig {
        listen_port: 9000,
        max_connections: 256,
        max_connections_per_peer: 2,
        ..Default::default()
    };

    let (mut service, cmd_tx) = P2pService::new(config).await?;

    // Start the service
    tokio::spawn(async move {
        service.start().await.expect("Service failed");
    });

    // Service is now running...

    Ok(())
}
```

#### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `listen_port` | 9000 | QUIC listening port |
| `max_connections` | 256 | Maximum total concurrent connections |
| `max_connections_per_peer` | 2 | Maximum connections per individual peer |
| `connection_timeout` | 30s | Idle connection timeout |
| `keypair_path` | None | Optional persistent keypair file path |
| `metrics_port` | 9100 | Prometheus metrics server port |

#### Metrics Exposed

- `icn_p2p_active_connections` - Currently active connections
- `icn_p2p_connected_peers` - Number of unique peers
- `icn_p2p_bytes_sent_total` - Total bytes sent
- `icn_p2p_bytes_received_total` - Total bytes received
- `icn_p2p_connections_established_total` - Successful connections
- `icn_p2p_connections_failed_total` - Failed connection attempts
- `icn_p2p_connections_closed_total` - Closed connections
- `icn_p2p_connection_limit` - Configured connection limit

#### PeerId ↔ AccountId32 Conversion

```rust
use icn_common::p2p::{generate_keypair, peer_id_to_account_id};
use libp2p::PeerId;

let keypair = generate_keypair();
let peer_id = PeerId::from(keypair.public());
let account_id = peer_id_to_account_id(&peer_id)?;

// Use account_id for Substrate extrinsics
```

## Testing

```bash
# Run all unit tests
cargo test -p icn-common --lib

# Run specific test
cargo test -p icn-common --lib p2p::identity::tests::test_keypair_generation

# Run with detailed output
cargo test -p icn-common --lib -- --nocapture
```

## Development

### Build

```bash
cargo build -p icn-common
```

### Lint

```bash
cargo clippy -p icn-common --lib -- -D warnings
```

### Format

```bash
cargo fmt -p icn-common
```

## Dependencies

- **libp2p**: 0.53 - P2P networking
- **sp-core**: 28.0 - Substrate types (AccountId32)
- **tokio**: 1.43 - Async runtime
- **prometheus**: 0.13 - Metrics
- **thiserror**: 1.0 - Error handling
- **tracing**: 0.1 - Structured logging

## Module Structure

```
icn-nodes/common/src/
├── lib.rs                  # Root module
├── chain/                  # Chain client utilities (stub)
├── types/                  # Shared type definitions (stub)
└── p2p/                    # P2P networking (complete)
    ├── mod.rs              # Public API exports
    ├── config.rs           # Configuration types
    ├── identity.rs         # Ed25519 keypair and PeerId↔AccountId
    ├── behaviour.rs        # libp2p NetworkBehaviour + ConnectionTracker
    ├── metrics.rs          # Prometheus metrics
    └── service.rs          # P2P service main loop
```

## License

MIT
