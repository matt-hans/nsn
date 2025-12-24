# ICN Off-Chain Nodes

Rust workspace containing the off-chain node implementations for the Interdimensional Cable Network.

## Components

| Crate | Binary | Description | Task |
|-------|--------|-------------|------|
| `icn-common` | (library) | Shared P2P, chain client, types | T021-T027 |
| `icn-director` | `icn-director` | GPU video generation + BFT | T009 |
| `icn-validator` | `icn-validator` | CLIP semantic verification | T010 |
| `icn-super-node` | `icn-super-node` | Tier 1 storage + erasure coding | T011 |
| `icn-relay` | `icn-relay` | Tier 2 regional distribution | T012 |

## Building

```bash
# Build all nodes
cargo build --release

# Build specific node
cargo build --release -p icn-director

# Run tests
cargo test
```

## Architecture

```
Director (Tier 0)
    ↓ video chunks
Super-Node (Tier 1) ←→ Super-Node
    ↓ cached content
Regional Relay (Tier 2)
    ↓ streams
Viewer (Tier 3)
```

All nodes connect to the Moonbeam chain via `subxt` and communicate via libp2p GossipSub.
