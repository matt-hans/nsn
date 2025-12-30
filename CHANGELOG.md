# Changelog

## [Unreleased]

### Removed
- **legacy-nodes/**: Deprecated and removed after P2P migration to `node-core/crates/p2p/`.
  - Reason: Architectural consolidation (single off-chain node implementation in node-core).
  - Migration: T042 (P2P core) + T043 (GossipSub, reputation, metrics).
  - Impact: All off-chain development should target `node-core/` and `nsn-node`.
