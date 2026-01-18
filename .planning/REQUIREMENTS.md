# Requirements: v1.1 Viewer Networking Integration

**Version:** 1.0
**Date:** 2026-01-18
**Status:** Draft

---

## Overview

This document defines the requirements for connecting the NSN viewer to the live testnet. The milestone bridges the protocol gap between the Rust libp2p mesh and browser-based viewer.

---

## Functional Requirements

### Video Bridge Service

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-VB-001 | Bridge SHALL connect to NSN mesh via libp2p | P0 | Successfully dials at least one mesh peer |
| REQ-VB-002 | Bridge SHALL subscribe to GossipSub topic `/nsn/video/1.0.0` | P0 | Receives VideoChunk messages from mesh |
| REQ-VB-003 | Bridge SHALL decode SCALE-encoded VideoChunk | P0 | Correctly parses header and payload fields |
| REQ-VB-004 | Bridge SHALL translate VideoChunk to 17-byte header format | P0 | Output matches `[slot:4][chunk_index:4][timestamp:8][is_keyframe:1][data]` |
| REQ-VB-005 | Bridge SHALL serve WebSocket connections on configurable port | P0 | Browsers can connect via `ws://bridge:port/video` |
| REQ-VB-006 | Bridge SHALL forward translated chunks to all connected clients | P0 | All connected browsers receive chunks |
| REQ-VB-007 | Bridge SHALL expose health endpoint | P0 | Returns 200 OK when mesh connected, 503 otherwise |
| REQ-VB-008 | Bridge SHALL handle client disconnects gracefully | P0 | Other clients unaffected by one disconnect |
| REQ-VB-009 | Bridge SHALL implement connection limits | P1 | Configurable max concurrent connections |
| REQ-VB-010 | Bridge SHALL log chunk statistics | P1 | Logs chunks/sec, connected clients periodically |

### Chain RPC Client

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-RPC-001 | Viewer SHALL connect to chain RPC endpoint | P0 | WebSocket connection to validator node |
| REQ-RPC-002 | Viewer SHALL query current epoch from `NsnDirector` pallet | P0 | Returns epoch ID, directors, status |
| REQ-RPC-003 | Viewer SHALL query elected directors for current slot | P0 | Returns list of director AccountIds |
| REQ-RPC-004 | Viewer SHALL subscribe to block events | P1 | Receives new block notifications |
| REQ-RPC-005 | Viewer SHALL handle RPC connection errors gracefully | P0 | Shows error state, attempts reconnection |
| REQ-RPC-006 | Viewer SHALL cache last known state | P1 | Displays cached data during reconnection |

### Viewer Integration

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-VI-001 | Viewer SHALL connect to video bridge via WebSocket | P0 | Establishes connection, receives chunks |
| REQ-VI-002 | Viewer SHALL parse 17-byte chunk headers | P0 | Correctly extracts slot, chunk_index, timestamp, is_keyframe |
| REQ-VI-003 | Viewer SHALL feed chunks to existing video pipeline | P0 | Video renders in canvas |
| REQ-VI-004 | Viewer SHALL remove mock video stream | P0 | No hardcoded/generated video data |
| REQ-VI-005 | Viewer SHALL implement WebSocket reconnection | P0 | Auto-reconnects with exponential backoff |
| REQ-VI-006 | Viewer SHALL display connection status | P0 | Shows connecting/connected/disconnected/error |

### Live Statistics

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-LS-001 | Viewer SHALL display real bitrate | P0 | Calculated from chunk data size over time |
| REQ-LS-002 | Viewer SHALL display real latency | P0 | Time from chunk timestamp to render |
| REQ-LS-003 | Viewer SHALL display connected peer count | P1 | Number of mesh peers (from bridge or estimated) |
| REQ-LS-004 | Viewer SHALL display buffer level | P0 | Seconds of video buffered ahead |
| REQ-LS-005 | Viewer SHALL display current director info | P1 | Director ID from chain query |
| REQ-LS-006 | Viewer SHALL remove mock statistics | P0 | No hardcoded stat values |

---

## Non-Functional Requirements

### Performance

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-NF-001 | Bridge chunk forwarding latency SHALL be < 10ms | P1 | 95th percentile under 10ms |
| REQ-NF-002 | Bridge SHALL handle 100 concurrent WebSocket clients | P2 | No degradation at 100 connections |
| REQ-NF-003 | Viewer SHALL render video within 100ms of chunk receipt | P1 | Glass-to-glass latency acceptable |

### Reliability

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-NF-004 | Bridge SHALL reconnect to mesh on peer disconnect | P0 | Auto-reconnection within 30 seconds |
| REQ-NF-005 | Bridge SHALL continue serving during peer churn | P0 | No service interruption during reconnects |
| REQ-NF-006 | Viewer SHALL handle chunk gaps gracefully | P0 | Video continues despite missing chunks |

### Deployment

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-NF-007 | Bridge SHALL be containerized | P0 | Dockerfile with health check |
| REQ-NF-008 | Bridge SHALL integrate with docker-compose testnet | P0 | Added to existing docker-compose.yml |
| REQ-NF-009 | Configuration SHALL use environment variables | P0 | MESH_BOOTNODES, WS_PORT, HEALTH_PORT configurable |

### Observability

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-NF-010 | Bridge SHALL expose Prometheus metrics | P2 | /metrics endpoint with chunks_received, clients_connected |
| REQ-NF-011 | Bridge SHALL log structured JSON | P1 | Machine-parseable log format |

---

## Constraints

| ID | Constraint | Rationale |
|----|------------|-----------|
| CON-001 | Bridge must use Rust with tokio-tungstenite | Consistency with node-core, reuse P2P infrastructure |
| CON-002 | No libp2p in browser | Browser WebRTC requires alpha rust-libp2p-webrtc |
| CON-003 | Chain RPC via standard WebSocket | Use @polkadot/api, not custom protocol |
| CON-004 | Docker Compose deployment only | No Kubernetes for testnet |
| CON-005 | No authentication required | Testnet is permissionless |

---

## Dependencies

| ID | Dependency | Required By |
|----|------------|-------------|
| DEP-001 | NSN mesh nodes running | REQ-VB-001 |
| DEP-002 | GossipSub topic active | REQ-VB-002 |
| DEP-003 | Chain RPC endpoint exposed | REQ-RPC-001 |
| DEP-004 | Directors publishing video | REQ-VB-002 |

---

## Out of Scope

- Browser-native libp2p (WebRTC-direct)
- Authentication/authorization
- Chunk generation or publishing
- BFT consensus participation
- Mobile client support
- Video transcoding
- Persistent chunk storage

---

## Traceability

| Requirement | Research Source |
|-------------|-----------------|
| REQ-VB-* | ARCHITECTURE.md, STACK.md |
| REQ-RPC-* | FEATURES.md (Chain RPC Client section) |
| REQ-VI-* | FEATURES.md (Live Statistics section), PITFALLS.md |
| REQ-LS-* | FEATURES.md (Live Statistics section) |
| REQ-NF-* | PITFALLS.md, ARCHITECTURE.md |

---

*Requirements document v1.0 - Ready for roadmap planning*
