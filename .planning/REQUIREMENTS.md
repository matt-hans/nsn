# Requirements: v1.1 Viewer Networking Integration (WebRTC-Direct)

**Version:** 2.0
**Date:** 2026-01-18
**Status:** Draft
**Approach:** WebRTC-Direct (browser connects directly to mesh nodes)

---

## Overview

This document defines the requirements for connecting the NSN viewer directly to the libp2p mesh via WebRTC. This approach eliminates the need for a separate bridge service by enabling Rust nodes to accept browser connections.

---

## Functional Requirements

### Rust Node WebRTC Support

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-WR-001 | Node SHALL enable WebRTC transport in libp2p | P0 | `libp2p = { features = ["webrtc"] }` compiles |
| REQ-WR-002 | Node SHALL persist WebRTC certificate to disk | P0 | Certificate survives node restart |
| REQ-WR-003 | Node SHALL listen on configurable UDP port for WebRTC | P0 | Default 9003, configurable via `--p2p-webrtc-port` |
| REQ-WR-004 | Node SHALL announce WebRTC multiaddr with certhash | P0 | Multiaddr includes `/webrtc/certhash/<HASH>` |
| REQ-WR-005 | Node SHALL support hybrid transport (TCP + WebRTC) | P0 | Both mesh peers and browsers can connect |
| REQ-WR-006 | Node SHALL announce external IP when configured | P1 | `--p2p-external-address` overrides internal IP |
| REQ-WR-007 | Node SHALL log WebRTC connection events | P1 | Connection/disconnection logged at INFO level |

### Discovery Endpoint

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-DISC-001 | Node SHALL expose HTTP endpoint at `/p2p/info` | P0 | Returns JSON with peer_id and multiaddrs |
| REQ-DISC-002 | Response SHALL include WebRTC multiaddr with certhash | P0 | Browser can parse and dial the address |
| REQ-DISC-003 | Response SHALL filter out internal Docker IPs | P0 | No 172.x.x.x or 127.0.0.1 in public response |
| REQ-DISC-004 | Endpoint SHALL include CORS headers | P0 | Browser fetch succeeds from different origin |
| REQ-DISC-005 | Endpoint SHALL return supported protocols | P1 | Includes `/nsn/video/1.0.0` in response |
| REQ-DISC-006 | Endpoint SHALL be on same HTTP server as metrics | P2 | No additional port needed |

### Viewer P2P Integration

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-VI-001 | Viewer SHALL initialize js-libp2p with WebRTC transport | P0 | libp2p node starts in browser |
| REQ-VI-002 | Viewer SHALL fetch discovery info from node HTTP endpoint | P0 | Parses `/p2p/info` response correctly |
| REQ-VI-003 | Viewer SHALL dial WebRTC multiaddr from discovery | P0 | Connection established to Rust node |
| REQ-VI-004 | Viewer SHALL subscribe to GossipSub `/nsn/video/1.0.0` | P0 | Receives published video chunks |
| REQ-VI-005 | Viewer SHALL decode SCALE-encoded VideoChunk | P0 | Header and payload correctly extracted |
| REQ-VI-006 | Viewer SHALL feed decoded chunks to video pipeline | P0 | Video renders on canvas |
| REQ-VI-007 | Viewer SHALL implement connection retry with backoff | P0 | Reconnects after disconnect |
| REQ-VI-008 | Viewer SHALL display connection status | P0 | Shows connecting/connected/disconnected |
| REQ-VI-009 | Viewer SHALL remove mock video stream | P0 | No hardcoded/generated video data |

### Chain RPC Client

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-RPC-001 | Viewer SHALL connect to chain RPC endpoint | P0 | WebSocket connection to validator node |
| REQ-RPC-002 | Viewer SHALL query current epoch from `NsnDirector` pallet | P0 | Returns epoch ID, directors, status |
| REQ-RPC-003 | Viewer SHALL query elected directors for current slot | P0 | Returns list of director AccountIds |
| REQ-RPC-004 | Viewer SHALL subscribe to block events | P1 | Receives new block notifications |
| REQ-RPC-005 | Viewer SHALL handle RPC connection errors gracefully | P0 | Shows error state, attempts reconnection |
| REQ-RPC-006 | Viewer SHALL cache last known state | P1 | Displays cached data during reconnection |

### Live Statistics

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-LS-001 | Viewer SHALL display real bitrate | P0 | Calculated from chunk data size over time |
| REQ-LS-002 | Viewer SHALL display real latency | P0 | Time from chunk timestamp to render |
| REQ-LS-003 | Viewer SHALL display connected peer count | P1 | From libp2p connection manager |
| REQ-LS-004 | Viewer SHALL display buffer level | P0 | Seconds of video buffered ahead |
| REQ-LS-005 | Viewer SHALL display current director info | P1 | Director ID from chain query |
| REQ-LS-006 | Viewer SHALL remove mock statistics | P0 | No hardcoded stat values |

---

## Non-Functional Requirements

### Performance

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-NF-001 | WebRTC connection latency SHALL be < 500ms | P1 | Dial to connected under 500ms |
| REQ-NF-002 | Node SHALL handle 50 concurrent WebRTC connections | P2 | No degradation at 50 browsers |
| REQ-NF-003 | Viewer SHALL render video within 100ms of chunk receipt | P1 | Glass-to-glass latency acceptable |
| REQ-NF-004 | Discovery endpoint SHALL respond in < 50ms | P1 | No blocking operations |

### Reliability

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-NF-005 | Certificate SHALL persist across node restarts | P0 | Same certhash after restart |
| REQ-NF-006 | Node SHALL continue mesh operations if WebRTC fails | P0 | Graceful degradation |
| REQ-NF-007 | Viewer SHALL handle chunk gaps gracefully | P0 | Video continues despite missing chunks |
| REQ-NF-008 | Viewer SHALL reconnect automatically after disconnect | P0 | Reconnection within 30 seconds |

### Deployment

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-NF-009 | WebRTC port SHALL be configurable via CLI | P0 | `--p2p-webrtc-port 9003` works |
| REQ-NF-010 | External address SHALL be configurable via CLI | P0 | `--p2p-external-address` works |
| REQ-NF-011 | Docker Compose SHALL expose UDP port for WebRTC | P0 | Port 9003/udp mapped |
| REQ-NF-012 | Data volume SHALL preserve certificate | P0 | Certificate persists across container restarts |

### Security

| REQ-ID | Requirement | Priority | Acceptance Criteria |
|--------|-------------|----------|---------------------|
| REQ-NF-013 | WebRTC SHALL use DTLS encryption | P0 | Built into libp2p WebRTC transport |
| REQ-NF-014 | Certificate hash SHALL be verified on connect | P0 | Prevents MITM attacks |
| REQ-NF-015 | Discovery endpoint SHALL be read-only | P0 | No state mutations via HTTP |

---

## Constraints

| ID | Constraint | Rationale |
|----|------------|-----------|
| CON-001 | Must use libp2p 0.53.x in Rust | WebRTC support included, no major upgrade |
| CON-002 | Browser uses js-libp2p WebRTC | Native browser RTCPeerConnection |
| CON-003 | No signaling server required | WebRTC-direct uses certhash in multiaddr |
| CON-004 | SCALE encoding for video chunks | Matches existing mesh protocol |
| CON-005 | Docker Compose deployment only | No Kubernetes for testnet |

---

## Dependencies

| ID | Dependency | Required By |
|----|------------|-------------|
| DEP-001 | libp2p 0.53 with webrtc feature | REQ-WR-001 |
| DEP-002 | axum or warp HTTP server | REQ-DISC-001 |
| DEP-003 | js-libp2p with @libp2p/webrtc | REQ-VI-001 |
| DEP-004 | @polkadot/types-codec | REQ-VI-005 |
| DEP-005 | NSN mesh nodes running | All REQ-VI-* |
| DEP-006 | GossipSub topic active | REQ-VI-004 |

---

## Out of Scope

- Separate bridge service (replaced by WebRTC-direct)
- Custom binary chunk format (browser decodes SCALE directly)
- STUN/TURN servers (not needed for testnet with public IPs)
- Mobile app support (browser only)
- Video transcoding in node (Vortex handles format)
- Authentication/authorization (testnet is permissionless)

---

## Traceability

| Requirement | Research Source | Phase |
|-------------|-----------------|-------|
| REQ-WR-* | ARCHITECTURE.md, STACK.md | Phase 1 |
| REQ-DISC-* | ARCHITECTURE.md (Discovery Bridge) | Phase 2 |
| REQ-VI-* | ARCHITECTURE.md (Browser Implementation) | Phase 3 |
| REQ-RPC-* | FEATURES.md (Chain RPC Client) | Phase 4 (parallel) |
| REQ-LS-* | FEATURES.md (Live Statistics) | Phase 4 |
| REQ-NF-* | PITFALLS.md, ARCHITECTURE.md | All phases |

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-18 | Initial requirements (Node.js bridge approach) |
| 2.0 | 2026-01-18 | Revised for WebRTC-direct approach |

---

*Requirements document v2.0 - WebRTC-Direct approach*
