# Research Summary: v1.1 Viewer Networking Integration (WebRTC-Direct)

**Research completed:** 2026-01-18
**Sources:** STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md
**Approach:** WebRTC-Direct (browser connects directly to mesh nodes)

---

## Executive Summary

The v1.1 milestone enables direct browser-to-mesh connectivity via WebRTC. By adding WebRTC transport to existing Rust nodes, browsers can connect as first-class libp2p peers without requiring a separate bridge service.

**Key advantages over bridge approach:**
- Lower latency (direct connection, no relay)
- Reduced infrastructure (no separate service to deploy)
- Future-proof (standard libp2p protocol)
- Same protocol (browser decodes SCALE, no format translation)

**Reliability Assessment:** MEDIUM-HIGH
- rust-libp2p 0.53 WebRTC: Stable, production-ready
- js-libp2p WebRTC: Well-tested, browser-native
- Cross-implementation interop: Requires testing but supported

---

## Approach Overview

### The Problem

Browsers cannot speak libp2p directly (no TCP/QUIC sockets). Two approaches exist:
1. **Bridge service** (original plan): Node.js intermediary translates protocols
2. **WebRTC-direct** (revised plan): Rust nodes accept WebRTC, browsers dial directly

### The Solution

Add WebRTC transport to Rust nodes, expose HTTP discovery endpoint, let browser connect directly.

```
NSN Mesh (Rust libp2p 0.53)
    │ TCP/Noise/Yamux (mesh interconnect)
    │ UDP/WebRTC (browser clients)
    ▼
Director/Validator Nodes
    │ HTTP /p2p/info → { peer_id, multiaddrs with certhash }
    ▼
Browser Viewer (js-libp2p + @libp2p/webrtc)
    │ Direct WebRTC connection
    │ GossipSub subscription
    │ SCALE-encoded VideoChunk
    ▼
Video Pipeline
```

---

## Technology Stack

### Rust (node-core)

| Component | Library | Notes |
|-----------|---------|-------|
| Core | libp2p 0.53.x | Enable `webrtc` feature |
| Transport | TCP + WebRTC | Hybrid for mesh + browsers |
| HTTP | axum or warp | Discovery endpoint |
| Cert | pem 3.0 | Certificate persistence |

### Browser (viewer)

| Component | Library | Notes |
|-----------|---------|-------|
| Core | libp2p 2.0.x | TypeScript-first |
| Transport | @libp2p/webrtc 5.x | Browser RTCPeerConnection |
| PubSub | @chainsafe/libp2p-gossipsub 14.x | Video topic subscription |
| Codec | @polkadot/types-codec 16.x | SCALE decoding |
| Chain | @polkadot/api 16.x | Director queries |

---

## Critical Design Decisions

### 1. Certificate Persistence

WebRTC multiaddrs include certificate hash:
```
/ip4/192.168.1.50/udp/9003/webrtc/certhash/uEiD...
```

If certificate regenerates on restart, all cached addresses break.

**Solution:** Persist certificate to `{data_dir}/webrtc_cert.pem`

### 2. Discovery Endpoint

Browsers cannot discover WebRTC addresses via mDNS/DHT.

**Solution:** HTTP endpoint `GET /p2p/info` returns:
```json
{
  "peer_id": "12D3KooW...",
  "multiaddrs": ["/ip4/.../udp/9003/webrtc/certhash/..."],
  "protocols": ["/nsn/video/1.0.0"]
}
```

### 3. Hybrid Transport

Nodes must support both:
- TCP/Noise/Yamux for mesh interconnect
- WebRTC for browser connections

### 4. SCALE Decoding in Browser

Browser decodes SCALE-encoded VideoChunk directly (no format translation).

**Benefits:**
- Same protocol as mesh nodes
- No bridge complexity
- Type-safe with @polkadot/types-codec

---

## Implementation Phases

| Phase | Name | Key Deliverables |
|-------|------|------------------|
| 1 | Rust Node Core Upgrade | WebRTC feature, cert persistence, transport |
| 2 | Discovery Bridge | `/p2p/info` endpoint, CORS |
| 3 | Viewer Implementation | js-libp2p WebRTC, discovery fetch |
| 4 | Video Streaming | GossipSub subscription, SCALE decode |
| 5 | Chain RPC | @polkadot/api integration (parallel) |
| 6 | Docker & Operations | UDP port, env vars (parallel) |
| 7 | Testing & Validation | E2E tests |

**Critical Path:** Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 7

---

## Key Pitfalls to Avoid

### Phase 1: Rust Node

| Pitfall | Prevention |
|---------|------------|
| Certificate regeneration | Persist to disk on first run |
| Docker internal IPs announced | Filter 172.x.x.x, use `--p2p-external-address` |
| Missing WebRTC feature | Add `webrtc` to libp2p features |

### Phase 2: Discovery

| Pitfall | Prevention |
|---------|------------|
| CORS blocking | Add `CorsLayer::permissive()` |
| Missing certhash in response | Filter addresses, include only full WebRTC addr |

### Phase 3: Browser

| Pitfall | Prevention |
|---------|------------|
| Hardcoded multiaddr | Always fetch from `/p2p/info` |
| Noise double encryption | WebRTC uses DTLS, don't add noise |
| Connection not waiting for mesh | Implement ready check before subscribing |

### Phase 4: Video

| Pitfall | Prevention |
|---------|------------|
| SCALE type mismatch | Match exact field order from Rust |
| GossipSub topic hash mismatch | Use identical topic string |

---

## Comparison: Bridge vs WebRTC-Direct

| Aspect | Bridge | WebRTC-Direct |
|--------|--------|---------------|
| Latency | +10-50ms | Minimal |
| Infrastructure | Extra service | Integrated |
| Protocol | Translate SCALE→binary | Same (SCALE) |
| Scalability | Bridge bottleneck | Distributed |
| Complexity | Medium | Higher (initial) |
| Maintenance | Two systems | One system |

**Verdict:** WebRTC-Direct is preferred for production. Higher initial complexity pays off with better latency and reduced infrastructure.

---

## Requirements Summary

### P0 (Must Have)

- WebRTC transport in Rust nodes
- Certificate persistence
- Discovery endpoint with CORS
- js-libp2p WebRTC client
- GossipSub video subscription
- SCALE VideoChunk decoding
- Connection status display
- Remove mock video

### P1 (Should Have)

- External address configuration
- Connection retry with backoff
- Chain RPC for directors
- Live statistics

### P2 (Nice to Have)

- 50+ concurrent connections
- Prometheus metrics

---

## Definition of Done

Milestone v1.1 is complete when:

1. ✓ Rust nodes accept WebRTC connections
2. ✓ Certificate persists across restarts
3. ✓ `/p2p/info` returns valid WebRTC multiaddr
4. ✓ Browser connects directly via js-libp2p
5. ✓ GossipSub delivers video chunks
6. ✓ SCALE decoding works
7. ✓ Video renders without mock data
8. ✓ Statistics are real
9. ✓ Chain queries work
10. ✓ E2E tests pass

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-18 | Initial summary (Node.js bridge approach) |
| 2.0 | 2026-01-18 | Revised for WebRTC-direct approach |

---

*Research synthesis v2.0 - WebRTC-Direct approach*
