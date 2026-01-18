# Research Summary: v1.1 Viewer Networking Integration

**Research completed:** 2026-01-18
**Sources:** STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md

---

## Executive Summary

The v1.1 milestone requires bridging the browser-based viewer to the NSN libp2p mesh. Browsers cannot speak libp2p directly, so a Node.js bridge service using js-libp2p will subscribe to GossipSub and relay video chunks to viewers via WebSocket. The approach is well-supported by mature libraries with verified cross-implementation interoperability.

**Reliability Assessment:** HIGH - All core technologies (js-libp2p, @polkadot/types-codec, ws) are production-proven with active maintenance.

---

## Recommended Technology Stack

| Component | Library | Version | Confidence |
|-----------|---------|---------|------------|
| Core libp2p | `libp2p` | ^3.1.2 | HIGH |
| Transport | `@libp2p/tcp` | ^9.0.13 | HIGH |
| Encryption | `@chainsafe/libp2p-noise` | ^16.1.4 | HIGH |
| Multiplexer | `@chainsafe/libp2p-yamux` | ^7.0.1 | HIGH |
| PubSub | `@chainsafe/libp2p-gossipsub` | ^14.1.2 | HIGH |
| SCALE Codec | `@polkadot/types-codec` | ^16.5.4 | HIGH |
| WebSocket | `ws` | ^8.19.0 | HIGH |
| Chain RPC | `@polkadot/api` | ^16.x | HIGH |

**Critical:** Use TCP+Noise+Yamux transport stack. QUIC has known js/rust interop issues.

---

## Architecture Decision

**Selected:** Rust Video Bridge (integrated with existing node-core)

The bridge will be implemented in Rust, reusing existing P2P infrastructure from node-core. This approach:
- Reuses `nsn_p2p::P2pService` for mesh connectivity
- Reuses `nsn_types::VideoChunk` for chunk handling
- Adds WebSocket server via `tokio-tungstenite`
- Maintains single codebase language (Rust)

**Alternative considered:** Node.js bridge with js-libp2p was researched but Rust is preferred for consistency.

---

## Scope Summary

### Table Stakes (Must Have)

**Video Bridge Service:**
1. GossipSub subscription to `/nsn/video/1.0.0`
2. SCALE decode VideoChunk to browser binary format
3. WebSocket server for browser connections
4. Chunk forwarding with preserved header fields
5. Connection to at least one mesh peer
6. Health endpoint for Docker healthcheck
7. Graceful shutdown

**Chain RPC Client:**
1. Connect to chain RPC endpoint (ws://validator:9944)
2. Query `NsnDirector::CurrentEpoch`
3. Query `NsnDirector::ElectedDirectors(slot)`
4. Query `NsnDirector::NextEpochDirectors`
5. Subscribe to new blocks for epoch transitions
6. Graceful error handling

**Live Statistics:**
1. Connected peer count
2. Current bitrate (Mbps)
3. Chunk latency (ms)
4. Buffer level (seconds)
5. Current director info
6. Connection status states

### Out of Scope (Anti-features)

- Full libp2p-in-browser via WebRTC (rust-libp2p-webrtc is alpha)
- Chunk generation/publishing (bridge is read-only)
- BFT consensus participation
- Persistent storage of chunks
- Authentication/authorization
- Video transcoding

---

## Critical Pitfalls to Address

### Phase 1: Video Bridge Setup

| Pitfall | Mitigation |
|---------|------------|
| Topic hashing mismatch | Configure IdentityHash or verify topic strings match exactly |
| Noise handshake pattern | Use XX pattern explicitly, not default |
| Insufficient GossipSub peers | Wait for mesh formation, implement ready check |
| JS-to-Rust stream closure | Proper async handling, connection keepalive |
| Silent WebSocket failures | Register all event handlers, implement heartbeat |
| Thundering herd reconnection | Exponential backoff with jitter |

### Phase 2: Chunk Reception

| Pitfall | Mitigation |
|---------|------------|
| SCALE type definition mismatch | Generate types from chain metadata, validate field order |
| Sequence number deduplication | Use content-based message ID, not sequence numbers |
| Background tab throttling | Document as known limitation, buffer recent chunks |

### Phase 3: Chain RPC

| Pitfall | Mitigation |
|---------|------------|
| API instance memory leak | Single instance, reuse everywhere |
| Subscription cleanup | Always store and call unsubscribe |
| Type clashes across pallets | Use explicit type aliases |

---

## Build Order (8 Phases)

| Phase | Component | Depends On | Deliverable |
|-------|-----------|------------|-------------|
| 1 | Types crate | - | Shared message definitions |
| 2 | Chunk Translator | Types | Translation function with tests |
| 3 | Mesh Subscriber | nsn-p2p, nsn-types | broadcast::Receiver<VideoChunk> |
| 4 | WebSocket Server | Types, Translator | BridgeServer accepting connections |
| 5 | Integration | All above | video-bridge binary |
| 6 | Viewer WebSocket mode | Types | Updated p2p.ts using WebSocket |
| 7 | Chain RPC (bridge-side) | subxt | Optional epoch updates push |
| 8 | Chain RPC (viewer-side) | @polkadot/api | Direct chain queries from viewer |

**Critical path:** Phases 1-5 are blocking for MVP. Phases 6-8 can be parallelized.

---

## Message Translation

**Input (SCALE VideoChunk from mesh):**
```rust
VideoChunk {
    header: VideoChunkHeader {
        version: u16,
        slot: u64,
        content_id: String,
        chunk_index: u32,
        total_chunks: u32,
        timestamp_ms: u64,
        is_keyframe: bool,
        payload_hash: [u8; 32],
    },
    payload: Vec<u8>,
    signer: Vec<u8>,
    signature: Vec<u8>,
}
```

**Output (Binary to browser):**
```
[slot:4 bytes, big-endian u32]
[chunk_index:4 bytes, big-endian u32]
[timestamp:8 bytes, big-endian u64]
[is_keyframe:1 byte, 0x00 or 0x01]
[data: remaining bytes]
```

Total header: 17 bytes + payload

---

## Data Flow

```
NSN Mesh (Rust libp2p)
    │ GossipSub /nsn/video/1.0.0 (SCALE VideoChunk)
    ▼
Video Bridge (Rust)
    │ Decode SCALE → Binary translation
    │ WebSocket (ws://bridge:9090/video)
    ▼
Browser Viewer (React)
    │ Parse 17-byte header + payload
    │ Feed to video pipeline
    ▼
Canvas Renderer

Optional:
NSN Chain ─── @polkadot/api ──► Viewer (Director queries)
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| js/rust libp2p interop issues | Low | High | Verified via multidim-interop suite |
| SCALE decode failures | Low | High | Type generation from metadata |
| WebSocket connection instability | Medium | Medium | Heartbeat + reconnection logic |
| Chain RPC availability | Low | Low | Graceful degradation to cached state |

---

## Recommendations

1. **Start with Rust bridge** - Leverage existing nsn-p2p infrastructure
2. **Implement health checks early** - Critical for Docker Compose integration
3. **Test interop incrementally** - Verify each layer before adding next
4. **Add metrics from start** - Prometheus endpoints for debugging
5. **Document WebSocket protocol** - Clear contract for viewer integration

---

*Research synthesis complete. Ready for requirements definition.*
