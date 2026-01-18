# Stack Research: js-libp2p Bridge for NSN Video Distribution

## Executive Summary

This document specifies the recommended technology stack for a Node.js bridge service that connects js-libp2p to the existing NSN Rust libp2p 0.53 mesh and relays SCALE-encoded video chunks to browser clients via WebSocket.

**Reliability Assessment**:
- js-libp2p/rust-libp2p interop: **HIGH** (well-tested via libp2p multidim-interop suite)
- SCALE codec TypeScript: **HIGH** (mature ecosystem via Polkadot tooling)
- Transport compatibility: **HIGH** (TCP+Noise+Yamux is standard interop path)

---

## Recommended Stack

| Component | Library | Version | Rationale |
|-----------|---------|---------|-----------|
| **Core libp2p** | `libp2p` | `^3.1.2` | Latest stable, active development, TypeScript-first |
| **Transport (TCP)** | `@libp2p/tcp` | `^9.0.13` | Required for Noise+Yamux interop with rust-libp2p |
| **Security** | `@chainsafe/libp2p-noise` | `^16.1.4` | XX handshake pattern for cross-impl interop |
| **Multiplexer** | `@chainsafe/libp2p-yamux` | `^7.0.1` | Recommended over deprecated mplex |
| **PubSub** | `@chainsafe/libp2p-gossipsub` | `^14.1.2` | GossipSub v1.1 spec, ChainSafe maintained |
| **SCALE Codec** | `@polkadot/types-codec` | `^16.5.4` | Production-proven, type-safe SCALE decoding |
| **WebSocket Server** | `ws` | `^8.19.0` | Blazing fast, RFC 6455 compliant |
| **Runtime** | Node.js | `^20.0.0` (LTS) | ESM support, stable async/await |
| **TypeScript** | `typescript` | `^5.3.0` | Modern type inference, ESM output |

---

## Core Dependencies

### 1. libp2p Core (`libp2p@^3.1.2`)

**Purpose**: Main networking stack for peer-to-peer communication.

**Rationale**:
- Latest stable release (September 2025)
- Full TypeScript support with generated types
- Modular architecture for transport/encryption/muxer selection
- Active maintenance (though with reduced maintainer count)

**Installation**:
```bash
npm install libp2p@^3.1.2
```

**Configuration Notes**:
- Use `createLibp2p()` factory with explicit service configuration
- Protocol handler signature changed in v3: `(stream, connection) => void | Promise<void>`

---

### 2. TCP Transport (`@libp2p/tcp@^9.0.13`)

**Purpose**: TCP transport for connecting to rust-libp2p mesh nodes.

**Rationale**:
- Required for Noise+Yamux interoperability with rust-libp2p
- QUIC support in js-libp2p is less mature for cross-implementation use
- NSN Rust mesh listens on both TCP and QUIC (service.rs lines 447-458)

**Why Not QUIC**:
- js-libp2p QUIC (@libp2p/quic-v1) has known interop issues with rust-libp2p
- TCP+Noise+Yamux is the guaranteed interop path per libp2p test suite

---

### 3. Noise Encryption (`@chainsafe/libp2p-noise@^16.1.4`)

**Purpose**: Noise protocol for encrypted communications.

**Rationale**:
- XX handshake pattern provides guaranteed cross-implementation interop
- ChainSafe-maintained with active release cadence
- Matches rust-libp2p noise configuration (service.rs line 334)

**Critical Configuration**:
```typescript
import { noise } from '@chainsafe/libp2p-noise'

// Use default XX handshake - IK/IX patterns may not interop
connectionEncrypters: [noise()]
```

---

### 4. Yamux Multiplexer (`@chainsafe/libp2p-yamux@^7.0.1`)

**Purpose**: Stream multiplexing over single connection.

**Rationale**:
- Recommended multiplexer (mplex is deprecated)
- Matches rust-libp2p yamux configuration (service.rs line 335)
- Known interop issue fixed in rust-yamux#156

**Configuration**:
```typescript
import { yamux } from '@chainsafe/libp2p-yamux'

streamMuxers: [yamux()]
```

---

### 5. GossipSub (`@chainsafe/libp2p-gossipsub@^14.1.2`)

**Purpose**: Subscribe to `/nsn/video/1.0.0` topic for video chunk distribution.

**Rationale**:
- Implements GossipSub v1.1 spec (matching rust-libp2p gossipsub)
- ChainSafe-maintained with active development
- Supports signed messages (required for NSN MessageAuthenticity::Signed)

**Critical Configuration**:
```typescript
import { gossipsub } from '@chainsafe/libp2p-gossipsub'

services: {
  pubsub: gossipsub({
    allowPublishToZeroTopicPeers: false,
    emitSelf: false,
    // Match NSN mesh parameters (gossipsub.rs)
    D: 6,           // mesh_n
    Dlo: 4,         // mesh_n_low
    Dhi: 12,        // mesh_n_high
    Dlazy: 6,       // gossip_lazy
    heartbeatInterval: 1000,
    // Signature validation - CRITICAL
    globalSignaturePolicy: 'StrictSign'
  })
}
```

**Topic Subscription**:
```typescript
// Match NSN topic string exactly (topics.rs line 15)
const VIDEO_TOPIC = '/nsn/video/1.0.0'
await libp2p.services.pubsub.subscribe(VIDEO_TOPIC)
```

---

### 6. SCALE Codec (`@polkadot/types-codec@^16.5.4`)

**Purpose**: Decode SCALE-encoded VideoChunk structs from GossipSub messages.

**Rationale**:
- Production-proven in Polkadot ecosystem
- Full TypeScript type safety
- Handles complex nested structs (VideoChunk contains VideoChunkHeader)

**Alternative Considered**: `@scale-codec/core` - lower-level, less ecosystem support

**VideoChunk Struct Definition** (from `node-core/crates/types/src/lib.rs:211-242`):

```typescript
import { Struct, u16, u64, Str, u32, bool, Bytes, createType } from '@polkadot/types-codec'
import { TypeRegistry } from '@polkadot/types'

const registry = new TypeRegistry()

// Register custom types matching Rust definitions
registry.register({
  VideoChunkHeader: {
    version: 'u16',
    slot: 'u64',
    content_id: 'Text',
    chunk_index: 'u32',
    total_chunks: 'u32',
    timestamp_ms: 'u64',
    is_keyframe: 'bool',
    payload_hash: '[u8; 32]'
  },
  VideoChunk: {
    header: 'VideoChunkHeader',
    payload: 'Bytes',
    signer: 'Bytes',
    signature: 'Bytes'
  }
})

// Decode incoming GossipSub message
function decodeVideoChunk(data: Uint8Array): VideoChunk {
  return registry.createType('VideoChunk', data)
}
```

---

### 7. WebSocket Server (`ws@^8.19.0`)

**Purpose**: Relay decoded video chunks to browser clients.

**Rationale**:
- Highest performance WebSocket library for Node.js
- Minimal overhead, RFC 6455 compliant
- 24,000+ dependent packages - battle-tested

**Configuration**:
```typescript
import { WebSocketServer } from 'ws'

const wss = new WebSocketServer({
  port: 8080,
  // Disable compression for video streaming (already compressed)
  perMessageDeflate: false,
  // Large messages for video chunks (up to 16MB per gossipsub.rs)
  maxPayload: 16 * 1024 * 1024
})
```

---

## Avoid

### Libraries NOT to Use

| Library | Reason |
|---------|--------|
| `@libp2p/quic-v1` | Interop issues with rust-libp2p 0.53; TCP+Noise+Yamux is safer |
| `@libp2p/mplex` | **Deprecated** - use yamux instead |
| `libp2p@^2.x` | Older API, noise/yamux packages require v3 compatibility |
| `@chainsafe/libp2p-noise@^17.x` | Requires libp2p@3.x minimum - pin to 16.x for stability |
| `scale-codec` (npm) | Less ecosystem support than @polkadot/types-codec |
| `socket.io` | Unnecessary abstraction layer for this use case |
| `uWebSockets.js` | Complex native bindings, ws is sufficient |
| `parity-scale-codec` (npm) | Deprecated, renamed to scale-codec |

### Patterns to Avoid

1. **Do NOT use QUIC as primary transport** - TCP+Noise+Yamux is the guaranteed interop path
2. **Do NOT skip signature validation** - NSN uses `MessageAuthenticity::Signed`
3. **Do NOT enable WebSocket compression** - video chunks are already compressed
4. **Do NOT use libp2p@2.x** - breaking API changes, noise/yamux version mismatches

---

## Version Compatibility Notes

### js-libp2p <-> rust-libp2p 0.53 Interoperability

**Status**: VERIFIED COMPATIBLE

The libp2p project maintains a [multidimensional interoperability test suite](https://blog.libp2p.io/multidim-interop/) that validates cross-implementation compatibility.

**Known Compatible Combinations**:
- Transport: TCP
- Encryption: Noise (XX handshake pattern only)
- Muxer: Yamux

**Known Issues (Resolved)**:
- rust-yamux#156: Fixed yamux interop between rust and JS
- rust-libp2p#2954: Fixed wrong default Noise handshake pattern

### Package Version Matrix

| js-libp2p Version | @chainsafe/libp2p-noise | @chainsafe/libp2p-yamux | @chainsafe/libp2p-gossipsub |
|-------------------|-------------------------|-------------------------|------------------------------|
| 3.x (recommended) | 16.x or 17.x | 7.x or 8.x | 14.x |
| 2.x (avoid) | 15.x | 6.x | 13.x |

### GossipSub Topic Hash Compatibility

**Critical**: js-libp2p and rust-libp2p use the same topic hashing algorithm (SHA256 of topic string).

```typescript
// This produces the same TopicHash as rust-libp2p
const topic = '/nsn/video/1.0.0'  // Matches topics.rs VIDEO_CHUNKS_TOPIC
```

### SCALE Codec Compatibility

The `@polkadot/types-codec` library uses the same SCALE specification as `parity-scale-codec` in Rust. The VideoChunk struct encoding/decoding is byte-compatible.

**Verification**: Test with sample encoded chunks from the Rust side during integration.

---

## Full package.json Dependencies

```json
{
  "name": "nsn-libp2p-bridge",
  "type": "module",
  "engines": {
    "node": ">=20.0.0"
  },
  "dependencies": {
    "libp2p": "^3.1.2",
    "@libp2p/tcp": "^9.0.13",
    "@chainsafe/libp2p-noise": "^16.1.4",
    "@chainsafe/libp2p-yamux": "^7.0.1",
    "@chainsafe/libp2p-gossipsub": "^14.1.2",
    "@polkadot/types-codec": "^16.5.4",
    "@polkadot/types": "^16.5.4",
    "ws": "^8.19.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "@types/ws": "^8.5.10",
    "@types/node": "^20.10.0"
  }
}
```

---

## Sources

- [js-libp2p v3.0.0 Release Announcement](https://blog.libp2p.io/2025-09-30-js-libp2p/)
- [libp2p Multidimensional Interoperability Testing](https://blog.libp2p.io/multidim-interop/)
- [js-libp2p GitHub Repository](https://github.com/libp2p/js-libp2p)
- [@chainsafe/libp2p-gossipsub npm](https://www.npmjs.com/package/@chainsafe/libp2p-gossipsub)
- [@chainsafe/libp2p-noise npm](https://www.npmjs.com/package/@chainsafe/libp2p-noise)
- [@chainsafe/libp2p-yamux npm](https://www.npmjs.com/package/@chainsafe/libp2p-yamux)
- [@polkadot/types-codec npm](https://www.npmjs.com/package/@polkadot/types-codec)
- [ws WebSocket Library](https://www.npmjs.com/package/ws)
- [rust-libp2p Releases](https://github.com/libp2p/rust-libp2p/releases)
- [Yamux Documentation](https://docs.libp2p.io/concepts/multiplex/yamux/)
- [Noise Protocol Documentation](https://docs.rs/libp2p/latest/libp2p/noise/index.html)

---

*Research completed: 2026-01-18*
*Target milestone: v1.1 (Node.js libp2p bridge)*
