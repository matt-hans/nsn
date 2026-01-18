# Phase 3: Viewer Implementation - Research

**Researched:** 2026-01-18
**Domain:** Browser P2P networking (js-libp2p, WebRTC-Direct, GossipSub, SCALE decoding)
**Confidence:** HIGH

## Summary

This phase transforms the viewer from a simple-peer + signaling server architecture to a native libp2p node using WebRTC-Direct. The key change is eliminating the signaling server by using the WebRTC-Direct protocol, which embeds certificate hashes in multiaddrs for self-signed certificate validation.

The standard approach uses:
1. **js-libp2p v3.x** with `@libp2p/webrtc` for transport
2. **@chainsafe/libp2p-gossipsub** for pub/sub messaging
3. **@polkadot/types** with TypeRegistry for SCALE decoding
4. **HTTP discovery** via the `/p2p/info` endpoint (implemented in Phase 2)

**Primary recommendation:** Use `webRTCDirect()` transport (not `webRTC()`) since browsers dial servers directly without circuit relay. This eliminates SDP signaling complexity and provides direct browser-to-Rust-node connectivity.

## Standard Stack

The established libraries for browser-to-mesh P2P connectivity:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| libp2p | ^3.1.2 | Core libp2p framework | Official js-libp2p, EventTarget-based streams in v3 |
| @libp2p/webrtc | ^6.0.2 | WebRTC-Direct transport | Browser-to-server without signaling |
| @chainsafe/libp2p-noise | ^16.x | Connection encryption | Standard libp2p encryption |
| @chainsafe/libp2p-yamux | ^7.x | Stream multiplexing | Standard libp2p muxer |
| @chainsafe/libp2p-gossipsub | ^14.1.2 | Pub/sub protocol | GossipSub v1.1 implementation |
| @libp2p/identify | ^3.x | Peer identification | Required for protocol negotiation |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| @multiformats/multiaddr | ^13.0.1 | Multiaddr parsing | Parse `/p2p/info` response multiaddrs |
| @polkadot/types | ^15.x | SCALE codec | Decode VideoChunk binary data |
| @polkadot/types-codec | ^16.x | Codec primitives | Lower-level SCALE primitives |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| webRTCDirect | webRTC + circuitRelay | Requires relay infrastructure, adds latency |
| @polkadot/types | Manual SCALE decode | More work, error-prone field ordering |
| @chainsafe/libp2p-gossipsub | Direct stream protocol | Loses mesh topology, fan-out benefits |

**Installation:**
```bash
cd viewer
pnpm add libp2p @libp2p/webrtc @chainsafe/libp2p-noise @chainsafe/libp2p-yamux @chainsafe/libp2p-gossipsub @libp2p/identify @multiformats/multiaddr @polkadot/types @polkadot/types-codec
```

**Packages to Remove:**
```bash
pnpm remove simple-peer @types/simple-peer
```

## Architecture Patterns

### Recommended Project Structure
```
viewer/src/
├── services/
│   ├── p2pClient.ts          # NEW: libp2p node wrapper (replaces p2p.ts)
│   ├── discovery.ts          # NEW: HTTP discovery + multiaddr parsing
│   ├── videoCodec.ts         # NEW: SCALE VideoChunk decoder
│   ├── videoPipeline.ts      # EXISTING: Feed decoded chunks here
│   ├── videoBuffer.ts        # EXISTING: Buffer management
│   └── webcodecs.ts          # EXISTING: WebCodecs decoder
├── hooks/
│   └── useP2PConnection.ts   # NEW: React hook for P2P state
├── store/
│   └── appStore.ts           # EXISTING: Extend with connection state
└── components/
    └── NetworkStatus.tsx     # NEW: Connection indicator widget
```

### Pattern 1: P2PClient Service Class
**What:** Singleton service managing libp2p node lifecycle
**When to use:** Central P2P management, shared across components

```typescript
// Source: Based on js-libp2p v3 EventTarget streams pattern
// https://blog.libp2p.io/2025-09-30-js-libp2p/

import { createLibp2p, type Libp2p } from 'libp2p'
import { webRTCDirect } from '@libp2p/webrtc'
import { noise } from '@chainsafe/libp2p-noise'
import { yamux } from '@chainsafe/libp2p-yamux'
import { gossipsub } from '@chainsafe/libp2p-gossipsub'
import { identify } from '@libp2p/identify'
import { multiaddr } from '@multiformats/multiaddr'

export class P2PClient {
  private node: Libp2p | null = null;
  private videoTopic = '/nsn/video/1.0.0';

  async initialize(): Promise<void> {
    this.node = await createLibp2p({
      transports: [webRTCDirect()],
      connectionEncrypters: [noise()],
      streamMuxers: [yamux()],
      services: {
        identify: identify(),
        pubsub: gossipsub({
          emitSelf: false,
          fallbackToFloodsub: false,
        }),
      },
    });
  }

  async connectToNode(webrtcMultiaddr: string): Promise<void> {
    if (!this.node) throw new Error('Node not initialized');
    const ma = multiaddr(webrtcMultiaddr);
    await this.node.dial(ma);
  }

  subscribeToVideo(handler: (data: Uint8Array) => void): void {
    if (!this.node) throw new Error('Node not initialized');

    this.node.services.pubsub.subscribe(this.videoTopic);
    this.node.services.pubsub.addEventListener('message', (evt) => {
      if (evt.detail.topic === this.videoTopic) {
        handler(evt.detail.data);
      }
    });
  }
}
```

### Pattern 2: Discovery Service
**What:** Fetch and parse `/p2p/info`, extract WebRTC multiaddr
**When to use:** Bootstrap connection to Rust nodes

```typescript
// Source: Based on Phase 2 P2pInfoResponse format
// /home/matt/nsn/node-core/crates/p2p/src/discovery.rs

interface P2pInfoResponse {
  success: boolean;
  data?: {
    peer_id: string;
    multiaddrs: string[];
    protocols: string[];
    features: {
      webrtc_enabled: boolean;
      role: string;
    };
  };
  error?: {
    code: string;
    message: string;
  };
}

export async function discoverNode(baseUrl: string): Promise<string | null> {
  const response = await fetch(`${baseUrl}/p2p/info`);
  if (!response.ok) {
    if (response.status === 503) {
      // Node still initializing, can retry
      return null;
    }
    throw new Error(`Discovery failed: ${response.status}`);
  }

  const info: P2pInfoResponse = await response.json();
  if (!info.success || !info.data) {
    throw new Error(info.error?.message || 'Discovery failed');
  }

  // Find WebRTC-Direct multiaddr with certhash
  const webrtcAddr = info.data.multiaddrs.find(
    addr => addr.includes('/webrtc-direct/') && addr.includes('/certhash/')
  );

  if (!webrtcAddr) {
    throw new Error('No WebRTC address available');
  }

  // Append peer ID if not present
  if (!webrtcAddr.includes('/p2p/')) {
    return `${webrtcAddr}/p2p/${info.data.peer_id}`;
  }
  return webrtcAddr;
}
```

### Pattern 3: SCALE Codec for VideoChunk
**What:** Decode SCALE-encoded VideoChunk using @polkadot/types
**When to use:** Process GossipSub messages

```typescript
// Source: Based on VideoChunk definition in nsn-types
// /home/matt/nsn/node-core/crates/types/src/lib.rs:212-242

import { TypeRegistry, Struct, u16, u32, u64, Bool, Bytes, Vec } from '@polkadot/types';

// Create registry and register types ONCE at module load
const registry = new TypeRegistry();

registry.register({
  VideoChunkHeader: {
    version: 'u16',
    slot: 'u64',
    content_id: 'Text',
    chunk_index: 'u32',
    total_chunks: 'u32',
    timestamp_ms: 'u64',
    is_keyframe: 'bool',
    payload_hash: '[u8; 32]',
  },
  VideoChunk: {
    header: 'VideoChunkHeader',
    payload: 'Bytes',
    signer: 'Bytes',
    signature: 'Bytes',
  },
});

export interface DecodedVideoChunk {
  slot: bigint;
  chunkIndex: number;
  totalChunks: number;
  timestampMs: bigint;
  isKeyframe: boolean;
  payload: Uint8Array;
  contentId: string;
}

export function decodeVideoChunk(data: Uint8Array): DecodedVideoChunk {
  const chunk = registry.createType('VideoChunk', data);
  const json = chunk.toJSON() as any;

  return {
    slot: BigInt(json.header.slot),
    chunkIndex: json.header.chunk_index,
    totalChunks: json.header.total_chunks,
    timestampMs: BigInt(json.header.timestamp_ms),
    isKeyframe: json.header.is_keyframe,
    payload: new Uint8Array(json.payload),
    contentId: json.header.content_id,
  };
}
```

### Anti-Patterns to Avoid
- **Using webRTC() instead of webRTCDirect():** webRTC requires circuit relay for signaling; webRTCDirect dials servers directly
- **Forgetting peer ID in multiaddr:** Always append `/p2p/<peer_id>` for proper connection
- **Wrong SCALE field order:** Fields MUST match Rust struct order exactly
- **Creating TypeRegistry per decode:** Create once, reuse for all decodes

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SCALE decoding | Manual binary parsing | @polkadot/types TypeRegistry | Field ordering, nested types, endianness |
| Multiaddr parsing | String splitting | @multiformats/multiaddr | Protocol components, validation |
| WebRTC signaling | Custom WebSocket server | webRTCDirect (no signaling) | Certhash eliminates SDP exchange |
| GossipSub mesh | Manual peer management | @chainsafe/libp2p-gossipsub | Mesh topology, scoring, fan-out |
| Connection encryption | Raw DTLS | @chainsafe/libp2p-noise | Protocol negotiation, key exchange |
| Reconnection backoff | setTimeout chains | Built-in libp2p reconnection | Exponential backoff, jitter |

**Key insight:** The libp2p stack handles connection management, peer discovery within the mesh, and message routing. Focus on integration, not reimplementation.

## Common Pitfalls

### Pitfall 1: Missing Peer ID in Dial Multiaddr
**What goes wrong:** `dial()` fails with "Invalid multiaddr" or similar
**Why it happens:** WebRTC-Direct requires peer ID for connection authentication
**How to avoid:** Always construct multiaddr as `/ip4/.../udp/.../webrtc-direct/certhash/.../p2p/<peer_id>`
**Warning signs:** Connection timeout, no error but no connection

### Pitfall 2: SCALE Field Order Mismatch
**What goes wrong:** Decoded values are garbage or decode fails
**Why it happens:** SCALE is position-based, no field names in binary
**How to avoid:** Match TypeRegistry definition EXACTLY to Rust struct order
**Warning signs:** Incorrect slot numbers, corrupted payload

### Pitfall 3: GossipSub Subscription Before Connection
**What goes wrong:** No messages received despite being "subscribed"
**Why it happens:** GossipSub requires peers to propagate subscription state
**How to avoid:** Wait for connection established before subscribing; check peer count
**Warning signs:** `pubsub.getSubscribers(topic)` returns empty array

### Pitfall 4: Not Handling 503 from Discovery
**What goes wrong:** Application crashes or shows permanent error on startup
**Why it happens:** Rust node returns 503 while swarm is initializing (~5 seconds)
**How to avoid:** Implement retry with backoff on 503; check `Retry-After` header
**Warning signs:** Works sometimes, fails on cold start

### Pitfall 5: Certhash Changes After Node Restart
**What goes wrong:** Previously working multiaddr fails to connect
**Why it happens:** Ephemeral certificates regenerate on restart without persistence
**How to avoid:** This is handled on Rust side (Phase 1); viewer should re-fetch discovery
**Warning signs:** "Certificate mismatch" or DTLS handshake failure

### Pitfall 6: Browser WebRTC Limitations
**What goes wrong:** Node starts but cannot receive connections
**Why it happens:** Browsers cannot listen for incoming connections (no server ports)
**How to avoid:** Browser ONLY dials; never configure listen addresses
**Warning signs:** libp2p errors about binding ports

## Code Examples

Verified patterns from official sources and Phase 2 implementation:

### Complete P2P Client Initialization
```typescript
// Source: Combines patterns from js-libp2p docs and project requirements

import { createLibp2p } from 'libp2p'
import { webRTCDirect } from '@libp2p/webrtc'
import { noise } from '@chainsafe/libp2p-noise'
import { yamux } from '@chainsafe/libp2p-yamux'
import { gossipsub } from '@chainsafe/libp2p-gossipsub'
import { identify } from '@libp2p/identify'

export async function createP2PNode() {
  return await createLibp2p({
    // NO listen addresses - browsers cannot listen
    transports: [
      webRTCDirect(),
    ],
    connectionEncrypters: [noise()],
    streamMuxers: [yamux()],
    services: {
      identify: identify(),
      pubsub: gossipsub({
        emitSelf: false,
        fallbackToFloodsub: false,
        // Match Rust node config for interop
        signMessages: true,
        strictSigning: true,
      }),
    },
    connectionManager: {
      maxConnections: 10,
      minConnections: 1,
    },
  });
}
```

### Discovery with Parallel Race Pattern
```typescript
// Source: Based on 03-CONTEXT.md Discovery Behavior section

interface DiscoveryCandidate {
  url: string;
  source: 'localStorage' | 'settings' | 'env' | 'hardcoded';
}

const HARDCODED_DEFAULTS = [
  'http://bootstrap1.nsn.network:9615',
  'http://bootstrap2.nsn.network:9615',
  'http://bootstrap3.nsn.network:9615',
];

export async function discoverWithRace(
  candidates: DiscoveryCandidate[],
  batchSize = 3,
  timeoutMs = 3000
): Promise<string> {
  // Shuffle hardcoded defaults to avoid hammering first node
  const shuffled = [...candidates];
  const hardcodedStart = shuffled.findIndex(c => c.source === 'hardcoded');
  if (hardcodedStart >= 0) {
    const hardcoded = shuffled.splice(hardcodedStart);
    for (let i = hardcoded.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [hardcoded[i], hardcoded[j]] = [hardcoded[j], hardcoded[i]];
    }
    shuffled.push(...hardcoded);
  }

  for (let i = 0; i < shuffled.length; i += batchSize) {
    const batch = shuffled.slice(i, i + batchSize);

    const raceResult = await Promise.race([
      ...batch.map(c => discoverNode(c.url)),
      new Promise<null>((_, reject) =>
        setTimeout(() => reject(new Error('Batch timeout')), timeoutMs)
      ),
    ].map(p => p.catch(() => null)));

    if (raceResult) {
      // Save successful node to localStorage
      localStorage.setItem('last_known_node', batch.find(
        c => discoverNode(c.url).then(r => r === raceResult)
      )?.url || '');
      return raceResult;
    }
  }

  throw new Error('All discovery candidates failed');
}
```

### Integration with Existing Video Pipeline
```typescript
// Source: Adapts existing videoPipeline.ts interface

import { decodeVideoChunk, type DecodedVideoChunk } from './videoCodec';
import type { VideoPipeline } from './videoPipeline';

export function connectP2PToPipeline(
  p2pClient: P2PClient,
  pipeline: VideoPipeline
): void {
  p2pClient.subscribeToVideo((data: Uint8Array) => {
    try {
      const chunk = decodeVideoChunk(data);

      // Adapt to existing VideoChunkMessage interface
      pipeline.handleIncomingChunk({
        slot: Number(chunk.slot),
        chunk_index: chunk.chunkIndex,
        data: chunk.payload,
        timestamp: Number(chunk.timestampMs) * 1000, // ms to microseconds
        is_keyframe: chunk.isKeyframe,
      });
    } catch (err) {
      console.error('Failed to decode video chunk:', err);
    }
  });
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| simple-peer + signaling | webRTCDirect (no signaling) | 2024 | Eliminates signaling server |
| Streaming iterables | EventTarget streams | js-libp2p v3.0 (Sep 2025) | Breaking change in stream API |
| @libp2p/webrtc-direct (old pkg) | @libp2p/webrtc (webRTCDirect) | 2023 | Old package archived |
| floodsub | gossipsub v1.1 | 2020 | Better scaling, peer scoring |
| Manual SCALE parsing | @polkadot/types | N/A | Type safety, less error-prone |

**Deprecated/outdated:**
- `@libp2p/webrtc-direct` package: Archived, use `@libp2p/webrtc` with `webRTCDirect()` export
- `@libp2p/webrtc-star`: Requires centralized signaling server, deprecated
- Streaming iterables in js-libp2p: v3.0 moved to EventTarget-based streams
- libp2p v2.x protocol handlers: Signature changed in v3.0

## Open Questions

Things that couldn't be fully resolved:

1. **GossipSub Message Signing Interop**
   - What we know: Rust node uses `MessageAuthenticity::Signed`, js-libp2p gossipsub supports `signMessages: true`
   - What's unclear: Whether signature format is 100% compatible between rust-libp2p and js-libp2p
   - Recommendation: Enable signing on both sides, test early; if issues arise, check protobuf encoding

2. **Browser Bundle Size Impact**
   - What we know: libp2p + polkadot/types add significant JS bundle
   - What's unclear: Exact size impact, tree-shaking effectiveness
   - Recommendation: Measure after implementation; consider lazy loading P2P module

3. **WebRTC-Direct DTLS Handshake Failures**
   - What we know: Certhash validation should work per spec
   - What's unclear: Real-world reliability across browsers/firewalls
   - Recommendation: Implement comprehensive error handling; add firewall detection heuristic

## Sources

### Primary (HIGH confidence)
- [js-libp2p v3.0.0 Release Blog](https://blog.libp2p.io/2025-09-30-js-libp2p/) - EventTarget streams, API changes
- [WebRTC with js-libp2p Guide](https://docs.libp2p.io/guides/getting-started/webrtc/) - Transport configuration
- [js-libp2p/packages/transport-webrtc](https://github.com/libp2p/js-libp2p/tree/main/packages/transport-webrtc) - webRTCDirect usage
- [ChainSafe/js-libp2p-gossipsub README](https://github.com/ChainSafe/js-libp2p-gossipsub) - Configuration options
- [polkadot.js Type Creation Docs](https://polkadot.js.org/docs/api/start/types.create/) - TypeRegistry API
- [polkadot.js Extending Types Docs](https://polkadot.js.org/docs/api/start/types.extend/) - Custom struct registration

### Secondary (MEDIUM confidence)
- [WebRTC Browser-to-Server Blog](https://blog.libp2p.io/libp2p-webrtc-browser-to-server/) - Architecture overview
- [js-libp2p Discussion #2406](https://github.com/libp2p/js-libp2p/discussions/2406) - Certhash extraction
- [@multiformats/multiaddr npm](https://www.npmjs.com/package/@multiformats/multiaddr) - Version info

### Tertiary (LOW confidence)
- WebSearch results for interoperability issues - May need validation through testing

### Project Sources (HIGH confidence)
- `/home/matt/nsn/node-core/crates/p2p/src/discovery.rs` - P2pInfoResponse format
- `/home/matt/nsn/node-core/crates/types/src/lib.rs` - VideoChunk/VideoChunkHeader structs
- `/home/matt/nsn/.planning/phases/03-viewer-implementation/03-CONTEXT.md` - UX decisions

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Official docs, npm packages verified
- Architecture: HIGH - Based on js-libp2p v3 patterns and project context
- Pitfalls: MEDIUM - Some based on general libp2p knowledge, not all verified in this exact setup
- SCALE decoding: HIGH - @polkadot/types well-documented, struct format from project source

**Research date:** 2026-01-18
**Valid until:** 2026-02-18 (30 days - libp2p ecosystem relatively stable)
