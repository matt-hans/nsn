# Architecture Research: WebRTC-Direct Browser-to-Mesh Connectivity

## Component Diagram

```
                              NSN ARCHITECTURE (WebRTC-Direct)
+====================================================================================+
|                                                                                    |
|   NSN MESH (Rust/libp2p 0.53)                                                      |
|   +----------------------------------+                                             |
|   |                                  |                                             |
|   |  Director/Validator Nodes        |                                             |
|   |  ┌────────────────────────────┐  |                                             |
|   |  │ Transport Stack:           │  |                                             |
|   |  │ - TCP/Noise/Yamux (mesh)   │  |                                             |
|   |  │ - WebRTC (browser clients) │  |                                             |
|   |  └────────────────────────────┘  |                                             |
|   |                                  |                                             |
|   |  Listening Addresses:            |                                             |
|   |  /ip4/0.0.0.0/tcp/9001           |  ← Mesh interconnect                        |
|   |  /ip4/0.0.0.0/udp/9003/webrtc/   |  ← Browser clients                          |
|   |        certhash/<HASH>           |                                             |
|   |                                  |                                             |
|   +----------------------------------+                                             |
|              │                    │                                                |
|              │ GossipSub          │ HTTP /p2p/info                                 |
|              │ /nsn/video/1.0.0   │ (Discovery)                                    |
|              │                    │                                                |
|              ▼                    ▼                                                |
|   +----------------------------------+         +-----------------------------+     |
|   |  VideoChunk Distribution         |         |  Discovery Bridge           |     |
|   |  (SCALE-encoded via GossipSub)   |         |  HTTP Server (:9615)        |     |
|   +----------------------------------+         |                             |     |
|                                                |  GET /p2p/info              |     |
|                                                |  → { peer_id, multiaddrs }  |     |
|                                                +-----------------------------+     |
|                                                             │                      |
|                                                             │ JSON                 |
|                                                             ▼                      |
|                                                +-----------------------------+     |
|   NSN CHAIN (Substrate)                        |  BROWSER VIEWER             |     |
|   +-------------------------+                  |  (js-libp2p + @libp2p/webrtc)|    |
|   |  nsn-director pallet    |                  |                             |     |
|   |                         |                  |  1. Fetch /p2p/info         |     |
|   |  Storage:               |                  |  2. Parse WebRTC multiaddr  |     |
|   |  - current_epoch()      |<-----------------|  3. Dial node directly      |     |
|   |  - next_epoch_directors |     JSON RPC     |  4. Subscribe GossipSub     |     |
|   |  - elected_directors(s) |                  |  5. Receive VideoChunks     |     |
|   |                         |                  |                             |     |
|   +-------------------------+                  +-----------------------------+     |
|                                                                                    |
+====================================================================================+
```

## Key Architecture Changes (vs Bridge Approach)

| Aspect | Bridge Approach | WebRTC-Direct Approach |
|--------|-----------------|------------------------|
| Browser connectivity | Via WebSocket relay | Direct libp2p WebRTC |
| Video chunk delivery | Bridge translates SCALE→binary | Browser decodes SCALE directly |
| Discovery | Hardcoded bridge URL | HTTP `/p2p/info` endpoint |
| Certificate management | N/A (TLS on bridge) | Persisted WebRTC cert with stable certhash |
| Network hops | Browser→Bridge→Mesh | Browser→Mesh (direct) |
| Latency | +10-50ms per hop | Minimal (direct connection) |
| Infrastructure | Separate bridge service | Integrated in existing nodes |

## Certificate Persistence (Critical)

### The Problem

WebRTC transport generates a self-signed certificate. The certificate's hash (`certhash`) is embedded in the multiaddr:

```
/ip4/192.168.1.50/udp/9003/webrtc/certhash/uEiD...
```

If the certificate regenerates on node restart, the `certhash` changes, breaking all client connections that cached the old address.

### The Solution

Persist the WebRTC certificate to disk, similar to PeerId keypair persistence.

**Implementation:**

```rust
// node-core/crates/p2p/src/cert.rs

use libp2p::webrtc::tokio::Certificate;
use std::path::Path;

const CERT_FILENAME: &str = "webrtc_cert.pem";

pub fn load_or_generate_cert(data_dir: &Path) -> Result<Certificate, Error> {
    let cert_path = data_dir.join(CERT_FILENAME);

    if cert_path.exists() {
        // Load existing certificate
        let pem = std::fs::read_to_string(&cert_path)?;
        Certificate::from_pem(&pem)
    } else {
        // Generate new certificate and save
        let cert = Certificate::generate(&mut rand::thread_rng())?;
        let pem = cert.serialize_pem();
        std::fs::write(&cert_path, &pem)?;
        Ok(cert)
    }
}
```

**Storage location:** `{data_dir}/webrtc_cert.pem`

## Discovery Bridge Design

### Why HTTP Discovery?

Browsers cannot:
- Scan UDP ports to find WebRTC endpoints
- Query mDNS/DHT without libp2p initialization
- Know the `certhash` without connecting first (chicken-egg problem)

The HTTP discovery endpoint solves this by providing a stable, fetchable URL.

### Endpoint Specification

**URL:** `GET http://{node-ip}:9615/p2p/info`

**Response:**
```json
{
  "peer_id": "12D3KooWExample...",
  "multiaddrs": [
    "/ip4/192.168.1.50/tcp/9001",
    "/ip4/192.168.1.50/udp/9003/webrtc/certhash/uEiD..."
  ],
  "protocols": [
    "/nsn/video/1.0.0",
    "/meshsub/1.1.0"
  ]
}
```

**Critical filtering:** The response MUST include the WebRTC address with `certhash`. Internal Docker IPs (172.x.x.x) should be filtered out; use `--p2p-external-address` to announce public IPs.

### Integration Point

Add to existing HTTP server in `node-core` (same server used for metrics/RPC):

```rust
// In HTTP server handler
async fn handle_p2p_info(swarm: &Swarm) -> impl Reply {
    let peer_id = swarm.local_peer_id().to_string();
    let multiaddrs: Vec<String> = swarm
        .listeners()
        .map(|a| a.to_string())
        .collect();

    json!({
        "peer_id": peer_id,
        "multiaddrs": multiaddrs,
        "protocols": swarm.behaviour().protocols()
    })
}
```

## Transport Configuration

### Hybrid Transport Stack

Nodes must support both TCP (mesh interconnect) and WebRTC (browser clients):

```rust
// node-core/crates/p2p/src/transport.rs

use libp2p::{
    core::Transport,
    noise, yamux, tcp, dns,
    webrtc::tokio::{Transport as WebRtcTransport, Config as WebRtcConfig},
};

pub fn build_transport(
    keypair: &Keypair,
    webrtc_cert: Certificate,
) -> Boxed<(PeerId, StreamMuxerBox)> {
    // TCP transport for mesh interconnect
    let tcp = tcp::tokio::Transport::new(tcp::Config::default())
        .upgrade(Version::V1)
        .authenticate(noise::Config::new(keypair).unwrap())
        .multiplex(yamux::Config::default());

    // WebRTC transport for browser clients
    let webrtc = WebRtcTransport::new(
        keypair.clone(),
        webrtc_cert,
    );

    // DNS resolution wrapper
    let transport = dns::tokio::Transport::system(
        tcp.or_transport(webrtc)
    ).unwrap();

    transport.boxed()
}
```

### Listening Addresses

```rust
// Configure swarm to listen on both transports
swarm.listen_on("/ip4/0.0.0.0/tcp/9001".parse()?)?;  // Mesh
swarm.listen_on("/ip4/0.0.0.0/udp/9003/webrtc".parse()?)?;  // Browsers
```

## Browser Connection Flow

### Hybrid Handshake Sequence

```
Browser                           HTTP Server                      Rust Node (libp2p)
   │                                   │                                   │
   │  1. fetch('/p2p/info')            │                                   │
   │──────────────────────────────────>│                                   │
   │                                   │                                   │
   │  2. { peer_id, multiaddrs }       │                                   │
   │<──────────────────────────────────│                                   │
   │                                   │                                   │
   │  3. Parse WebRTC addr with certhash                                   │
   │                                   │                                   │
   │  4. node.dial(multiaddr)          │                                   │
   │───────────────────────────────────────────────────────────────────────>│
   │                                   │              WebRTC/DTLS handshake │
   │<───────────────────────────────────────────────────────────────────────│
   │                                   │                                   │
   │  5. Connection established!                                           │
   │                                   │                                   │
   │  6. Subscribe to /nsn/video/1.0.0 topic                               │
   │───────────────────────────────────────────────────────────────────────>│
   │                                   │                                   │
   │  7. Receive VideoChunk messages (SCALE-encoded)                       │
   │<───────────────────────────────────────────────────────────────────────│
```

### Browser Implementation

```typescript
// viewer/src/services/p2pClient.ts

import { createLibp2p } from 'libp2p'
import { webRTC } from '@libp2p/webrtc'
import { noise } from '@chainsafe/libp2p-noise'
import { gossipsub } from '@chainsafe/libp2p-gossipsub'
import { multiaddr } from '@multiformats/multiaddr'

export class P2PClient {
  private node: Libp2p | null = null;

  async connect(discoveryUrl: string): Promise<void> {
    // 1. Initialize libp2p node
    this.node = await createLibp2p({
      transports: [webRTC()],
      connectionEncrypters: [noise()],
      services: {
        pubsub: gossipsub({
          emitSelf: false,
          globalSignaturePolicy: 'StrictSign',
        }),
      },
    });

    await this.node.start();

    // 2. Fetch discovery info
    const response = await fetch(`${discoveryUrl}/p2p/info`);
    const info = await response.json();

    // 3. Find WebRTC address with certhash
    const webrtcAddr = info.multiaddrs.find(
      (a: string) => a.includes('/webrtc/')
    );
    if (!webrtcAddr) {
      throw new Error('No WebRTC address available');
    }

    // 4. Dial the node
    await this.node.dial(multiaddr(webrtcAddr));

    // 5. Subscribe to video topic
    this.node.services.pubsub.subscribe('/nsn/video/1.0.0');
    this.node.services.pubsub.addEventListener('message', (evt) => {
      if (evt.detail.topic === '/nsn/video/1.0.0') {
        this.handleVideoChunk(evt.detail.data);
      }
    });
  }

  private handleVideoChunk(data: Uint8Array): void {
    // Decode SCALE-encoded VideoChunk
    // ... (use @polkadot/types-codec)
  }
}
```

## Video Chunk Protocol

### Option 1: GossipSub (Recommended)

Browser subscribes to `/nsn/video/1.0.0` topic and receives same VideoChunk messages as mesh nodes.

**Pros:**
- Reuses existing protocol
- Same data path as validators
- No custom protocol needed

**Cons:**
- Browser receives ALL chunks (filtering needed)
- May receive chunks from multiple directors

### Option 2: Request-Response Protocol

Custom protocol for on-demand video retrieval.

**Protocol ID:** `/nsn/video-stream/1.0.0`

**Request:** `{ slot: u64 }`
**Response:** Stream of video bytes

**Use case:** VOD playback, seeking to specific slots.

### Recommendation

Use GossipSub for live streaming (primary use case), add Request-Response later for VOD functionality.

## Build Order (Updated)

| Phase | Component | Depends On | Deliverable |
|-------|-----------|------------|-------------|
| 1 | WebRTC Transport | - | Certificate persistence, hybrid transport |
| 2 | Discovery Endpoint | Phase 1 | HTTP `/p2p/info` endpoint |
| 3 | Browser P2P Client | Phase 2 | js-libp2p WebRTC connection |
| 4 | Video Streaming | Phase 3 | GossipSub subscription, SCALE decode |
| 5 | Chain RPC | - | @polkadot/api integration (parallel) |
| 6 | Docker/Ops | Phase 2 | UDP port exposure, configs |
| 7 | Testing | All | E2E validation |

**Critical path:** Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 7

## Key Implementation Notes

### NAT Traversal

1. **Public nodes (Directors/Validators):** Must announce public IP via `--p2p-external-address`
2. **Docker:** Map UDP port 9003 to host
3. **Firewall:** Ensure UDP 9003 is open

### Browser Security

- `certhash` in multiaddr allows encrypted connection without CA-signed cert
- Self-signed cert is acceptable for P2P (no central authority needed)
- CORS headers needed on `/p2p/info` endpoint

### Video Codec Compatibility

Browser requires H.264/MP4. If Vortex outputs raw video:
- Add FFmpeg transcoding step in sidecar
- Or ensure Vortex outputs browser-compatible format

### Error Handling

```typescript
// Browser should handle connection failures gracefully
try {
  await p2pClient.connect('http://node:9615');
} catch (err) {
  if (err.message.includes('No WebRTC address')) {
    // Node doesn't support WebRTC yet
    showError('Node upgrade required');
  } else if (err.message.includes('dial failed')) {
    // Network/NAT issue
    showError('Connection failed - check network');
  }
}
```

## Comparison: Bridge vs WebRTC-Direct

| Metric | Bridge | WebRTC-Direct |
|--------|--------|---------------|
| Latency | +10-50ms | Minimal |
| Scalability | Bridge is bottleneck | Distributed |
| Infrastructure | Extra service | Integrated |
| Complexity | Moderate | Higher (initial) |
| Maintenance | Two systems | One system |
| Browser support | All browsers | Modern browsers |
| NAT traversal | Bridge handles | Requires STUN/TURN for edge cases |

**Recommendation:** WebRTC-Direct is preferred for production. Higher initial complexity pays off with better latency, scalability, and reduced infrastructure.

---

*Architecture updated: 2026-01-18*
*Approach: WebRTC-Direct (supersedes Node.js bridge)*
