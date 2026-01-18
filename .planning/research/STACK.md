# Stack Research: WebRTC-Direct Browser-to-Mesh Connectivity

## Executive Summary

This document specifies the technology stack for enabling direct WebRTC connections between browsers and the NSN Rust libp2p mesh. The approach eliminates the need for a separate bridge service by adding WebRTC transport to existing nodes.

**Reliability Assessment**:
- rust-libp2p WebRTC: **MEDIUM-HIGH** (stable in 0.53, production-ready)
- js-libp2p WebRTC: **HIGH** (well-tested, browser-native)
- Certificate persistence: **HIGH** (standard pattern)
- Cross-implementation interop: **MEDIUM** (requires testing)

---

## Recommended Stack

### Rust (node-core)

| Component | Library | Version | Rationale |
|-----------|---------|---------|-----------|
| **Core libp2p** | `libp2p` | `0.53.x` | Current version in use, WebRTC support included |
| **WebRTC Transport** | `libp2p::webrtc` | (bundled) | Server-side WebRTC for browser connections |
| **TCP Transport** | `libp2p::tcp` | (bundled) | Mesh interconnect (existing) |
| **Noise Encryption** | `libp2p::noise` | (bundled) | Secure channel (existing) |
| **Yamux Muxer** | `libp2p::yamux` | (bundled) | Stream multiplexing (existing) |
| **HTTP Server** | `axum` or `warp` | `^0.7` / `^0.3` | Discovery endpoint `/p2p/info` |

### Browser (viewer)

| Component | Library | Version | Rationale |
|-----------|---------|---------|-----------|
| **Core libp2p** | `libp2p` | `^2.0.0` | Latest stable, TypeScript-first |
| **WebRTC Transport** | `@libp2p/webrtc` | `^5.0.0` | Browser WebRTC via RTCPeerConnection |
| **Noise Encryption** | `@chainsafe/libp2p-noise` | `^16.1.0` | Cross-implementation compatible |
| **GossipSub** | `@chainsafe/libp2p-gossipsub` | `^14.1.0` | Video chunk subscription |
| **SCALE Codec** | `@polkadot/types-codec` | `^16.5.0` | Decode VideoChunk from mesh |
| **Multiaddr** | `@multiformats/multiaddr` | `^12.0.0` | Parse multiaddrs from discovery |
| **Chain RPC** | `@polkadot/api` | `^16.x` | Director/epoch queries |

---

## Rust Dependencies

### Cargo.toml Configuration

```toml
[dependencies]
# libp2p with WebRTC feature enabled
libp2p = { version = "0.53", features = [
    "tokio",
    "tcp",
    "noise",
    "yamux",
    "webrtc",      # NEW: Enable WebRTC transport
    "dns",
    "macros",
    "gossipsub",
    "identify",
    "ping",
] }

# WebRTC-specific dependencies (pulled in by feature, but may need explicit)
rcgen = "0.12"           # Certificate generation
webrtc = "0.9"           # WebRTC implementation (via libp2p)

# HTTP server for discovery endpoint
axum = "0.7"             # OR warp = "0.3"
tower-http = { version = "0.5", features = ["cors"] }

# Certificate persistence
pem = "3.0"              # PEM encoding/decoding
```

### WebRTC Feature Notes

- `libp2p = { features = ["webrtc"] }` enables server-side WebRTC
- This is different from the deprecated `webrtc-direct` feature
- WebRTC transport uses DTLS for encryption (built-in, no separate Noise layer)
- Certificate is self-signed, hash embedded in multiaddr

---

## Browser Dependencies

### package.json

```json
{
  "name": "nsn-viewer",
  "type": "module",
  "dependencies": {
    "libp2p": "^2.0.0",
    "@libp2p/webrtc": "^5.0.0",
    "@chainsafe/libp2p-noise": "^16.1.0",
    "@chainsafe/libp2p-gossipsub": "^14.1.0",
    "@polkadot/types-codec": "^16.5.0",
    "@polkadot/types": "^16.5.0",
    "@polkadot/api": "^16.0.0",
    "@multiformats/multiaddr": "^12.0.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "@types/node": "^20.10.0"
  }
}
```

### Browser Compatibility

| Browser | WebRTC Support | Notes |
|---------|---------------|-------|
| Chrome 90+ | Full | RTCPeerConnection, DataChannels |
| Firefox 85+ | Full | RTCPeerConnection, DataChannels |
| Safari 15+ | Full | RTCPeerConnection, DataChannels |
| Edge 90+ | Full | Chromium-based |
| Mobile Chrome | Full | Android 5+ |
| Mobile Safari | Full | iOS 14.5+ |

**Note:** Older browsers without WebRTC DataChannel support cannot connect.

---

## Certificate Persistence

### Why Persistence Matters

WebRTC multiaddrs include a certificate hash (`certhash`):
```
/ip4/192.168.1.50/udp/9003/webrtc/certhash/uEiD...
```

If the certificate regenerates on restart:
1. Old `certhash` becomes invalid
2. Cached addresses in browsers fail
3. Discovery endpoint returns new hash
4. All clients must re-fetch discovery

### Implementation Pattern

```rust
// node-core/crates/p2p/src/cert.rs

use libp2p::webrtc::tokio::Certificate;
use pem::Pem;
use std::path::PathBuf;

pub struct CertificateManager {
    data_dir: PathBuf,
}

impl CertificateManager {
    const CERT_FILE: &'static str = "webrtc_cert.pem";

    pub fn new(data_dir: PathBuf) -> Self {
        Self { data_dir }
    }

    pub fn load_or_generate(&self) -> Result<Certificate, CertError> {
        let path = self.data_dir.join(Self::CERT_FILE);

        if path.exists() {
            self.load_from_file(&path)
        } else {
            let cert = self.generate_and_save(&path)?;
            Ok(cert)
        }
    }

    fn load_from_file(&self, path: &PathBuf) -> Result<Certificate, CertError> {
        let pem_str = std::fs::read_to_string(path)?;
        Certificate::from_pem(&pem_str)
            .map_err(|e| CertError::ParseError(e.to_string()))
    }

    fn generate_and_save(&self, path: &PathBuf) -> Result<Certificate, CertError> {
        let cert = Certificate::generate(&mut rand::thread_rng())?;
        let pem_str = cert.serialize_pem();
        std::fs::write(path, &pem_str)?;
        tracing::info!("Generated new WebRTC certificate: {:?}", path);
        Ok(cert)
    }
}
```

### Certificate Lifecycle

1. **First run:** Generate certificate, save to `{data_dir}/webrtc_cert.pem`
2. **Subsequent runs:** Load existing certificate
3. **Manual rotation:** Delete file, restart node
4. **Docker:** Mount data volume to persist across container restarts

---

## Discovery Endpoint

### HTTP Server Integration

Add to existing metrics/RPC HTTP server:

```rust
// node-core/bin/nsn-node/src/http.rs

use axum::{routing::get, Router, Json};
use tower_http::cors::CorsLayer;

#[derive(Serialize)]
struct P2pInfo {
    peer_id: String,
    multiaddrs: Vec<String>,
    protocols: Vec<String>,
}

async fn get_p2p_info(
    State(swarm): State<SharedSwarm>,
) -> Json<P2pInfo> {
    let swarm = swarm.lock().await;

    Json(P2pInfo {
        peer_id: swarm.local_peer_id().to_string(),
        multiaddrs: swarm
            .listeners()
            .filter(|a| !a.to_string().contains("127.0.0.1"))
            .filter(|a| !a.to_string().contains("172."))
            .map(|a| a.to_string())
            .collect(),
        protocols: vec![
            "/nsn/video/1.0.0".into(),
            "/meshsub/1.1.0".into(),
        ],
    })
}

pub fn create_router(swarm: SharedSwarm) -> Router {
    Router::new()
        .route("/p2p/info", get(get_p2p_info))
        .route("/health", get(health_check))
        .route("/metrics", get(metrics))
        .layer(CorsLayer::permissive())  // Allow browser fetch
        .with_state(swarm)
}
```

### CORS Configuration

Browser fetch requires CORS headers:

```rust
use tower_http::cors::{CorsLayer, Any};

let cors = CorsLayer::new()
    .allow_origin(Any)
    .allow_methods([Method::GET])
    .allow_headers(Any);
```

---

## Transport Stack Configuration

### Hybrid Transport (TCP + WebRTC)

```rust
// node-core/crates/p2p/src/transport.rs

use libp2p::{
    core::{transport::OrTransport, upgrade::Version},
    identity::Keypair,
    noise, tcp, yamux,
    webrtc::tokio::{Certificate, Transport as WebRtcTransport},
    PeerId, Transport,
};

pub fn build_transport(
    keypair: &Keypair,
    webrtc_cert: Certificate,
) -> impl Transport<Output = (PeerId, impl StreamMuxer)> + Clone {
    // TCP for mesh-to-mesh communication
    let tcp = tcp::tokio::Transport::new(tcp::Config::default())
        .upgrade(Version::V1)
        .authenticate(noise::Config::new(keypair).expect("noise config"))
        .multiplex(yamux::Config::default());

    // WebRTC for browser connections
    // Note: WebRTC uses DTLS internally, no separate noise layer needed
    let webrtc = WebRtcTransport::new(keypair.clone(), webrtc_cert);

    // Combined transport
    OrTransport::new(tcp, webrtc)
}
```

### Listening Configuration

```rust
// In swarm initialization
let mut swarm = Swarm::new(transport, behaviour, local_peer_id, config);

// TCP for mesh (existing)
swarm.listen_on("/ip4/0.0.0.0/tcp/9001".parse()?)?;

// WebRTC for browsers (new)
swarm.listen_on("/ip4/0.0.0.0/udp/9003/webrtc".parse()?)?;

// After listening, log the full address with certhash
for addr in swarm.listeners() {
    tracing::info!("Listening on: {}", addr);
}
```

---

## Browser libp2p Configuration

### Full Configuration

```typescript
// viewer/src/services/p2pClient.ts

import { createLibp2p, Libp2p } from 'libp2p'
import { webRTC } from '@libp2p/webrtc'
import { noise } from '@chainsafe/libp2p-noise'
import { gossipsub } from '@chainsafe/libp2p-gossipsub'
import { multiaddr } from '@multiformats/multiaddr'
import { TypeRegistry } from '@polkadot/types'
import type { VideoChunk } from './types'

// SCALE type registry for VideoChunk decoding
const registry = new TypeRegistry()
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

export async function createP2PNode(): Promise<Libp2p> {
  return await createLibp2p({
    transports: [
      webRTC({
        // Browser-side WebRTC configuration
        rtcConfiguration: {
          iceServers: [
            // Only needed for NAT traversal in production
            // { urls: 'stun:stun.l.google.com:19302' }
          ]
        }
      })
    ],
    connectionEncrypters: [noise()],
    services: {
      pubsub: gossipsub({
        emitSelf: false,
        globalSignaturePolicy: 'StrictSign',
        allowPublishToZeroTopicPeers: false,
        // Match NSN mesh parameters
        D: 6,
        Dlo: 4,
        Dhi: 12,
      })
    }
  })
}

export function decodeVideoChunk(data: Uint8Array): VideoChunk {
  return registry.createType('VideoChunk', data).toJSON() as VideoChunk
}
```

---

## Avoid

### Libraries NOT to Use

| Library | Reason |
|---------|--------|
| `@libp2p/websockets` | Not needed for WebRTC-direct approach |
| `@libp2p/tcp` | Browser cannot use TCP |
| `@libp2p/mplex` | Deprecated, use WebRTC native channels |
| `simple-peer` | Replaced by @libp2p/webrtc |
| `socket.io` | Not needed, using libp2p protocols |
| `ws` (WebSocket library) | Only needed for chain RPC, not P2P |

### Patterns to Avoid

1. **Generating certificates on every restart** - breaks certhash stability
2. **Announcing Docker internal IPs** - browsers can't reach 172.x.x.x
3. **Skipping CORS headers** - browser fetch will fail
4. **Mixing noise with WebRTC** - WebRTC uses DTLS, noise causes double encryption
5. **Hardcoding multiaddrs in viewer** - use discovery endpoint

---

## Version Compatibility Matrix

### rust-libp2p ↔ js-libp2p WebRTC Interop

| rust-libp2p | js-libp2p | @libp2p/webrtc | Status |
|-------------|-----------|----------------|--------|
| 0.53.x | 2.0.x | 5.x | Compatible |
| 0.52.x | 1.x | 4.x | Compatible |
| < 0.52 | * | * | Not supported |

### GossipSub Compatibility

Both implementations use GossipSub 1.1 spec. Topic hashing is consistent.

**Topic string:** `/nsn/video/1.0.0` produces identical topic hash on both sides.

---

## Network Configuration

### Docker Compose

```yaml
# docker/testnet/docker-compose.yml

services:
  director-alice:
    ports:
      - "9001:9001"     # TCP (mesh)
      - "9003:9003/udp" # WebRTC (browsers) - NEW
      - "9615:9615"     # HTTP (discovery, metrics)
    environment:
      - P2P_WEBRTC_PORT=9003
      - P2P_EXTERNAL_ADDRESS=/ip4/${HOST_IP}/udp/9003/webrtc
```

### Firewall Rules

```bash
# Allow WebRTC UDP
ufw allow 9003/udp

# Allow discovery HTTP
ufw allow 9615/tcp
```

---

## Sources

- [rust-libp2p WebRTC Documentation](https://docs.rs/libp2p/0.53/libp2p/webrtc/index.html)
- [js-libp2p WebRTC](https://github.com/libp2p/js-libp2p/tree/main/packages/transport-webrtc)
- [libp2p WebRTC Spec](https://github.com/libp2p/specs/tree/master/webrtc)
- [@polkadot/types-codec](https://www.npmjs.com/package/@polkadot/types-codec)
- [GossipSub Spec](https://github.com/libp2p/specs/tree/master/pubsub/gossipsub)

---

*Research updated: 2026-01-18*
*Approach: WebRTC-Direct (supersedes Node.js bridge)*
