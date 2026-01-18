# Roadmap: v1.1 Viewer Networking Integration (WebRTC-Direct)

**Milestone:** v1.1
**Created:** 2026-01-18
**Status:** Planning
**Approach:** WebRTC-Direct (browser connects directly to mesh nodes)

---

## Phase Overview

| Phase | Name | Status | Dependencies | Complexity | Plans |
|-------|------|--------|--------------|------------|-------|
| 1 | Rust Node Core Upgrade | ✅ Complete | - | Medium | 2 plans |
| 2 | Discovery Bridge (HTTP Sidecar) | Ready | Phase 1 | Low | TBD |
| 3 | Viewer Implementation | Pending | Phase 2 | Medium | TBD |
| 4 | Video Streaming Protocol | Pending | Phase 3 | High | TBD |
| 5 | Chain RPC Integration | Pending | - (parallel) | Medium | TBD |
| 6 | Docker & Operations | Pending | Phase 2 | Low | TBD |
| 7 | Testing & Validation | Pending | All phases | Medium | TBD |

---

## Phase 1: Rust Node Core Upgrade

**Goal:** Enable Director and Validator nodes to accept incoming WebRTC connections from browsers.

**Requirements:** REQ-WR-001 through REQ-WR-007

**Plans:** 2 plans

Plans:
- [x] 01-01-PLAN.md - Dependencies and certificate persistence
- [x] 01-02-PLAN.md - WebRTC transport integration and CLI

### Deliverables

1. Update `node-core/crates/p2p/Cargo.toml` with WebRTC feature
2. Certificate persistence module (`cert.rs`)
3. Hybrid transport configuration (TCP + WebRTC)
4. CLI flags for WebRTC configuration
5. Unit tests for certificate persistence

### 1.1 Dependency Update

Update `node-core/crates/p2p/Cargo.toml`:

```toml
[dependencies]
libp2p = { version = "0.53", features = [
    "tokio", "tcp", "noise", "yamux",
    "webrtc",  # NEW
    "dns", "macros", "gossipsub", "identify", "ping"
] }
pem = "3.0"  # Certificate serialization
```

### 1.2 Certificate Persistence

Create `node-core/crates/p2p/src/cert.rs`:

```rust
pub struct CertificateManager {
    data_dir: PathBuf,
}

impl CertificateManager {
    pub fn load_or_generate(&self) -> Result<Certificate, CertError>;
}
```

**Persistence logic:**
1. Check `{data_dir}/webrtc_cert.pem`
2. If missing, generate with `Certificate::generate()`
3. Save PEM to disk
4. If present, load from file

### 1.3 Transport Configuration

Modify `node-core/crates/p2p/src/transport.rs`:

```rust
let webrtc_cert = cert_manager.load_or_generate()?;

let transport = tcp_transport
    .or_transport(webrtc_transport(keypair, webrtc_cert))
    .boxed();
```

### 1.4 CLI Flags

Add to `node-core/bin/nsn-node/src/cli.rs`:

| Flag | Default | Description |
|------|---------|-------------|
| `--p2p-webrtc-port` | 9003 | UDP port for WebRTC connections |
| `--p2p-external-address` | None | Override announced address (for NAT) |

### Acceptance Criteria

- [x] `cargo build` succeeds with WebRTC feature
- [x] Certificate persists across node restarts
- [x] Node listens on `/ip4/0.0.0.0/udp/9003/webrtc`
- [x] Multiaddr logged includes `certhash`
- [x] Unit tests pass for certificate persistence

---

## Phase 2: Discovery Bridge (HTTP Sidecar)

**Goal:** Provide HTTP endpoint for browsers to discover WebRTC address.

**Requirements:** REQ-DISC-001 through REQ-DISC-006

### Deliverables

1. New endpoint `GET /p2p/info` on existing HTTP server
2. Response filtering (exclude internal IPs)
3. CORS headers for browser access
4. Integration with swarm state

### 2.1 Endpoint Implementation

Add to existing HTTP server (`node-core/bin/nsn-node/src/http.rs`):

```rust
#[derive(Serialize)]
struct P2pInfo {
    peer_id: String,
    multiaddrs: Vec<String>,
    protocols: Vec<String>,
}

async fn get_p2p_info(State(swarm): State<SharedSwarm>) -> Json<P2pInfo> {
    // Filter out 127.0.0.1, 172.x.x.x addresses
    // Include only WebRTC address with certhash
}
```

### 2.2 Response Format

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

### 2.3 CORS Configuration

```rust
.layer(CorsLayer::permissive())
```

### Acceptance Criteria

- [ ] `curl http://node:9615/p2p/info` returns valid JSON
- [ ] Response includes WebRTC address with certhash
- [ ] No internal Docker IPs in response
- [ ] Browser can fetch without CORS errors
- [ ] Same port as existing metrics endpoint

---

## Phase 3: Viewer Implementation

**Goal:** Transform viewer into a lightweight P2P node that connects directly to mesh.

**Requirements:** REQ-VI-001 through REQ-VI-009

### Deliverables

1. Install js-libp2p dependencies
2. P2P client service (`viewer/src/services/p2pClient.ts`)
3. Discovery fetch and multiaddr parsing
4. GossipSub subscription to video topic
5. SCALE decoder for VideoChunk
6. Connection status management
7. Reconnection with exponential backoff
8. Remove mock video stream

### 3.1 Dependencies

```bash
cd viewer
pnpm add libp2p @libp2p/webrtc @chainsafe/libp2p-noise @chainsafe/libp2p-gossipsub
pnpm add @polkadot/types-codec @polkadot/types @multiformats/multiaddr
```

### 3.2 P2P Client Service

```typescript
// viewer/src/services/p2pClient.ts

export class P2PClient {
  private node: Libp2p | null = null;
  private reconnectAttempts = 0;

  async connect(discoveryUrl: string): Promise<void> {
    // 1. Create libp2p node with WebRTC transport
    // 2. Fetch /p2p/info from discoveryUrl
    // 3. Parse WebRTC multiaddr with certhash
    // 4. Dial the Rust node
    // 5. Subscribe to /nsn/video/1.0.0
  }

  private handleVideoChunk(data: Uint8Array): void {
    // Decode SCALE-encoded VideoChunk
    // Feed to video pipeline
  }
}
```

### 3.3 SCALE Decoding

```typescript
import { TypeRegistry } from '@polkadot/types';

const registry = new TypeRegistry();
registry.register({
  VideoChunkHeader: { version: 'u16', slot: 'u64', /* ... */ },
  VideoChunk: { header: 'VideoChunkHeader', payload: 'Bytes', /* ... */ }
});

function decodeVideoChunk(data: Uint8Array): VideoChunk {
  return registry.createType('VideoChunk', data).toJSON();
}
```

### 3.4 Connection State Management

Update `viewer/src/store/appStore.ts`:

```typescript
interface ConnectionState {
  status: 'disconnected' | 'connecting' | 'connected' | 'error';
  peerId: string | null;
  connectedAt: number | null;
  error: string | null;
}
```

### 3.5 Remove Mock Video

Delete or disable:
- Mock video generator in `p2p.ts`
- Hardcoded video chunks
- Fake peer connections

### Acceptance Criteria

- [ ] libp2p node initializes in browser
- [ ] Fetch `/p2p/info` succeeds
- [ ] WebRTC connection established to Rust node
- [ ] GossipSub subscription active
- [ ] SCALE decoding works for VideoChunk
- [ ] Video renders from mesh data
- [ ] Mock video completely removed
- [ ] Reconnection works after disconnect

---

## Phase 4: Video Streaming Protocol

**Goal:** Establish reliable video chunk delivery from mesh to browser.

**Requirements:** REQ-VI-004, REQ-VI-005, REQ-VI-006, REQ-LS-001, REQ-LS-002

### Deliverables

1. GossipSub message handler in viewer
2. VideoChunk parsing and validation
3. Integration with existing video pipeline
4. Bitrate and latency calculation
5. Buffer management

### 4.1 GossipSub Handler

```typescript
node.services.pubsub.addEventListener('message', (evt) => {
  if (evt.detail.topic === '/nsn/video/1.0.0') {
    const chunk = decodeVideoChunk(evt.detail.data);
    this.processChunk(chunk);
  }
});
```

### 4.2 Chunk Processing

```typescript
private processChunk(chunk: VideoChunk): void {
  // 1. Validate chunk (hash, signature)
  // 2. Check if this slot is relevant
  // 3. Feed to video buffer
  // 4. Update statistics
  this.statsCalculator.recordChunk(chunk);
  this.videoBuffer.append(chunk);
}
```

### 4.3 Statistics Calculation

```typescript
export class StatsCalculator {
  private chunks: { time: number; size: number }[] = [];

  recordChunk(chunk: VideoChunk): void {
    this.chunks.push({ time: Date.now(), size: chunk.payload.length });
    this.prune();
  }

  getBitrateMbps(): number {
    // Calculate from last N seconds of chunks
  }

  getLatencyMs(): number {
    // chunk.header.timestamp_ms vs now
  }
}
```

### Acceptance Criteria

- [ ] Chunks received via GossipSub
- [ ] SCALE decoding successful
- [ ] Video pipeline receives chunks
- [ ] Canvas renders video frames
- [ ] Bitrate displayed accurately
- [ ] Latency calculated from timestamps

---

## Phase 5: Chain RPC Integration

**Goal:** Query NSN chain for director and epoch information.

**Requirements:** REQ-RPC-001 through REQ-RPC-006

**Note:** Can run in parallel with Phases 3-4.

### Deliverables

1. Chain client service (`viewer/src/services/chainClient.ts`)
2. Type definitions for nsn-director pallet
3. React hook `useChainData()`
4. Error handling and reconnection
5. State caching

### 5.1 Chain Client

```typescript
import { ApiPromise, WsProvider } from '@polkadot/api';

export class ChainClient {
  private api: ApiPromise | null = null;

  async connect(endpoint: string): Promise<void>;
  async getCurrentEpoch(): Promise<Epoch | null>;
  async getElectedDirectors(slot: number): Promise<AccountId[]>;
  async subscribeToBlocks(callback: BlockCallback): Promise<Unsubscribe>;
}
```

### 5.2 React Hook

```typescript
export function useChainData() {
  const [epoch, setEpoch] = useState<Epoch | null>(null);
  const [directors, setDirectors] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // ...
}
```

### Acceptance Criteria

- [ ] Connects to chain RPC endpoint
- [ ] Queries CurrentEpoch successfully
- [ ] Queries ElectedDirectors
- [ ] Block subscription works
- [ ] Graceful error handling
- [ ] Caches last known state

---

## Phase 6: Docker & Operations

**Goal:** Update deployment configuration for WebRTC support.

**Requirements:** REQ-NF-009 through REQ-NF-012

### Deliverables

1. Update `docker-compose.yml` with UDP port
2. Environment variable configuration
3. Data volume for certificate persistence
4. Updated documentation

### 6.1 Docker Compose Update

```yaml
services:
  director-alice:
    ports:
      - "9001:9001"       # TCP (mesh)
      - "9003:9003/udp"   # WebRTC (browsers) - NEW
      - "9615:9615"       # HTTP (discovery, metrics)
    environment:
      - P2P_WEBRTC_PORT=9003
      - P2P_EXTERNAL_ADDRESS=/ip4/${HOST_IP}/udp/9003/webrtc
    volumes:
      - alice-data:/data  # Preserves certificate
```

### 6.2 Firewall Configuration

```bash
# Required for WebRTC
ufw allow 9003/udp
```

### Acceptance Criteria

- [ ] UDP port 9003 exposed in Docker
- [ ] Environment variables configure WebRTC
- [ ] Certificate persists in volume
- [ ] External address configuration works

---

## Phase 7: Testing & Validation

**Goal:** End-to-end validation of WebRTC-direct viewer connectivity.

**Requirements:** All

### Deliverables

1. Integration tests for certificate persistence
2. E2E test: browser → WebRTC → node → GossipSub
3. Connection recovery tests
4. Performance validation
5. Updated documentation

### Test Scenarios

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| T-001 | Certificate persists across restart | Same certhash after restart |
| T-002 | Discovery endpoint returns valid data | JSON with WebRTC multiaddr |
| T-003 | Browser connects via WebRTC | Connection established |
| T-004 | GossipSub subscription works | Chunks received in browser |
| T-005 | SCALE decoding succeeds | VideoChunk parsed correctly |
| T-006 | Video renders | Canvas displays frames |
| T-007 | Statistics are real | No mock values |
| T-008 | Reconnection works | Connection restored after drop |
| T-009 | Chain RPC queries work | Epoch/directors displayed |
| T-010 | Full testnet E2E | Video streams end-to-end |

### Acceptance Criteria

- [ ] All integration tests pass
- [ ] E2E test demonstrates full flow
- [ ] Recovery tests pass
- [ ] Performance meets requirements
- [ ] Documentation updated

---

## Parallel Work Streams

```
        Phase 1: Rust Node Core Upgrade
                    │
                    ▼
        Phase 2: Discovery Bridge
                    │
        ┌───────────┼───────────┐
        ▼           │           ▼
    Phase 6:        │       Phase 5:
    Docker/Ops      │       Chain RPC
                    ▼           │
        Phase 3: Viewer         │
                    │           │
                    ▼           │
        Phase 4: Video          │
                Streaming       │
                    │           │
                    └─────┬─────┘
                          ▼
                Phase 7: Testing
```

**Critical Path:** Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 7

**Parallelizable:**
- Phase 5 (Chain RPC) can start after Phase 2
- Phase 6 (Docker) can start after Phase 2

---

## Implementation Checklist

| Step | Component | Task | Complexity | Status |
|------|-----------|------|------------|--------|
| 1.1 | Rust | Update Cargo.toml with WebRTC feature | Low | ✅ Done |
| 1.2 | Rust | Implement certificate persistence | Medium | ✅ Done |
| 1.3 | Rust | Add WebRTC transport to swarm | Medium | ✅ Done |
| 1.4 | Rust | Add CLI flags | Low | ✅ Done |
| 2.1 | Rust | Add `/p2p/info` endpoint | Low | Pending |
| 2.2 | Rust | Configure CORS | Low | Pending |
| 3.1 | JS | Install libp2p dependencies | Low | Pending |
| 3.2 | JS | Implement P2PClient | Medium | Pending |
| 3.3 | JS | Implement SCALE decoder | Medium | Pending |
| 3.4 | JS | Remove mock video | Low | Pending |
| 4.1 | JS | GossipSub handler | Medium | Pending |
| 4.2 | JS | Video pipeline integration | High | Pending |
| 4.3 | JS | Statistics calculator | Medium | Pending |
| 5.1 | JS | Chain RPC client | Medium | Pending |
| 6.1 | Ops | Update Docker Compose | Low | Pending |
| 7.1 | Test | E2E validation | Medium | Pending |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| WebRTC interop issues | Medium | High | Test early with minimal setup |
| Certificate persistence fails | Low | High | Unit tests, integration tests |
| NAT traversal problems | Medium | Medium | `--p2p-external-address` flag |
| SCALE decode errors | Low | High | Type registry from metadata |
| Browser compatibility | Low | Medium | Test major browsers early |

---

## Definition of Done

Milestone v1.1 is complete when:

1. Rust nodes accept WebRTC connections from browsers
2. Certificate persists across restarts (stable certhash)
3. Discovery endpoint returns valid WebRTC multiaddr
4. Viewer connects directly to mesh via libp2p WebRTC
5. GossipSub delivers video chunks to browser
6. SCALE-encoded chunks decoded correctly
7. Video renders without mock data
8. Statistics display real network data
9. Chain queries return director information
10. All E2E tests pass reliably
11. Documentation updated

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-18 | Initial roadmap (Node.js bridge approach) |
| 2.0 | 2026-01-18 | Restructured for WebRTC-direct approach |
| 2.1 | 2026-01-18 | Phase 1 planned: 2 plans in 2 waves |
| 2.2 | 2026-01-18 | Phase 1 complete: WebRTC transport and CLI |

---

*Roadmap v2.2 - WebRTC-Direct approach*
