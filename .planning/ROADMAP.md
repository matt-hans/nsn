# Roadmap: v1.1 Viewer Networking Integration

**Milestone:** v1.1
**Created:** 2026-01-18
**Status:** Planning

---

## Phase Overview

| Phase | Name | Status | Dependencies |
|-------|------|--------|--------------|
| 1 | Video Bridge Core | Pending | - |
| 2 | WebSocket Server | Pending | Phase 1 |
| 3 | Viewer WebSocket Client | Pending | Phase 2 |
| 4 | Chain RPC Integration | Pending | - |
| 5 | Live Statistics | Pending | Phase 3, Phase 4 |
| 6 | Docker Integration | Pending | Phase 2 |
| 7 | Testing & Validation | Pending | All phases |

---

## Phase 1: Video Bridge Core

**Goal:** Create Rust crate that subscribes to NSN mesh and receives video chunks.

**Requirements:** REQ-VB-001, REQ-VB-002, REQ-VB-003, REQ-VB-004

### Deliverables

1. `video-bridge` crate in `node-core/crates/`
2. MeshSubscriber component using existing `nsn_p2p::P2pService`
3. ChunkTranslator: SCALE VideoChunk → 17-byte binary format
4. Unit tests for chunk translation
5. Integration test with mock GossipSub

### Technical Approach

```
node-core/crates/video-bridge/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── subscriber.rs    # MeshSubscriber
│   ├── translator.rs    # ChunkTranslator
│   └── config.rs        # BridgeConfig
```

**Key interfaces:**
```rust
pub struct MeshSubscriber {
    p2p_service: P2pService,
    chunk_tx: broadcast::Sender<VideoChunk>,
}

pub fn translate_chunk(chunk: &VideoChunk) -> Vec<u8>;
```

### Acceptance Criteria

- [ ] Crate compiles with `cargo build -p video-bridge`
- [ ] Unit tests pass for chunk translation
- [ ] MeshSubscriber receives chunks from local mesh
- [ ] Translated chunks match expected 17-byte header format

---

## Phase 2: WebSocket Server

**Goal:** Add WebSocket server to video bridge for browser connections.

**Requirements:** REQ-VB-005, REQ-VB-006, REQ-VB-007, REQ-VB-008, REQ-VB-009, REQ-VB-010

### Deliverables

1. WebSocket server using `tokio-tungstenite`
2. Connection management with client tracking
3. Health HTTP endpoint (`/health`)
4. Binary message forwarding
5. Connection limit enforcement

### Technical Approach

```rust
pub struct BridgeServer {
    chunk_rx: broadcast::Receiver<VideoChunk>,
    clients: Arc<RwLock<HashMap<ClientId, WebSocketSink>>>,
    config: BridgeConfig,
}

impl BridgeServer {
    pub async fn run(&self, addr: SocketAddr) -> Result<()>;
    async fn handle_connection(&self, stream: TcpStream);
    async fn broadcast_chunk(&self, chunk: Vec<u8>);
}
```

### Acceptance Criteria

- [ ] WebSocket server accepts connections on configured port
- [ ] `/health` returns 200 when mesh connected
- [ ] Chunks forwarded to all connected clients
- [ ] Client disconnect doesn't affect other clients
- [ ] Connection limit enforced (configurable)

---

## Phase 3: Viewer WebSocket Client

**Goal:** Update viewer to receive video from bridge instead of mock data.

**Requirements:** REQ-VI-001, REQ-VI-002, REQ-VI-003, REQ-VI-004, REQ-VI-005, REQ-VI-006

### Deliverables

1. WebSocket client in `viewer/src/services/bridgeClient.ts`
2. Chunk parser for 17-byte header format
3. Integration with existing video pipeline
4. Reconnection logic with exponential backoff
5. Connection status in Zustand store
6. Remove mock video stream code

### Technical Approach

```typescript
// viewer/src/services/bridgeClient.ts
export class BridgeClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;

  connect(url: string): void;
  disconnect(): void;
  private handleMessage(data: ArrayBuffer): void;
  private parseChunk(buffer: ArrayBuffer): VideoChunk;
  private scheduleReconnect(): void;
}

interface VideoChunk {
  slot: number;
  chunkIndex: number;
  timestamp: bigint;
  isKeyframe: boolean;
  data: Uint8Array;
}
```

### Acceptance Criteria

- [ ] Viewer connects to bridge WebSocket
- [ ] Chunks parsed correctly (17-byte header)
- [ ] Video renders from real mesh data
- [ ] Mock video stream removed
- [ ] Auto-reconnection works with backoff
- [ ] Connection status displayed in UI

---

## Phase 4: Chain RPC Integration

**Goal:** Query NSN chain for director information.

**Requirements:** REQ-RPC-001, REQ-RPC-002, REQ-RPC-003, REQ-RPC-004, REQ-RPC-005, REQ-RPC-006

### Deliverables

1. Chain client in `viewer/src/services/chainClient.ts`
2. Type definitions for nsn-director pallet queries
3. React hook `useChainData()` for component access
4. Error handling and reconnection
5. Caching of last known state

### Technical Approach

```typescript
// viewer/src/services/chainClient.ts
import { ApiPromise, WsProvider } from '@polkadot/api';

export class ChainClient {
  private api: ApiPromise | null = null;

  async connect(endpoint: string): Promise<void>;
  async getCurrentEpoch(): Promise<Epoch | null>;
  async getElectedDirectors(slot: number): Promise<AccountId[]>;
  async subscribeToBlocks(callback: (block: Block) => void): Promise<() => void>;
}

// viewer/src/hooks/useChainData.ts
export function useChainData() {
  const [epoch, setEpoch] = useState<Epoch | null>(null);
  const [directors, setDirectors] = useState<AccountId[]>([]);
  // ...
}
```

### Acceptance Criteria

- [ ] Connects to chain RPC endpoint
- [ ] Queries CurrentEpoch successfully
- [ ] Queries ElectedDirectors for slot
- [ ] Block subscription works
- [ ] Handles connection errors gracefully
- [ ] Caches last known state

---

## Phase 5: Live Statistics

**Goal:** Replace mock statistics with real network data.

**Requirements:** REQ-LS-001, REQ-LS-002, REQ-LS-003, REQ-LS-004, REQ-LS-005, REQ-LS-006

### Deliverables

1. Bitrate calculation from chunk data
2. Latency calculation from timestamps
3. Buffer level tracking
4. Director info from chain queries
5. Updated stats display components
6. Remove all mock stat values

### Technical Approach

```typescript
// viewer/src/services/statsCalculator.ts
export class StatsCalculator {
  private chunkSizes: { time: number; size: number }[] = [];
  private latencies: number[] = [];

  recordChunk(chunk: VideoChunk): void;
  getBitrateMbps(): number;
  getAverageLatencyMs(): number;
}

// Update appStore.ts to use real values
interface NetworkStats {
  bitrate: number;        // Real Mbps from chunks
  latency: number;        // Real ms from timestamps
  connectedPeers: number; // From bridge or estimate
  bufferSeconds: number;  // From video pipeline
  directorPeerId: string; // From chain query
}
```

### Acceptance Criteria

- [ ] Bitrate calculated from actual chunk throughput
- [ ] Latency measured from chunk timestamps
- [ ] Buffer level reflects actual buffered video
- [ ] Director info from chain (not hardcoded)
- [ ] All mock values removed
- [ ] Stats update smoothly (1-2 second intervals)

---

## Phase 6: Docker Integration

**Goal:** Add video bridge to testnet Docker Compose.

**Requirements:** REQ-NF-007, REQ-NF-008, REQ-NF-009

### Deliverables

1. Dockerfile for video-bridge
2. Docker Compose service definition
3. Environment variable configuration
4. Health check integration
5. Network configuration for mesh access

### Technical Approach

```dockerfile
# docker/video-bridge/Dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release -p video-bridge

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/video-bridge /usr/local/bin/
EXPOSE 9090 9091
HEALTHCHECK CMD curl -f http://localhost:9091/health || exit 1
ENTRYPOINT ["video-bridge"]
```

```yaml
# docker/testnet/docker-compose.yml (addition)
video-bridge:
  build:
    context: ../..
    dockerfile: docker/video-bridge/Dockerfile
  environment:
    - MESH_BOOTNODES=/ip4/sidecar-alice/tcp/4001
    - WS_PORT=9090
    - HEALTH_PORT=9091
  ports:
    - "9090:9090"
  depends_on:
    sidecar-alice:
      condition: service_healthy
  networks:
    - nsn-testnet
```

### Acceptance Criteria

- [ ] Dockerfile builds successfully
- [ ] Bridge runs in Docker Compose testnet
- [ ] Health check passes when mesh connected
- [ ] Environment variables configure ports and bootnodes
- [ ] Bridge accessible from host at localhost:9090

---

## Phase 7: Testing & Validation

**Goal:** End-to-end validation of viewer networking.

**Requirements:** All REQ-* requirements

### Deliverables

1. Integration tests for video bridge
2. E2E test: mesh → bridge → viewer
3. Connection failure recovery tests
4. Performance validation (latency, throughput)
5. Updated documentation

### Test Scenarios

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| T-001 | Bridge connects to mesh | Peer connection established |
| T-002 | Bridge receives chunks | Chunks logged/counted |
| T-003 | Browser connects to bridge | WebSocket open event |
| T-004 | Viewer receives chunks | Chunks parsed correctly |
| T-005 | Video renders | Canvas displays frames |
| T-006 | Stats display real values | No mock data visible |
| T-007 | Bridge reconnects on peer loss | Service restored < 30s |
| T-008 | Viewer reconnects on WS close | Connection restored |
| T-009 | Chain queries work | Epoch/director data displayed |
| T-010 | Full testnet E2E | Video streams end-to-end |

### Acceptance Criteria

- [ ] All integration tests pass
- [ ] E2E test demonstrates full flow
- [ ] Recovery tests pass
- [ ] Performance meets REQ-NF-001, REQ-NF-003
- [ ] Documentation updated

---

## Parallel Work Streams

```
                    Phase 1: Bridge Core
                           │
                           ▼
                    Phase 2: WebSocket Server ──────┐
                           │                        │
              ┌────────────┴────────────┐           │
              ▼                         ▼           ▼
    Phase 3: Viewer WS       Phase 4: Chain RPC    Phase 6: Docker
              │                         │
              └────────────┬────────────┘
                           ▼
                    Phase 5: Live Stats
                           │
                           ▼
                    Phase 7: Testing
```

**Critical Path:** Phase 1 → Phase 2 → Phase 3 → Phase 5 → Phase 7

**Parallelizable:** Phase 4 and Phase 6 can run alongside Phase 3

---

## Risk Mitigation

| Risk | Mitigation | Phase |
|------|------------|-------|
| libp2p interop issues | Verify with isolated test first | Phase 1 |
| SCALE decode failures | Unit tests with real mesh data | Phase 1 |
| WebSocket stability | Implement heartbeat, test reconnection | Phase 2, 3 |
| Chain RPC availability | Graceful degradation, caching | Phase 4 |

---

## Definition of Done

Milestone v1.1 is complete when:

1. Video bridge runs in Docker Compose testnet
2. Viewer displays real video from mesh (no mocks)
3. Statistics show real network data
4. Chain queries return director information
5. All acceptance criteria for all phases met
6. E2E test passes reliably
7. Documentation updated

---

*Roadmap v1.0 - Ready for phase planning*
