# Architecture Research: libp2p-to-Browser Video Bridge

## Component Diagram

```
                              NSN ARCHITECTURE
+============================================================================+
|                                                                            |
|   NSN MESH (Rust/libp2p)              VIDEO BRIDGE (Rust)                  |
|   +-------------------------+         +-----------------------------+      |
|   |                         |         |                             |      |
|   |  GossipSub              |         |  MeshSubscriber             |      |
|   |  /nsn/video/1.0.0       |-------->|  (libp2p client)            |      |
|   |                         |  SCALE  |                             |      |
|   |  VideoChunk {           |         |  ChunkTranslator            |      |
|   |    header: {            |         |  (SCALE -> Binary)          |      |
|   |      version, slot,     |         |                             |      |
|   |      content_id,        |         |  WebSocketServer            |      |
|   |      chunk_index,       |         |  (tokio-tungstenite)        |      |
|   |      total_chunks,      |         |                             |      |
|   |      timestamp_ms,      |         +-----------------------------+      |
|   |      is_keyframe,       |                    |                         |
|   |      payload_hash       |                    | Binary [slot:4]         |
|   |    }                    |                    |        [chunk_index:4]  |
|   |    payload: Vec<u8>     |                    |        [timestamp:8]    |
|   |    signer, signature    |                    |        [is_keyframe:1]  |
|   |  }                      |                    |        [data:rest]      |
|   |                         |                    v                         |
|   +-------------------------+         +-----------------------------+      |
|                                       |  BROWSER VIEWERS            |      |
|                                       |                             |      |
|   NSN CHAIN (Substrate)               |  WebSocket Client           |      |
|   +-------------------------+         |  VideoBuffer                |      |
|   |  nsn-director pallet    |         |  WebCodecs Decoder          |      |
|   |                         |         |  Canvas Renderer            |      |
|   |  Storage:               |         |                             |      |
|   |  - current_epoch()      |<--------|  Chain RPC (optional)       |      |
|   |  - next_epoch_directors |  JSON   |  Director status queries    |      |
|   |  - elected_directors(s) |  RPC    |                             |      |
|   |                         |         +-----------------------------+      |
|   +-------------------------+                                              |
|                                                                            |
+============================================================================+
```

## Video Bridge Design

### Connection to Mesh

The video bridge connects to the NSN P2P mesh as a **subscriber-only node**. It does not need to publish video chunks or participate in BFT consensus.

**Implementation approach:**

1. **Use existing P2P service** (`node-core/crates/p2p/src/service.rs`)
   - Create a minimal configuration with only VideoChunks topic subscription
   - Reuse `P2pService::subscribe_video_latency()` broadcast channel for chunk notifications
   - Or create a new `ServiceCommand::SubscribeVideoStream` that returns a `broadcast::Receiver<VideoChunk>`

2. **Connection flow:**
   ```
   1. Bridge starts P2pService with reduced config:
      - No publishing capability needed
      - Subscribe only to TopicCategory::VideoChunks
      - Use mDNS for local discovery (testnet) or bootstrap peers (mainnet)

   2. On GossipSub message receipt:
      - decode_video_chunk() parses SCALE-encoded VideoChunk
      - verify_video_chunk() validates signature and payload hash
      - Chunk forwarded to WebSocket broadcast
   ```

3. **Key files to integrate with:**
   - `/home/matt/nsn/node-core/crates/p2p/src/video.rs` - chunk decoding/verification
   - `/home/matt/nsn/node-core/crates/p2p/src/topics.rs` - VIDEO_CHUNKS_TOPIC constant
   - `/home/matt/nsn/node-core/crates/types/src/lib.rs` - VideoChunk, VideoChunkHeader types

**Mesh subscription pseudocode:**
```rust
// In video_bridge crate
use nsn_p2p::{P2pService, ServiceCommand, TopicCategory};
use nsn_types::VideoChunk;

async fn subscribe_to_mesh(p2p_service: &P2pService) -> broadcast::Receiver<VideoChunk> {
    // Service already handles GossipSub events
    // Extend handle_gossipsub_event() to forward decoded chunks
    p2p_service.subscribe_video_chunks()
}
```

### WebSocket Server

The bridge exposes a WebSocket server for browser connections.

**Protocol design:**

1. **Connection endpoint:** `ws://bridge-host:9090/video`

2. **Message types (server -> client):**
   ```
   Binary frame: Video chunk
   [slot:4 bytes, big-endian u32]
   [chunk_index:4 bytes, big-endian u32]
   [timestamp:8 bytes, big-endian u64, ms since epoch]
   [is_keyframe:1 byte, 0x00 or 0x01]
   [data: remaining bytes, raw video payload]

   Text frame: Control message (JSON)
   {
     "type": "epoch_update",
     "epoch_id": 42,
     "directors": ["5GrwvaEF...", "5FHneW46..."],
     "slot": 1234
   }
   ```

3. **Message types (client -> server):**
   ```
   Text frame: Subscription control (JSON)
   {
     "type": "subscribe",
     "slot": 1234  // optional: subscribe to specific slot
   }

   {
     "type": "unsubscribe"
   }
   ```

4. **Implementation with tokio-tungstenite:**
   ```rust
   use tokio_tungstenite::{accept_async, tungstenite::Message};
   use futures::{SinkExt, StreamExt};

   async fn handle_ws_connection(
       ws_stream: WebSocketStream<TcpStream>,
       mut chunk_rx: broadcast::Receiver<VideoChunk>,
   ) {
       let (mut ws_sender, mut ws_receiver) = ws_stream.split();

       loop {
           tokio::select! {
               // Forward chunks to browser
               Ok(chunk) = chunk_rx.recv() => {
                   let binary = translate_chunk_to_browser_format(&chunk);
                   ws_sender.send(Message::Binary(binary)).await?;
               }
               // Handle browser messages
               Some(msg) = ws_receiver.next() => {
                   // Process subscription control
               }
           }
       }
   }
   ```

### Message Translation

Translation between SCALE-encoded mesh format and browser binary format.

**SCALE VideoChunk (mesh):**
```rust
// From nsn_types::VideoChunk
pub struct VideoChunk {
    pub header: VideoChunkHeader,  // SCALE-encoded struct
    pub payload: Vec<u8>,          // Raw video bytes
    pub signer: Vec<u8>,           // libp2p public key (protobuf)
    pub signature: Vec<u8>,        // Ed25519 signature
}

pub struct VideoChunkHeader {
    pub version: u16,
    pub slot: u64,
    pub content_id: String,
    pub chunk_index: u32,
    pub total_chunks: u32,
    pub timestamp_ms: u64,
    pub is_keyframe: bool,
    pub payload_hash: [u8; 32],  // Blake3
}
```

**Browser binary format (existing viewer expectation):**
```typescript
// From viewer/src/services/p2p.ts
// [slot:4][chunk_index:4][timestamp:8][is_keyframe:1][data:rest]
const CHUNK_HEADER_SIZE = 17;
```

**Translation function:**
```rust
fn translate_chunk_to_browser_format(chunk: &VideoChunk) -> Vec<u8> {
    let header_size = 17;
    let total_size = header_size + chunk.payload.len();
    let mut buffer = Vec::with_capacity(total_size);

    // slot: 4 bytes, big-endian u32 (truncate from u64)
    buffer.extend_from_slice(&(chunk.header.slot as u32).to_be_bytes());

    // chunk_index: 4 bytes, big-endian u32
    buffer.extend_from_slice(&chunk.header.chunk_index.to_be_bytes());

    // timestamp: 8 bytes, big-endian u64
    buffer.extend_from_slice(&chunk.header.timestamp_ms.to_be_bytes());

    // is_keyframe: 1 byte
    buffer.push(if chunk.header.is_keyframe { 1 } else { 0 });

    // data: raw payload
    buffer.extend_from_slice(&chunk.payload);

    buffer
}
```

**Note on slot truncation:** The mesh uses `u64` for slot numbers, but the browser format uses `u32`. This is acceptable for the foreseeable future (u32 max = 4.2 billion slots, at 8 blocks/slot and 6s/block = ~800 years).

## Chain RPC Integration

### Connection

Browser viewers can optionally query the NSN chain for director information.

**RPC endpoint:** `ws://chain-host:9944` (standard Substrate JSON-RPC)

**Connection options:**

1. **Direct from browser** (simple but exposes RPC):
   ```typescript
   import { ApiPromise, WsProvider } from '@polkadot/api';

   const provider = new WsProvider('ws://localhost:9944');
   const api = await ApiPromise.create({ provider });
   ```

2. **Via bridge proxy** (recommended for production):
   - Bridge queries chain and pushes updates via WebSocket control messages
   - Reduces browser complexity and RPC exposure
   - Bridge can cache and batch queries

### Queries

**Current epoch directors:**
```typescript
// Storage: CurrentEpoch<T>
const epoch = await api.query.nsnDirector.currentEpoch();
if (epoch.isSome) {
    const { id, directors, status, start_block, end_block } = epoch.unwrap();
    console.log(`Epoch ${id}: ${directors.length} directors`);
}
```

**Next epoch directors (On-Deck):**
```typescript
// Storage: NextEpochDirectors<T>
const nextDirectors = await api.query.nsnDirector.nextEpochDirectors();
console.log(`On-Deck: ${nextDirectors.length} directors`);
```

**Elected directors for specific slot:**
```typescript
// Storage: ElectedDirectors<T> - Map<slot, Vec<AccountId>>
const slotDirectors = await api.query.nsnDirector.electedDirectors(slotNumber);
```

**Current slot:**
```typescript
// Storage: CurrentSlot<T>
const currentSlot = await api.query.nsnDirector.currentSlot();
```

**Subscribe to epoch changes (recommended):**
```typescript
// Subscribe to CurrentEpoch storage changes
const unsub = await api.query.nsnDirector.currentEpoch((epoch) => {
    if (epoch.isSome) {
        const data = epoch.unwrap();
        notifyViewers({ type: 'epoch_update', ...data });
    }
});
```

## Build Order

The components have the following dependencies:

```
1. Types crate (shared types)
   - No dependencies on other new components
   - Exports: BridgeChunk, ControlMessage, SubscriptionRequest

2. Mesh Subscriber
   - Depends on: nsn-p2p, nsn-types
   - Produces: broadcast::Sender<VideoChunk>

3. Chunk Translator
   - Depends on: nsn-types, types crate
   - Pure function, no runtime dependencies

4. WebSocket Server
   - Depends on: types crate, chunk translator
   - Consumes: broadcast::Receiver<VideoChunk>
   - Produces: WebSocket connections

5. Chain RPC Client (optional for bridge)
   - Depends on: subxt or substrate-api-client
   - Can be built independently

6. Viewer Updates (existing code)
   - Minor changes to accept WebSocket instead of WebRTC
   - Replace simple-peer with native WebSocket
```

**Recommended build order:**

| Phase | Component | Depends On | Deliverable |
|-------|-----------|------------|-------------|
| 1 | **Types crate** | - | `video-bridge-types` crate with shared message definitions |
| 2 | **Chunk Translator** | Types | `translate_chunk_to_browser_format()` function with tests |
| 3 | **Mesh Subscriber** | nsn-p2p, nsn-types | `MeshSubscriber` that outputs `broadcast::Receiver<VideoChunk>` |
| 4 | **WebSocket Server** | Types, Translator | `BridgeServer` that accepts WS connections and forwards chunks |
| 5 | **Integration** | All above | `video-bridge` binary combining subscriber + server |
| 6 | **Viewer WebSocket mode** | Types | Update `viewer/src/services/p2p.ts` to use WebSocket |
| 7 | **Chain RPC (bridge-side)** | subxt | Optional: Bridge queries chain, pushes epoch updates |
| 8 | **Chain RPC (viewer-side)** | @polkadot/api | Optional: Direct chain queries from viewer |

**Critical path:** Phases 1-5 are blocking for MVP. Phases 6-8 can be parallelized once Phase 4 is complete.

## Key Implementation Notes

1. **Signature verification in bridge:** The bridge should verify chunk signatures before forwarding to browsers. This prevents malicious mesh nodes from injecting invalid chunks. Use `nsn_p2p::verify_video_chunk()`.

2. **Slot filtering:** The bridge may receive chunks for multiple slots (during transitions). It should either:
   - Forward all chunks and let browser filter
   - Accept slot subscription from browser and filter server-side

3. **Connection limits:** The WebSocket server should limit concurrent connections to prevent resource exhaustion. Use `tokio::sync::Semaphore` or similar.

4. **Backpressure:** If a browser cannot keep up with chunk rate, the bridge should either drop older chunks or disconnect slow clients rather than buffering indefinitely.

5. **Metrics:** Add Prometheus metrics for:
   - Chunks received from mesh
   - Chunks forwarded to browsers
   - Active WebSocket connections
   - Translation latency

6. **Error handling:** Browser disconnects should not affect other connections or mesh subscription. Use isolated tasks per connection.
