# Features Research

**Research Date:** 2026-01-18
**Milestone:** v1.1 Viewer Networking Integration
**Purpose:** Define feature scope for video bridge service, chain RPC client, and live statistics display

---

## Video Bridge Service

The video bridge connects the NSN libp2p mesh (Rust nodes) to browser viewers via WebSocket. Browsers cannot speak libp2p directly, so a Node.js bridge using js-libp2p subscribes to GossipSub and relays video chunks to WebSocket clients.

### Table Stakes

| Feature | Rationale | Evidence |
|---------|-----------|----------|
| **GossipSub subscription to `/nsn/video/1.0.0`** | Lane 0 video chunks flow through this topic | `node-core/crates/p2p/src/topics.rs:15` |
| **SCALE decode VideoChunk to simple binary** | Viewer uses 17-byte header format, mesh uses SCALE-encoded VideoChunk with signature | `viewer/src/services/p2p.ts:46-47`, `node-core/crates/types/src/lib.rs:231-242` |
| **WebSocket server for browser connections** | Browser cannot connect to libp2p directly | `PROJECT.md:86-92` architecture diagram |
| **Chunk forwarding with preserved header fields** | slot, chunk_index, timestamp, is_keyframe must survive translation | `viewer/src/services/p2p.ts:249-261` parse logic |
| **Connection to at least one mesh peer** | Bridge must bootstrap into the NSN mesh to receive chunks | `docker/testnet/docker-compose.yml:99` bootnodes pattern |
| **Health endpoint** | Required for Docker healthcheck and monitoring | `docker/testnet/docker-compose.yml:346-350` |
| **Graceful shutdown** | Clean peer disconnection and resource release | Production deployment requirement |

### Differentiators

| Feature | Rationale | Priority |
|---------|-----------|----------|
| **Multi-topic subscription** | Subscribe to recipes + attestations for richer UI context | Medium |
| **Chunk caching/deduplication** | Avoid sending duplicate chunks to the same WebSocket client | Medium |
| **Client-side topic filtering** | Let clients subscribe to specific slots/directors | Low |
| **Backpressure handling** | Slow clients should not block bridge or other clients | Medium |
| **Peer discovery status broadcast** | Inform clients of mesh health via WebSocket control messages | Low |
| **Signature verification** | Verify VideoChunk signatures before forwarding (security hardening) | Low - validation happens on mesh |
| **Rate limiting per client** | Prevent single client from overwhelming the bridge | Medium |
| **TLS support** | wss:// for production deployments | Medium (required for prod) |

### Anti-features

| Anti-feature | Reason to NOT Build |
|--------------|---------------------|
| **Full libp2p-in-browser via WebRTC** | rust-libp2p-webrtc is alpha (0.9.0-alpha.1), explicitly deferred per `PROJECT.md:64` |
| **Chunk generation/publishing** | Bridge is read-only; directors publish chunks via Rust mesh |
| **BFT consensus participation** | Bridge is a passive relay, not a validator or director |
| **Persistent storage of chunks** | Viewer handles buffering; bridge is stateless relay |
| **Authentication/authorization** | Testnet is permissionless; auth layer is out of scope |
| **Video transcoding** | Chunks are already encoded; viewer decodes via WebCodecs |
| **Custom binary protocol over WebSocket** | JSON control + binary chunks is sufficient for MVP |

---

## Chain RPC Client

Browser queries the NSN chain via @polkadot/api to discover elected directors, epoch state, and network information. This replaces hardcoded mock data with live chain state.

### Table Stakes

| Feature | Rationale | Evidence |
|---------|-----------|----------|
| **Connect to chain RPC endpoint** | Query on-chain storage via WebSocket | `docker/testnet/docker-compose.yml:55` exposes RPC at 9944 |
| **Query `NsnDirector::CurrentEpoch`** | Get active epoch ID, start/end blocks, director list | `nsn-chain/pallets/nsn-director/src/lib.rs:245-249` |
| **Query `NsnDirector::ElectedDirectors(slot)`** | Get director AccountIds for current/future slots | `nsn-chain/pallets/nsn-director/src/lib.rs:177-185` |
| **Query `NsnDirector::NextEpochDirectors`** | Get On-Deck directors for upcoming epoch | `nsn-chain/pallets/nsn-director/src/lib.rs:256-258` |
| **Query `NsnReputation::ReputationScores(account)`** | Get director reputation for display | `node-core/crates/chain-client/src/lib.rs:206-222` |
| **Subscribe to new blocks** | Trigger UI updates on epoch transitions | Standard @polkadot/api pattern |
| **Handle connection errors gracefully** | Fallback to cached state or show error UI | Reliability requirement |

### Differentiators

| Feature | Rationale | Priority |
|---------|-----------|----------|
| **Query `NsnStake::Stakes(account)`** | Show director stake amount alongside reputation | Low |
| **Subscribe to specific events** | `DirectorsElected`, `EpochStarted`, `OnDeckElection` for real-time updates | Medium |
| **Query `NsnBft::BftResults(slot)`** | Show consensus status for recent slots | Low |
| **Multi-endpoint failover** | Connect to multiple validators for redundancy | Medium |
| **Light client mode** | Use smoldot for trustless queries (experimental) | Low - alpha maturity |
| **Cache chain state locally** | Reduce RPC load with localStorage caching | Medium |
| **TypeScript type generation** | Generate types from chain metadata for type safety | Medium |

### Anti-features

| Anti-feature | Reason to NOT Build |
|--------------|---------------------|
| **Transaction submission** | Viewer is read-only; no staking/voting from browser |
| **Key management in browser** | No wallet integration for testnet MVP |
| **Full node in browser** | Far too resource-intensive for a viewer app |
| **Offline-first with sync** | Testnet viewers need live data; offline mode adds complexity |
| **Historical state queries** | Current epoch/slot is sufficient; archive queries not needed |
| **Multi-chain support** | NSN is single-chain (solochain); no relay chain queries |

---

## Live Statistics

Real network statistics replace mock data in the Zustand store. Data flows from P2P bridge (bitrate, latency, peers) and chain client (director info, epoch).

### Table Stakes

| Feature | Rationale | Evidence |
|---------|-----------|----------|
| **Connected peer count** | Number of WebSocket clients connected to bridge | `viewer/src/store/appStore.ts:23` `connectedPeers` state |
| **Current bitrate (Mbps)** | Calculate from chunk data size / time | `viewer/src/store/appStore.ts:21` `bitrate` state |
| **Chunk latency (ms)** | Time from chunk timestamp to render | `viewer/src/store/appStore.ts:22` `latency` state |
| **Buffer level (seconds)** | How much video is buffered ahead | `viewer/src/store/appStore.ts:24` `bufferSeconds` state |
| **Current director peer ID** | Show who is generating the current slot | `viewer/src/store/appStore.ts:31` `directorPeerId` state |
| **Director reputation score** | Display reputation from chain query | `viewer/src/store/appStore.ts:32` `directorReputation` state |
| **Connection status** | disconnected/connecting/connected/error states | `viewer/src/store/appStore.ts:16` `connectionStatus` state |
| **Current epoch/slot** | Display from chain subscription | Standard chain data |

### Differentiators

| Feature | Rationale | Priority |
|---------|-----------|----------|
| **Epoch countdown timer** | Show time remaining in current epoch | Medium |
| **On-Deck director preview** | Show upcoming directors from chain | Low |
| **Quality auto-detection display** | Show current ABR quality level | Already implemented in `videoPipeline.ts` |
| **Network graph visualization** | Show mesh topology (peer connections) | Low - significant UI work |
| **Historical latency chart** | Sparkline of recent latency values | Medium |
| **Uploaded bytes (seeding)** | Track P2P seeding contribution | Low - `uploadedBytes` exists but seeding not implemented |
| **Error rate tracking** | Count decode failures, dropped chunks | Medium |
| **Regional latency breakdown** | Show latency by relay region | Low |

### Anti-features

| Anti-feature | Reason to NOT Build |
|--------------|---------------------|
| **Full Prometheus integration in browser** | Metrics are for server-side monitoring, not browser display |
| **Grafana embedding** | Overkill for viewer; simple stats panel suffices |
| **Debug-level logging to UI** | Developers use browser console; end-users see clean stats |
| **Network diagnostic tools** | MTR, traceroute, etc. are not viewer concerns |
| **Cross-session statistics persistence** | Each viewing session is independent |
| **Comparative analytics** | "Better than X%" requires aggregation infrastructure |

---

## Data Flow Summary

```
Chain Layer (nsn-chain)
    |
    | @polkadot/api WebSocket (wss://validator:9944)
    v
+-------------------+
| Chain RPC Client  | --> Epoch, Directors, Reputation
| (in browser)      |
+-------------------+
    |
    | Updates Zustand store
    v
+-------------------+
| Zustand Store     | <-- Stats from P2P
| (appStore.ts)     |
+-------------------+
    ^
    | Chunk stats
    |
+-------------------+
| Video Bridge      | <-- GossipSub /nsn/video/1.0.0
| (Node.js)         |
+-------------------+
    |
    | WebSocket (ws://bridge:8080)
    v
+-------------------+
| P2P Service       | --> Video chunks to pipeline
| (in browser)      |
+-------------------+
```

---

## Implementation Recommendations

### Video Bridge Service

1. **Technology**: Node.js + js-libp2p + ws (WebSocket library)
2. **Entry point**: Standalone service in `video-bridge/` directory
3. **Configuration**: Environment variables for mesh bootnodes, WebSocket port
4. **Docker**: Add to `docker/testnet/docker-compose.yml` alongside signaling

### Chain RPC Client

1. **Technology**: @polkadot/api with auto-connect and reconnection
2. **Location**: New service in `viewer/src/services/chainClient.ts`
3. **Hook integration**: React hook `useChainData()` for component access
4. **Type generation**: Consider @polkadot/typegen for chain-specific types

### Live Statistics

1. **Data sources**: Bridge WebSocket + Chain RPC client
2. **Update frequency**: 1-2 second intervals for smooth UI updates
3. **Error handling**: Graceful degradation to "unknown" states
4. **Component**: New `<NetworkStats />` component replacing mock data

---

## Open Questions for Requirements Phase

1. **Bridge location**: Should video-bridge run as sidecar to viewer or standalone?
2. **Multi-bridge**: Support for connecting to multiple bridges for redundancy?
3. **Chunk format versioning**: How to handle version mismatches between bridge and mesh?
4. **Epoch transition handling**: How should viewer behave when directors change mid-stream?

---

*Research compiled from codebase analysis*
*Sources: PROJECT.md, nsn-director pallet, p2p crate, viewer services, docker-compose.yml*
