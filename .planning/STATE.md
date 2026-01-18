# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-18)

**Core value:** End-to-end video generation flow works reliably: prompt in, verified video out, delivered to viewers.
**Current focus:** v1.1 Viewer Networking Integration

## Current Milestone

**Milestone:** v1.1 Viewer Networking Integration
**Status:** Executing (Phase 2 complete)
**Goal:** Wire the viewer to the live NSN testnet via direct WebRTC-to-libp2p connectivity
**Approach:** WebRTC-Direct — browser connects directly to Rust mesh nodes

## Current Position

Phase: 3 of 7 (Viewer Implementation)
Plan: 5 of 5 complete (awaiting checkpoint verification)
Status: Awaiting checkpoint for Plan 03-05
Last activity: 2026-01-18 — Completed Plan 03-05 (Mock Removal & P2P Integration)

Progress: ███████░░░ 43% (7/14 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: ~1 session
- Total execution time: 8 sessions

**By Phase (v1.0):**

| Phase | Plans | Total | Status |
|-------|-------|-------|--------|
| Phase 1 | 1 | 55 tests | ✅ Complete |
| Phase 2 | 1 | 49 tests | ✅ Complete |
| Phase 3 | 1 | 29 tests | ✅ Complete |
| Phase 4 | 1 | 19 tests | ✅ Complete |
| Phase 5 | 1/1 | 24 tests | ✅ Complete |
| Phase 6 | 1/1 | 8 tasks | ✅ Complete |

**By Phase (v1.1 WebRTC-Direct):**

| Phase | Plans | Tests Added | Status |
|-------|-------|-------------|--------|
| Phase 1 | 2/2 | 10 tests | ✅ Complete |
| Phase 2 | 1/1 | 13 tests | ✅ Complete |
| Phase 3 | 3/5 | - | 🔄 Executing (awaiting checkpoint) |

**Recent Trend:**
- Last 9 plans: 9 completed
- Trend: Steady progress

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Surgical integration approach: Wire existing 62.5% complete components rather than building new features
- 6 phases focused on: pallet validation → Lane 0 → Lane 1 → viewer extraction → E2E testing → deployment
- Integration tests placed in `nsn-chain/integration-tests/` crate (separate from runtime)
- Lane 0 BFT timeout: 5000ms default, configurable
- Lane 0 CLIP embedding: 512 dimensions (dual-CLIP ensemble)
- Lane 0 consensus threshold: 3-of-5 directors, cosine similarity ≥ 0.85
- Lane 1 execution timeout: 300,000ms (5 minutes) default
- Lane 1 serial execution: 1 task at a time for MVP
- WebRTC alpha version: Use libp2p-webrtc v0.7.1-alpha (compatible with libp2p 0.53)
- Browser libp2p: WebRTC-Direct transport, no listen addresses, noise encryption, yamux muxing
- js-libp2p v3.1.3 for browser P2P with @libp2p/webrtc v6.0.11 transport

### Deferred Issues

None yet.

### Blockers/Concerns

None yet.

## Phase 1 Summary

**Completed:** 2026-01-08

**Deliverables:**
- `nsn-chain/integration-tests/` crate with mock runtime and 6 test modules
- 55 integration tests validating all critical pallet interaction chains

**Integration chains validated:**
1. ✅ Stake → Director Election (NodeModeUpdater, NodeRoleUpdater traits)
2. ✅ Director → Reputation (ReputationEventType callbacks)
3. ✅ Director → BFT Storage (consensus statistics tracking)
4. ✅ Task Market → Stake (LaneNodeProvider, TaskSlashHandler)
5. ✅ Treasury → Work Recording (contribution accumulation)
6. ✅ Full epoch lifecycle (end-to-end mode/role transitions)

**Commits:**
- `9c2baea` test(1-1): create integration test infrastructure
- `6c81906` test(1-2..7): add integration tests for Phase 1 pallet validation

## Phase 2 Summary

**Completed:** 2026-01-08

**Deliverables:**
- `node-core/crates/lane0/` crate with 6 modules (~1,825 lines)
- DirectorService with full lifecycle state machine
- RecipeProcessor for P2P recipe ingestion
- VortexClient for sidecar gRPC integration
- BftParticipant for CLIP embedding consensus
- ChunkPublisher for video distribution
- 49 tests (33 unit + 15 integration + 1 doc)

**Components implemented:**
1. ✅ DirectorService (Standby → OnDeck → Active → Draining → Standby)
2. ✅ RecipeProcessor (validation, queuing, P2P subscription)
3. ✅ VortexClient (sidecar gRPC, response parsing)
4. ✅ BftParticipant (CLIP consensus, signature verification)
5. ✅ ChunkPublisher (video chunking, signing, P2P publish)
6. ✅ Error types (thiserror, comprehensive failure modes)
7. ✅ Integration tests (mock-based pipeline testing)

**Commits:**
- `72c4200` feat(2-1): implement Lane 0 pipeline stitching crate

## Phase 3 Summary

**Completed:** 2026-01-08

**Deliverables:**
- `node-core/crates/lane1/` crate with 5 modules (~1,894 lines)
- ChainListener for task-market event subscription
- ExecutionRunner for sidecar gRPC wrapper
- ResultSubmitter for chain extrinsic submission
- TaskExecutorService with event-driven state machine
- 29 tests (20 unit + 9 integration)

**Components implemented:**
1. ✅ ChainListener (TaskCreated, TaskAssigned, TaskVerified, TaskFailed)
2. ✅ ExecutionRunner (execute, poll_status, cancel)
3. ✅ ResultSubmitter (start_task, submit_result, fail_task)
4. ✅ TaskExecutorService (Idle → Executing → Submitting → Idle)
5. ✅ Error types (Lane1Error, ListenerError, ExecutionError, SubmissionError)
6. ✅ Integration tests (task lifecycle, priority ordering)

**Commits:**
- `675f417` feat(3-1): implement Lane 1 pipeline stitching crate

## Phase 4 Summary

**Completed:** 2026-01-09

**Deliverables:**
- Standalone React web application (extracted from Tauri)
- WebRTC signaling client with WebSocket transport
- P2P service using simple-peer for DataChannel video delivery
- Development signaling server (Node.js)
- Web-optimized Vite build configuration
- 19 new integration tests for signaling/P2P

**Components implemented:**
1. ✅ SignalingClient (WebSocket state machine, peer discovery)
2. ✅ P2PService (simple-peer, binary video chunk parsing)
3. ✅ Signaling server (join/leave/offer/answer/ice-candidate)
4. ✅ Test mocks (WebSocket, RTCPeerConnection, WebCodecs)
5. ✅ Vite config (chunk splitting, web-only build)

**Commits:**
- `2c2653f` chore(4-1): remove Tauri dependencies, add simple-peer
- `9d535c5` refactor(4-1): remove Tauri IPC, use browser APIs
- `42f9feb` feat(4-1): add WebRTC signaling client
- `da019f0` feat(4-1): implement WebRTC P2P service
- `d1e802e` feat(4-1): add development signaling server
- `cfde70b` test(4-1): update test mocks for web environment
- `be6a8e4` chore(4-1): configure Vite for standalone web
- `70f694b` test(4-1): add integration tests

## Phase 5 Summary

**Completed:** 2026-01-11

**Deliverables:**
- `node-core/crates/simulation/` crate with 13 files (~3,800 lines)
- SimulatedNetwork for in-memory message routing
- TestHarness for multi-node scenario orchestration
- 4 reusable mocks (MockVortexClient, MockBftParticipant, MockChunkPublisher, MockChainClient)
- 8 pre-defined scenarios (baseline, byzantine, partition, etc.)
- 24 integration tests (exceeds 15+ target)

**Components implemented:**
1. ✅ SimulatedNetwork (in-memory routing, latency profiles, partitions)
2. ✅ TestHarness (node management, epoch events, slot execution)
3. ✅ MockVortexClient (success/failure slots, latency injection)
4. ✅ MockBftParticipant (Byzantine modes: Crash, Delay, DivergentEmbedding)
5. ✅ MockChunkPublisher (event tracking, configurable chunk size)
6. ✅ MockChainClient (task/epoch event injection, extrinsic tracking)
7. ✅ 8 Scenarios (BaselineConsensus, ByzantineDirector, NetworkPartition, etc.)
8. ✅ Integration tests (consensus, faults, lifecycle)

**Commits:**
- `d9cc41c` feat(5-1): implement multi-node E2E simulation harness

## Phase 6 Summary

**Completed:** 2026-01-11

**Deliverables:**
- `docker/testnet/` directory with complete deployment configuration
- Multi-node Docker Compose (3 validators, 2 directors, GPU, signaling)
- Production Dockerfiles (nsn-offchain, signaling)
- Chain spec generation and bootnode setup scripts
- Prometheus alerting rules and Grafana dashboard
- Comprehensive deployment documentation

**Components implemented:**
1. ✅ docker-compose.yml (multi-node testnet with all services)
2. ✅ Chain spec generation script
3. ✅ Environment templates (.env.example, director.toml, validator.toml)
4. ✅ Bootstrap node configuration
5. ✅ Dockerfile.nsn-offchain (multi-stage Rust build)
6. ✅ Dockerfile.signaling (minimal Node.js)
7. ✅ Prometheus/Grafana configuration with alerts
8. ✅ Deployment documentation and operational scripts

**Commits:**
- `a9a64f6` feat(6-1): create testnet docker-compose configuration
- `d98f0ae` feat(6-1): add testnet chain spec and generation script
- `59a42ba` feat(6-1): add environment and service configuration templates
- `65284c4` feat(6-1): add bootstrap node configuration
- `2bedf6e` feat(6-1): add Dockerfile for off-chain node
- `2eb302c` feat(6-1): add Dockerfile for signaling server
- `2b445db` feat(6-1): add Prometheus and Grafana configuration
- `b4999c0` docs(6-1): add deployment documentation and operational scripts

## v1.1 Milestone Overview (WebRTC-Direct Approach)

**7 Phases Defined:**
1. Rust Node Core Upgrade - WebRTC transport, certificate persistence
2. Discovery Bridge - HTTP `/p2p/info` endpoint
3. Viewer Implementation - js-libp2p WebRTC client
4. Video Streaming Protocol - GossipSub + SCALE decoding
5. Chain RPC Integration - Director discovery (parallel)
6. Docker & Operations - UDP port, env vars (parallel)
7. Testing & Validation - E2E verification

**Critical Path:** Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 7

**Research Updated for WebRTC-Direct:**
- STACK.md - Rust libp2p 0.53 WebRTC, js-libp2p 2.0 with @libp2p/webrtc
- FEATURES.md - Scope definition (table stakes vs anti-features)
- ARCHITECTURE.md - Direct browser-to-mesh connectivity
- PITFALLS.md - Certificate persistence, discovery endpoint, CORS

**Key Decisions:**
- WebRTC-direct over Node.js bridge (better latency, no extra service)
- Certificate persistence for stable certhash
- HTTP discovery endpoint for browser bootstrap
- Browser decodes SCALE directly (no format translation)

## v1.1 Phase 1 Plan 01 Summary

**Completed:** 2026-01-18

**Deliverables:**
- libp2p-webrtc v0.9.0-alpha.1 dependency added to workspace
- CertificateManager module for WebRTC certificate persistence
- P2pConfig extended with WebRTC configuration fields
- 8 new tests (5 cert + 3 config)

**Components implemented:**
1. ✅ libp2p-webrtc workspace dependency
2. ✅ CertificateManager (load_or_generate, PEM persistence)
3. ✅ CertError (Io, Generation, Parse variants)
4. ✅ P2pConfig WebRTC fields (enable_webrtc, webrtc_port, data_dir, external_address)

**Commits:**
- `f4dd233` feat(01-01): add libp2p-webrtc dependency for WebRTC transport
- `dc36778` feat(01-01): add WebRTC certificate persistence module
- `dbd2e1f` feat(01-01): extend P2pConfig with WebRTC transport fields

## v1.1 Phase 1 Plan 02 Summary

**Completed:** 2026-01-18

**Deliverables:**
- WebRTC transport integrated into P2pService swarm
- CLI flags for WebRTC configuration
- 2 integration tests for WebRTC transport

**Components implemented:**
1. ✅ WebRTC transport composition with SwarmBuilder
2. ✅ Certificate loading via CertificateManager
3. ✅ WebRTC listener on configurable UDP port
4. ✅ External address advertisement for NAT/Docker
5. ✅ CLI flags (--p2p-enable-webrtc, --p2p-webrtc-port, --p2p-external-address, --data-dir)

**Commits:**
- `e800730` feat(01-02): add WebRTC transport to P2pService swarm
- `6575642` feat(01-02): add CLI flags for WebRTC configuration
- `ed95ee0` test(01-02): add integration tests for WebRTC transport

**Deviations:**
- Downgraded libp2p-webrtc from 0.9.0-alpha.1 to 0.7.1-alpha for libp2p-core 0.41 compatibility

## v1.1 Phase 2 Plan 01 Summary

**Completed:** 2026-01-18

**Deliverables:**
- P2P discovery module with response types and address filtering
- HTTP `/p2p/info` endpoint with CORS and 503 handling
- 13 new tests (10 unit + 3 integration)

**Components implemented:**
1. ✅ P2pInfoResponse envelope (success/data/error pattern)
2. ✅ P2pInfoData with peer_id, multiaddrs, protocols, features
3. ✅ filter_addresses() with external override, IPv6 link-local filtering
4. ✅ HttpState for shared swarm state with HTTP server
5. ✅ serve_http() handling /metrics and /p2p/info
6. ✅ 503 response with Retry-After:5 when swarm not ready
7. ✅ CORS headers (Access-Control-Allow-Origin: *)
8. ✅ Cache-Control: no-store, max-age=0

**Commits:**
- `2a588bf` feat(02-01): add P2P discovery module with types and address filtering
- `ff07ab1` feat(02-01): add /p2p/info HTTP endpoint with CORS and 503 handling
- `c764283` test(02-01): add integration tests for discovery endpoint

## v1.1 Phase 3 Plan 01 Summary

**Completed:** 2026-01-18

**Deliverables:**
- P2PClient service class with WebRTC-Direct transport (205 lines)
- js-libp2p ecosystem dependencies installed
- simple-peer dependency removed

**Components implemented:**
1. ✅ P2PClient class with WebRTC-Direct transport
2. ✅ GossipSub pubsub with emitSelf: false
3. ✅ noise encryption and yamux muxing
4. ✅ identify service for protocol negotiation
5. ✅ No listen addresses (browser outbound-only)
6. ✅ Lifecycle: initialize() → dial() → subscribe() → publish() → stop()

**Commits:**
- `ed0d32a` chore(03-01): install js-libp2p dependencies, remove simple-peer
- `e611067` feat(03-01): create P2PClient service class with libp2p

## v1.1 Phase 3 Plan 02 Summary

**Completed:** 2026-01-18

**Deliverables:**
- HTTP-based node discovery service with parallel race pattern (258 lines)
- SCALE VideoChunk decoder matching Rust struct definition (208 lines)

**Components implemented:**
1. ✅ discovery.ts with discoverNode(), discoverWithRace(), buildCandidateList()
2. ✅ videoCodec.ts with decodeVideoChunk() using @polkadot/types TypeRegistry
3. ✅ Tiered configuration: localStorage → settings → env → hardcoded
4. ✅ WebRTC multiaddr extraction with certhash validation
5. ✅ 503 retry handling for node initialization
6. ✅ Hardcoded shuffling to avoid hammering first node

**Commits:**
- `2b77b61` feat(03-02): create discovery service and SCALE codec

## v1.1 Phase 3 Plan 03 Summary

**Completed:** 2026-01-18

**Deliverables:**
- GossipSub video topic subscription in P2PClient
- P2P-to-video pipeline adapter with SCALE decoding
- Chunk stats tracking for bitrate calculation

**Components implemented:**
1. ✅ VIDEO_TOPIC constant (/nsn/video/1.0.0)
2. ✅ subscribeToVideoTopic() with GossipSub subscription
3. ✅ unsubscribeFromVideoTopic() for cleanup
4. ✅ types.ts shared VideoChunkMessage interface
5. ✅ connectP2PToPipeline() adapter function
6. ✅ chunkStats tracking in VideoPipeline
7. ✅ getBitrateMbps() calculation method

**Deviations (Rule 3 - Blocking Issues Fixed):**
1. Created types.ts to decouple VideoChunkMessage from legacy p2p.ts
2. Converted p2p.ts to legacy stub (simple-peer removed)
3. Disabled p2p-service.test.ts and signaling.test.ts
4. Updated vite.config.ts manual chunks (removed simple-peer)

**Commits:**
- `cc7618a` feat(03-03): add video subscription to P2PClient and P2P-pipeline adapter

## v1.1 Phase 3 Plan 04 Summary

**Completed:** 2026-01-18

**Deliverables:**
- Zustand store extensions for P2P connection state
- useP2PConnection React hook with exponential backoff reconnection
- NetworkStatus widget with color-coded health indicator
- Connection lifecycle management (connect/disconnect)

**Components implemented:**
1. ✅ BootstrapProgress interface for phase tracking
2. ✅ P2P state properties (connectedPeerId, meshPeerCount, connectionError, lastConnectedNodeUrl, bootstrapProgress)
3. ✅ Store actions for P2P state updates
4. ✅ useP2PConnection hook with connect(), disconnect(), scheduleReconnect()
5. ✅ Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s max
6. ✅ NetworkStatus widget (green/yellow/red indicator, hover tooltip)
7. ✅ useEffect cleanup on unmount

**Commits:**
- `74c21aa` feat(03-04): extend Zustand store with P2P connection state
- `102c629` feat(03-04): create useP2PConnection hook with reconnection
- `f3a2708` feat(03-04): create NetworkStatus widget for connection health
- `bfc9e0c` fix(03-04): remove unused imports for TypeScript compliance
- `65c699d` style(03-04): apply biome formatter to appStore.ts

## v1.1 Phase 3 Plan 05 Summary

**Completed:** 2026-01-18

**Deliverables:**
- Mock video stream completely removed (startMockVideoStream deleted)
- Signaling service deleted (SignalingClient, signaling.ts removed)
- P2P integration in App.tsx with auto-connect on mount
- Bootstrap overlay UI for connection phases (BootstrapOverlay component)
- NetworkStatus widget in TopBar (replaced old connection-status div)

**Components implemented:**
1. ✅ Mock video generator removed (startMockVideoStream deleted)
2. ✅ Signaling service deleted (SignalingClient, signaling.ts)
3. ✅ Deleted obsolete test files (p2p-service.test.ts, signaling.test.ts, p2p.test.ts)
4. ✅ Rewrote p2p.ts as compatibility layer (delegates to P2PClient)
5. ✅ Updated App.tsx to use useP2PConnection hook
6. ✅ Created BootstrapOverlay component (terminal-style aesthetic)
7. ✅ Updated TopBar to include NetworkStatus widget
8. ✅ Updated VideoPlayer to remove P2P connection management
9. ✅ Updated tests for new architecture

**Commits:**
- `69cfec5` feat(03-05): remove mock video and delete signaling service
- `aba7743` feat(03-05): integrate P2P into App and add bootstrap overlay

## Session Continuity

Last session: 2026-01-18
Stopped at: Completed Phase 3 Plan 05 (Mock Removal & P2P Integration) - awaiting checkpoint
Resume file: None
Next step: Verify checkpoint for Plan 03-05, then proceed to Phase 4 (Video Streaming Protocol)

