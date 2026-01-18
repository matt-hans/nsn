# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-18)

**Core value:** End-to-end video generation flow works reliably: prompt in, verified video out, delivered to viewers.
**Current focus:** v1.1 Viewer Networking Integration

## Current Milestone

**Milestone:** v1.1 Viewer Networking Integration
**Status:** Planning complete (WebRTC-Direct approach), ready for Phase 1
**Goal:** Wire the viewer to the live NSN testnet via direct WebRTC-to-libp2p connectivity
**Approach:** WebRTC-Direct — browser connects directly to Rust mesh nodes

## Current Position

Phase: 1 of 7 (Rust Node Core Upgrade)
Plan: 1 of 2 complete
Status: In progress
Last activity: 2026-01-18 — Completed 01-01-PLAN.md

Progress: █░░░░░░░░░ 7% (1/14 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: ~1 session
- Total execution time: 7 sessions

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
| Phase 1 | 1/2 | 8 tests | 🔄 In Progress |

**Recent Trend:**
- Last 7 plans: 7 completed
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
- WebRTC alpha version: Use libp2p-webrtc v0.9.0-alpha.1 (latest available)

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

## Session Continuity

Last session: 2026-01-18
Stopped at: Completed 01-01-PLAN.md
Resume file: None
Next step: Execute 01-02-PLAN.md (WebRTC transport setup)
