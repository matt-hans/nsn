# Roadmap: NSN Testnet

## Overview

Wire the existing 62.5% complete codebase into a functioning testnet. Each phase focuses on surgical integration of existing components rather than new feature development. The goal is end-to-end flows that work reliably: prompts in, verified outputs out.

## Domain Expertise

None

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Pallet Integration Validation** - Cross-pallet integration tests for on-chain logic
- [x] **Phase 2: Lane 0 Pipeline Stitching** - Wire video generation from prompt to playback
- [x] **Phase 3: Lane 1 Pipeline Stitching** - Wire task marketplace from submission to result
- [x] **Phase 4: Viewer Web Extraction** - Extract React frontend, add WebRTC P2P
- [x] **Phase 5: Multi-Node E2E Simulation** - Network simulation testing infrastructure
- [x] **Phase 6: Testnet Deployment Config** - Docker Compose production packaging

## Phase Details

### Phase 1: Pallet Integration Validation
**Goal**: Validate on-chain pallet interactions work correctly together (stake→reputation→director→bft chain)
**Depends on**: Nothing (first phase)
**Research**: Unlikely (testing existing pallets with established FRAME patterns)
**Plans**: 1 (01-PLAN.md created 2026-01-08)

Key integration points:
- nsn-stake eligibility checks consumed by nsn-director
- nsn-reputation scoring affects nsn-director elections
- nsn-director election results drive nsn-bft validator sets
- nsn-treasury reward distribution based on bft participation

### Phase 2: Lane 0 Pipeline Stitching
**Goal**: Complete video generation flow: prompt → vortex → BFT consensus → P2P delivery → viewer playback
**Depends on**: Phase 1
**Research**: Unlikely (components exist — wiring interfaces)
**Plans**: 1 (01-PLAN.md created 2026-01-08)

Integration chain:
- nsn-node receives prompt via RPC
- Scheduler dispatches to vortex pipeline
- Vortex generates video chunks
- Chunks submitted to BFT consensus
- Finalized chunks broadcast via GossipSub
- Viewer receives and plays back stream

### Phase 3: Lane 1 Pipeline Stitching
**Goal**: Complete task marketplace flow: task submission → scheduler → sidecar execution → result delivery
**Depends on**: Phase 1
**Research**: Unlikely (existing scheduler, sidecar, task-market — integration only)
**Plans**: 1 (01-PLAN.md created 2026-01-08)

Integration chain:
- Task submitted via nsn-task-market pallet
- Scheduler picks up task, matches to capable node
- Sidecar executes workload
- Result submitted back to chain
- Requester notified of completion

### Phase 4: Viewer Web Extraction
**Goal**: Extract React frontend from Tauri shell, add WebRTC for P2P video chunk delivery
**Depends on**: Phase 2 (needs Lane 0 flow working)
**Research**: Completed (WebRTC DataChannel, simple-peer library, signaling patterns)
**Research findings**: simple-peer for WebRTC abstraction, WebSocket signaling server, 64KB max chunk size, public STUN servers for NAT traversal
**Plans**: 1 (01-PLAN.md created 2026-01-08)

Extraction scope:
- Remove Tauri-specific APIs
- Add WebRTC signaling for P2P mesh
- Configure Vite for web-only build
- Maintain existing React component structure

### Phase 5: Multi-Node E2E Simulation
**Goal**: Network simulation harness for epoch elections, BFT consensus, chunk propagation
**Depends on**: Phases 2, 3 (needs both lanes functional)
**Research**: Completed (turmoil for deterministic simulation, libp2p-swarm-test for in-memory P2P)
**Research findings**: Turmoil provides tokio-native deterministic simulation with network fault injection. libp2p-swarm-test enables in-memory P2P without real sockets. Existing mocks in lane0/lane1 tests can be extracted and reused.
**Plans**: 1 (01-PLAN.md created 2026-01-09)

Simulation scenarios:
- Multi-node epoch transitions
- Director election with competing candidates
- BFT consensus with Byzantine nodes
- Chunk propagation latency under load
- Network partition recovery

Implementation approach:
- New `node-core/crates/simulation/` crate
- SimulatedNetwork for in-memory message routing
- TestHarness for multi-node orchestration
- 8 reusable scenarios (baseline, byzantine, partition, etc.)
- 15+ integration tests using harness

### Phase 6: Testnet Deployment Config
**Goal**: Docker Compose manifests, environment configuration, genesis chain spec for testnet deployment
**Depends on**: Phase 5 (needs E2E validation passing)
**Research**: Not required (standard Docker Compose, existing Polkadot SDK patterns)
**Plans**: 1 (01-PLAN.md created 2026-01-11)

Deployment artifacts:
- docker-compose.yml for full stack (multi-validator testnet)
- Genesis chain spec with test accounts (nsn-testnet preset)
- Environment templates (.env.example)
- Bootstrap node configuration
- Health check endpoints
- Off-chain node Dockerfile
- Signaling server container
- Prometheus alerting rules
- Grafana dashboards
- Deployment documentation and scripts

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Pallet Integration Validation | 1/1 | ✅ Complete | 2026-01-08 |
| 2. Lane 0 Pipeline Stitching | 1/1 | ✅ Complete | 2026-01-08 |
| 3. Lane 1 Pipeline Stitching | 1/1 | ✅ Complete | 2026-01-08 |
| 4. Viewer Web Extraction | 1/1 | ✅ Complete | 2026-01-09 |
| 5. Multi-Node E2E Simulation | 1/1 | ✅ Complete | 2026-01-11 |
| 6. Testnet Deployment Config | 1/1 | ✅ Complete | 2026-01-11 |

**All phases complete!** NSN testnet is ready for deployment.
