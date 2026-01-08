# Roadmap: NSN Testnet

## Overview

Wire the existing 62.5% complete codebase into a functioning testnet. Each phase focuses on surgical integration of existing components rather than new feature development. The goal is end-to-end flows that work reliably: prompts in, verified outputs out.

## Domain Expertise

None

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Pallet Integration Validation** - Cross-pallet integration tests for on-chain logic
- [ ] **Phase 2: Lane 0 Pipeline Stitching** - Wire video generation from prompt to playback
- [ ] **Phase 3: Lane 1 Pipeline Stitching** - Wire task marketplace from submission to result
- [ ] **Phase 4: Viewer Web Extraction** - Extract React frontend, add WebRTC P2P
- [ ] **Phase 5: Multi-Node E2E Simulation** - Network simulation testing infrastructure
- [ ] **Phase 6: Testnet Deployment Config** - Docker Compose production packaging

## Phase Details

### Phase 1: Pallet Integration Validation
**Goal**: Validate on-chain pallet interactions work correctly together (stake→reputation→director→bft chain)
**Depends on**: Nothing (first phase)
**Research**: Unlikely (testing existing pallets with established FRAME patterns)
**Plans**: TBD

Key integration points:
- nsn-stake eligibility checks consumed by nsn-director
- nsn-reputation scoring affects nsn-director elections
- nsn-director election results drive nsn-bft validator sets
- nsn-treasury reward distribution based on bft participation

### Phase 2: Lane 0 Pipeline Stitching
**Goal**: Complete video generation flow: prompt → vortex → BFT consensus → P2P delivery → viewer playback
**Depends on**: Phase 1
**Research**: Unlikely (components exist — wiring interfaces)
**Plans**: TBD

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
**Plans**: TBD

Integration chain:
- Task submitted via nsn-task-market pallet
- Scheduler picks up task, matches to capable node
- Sidecar executes workload
- Result submitted back to chain
- Requester notified of completion

### Phase 4: Viewer Web Extraction
**Goal**: Extract React frontend from Tauri shell, add WebRTC for P2P video chunk delivery
**Depends on**: Phase 2 (needs Lane 0 flow working)
**Research**: Likely (WebRTC P2P patterns, Tauri→web extraction)
**Research topics**: WebRTC DataChannel for binary chunk streaming, libp2p-webrtc browser compatibility, Vite build config for standalone web app
**Plans**: TBD

Extraction scope:
- Remove Tauri-specific APIs
- Add WebRTC signaling for P2P mesh
- Configure Vite for web-only build
- Maintain existing React component structure

### Phase 5: Multi-Node E2E Simulation
**Goal**: Network simulation harness for epoch elections, BFT consensus, chunk propagation
**Depends on**: Phases 2, 3 (needs both lanes functional)
**Research**: Likely (network simulation patterns, multi-node orchestration)
**Research topics**: Rust test harness for multi-process orchestration, libp2p network simulation patterns, epoch/BFT timing simulation
**Plans**: TBD

Simulation scenarios:
- Multi-node epoch transitions
- Director election with competing candidates
- BFT consensus with Byzantine nodes
- Chunk propagation latency under load
- Network partition recovery

### Phase 6: Testnet Deployment Config
**Goal**: Docker Compose manifests, environment configuration, genesis chain spec for testnet deployment
**Depends on**: Phase 5 (needs E2E validation passing)
**Research**: Unlikely (standard Docker Compose, existing Polkadot SDK patterns)
**Plans**: TBD

Deployment artifacts:
- docker-compose.yml for full stack
- Genesis chain spec with test accounts
- Environment templates (.env.example)
- Bootstrap node configuration
- Health check endpoints

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Pallet Integration Validation | 0/TBD | Not started | - |
| 2. Lane 0 Pipeline Stitching | 0/TBD | Not started | - |
| 3. Lane 1 Pipeline Stitching | 0/TBD | Not started | - |
| 4. Viewer Web Extraction | 0/TBD | Not started | - |
| 5. Multi-Node E2E Simulation | 0/TBD | Not started | - |
| 6. Testnet Deployment Config | 0/TBD | Not started | - |
