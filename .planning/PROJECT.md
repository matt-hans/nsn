# Neural Sovereign Network (NSN)

## What This Is

NSN is a decentralized AI compute marketplace built as a Polkadot SDK solochain with dual-lane architecture. Lane 0 provides verified AI video generation with epoch-based director elections and BFT consensus. Lane 1 offers a general AI compute marketplace for arbitrary tasks. The network combines on-chain coordination with off-chain P2P compute delivery.

## Core Value

End-to-end video generation flow works reliably: prompt in, verified video out, delivered to viewers.

## Current Milestone: v1.1 Viewer Networking Integration

**Goal:** Wire the viewer to the live NSN testnet via direct WebRTC-to-libp2p connectivity.

**Approach:** WebRTC-Direct — browser connects directly to Rust mesh nodes without intermediate bridge.

**Target features:**
- Rust node WebRTC transport (libp2p 0.53 `webrtc` feature)
- Certificate persistence for stable certhash across restarts
- HTTP discovery endpoint (`/p2p/info`) for browser bootstrap
- js-libp2p WebRTC in browser connecting directly to mesh
- Chain RPC client for director discovery (@polkadot/api)
- Real video chunk reception via GossipSub (SCALE-encoded)
- Live network statistics (actual bitrate, latency, peer count)

## Requirements

### Validated

<!-- Existing capabilities from completed development work -->

- ✓ Polkadot SDK blockchain runtime with custom pallets — existing
- ✓ Staking pallet (nsn-stake) with slashing and role eligibility — existing
- ✓ Reputation pallet (nsn-reputation) with Merkle proofs — existing
- ✓ Director election pallet (nsn-director) with On-Deck protocol — existing
- ✓ BFT consensus pallet (nsn-bft) for chunk finalization — existing
- ✓ Storage pallet (nsn-storage) for erasure coding deals — existing
- ✓ Treasury pallet (nsn-treasury) for reward distribution — existing
- ✓ Task market pallet (nsn-task-market) for Lane 1 — existing
- ✓ Model registry pallet (nsn-model-registry) — existing
- ✓ P2P networking with libp2p GossipSub (node-core) — existing
- ✓ Vortex AI pipeline with Flux-Schnell generation — existing
- ✓ LivePortrait facial animation integration — existing
- ✓ Kokoro TTS audio synthesis — existing
- ✓ CLIP ensemble semantic verification — existing
- ✓ React/TypeScript viewer scaffold — existing
- ✓ Cross-pallet integration tests (55 tests) — v1.0
- ✓ Lane 0 pipeline stitching crate (49 tests) — v1.0
- ✓ Lane 1 pipeline stitching crate (29 tests) — v1.0
- ✓ Viewer web extraction with WebRTC signaling — v1.0
- ✓ Multi-node E2E simulation harness (24 tests) — v1.0
- ✓ Docker Compose testnet deployment config — v1.0

### Active

<!-- Current scope for v1.1 (WebRTC-Direct approach) -->

- [ ] Enable WebRTC transport in Rust nodes (libp2p 0.53 `webrtc` feature)
- [ ] Persist WebRTC certificates for stable certhash across restarts
- [ ] HTTP discovery endpoint (`/p2p/info`) returning multiaddrs with certhash
- [ ] js-libp2p WebRTC client in viewer connecting directly to mesh
- [ ] SCALE decoding of VideoChunk in browser (@polkadot/types-codec)
- [ ] Chain RPC queries for elected directors and epoch info
- [ ] Live network statistics (bitrate, latency, connected peers)
- [ ] Remove mock video stream and hardcoded stats

### Out of Scope

<!-- Explicit boundaries for v1.1 -->

- Parachain migration — stay solochain for testnet
- Separate bridge service — WebRTC-direct eliminates the need
- STUN/TURN servers — not needed for testnet with public IPs
- Mobile clients — desktop/web only
- Video transcoding — Vortex outputs browser-compatible format

## Context

**Project Status:** v1.0 testnet deployment complete. v1.1 focuses on viewer-to-network integration.

**Architecture:** Four-layer system:
1. On-chain (nsn-chain): Polkadot SDK blockchain with 9 custom pallets
2. Off-chain (node-core): P2P mesh with libp2p, scheduler, lane orchestrators
3. AI (vortex): GPU-resident pipeline with Flux, LivePortrait, Kokoro, CLIP
4. Client (viewer): React-based streaming interface

**Tech Stack:**
- Rust 2021 (nsn-chain, node-core) with Polkadot SDK 2512.0.0, libp2p 0.53 with WebRTC
- Python 3.11+ (vortex) with PyTorch 2.1+
- TypeScript 5.6+ (viewer) with React 18, Vite 6, js-libp2p 2.0 with @libp2p/webrtc

**WebRTC-Direct Architecture (v1.1):**
```
NSN Mesh (Rust libp2p 0.53)
    │ TCP/Noise/Yamux (mesh interconnect)
    │ UDP/WebRTC (browser clients)
    ▼
Director/Validator Nodes
    │ HTTP /p2p/info → { peer_id, multiaddrs with certhash }
    ▼
Browser Viewer (js-libp2p + @libp2p/webrtc)
    │ Direct WebRTC connection
    │ GossipSub /nsn/video/1.0.0
    │ SCALE-encoded VideoChunk
    ▼
Video Pipeline (WebCodecs → Canvas)
```

**Known Technical Debt:**
- 11.8GB VRAM requirement limits hardware compatibility
- WebRTC certificate must persist for stable certhash

## Constraints

- **Team Capacity**: Solo/limited contributors — prioritize ruthlessly, avoid scope creep
- **GPU Requirements**: RTX 3060 12GB minimum for Lane 0 validators
- **Deployment**: Docker Compose based — no Kubernetes for testnet
- **libp2p Version**: Stay on 0.53.x for WebRTC support, no major upgrade

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Solochain for testnet | Faster iteration, defer parachain complexity | ✅ v1.0 |
| Web-based viewer | Lower friction for testers | ✅ v1.0 |
| Both lanes for testnet | Demonstrate full capability | ✅ v1.0 |
| E2E simulation required | Catch integration issues | ✅ v1.0 |
| WebRTC-direct over Node.js bridge | Better latency, no extra service, future-proof | v1.1 |
| HTTP discovery endpoint | Solves certhash bootstrapping problem | v1.1 |
| Certificate persistence | Stable certhash across node restarts | v1.1 |
| Browser SCALE decoding | No format translation, same protocol as mesh | v1.1 |
| Chain RPC for discovery | Viewer queries nsn-director pallet for directors | v1.1 |

---
*Last updated: 2026-01-18 after v1.1 WebRTC-direct approach decision*
