# Neural Sovereign Network (NSN)

## What This Is

NSN is a decentralized AI compute marketplace built as a Polkadot SDK solochain with dual-lane architecture. Lane 0 provides verified AI video generation with epoch-based director elections and BFT consensus. Lane 1 offers a general AI compute marketplace for arbitrary tasks. The network combines on-chain coordination with off-chain P2P compute delivery.

## Core Value

End-to-end video generation flow works reliably: prompt in, verified video out, delivered to viewers.

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

### Active

<!-- Current scope for testnet launch -->

- [ ] Complete end-to-end Lane 0 video flow (prompt → generation → BFT → delivery → playback)
- [ ] Complete end-to-end Lane 1 task marketplace flow
- [ ] Web-based viewer client (extract from Tauri, use WebRTC for P2P)
- [ ] E2E network simulation testing with multi-node scenarios
- [ ] Docker Compose deployment configuration for testnet
- [ ] Cross-pallet integration tests
- [ ] Aggregate remaining 24 tasks into prioritized phases

### Out of Scope

<!-- Explicit boundaries for testnet -->

- Parachain migration — stay solochain for testnet, defer Cumulus integration
- Native desktop app — web-based viewer for lower friction during testnet
- Production security audit — testnet with test tokens, audit before mainnet
- Mobile clients — desktop/web only for testnet

## Context

**Project Status:** 64 total tasks in manifest, 40 completed (62.5%), 24 pending.

**Architecture:** Four-layer system:
1. On-chain (nsn-chain): Polkadot SDK blockchain with 9 custom pallets
2. Off-chain (node-core): P2P mesh with libp2p, scheduler, lane orchestrators
3. AI (vortex): GPU-resident pipeline with Flux, LivePortrait, Kokoro, CLIP
4. Client (viewer): React-based streaming interface

**Tech Stack:**
- Rust 2021 (nsn-chain, node-core) with Polkadot SDK 2512.0.0
- Python 3.11+ (vortex) with PyTorch 2.1+
- TypeScript 5.6+ (viewer) with React 18, Vite 6

**Known Technical Debt:**
- Cross-pallet integration tests needed
- P2P network simulation testing gap
- 11.8GB VRAM requirement limits hardware compatibility

## Constraints

- **Team Capacity**: Solo/limited contributors — prioritize ruthlessly, avoid scope creep
- **GPU Requirements**: RTX 3060 12GB minimum for Lane 0 validators
- **Deployment**: Docker Compose based — no Kubernetes for testnet

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Solochain for testnet | Faster iteration, defer parachain complexity | — Pending |
| Web-based viewer | Lower friction for testers, WebRTC for P2P | — Pending |
| Both lanes for testnet | Demonstrate full capability, not partial product | — Pending |
| E2E simulation required | Catch integration issues before public testnet | — Pending |
| Aggregate tasks into phases | Use gsd:plan-phase to organize remaining work | — Pending |

---
*Last updated: 2026-01-08 after initialization*
