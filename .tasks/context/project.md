# NSN Project Context

## Overview

The Neural Sovereign Network (NSN) is a decentralized AI compute marketplace built as its own Polkadot SDK chain with dual-lane architecture: Lane 0 for deterministic video generation and Lane 1 for general AI compute tasks.

## Vision & Goals

- **Vision**: Create the world's first truly decentralized AI compute network with economic incentives for content generation, AI task execution, validation, and distribution
- **Primary Goal**: Build NSN as its own Polkadot SDK chain with custom FRAME pallets to enable trustless coordination of AI workloads via dual-lane architecture
- **Secondary Goals**:
  - **Lane 0**: Achieve <45s glass-to-glass latency for AI video generation and playback
  - **Lane 1**: Enable open marketplace for arbitrary AI tasks (inference, training, fine-tuning)
  - Support 500+ distributed nodes across 7 geographic regions
  - Maintain 99.5% streaming availability through hierarchical P2P swarm architecture
  - Enable staged deployment: Solochain → Parachain → Coretime scaling

## Target Users

1. **Content Consumers (Viewers)**: End users seeking unique AI-generated video streams (Lane 0)
2. **Task Submitters**: Users and applications submitting AI compute tasks (Lane 1)
3. **Node Operators (Directors/Super-Nodes/Validators)**: Stakeholders running infrastructure for NSN token rewards
4. **Chain Validators**: Operators running NSN Chain validator nodes
5. **Developers**: Third-party integrators building on NSN's Substrate API (and optional EVM interface)

## Success Criteria

**Phase A (NSN Solochain - Weeks 1-8)**:
- NSN Chain bootstrapped from Polkadot SDK template
- All 8 custom pallets compile and pass unit tests
- NSN Testnet deployed with controlled validator set (3-5 validators)
- 10+ off-chain test nodes operational
- Lane 0: Complete staking → epoch election → reputation → BFT flow demonstrated
- Lane 1: Task marketplace operational with model registry

**Phase B (NSN Mainnet - Weeks 9-16)**:
- Security audit passed (Oak Security/SRLabs)
- Validator onboarding and genesis configuration finalized
- NSN native token launched
- 50+ mainnet nodes, 500+ community members
- 99.5%+ validator uptime

**Phase C (Parachain - Post-adoption)**:
- Cumulus integration complete
- Parachain slot acquired via Polkadot governance
- Migration to Polkadot shared security

## Key Constraints

- **Budget**: $80k-$200k (2-3 Rust/Substrate devs for 3-6 months)
- **Timeline**: 3-6 months to MVP (full sovereignty, no external governance dependency)
- **Hardware**: RTX 3060 12GB minimum for Director nodes (11.8GB VRAM budget)
- **TPS**: 100+ TPS on NSN Chain (no external TPS limits)
- **Security**: Solo phase relies on trusted validators; parachain phase inherits Polkadot security

## Timeline

- **Weeks 1-2**: Foundation (NSN Chain bootstrap, dev environment, chain spec, pallet-nsn-stake)
- **Weeks 3-4**: Reputation system (pallet-nsn-reputation, Merkle trees)
- **Weeks 5-6**: Director logic (pallet-nsn-director, epoch-based elections, BFT challenge mechanism)
- **Weeks 7-8**: Lane 1 (pallet-nsn-task-market, pallet-nsn-model-registry, node-core)
- **Weeks 9-10**: NSN Testnet deployment and integration testing
- **Weeks 11-16**: Security audit, validator onboarding, token genesis, mainnet launch

## Deployment Model

1. **Phase A: NSN Solochain** - Controlled 3-5 validator set, fast iteration, no external dependencies
2. **Phase B: NSN Mainnet** - Public validator onboarding, production deployment
3. **Phase C: Parachain** - Cumulus integration, Polkadot shared security
4. **Phase D: Scaling** - Coretime acquisition (on-demand → bulk)

## Dual-Lane Architecture

- **Lane 0 (Video Generation)**: Deterministic video streaming with epoch-based elections and BFT consensus
  - On-Deck protocol: 20 Director candidates, 5 elected per epoch (100 blocks)
  - Vortex pipeline: Flux, LivePortrait, Kokoro, CLIP verification

- **Lane 1 (General AI Compute)**: Open marketplace for arbitrary AI tasks
  - Task marketplace via pallet-nsn-task-market
  - Model registry via pallet-nsn-model-registry
  - node-core orchestration (scheduler + sidecar)
