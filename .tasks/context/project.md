# ICN Project Context

## Overview

The Interdimensional Cable Network (ICN) is a decentralized AI-powered video streaming platform that generates endless unique video content through Byzantine Fault Tolerant (BFT) consensus among director nodes.

## Vision & Goals

- **Vision**: Create the world's first truly decentralized AI-generated streaming network with economic incentives for content generation, validation, and distribution
- **Primary Goal**: Build ICN as its own Polkadot SDK chain with custom FRAME pallets to enable trustless coordination of AI video generation with 3-of-5 director consensus
- **Secondary Goals**:
  - Achieve <45s glass-to-glass latency for AI video generation and playback
  - Support 500+ distributed nodes across 7 geographic regions
  - Maintain 99.5% streaming availability through hierarchical P2P swarm architecture
  - Enable staged deployment: Solochain → Parachain → Coretime scaling

## Target Users

1. **Content Consumers (Viewers)**: End users seeking unique AI-generated video streams
2. **Node Operators (Directors/Super-Nodes/Validators)**: Stakeholders running infrastructure for ICN token rewards
3. **Chain Validators**: Operators running ICN Chain validator nodes
4. **Developers**: Third-party integrators building on ICN's Substrate API (and optional EVM interface)

## Success Criteria

**Phase A (ICN Solochain - Weeks 1-8)**:
- ICN Chain bootstrapped from Polkadot SDK template
- All 6 custom pallets compile and pass unit tests
- ICN Testnet deployed with controlled validator set (3-5 validators)
- 10+ off-chain test nodes operational
- Complete staking → election → reputation → BFT flow demonstrated

**Phase B (ICN Mainnet - Weeks 9-16)**:
- Security audit passed (Oak Security/SRLabs)
- Validator onboarding and genesis configuration finalized
- ICN native token launched
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
- **TPS**: 100+ TPS on ICN Chain (no external TPS limits)
- **Security**: Solo phase relies on trusted validators; parachain phase inherits Polkadot security

## Timeline

- **Weeks 1-2**: Foundation (ICN Chain bootstrap, dev environment, chain spec, pallet-icn-stake)
- **Weeks 3-4**: Reputation system (pallet-icn-reputation, Merkle trees)
- **Weeks 5-6**: Director logic (pallet-icn-director, VRF elections, BFT challenge mechanism)
- **Weeks 7-8**: ICN Testnet deployment and integration testing
- **Weeks 9-16**: Security audit, validator onboarding, token genesis, mainnet launch

## Deployment Model

1. **Phase A: ICN Solochain** - Controlled 3-5 validator set, fast iteration, no external dependencies
2. **Phase B: ICN Mainnet** - Public validator onboarding, production deployment
3. **Phase C: Parachain** - Cumulus integration, Polkadot shared security
4. **Phase D: Scaling** - Coretime acquisition (on-demand → bulk)
