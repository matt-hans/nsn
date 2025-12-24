# ICN Project Context

## Overview

The Interdimensional Cable Network (ICN) is a decentralized AI-powered video streaming platform that generates endless unique video content through Byzantine Fault Tolerant (BFT) consensus among director nodes.

## Vision & Goals

- **Vision**: Create the world's first truly decentralized AI-generated streaming network with economic incentives for content generation, validation, and distribution
- **Primary Goal**: Deploy custom Substrate pallets on Moonbeam parachain to enable trustless coordination of AI video generation with 3-of-5 director consensus
- **Secondary Goals**:
  - Achieve <45s glass-to-glass latency for AI video generation and playback
  - Support 500+ distributed nodes across 7 geographic regions
  - Maintain 99.5% streaming availability through hierarchical P2P swarm architecture

## Target Users

1. **Content Consumers (Viewers)**: End users seeking unique AI-generated video streams
2. **Node Operators (Directors/Super-Nodes/Validators)**: Stakeholders running infrastructure for ICN token rewards
3. **Developers**: Third-party integrators building on ICN's EVM + Substrate dual interface

## Success Criteria

**Phase 1 (Moonriver Testnet - Weeks 1-8)**:
- All 6 custom pallets compile and pass unit tests
- Runtime deployed to Moonriver with 10+ test nodes
- Complete staking → election → reputation → BFT flow demonstrated

**Phase 2 (Moonbeam Mainnet - Weeks 9-16)**:
- Security audit passed (Oak Security/SRLabs)
- Governance proposal approved by GLMR holders
- ICN ERC-20 token launched with DEX liquidity
- 50+ mainnet nodes, 500+ community members

## Key Constraints

- **Budget**: $80k-$200k (2-3 Rust/Substrate devs for 3-6 months)
- **Timeline**: 3-6 months to MVP (vs 9-18 months for custom parachain)
- **Hardware**: RTX 3060 12GB minimum for Director nodes (11.8GB VRAM budget)
- **TPS**: Limited to Moonbeam's ~50 TPS (optimized via off-chain reputation batching)
- **Governance**: Runtime upgrades require Moonbeam OpenGov approval

## Timeline

- **Weeks 1-2**: Foundation (Moonbeam fork, dev environment, pallet-icn-stake)
- **Weeks 3-4**: Reputation system (pallet-icn-reputation, Merkle trees)
- **Weeks 5-6**: Director logic (pallet-icn-director, VRF elections, BFT challenge mechanism)
- **Weeks 7-8**: Moonriver deployment and integration testing
- **Weeks 9-16**: Security audit, governance, token launch, mainnet deployment
