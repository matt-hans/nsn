# Neural Sovereign Network (NSN)
# Product Requirements Document v10.0

**Version:** 10.0.0
**Date:** 2025-12-29
**Status:** Dual-Lane Architecture - NSN Polkadot SDK Chain
**Classification:** Approved for Development

---

## Document Control

### Strategic Evolution (v9.0 → v10.0)

| Aspect | v9.0 Approach | v10.0 Dual-Lane Architecture |
|--------|---------------|------------------------------|
| **Architecture** | Video-only (single lane) | **Dual-lane (Lane 0 video + Lane 1 general AI)** |
| **Elections** | Per-slot VRF | **Epoch-based with On-Deck protocol (100 blocks)** |
| **Task Scope** | Video generation only | **Lane 0: video, Lane 1: arbitrary AI compute** |
| **Marketplace** | N/A | **nsn-task-market pallet for Lane 1** |
| **Model Discovery** | Hardcoded | **nsn-model-registry pallet** |
| **Orchestration** | Vortex only | **Vortex (Lane 0) + node-core (Lane 1)** |

### v10.0 Key Changes

- **Dual-Lane Architecture:** Separates deterministic video generation (Lane 0) from open compute marketplace (Lane 1)
- **Epoch-Based Elections:** On-Deck set (20 Directors) with 5 elected per epoch (100 blocks)
- **Task Marketplace:** nsn-task-market pallet enables arbitrary AI workloads (inference, training, fine-tuning)
- **Model Registry:** nsn-model-registry pallet for capability discovery and matching
- **node-core:** Universal compute orchestration layer (scheduler + sidecar) for Lane 1
- **Unified Infrastructure:** Shared stake, reputation, and P2P systems across both lanes

### v8.0.1 Enhancements (Retained)

- **BFT Challenge Period:** On-chain dispute mechanism with 50-block window, stake slashing
- **VRF Election Randomness:** Cryptographically secure director selection using chain randomness
- **Governance-Adjustable Retention:** Reputation pruning as governance parameter
- **Off-Chain Reputation Batching:** TPS optimization via aggregated events
- **Stake-Weighted Audit Probability:** Higher stake = lower audit frequency
- **Reputation-Integrated GossipSub:** On-chain reputation influences P2P peer scoring
- **Dual CLIP Self-Verification:** Directors use CLIP-B + CLIP-L ensemble before BFT

---

## 1. Executive Summary

### 1.1 The Problem

**v8.0 Reality Check:**
- Moonbeam pallet deployment requires **governance referendum approval**
- Creates platform dependency conflicting with "deployable by anyone" vision
- Runtime upgrade control lies with external party (Moonbeam token holders)
- Tight coupling between ICN protocol evolution and external governance

**v9.0 Solution:** Build NSN as its own **Polkadot SDK chain** with custom FRAME pallets, enabling full sovereignty over runtime upgrades and deployment.

**v10.0 Evolution:** Expand NSN from video-only to dual-lane architecture:
- **Lane 0:** Deterministic video generation with epoch-based elections
- **Lane 1:** Open marketplace for arbitrary AI compute tasks

### 1.2 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│              POLKADOT ECOSYSTEM (Future: Shared Security)           │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐    │
│  │  Relay Chain     │ │  Coretime Chain  │ │   Bridge Hub     │    │
│  │  (Security)      │ │  (Scaling)       │ │   (Snowbridge)   │    │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                          │ Future Integration
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ICN CHAIN (Polkadot SDK Runtime)                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              ICN CUSTOM PALLETS (Rust/FRAME)                 │   │
│  │  pallet-nsn-stake | pallet-nsn-reputation | pallet-nsn-pinning │
│  │  pallet-nsn-director | pallet-nsn-bft | pallet-nsn-treasury   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │       OPTIONAL: FRONTIER EVM (ICN Token ERC-20 Interface)    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │ On-Chain Events (subxt)
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OFF-CHAIN LAYER (P2P)                          │
│  Director Nodes (Vortex) | Validator Nodes (CLIP) | Super-Nodes    │
│  Regional Relays | Viewer Nodes                                     │
│  libp2p + QUIC + GossipSub + Kademlia DHT                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Technical Targets

| Metric | Target |
|--------|--------|
| Time to MVP | 3-6 months |
| Development Cost | $80k-$200k |
| Initial Deployment | NSN Solochain (controlled validators) |
| Future Scaling | Parachain + Coretime (on-demand/bulk) |
| Ethereum Access | Optional Frontier EVM or Snowbridge |

### 1.4 Core Features Preserved (from v8.0)

- ✅ Multi-Director BFT (3-of-5 consensus)
- ✅ Semantic verification (CLIP ensemble)
- ✅ Static resident VRAM management
- ✅ Hierarchical swarm architecture
- ✅ NAT traversal stack
- ✅ Erasure coding + pinning incentives
- ✅ Reputation decay + anti-cartel mechanics

---

## 2. Chain Strategy

### 2.1 Why Polkadot SDK Solochain?

| Feature | Benefit |
|---------|---------|
| Full sovereignty | No external governance dependency |
| Permissionless deployment | Anyone can launch ICN network from repo artifacts |
| Fast iteration | Runtime upgrades controlled by ICN team/community |
| Polkadot compatibility | Future parachain migration path via Cumulus |
| Coretime-ready | Designed for Agile Coretime model when scaling needed |
| Substrate ecosystem | Battle-tested FRAME pallets, tooling, libraries |

### 2.2 Staged Deployment Model

```
Phase A: Solochain MVP          Phase B: Parachain           Phase C: Coretime
┌────────────────────┐          ┌────────────────────┐       ┌────────────────────┐
│ Controlled         │          │ Cumulus-enabled    │       │ Bulk/on-demand     │
│ validator set      │   ──▶    │ Polkadot shared    │  ──▶  │ coretime scaling   │
│ (3-5 validators)   │          │ security           │       │ via Broker pallet  │
└────────────────────┘          └────────────────────┘       └────────────────────┘
   Weeks 1-8                       Post-adoption               Post-parachain
```

**Phase A: NSN Solochain MVP (Weeks 1-8)**
- NSN Chain runs with controlled validator set (3-5 trusted operators)
- Full pallet functionality operational
- Fast iteration on chain logic
- Low operational overhead

**Phase B: Parachain Readiness (Post-adoption)**
- Add Cumulus for Polkadot relay chain integration
- Obtain parachain slot via auction or governance
- Inherit Polkadot's ~$20B+ economic security

**Phase C: Coretime Scaling (Post-parachain)**
- Acquire coretime via Broker pallet on Coretime system chain
- On-demand coretime for launch, bulk coretime as adoption grows
- Elastic scaling as needed

### 2.3 ICN Runtime Architecture

```rust
// ICN Runtime Configuration
construct_runtime!(
    pub enum Runtime {
        // System pallets
        System: frame_system,
        Timestamp: pallet_timestamp,
        Balances: pallet_balances,
        TransactionPayment: pallet_transaction_payment,
        Sudo: pallet_sudo,  // Bootstrap governance
        
        // ICN custom pallets
        IcnStake: pallet_nsn_stake,
        IcnReputation: pallet_nsn_reputation,
        IcnDirector: pallet_nsn_director,
        IcnBft: pallet_nsn_bft,
        IcnPinning: pallet_nsn_pinning,
        IcnTreasury: pallet_nsn_treasury,
        
        // Optional: Frontier EVM (Phase D)
        // EVM: pallet_evm,
        // Ethereum: pallet_ethereum,
    }
);
```

### 2.4 Bootstrap Governance Model

| Phase | Governance Model | Upgrade Authority |
|-------|------------------|-------------------|
| MVP (Weeks 1-8) | Sudo (single key) | Core team |
| Testnet (Weeks 9-12) | Multisig (3-of-5) | Trusted operators |
| Mainnet Launch | Council + Technical Committee | Elected members |
| Mature (6+ months) | OpenGov | ICN token holders |

---

## 3. Custom Pallet Specifications

### 3.1 pallet-nsn-stake

**Purpose:** Token staking, slashing, role eligibility, delegation

**Key Storage:**
- `Stakes`: Account → StakeInfo (amount, locked_until, role, region, delegated_to_me)
- `TotalStaked`: Total staked in network
- `RegionStakes`: Region → Balance (anti-centralization)
- `Delegations`: Delegator → (Validator, Amount)

**Node Roles & Minimum Stakes:**
| Role | Min Stake | Max Stake |
|------|-----------|-----------|
| Director | 100 NSN | 1,000 ICN |
| SuperNode | 50 NSN | 500 ICN |
| Validator | 10 NSN | 100 NSN |
| Relay | 5 NSN | 50 NSN |

**Key Extrinsics:**
- `deposit_stake(amount, lock_blocks, region)` - Stake tokens, verify per-node cap (1000 ICN), per-region cap (20%)
- `delegate(validator, amount)` - Delegate to validator (max 5× validator's own stake)
- `slash(offender, amount, reason)` - Slash for protocol violations (root only)

**Anti-Centralization:**
- Per-node cap: 1,000 ICN maximum
- Per-region cap: 20% of total stake
- Delegation cap: 5× validator's own stake

### 3.2 pallet-nsn-reputation

**Purpose:** Verifiable reputation events with Merkle proofs, pruning

**Key Storage:**
- `ReputationScores`: Account → (director_score, validator_score, seeder_score, last_activity)
- `MerkleRoots`: Block → Hash (for proof generation)
- `Checkpoints`: Block → CheckpointData (every 1000 blocks)
- `RetentionPeriod`: Governance-adjustable (~6 months default)
- `AggregatedEvents`: Account → AggregatedReputation (TPS optimization)

**Reputation Events & Deltas:**
| Event | Delta |
|-------|-------|
| DirectorSlotAccepted | +100 |
| DirectorSlotRejected | -200 |
| DirectorSlotMissed | -150 |
| ValidatorVoteCorrect | +5 |
| ValidatorVoteIncorrect | -10 |
| SeederChunkServed | +1 |
| PinningAuditPassed | +10 |
| PinningAuditFailed | -50 |

**Weighted Score:** 50% director + 30% validator + 20% seeder

**Decay:** ~1% per inactive week

**Key Functions:**
- `record_event()` - Record reputation event, update score, add to Merkle tree
- `on_finalize()` - Compute Merkle root, create checkpoints, prune old events

### 3.3 pallet-nsn-director

**Purpose:** Multi-director election, BFT coordination, challenges

**Key Constants:**
- DIRECTORS_PER_SLOT = 5
- BFT_THRESHOLD = 3 (3-of-5)
- COOLDOWN_SLOTS = 20
- JITTER_FACTOR = 20% (±)
- CHALLENGE_PERIOD_BLOCKS = 50 (~5 minutes)
- CHALLENGE_STAKE = 25 NSN

**Key Storage:**
- `CurrentSlot`: Current slot number
- `ElectedDirectors`: Directors for current slot
- `Cooldowns`: Account → Last directed slot
- `BftResults`: Slot → BftConsensusResult
- `PendingChallenges`: Slot → BftChallenge
- `FinalizedSlots`: Slot → bool

**VRF-Based Election:**
1. Get eligible candidates (Director role, past cooldown)
2. Calculate weights: reputation (sublinear scaling) + jitter
3. Boost under-represented regions (max 2 directors per region)
4. Use NSN Chain `T::Randomness` for cryptographically secure selection
5. Select 5 directors for slot + 2 (pipeline ahead)

**BFT Result Submission:**
1. Verify submitter is elected director
2. Verify minimum agreement (3-of-5)
3. Store pending result
4. Start 50-block challenge period
5. After period (no challenge): finalize and record reputation events

**Challenge Mechanism:**
- Any staker can challenge with 25 NSN bond
- Validators provide attestations
- If upheld: Slash 100 NSN per director, refund challenger + 10 NSN reward
- If rejected: Slash challenger bond, finalize original result

### 3.4 pallet-nsn-pinning

**Purpose:** Erasure shard pinning deals, rewards, audits

**Key Constants:**
- REPLICATION_FACTOR = 5
- SHARD_REWARD_PER_BLOCK = 0.001 ICN
- AUDIT_SLASH_AMOUNT = 10 NSN
- BASE_AUDIT_PROBABILITY = 1%/hour
- Stake-weighted: Higher stake = lower audit frequency

**Key Storage:**
- `PinningDeals`: DealId → PinningDeal
- `ShardAssignments`: ShardHash → Vec<Pinners>
- `PinnerRewards`: Account → Balance
- `PendingAudits`: AuditId → PinningAudit

**Key Extrinsics:**
- `create_deal(shards, duration, payment)` - Create pinning deal, assign to top-reputation super-nodes
- `initiate_audit(pinner, shard)` - Random audit with VRF-generated challenge
- `submit_audit_proof(audit_id, proof)` - Pinner responds with proof

**Audit Flow:**
1. Random audit initiated (stake-weighted probability)
2. Pinner has 100 blocks (~10 min) to respond
3. Valid proof → +10 reputation
4. Invalid/timeout → Slash 10 NSN, -50 reputation

### 3.5 pallet-nsn-treasury & pallet-nsn-bft

**Treasury:** Reward distribution, funding

**BFT:** Embeddings hash storage, consensus round tracking

---

## 4. Off-Chain Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Director Nodes | Rust + Vortex Engine | Generate video (Flux + LivePortrait) |
| Validator Nodes | Rust + CLIP | Semantic verification |
| Super-Nodes | Rust + libp2p | Tier 1 relay + erasure storage |
| Regional Relays | Rust + QUIC | Tier 2 distribution |
| Viewer Clients | Tauri + React | Consumption + optional seeding |

### 4.1 Chain Integration

Off-chain nodes use **subxt** to:
- Subscribe to finalized blocks
- Listen for `DirectorsElected` events
- Submit `submit_bft_result()` extrinsics
- Query reputation scores

---

## 5. Ethereum Access Strategy

### 5.1 Track A: Frontier EVM on NSN Chain (Optional)

If EVM compatibility desired for dApp developers:
- Integrate **Frontier** (pallet-evm + pallet-ethereum)
- EVM execution within NSN Chain
- Familiar tooling (Hardhat, Foundry, ethers.js)
- ICN token accessible as ERC-20 via precompile

### 5.2 Track B: Snowbridge (Optional)

If Ethereum mainnet bridging needed:
- Use **Bridge Hub** system chain + Snowbridge
- Trustless Ethereum ↔ Polkadot bridge
- Gateway contract on Ethereum for token/message routing
- Requires ICN to be parachain (Phase B)

### 5.3 ICN Token (Native)

ICN is the **native token** of NSN Chain:
- Total supply: 1B NSN
- Used for transaction fees, staking, slashing
- No ERC-20 required for core functionality
- Optional ERC-20 representation via Frontier (Track A)

---

## 6. Deployment Phases

### Phase A: NSN Solochain MVP (Weeks 1-8)

| Week | Milestone |
|------|-----------|
| 1-2 | NSN Chain bootstrap, dev environment, chain spec |
| 3-4 | pallet-nsn-stake, pallet-nsn-reputation |
| 5-6 | pallet-nsn-director, pallet-nsn-bft |
| 7-8 | NSN Testnet deployment, integration testing |

**Exit Criteria:** All pallets pass tests, NSN Testnet running with 10+ nodes

### Phase B: NSN Mainnet (Weeks 9-16)

| Week | Milestone |
|------|-----------|
| 9-10 | Security audit |
| 11-12 | Genesis configuration, validator onboarding |
| 13-14 | Token genesis event |
| 15-16 | Mainnet launch |

**Governance:** sudo initially, migrate to multisig by week 12

### Phase C: Parachain Migration (Post-adoption)

- Add Cumulus integration
- Acquire parachain slot
- Migrate to Polkadot shared security
- Begin coretime acquisition planning

### Phase D: Ethereum Integration (When Needed)

- Option A: Enable Frontier EVM on NSN Chain
- Option B: Snowbridge integration via Bridge Hub

---

## 7. Cost & Resource Estimate

| Category | Cost (USD) |
|----------|------------|
| Development (2-3 devs × 6 months) | $60k-$120k |
| Security Audit | $20k-$60k |
| Infrastructure (validators, RPC nodes) | $5k-$15k |
| Legal & Token | $5k-$10k |
| Contingency (15%) | $15k-$30k |
| **TOTAL** | **$105k-$235k** |

### Comparison with Moonbeam Approach

| Aspect | Moonbeam Pallets (v8.0) | NSN Chain (v9.0) |
|--------|-------------------------|------------------|
| Development | $60k-$120k | $70k-$130k |
| Time to MVP | 3-6 months + governance | 3-6 months (no dependency) |
| Governance Risk | HIGH (referendum required) | NONE (own chain) |
| Year 1 Operations | ~$50k (Moonbeam gas) | ~$20k (own validators) |

---

## 8. Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Solo chain security (small validator set) | High | Conservative slashing, low initial TVL, trusted operators |
| Validator key compromise | High | HSM, multisig rotation, geographic distribution |
| Runtime bugs | High | Testnet phase, staged rollout, sudo recovery |
| Parachain migration complexity | Medium | Design pallets Cumulus-compatible from start |
| Off-chain BFT trust | High | Challenge period, slashing |
| Token regulatory | High | Legal review, utility focus |
| Director collusion | High | Multi-region requirement, challenges |

### Key Mitigation Strategies

**Solo Chain Security:** Start with low TVL, use trusted validator set, implement conservative slashing parameters. Migrate to parachain when adoption justifies shared security cost.

**Runtime Upgrades:** Sudo/multisig can perform emergency upgrades. Forkless runtime upgrades via `set_code`. No external approval needed.

**Validator Operations:** Document runbooks, 24/7 monitoring, automated alerting, backup validators ready.

---

## 9. Success Criteria

| Metric | Phase A (NSN Testnet) | Phase B (NSN Mainnet) |
|--------|----------------------|----------------------|
| Chain operational | ✓ | ✓ |
| Staking functional | 100% test coverage | Live tokens |
| Elections working | 10+ test nodes | 50+ mainnet nodes |
| BFT consensus | Simulated | Live 3-of-5 |
| Community | 50+ testnet users | 500+ mainnet users |
| Validator uptime | 95%+ | 99.5%+ |

---

## 10. Vortex Generation Engine

### 10.1 Static Resident VRAM Layout

**Critical:** All models remain loaded in VRAM at all times. No swapping.

| Component | Model | Precision | VRAM |
|-----------|-------|-----------|------|
| Actor Generation | Flux-Schnell | NF4 (4-bit) | ~6.0 GB |
| Video Warping | LivePortrait | FP16 | ~3.5 GB |
| Text-to-Speech | Kokoro-82M | FP32 | ~0.4 GB |
| Semantic Verify (Primary) | CLIP-ViT-B-32 | INT8 | ~0.3 GB |
| Semantic Verify (Secondary) | CLIP-ViT-L-14 | INT8 | ~0.6 GB |
| System Overhead | PyTorch/CUDA | - | ~1.0 GB |
| **TOTAL** | | | **~11.8 GB** |

**Minimum GPU:** RTX 3060 12GB

### 10.2 Generation Pipeline

1. **Parallel Phase (0-12s):** Audio (Kokoro) + Actor image (Flux) generated simultaneously
2. **Sequential Phase (12-15s):** Video warping (LivePortrait) using both outputs
3. **Verification (15-17s):** Dual CLIP embedding computation + self-check

**Dual CLIP Self-Verification:**
- CLIP-B-32 (weight 0.4) + CLIP-L-14 (weight 0.6)
- Both must pass thresholds (B: 0.70, L: 0.72)
- Reduces off-chain disputes by ~40%
- Combined embedding used for BFT exchange

### 10.3 Slot Timing Budget (45 seconds)

| Phase | Time | Activities |
|-------|------|------------|
| Generation | 0-12s | Audio + Actor (parallel), Video warp |
| BFT | 12-17s | Exchange embeddings, compute agreement, submit to chain |
| Propagation | 17-30s | Super-nodes → Relays → Viewers |
| Buffer | 30-45s | Viewer playback buffer |

---

## 11. Security Model

### 11.1 Security Layers

| Layer | Component | Protection |
|-------|-----------|------------|
| 1 | ICN Validator Set | Consensus security (solo phase) |
| 2 | Polkadot Relay Chain | $20B+ economic security (parachain phase) |
| 3 | ICN Staking (Pallet) | Sybil resistance |
| 4 | Reputation Slashing | Punishment |
| 5 | CLIP Semantic Verify | Content quality |
| 6 | Ed25519 Signatures | Authenticity |
| 7 | Sandboxed Vortex | RCE protection |
| 8 | E2E Encryption | Viewer privacy |

### 11.2 Slash Reasons

- BftFailure, AuditTimeout, AuditInvalid, MissedSlot, ContentViolation

### 11.3 CLIP Adversarial Hardening

**Ensemble approach:**
- CLIP-ViT-B-32 (0.3 weight) + CLIP-ViT-L-14 (0.5 weight) + CLIP-RN50 (0.2 weight, future)
- Outlier detection on embedding variance
- Prompt sanitization
- Human escalation for borderline cases

---

## 12. Tokenomics

### 12.1 Token Distribution

| Allocation | Percentage | Vesting |
|------------|------------|---------|
| Community Rewards | 40% | 4-year linear |
| Development Fund | 20% | 2-year cliff, 2-year vest |
| Ecosystem Growth | 15% | Grants, partnerships |
| Team & Advisors | 15% | 1-year cliff, 3-year vest |
| Initial Liquidity | 10% | Immediate |

**Total Supply:** 1B NSN (native token of NSN Chain)

### 12.2 Token Utility

- **Staking:** Lock ICN for roles
- **Slashing:** Forfeit for violations
- **Delegation:** Share rewards
- **Pinning Rewards:** Earn for storage
- **Governance:** Vote on proposals
- **Transaction Fees:** Pay for on-chain operations

### 12.3 Emission Schedule

Year 1: 100M NSN, then 15% annual decay

**Distribution:** 40% directors, 25% validators, 20% pinners, 15% treasury

### 12.4 Staking Details

| Role | Min Stake | Lock Period | Slashing Risk |
|------|-----------|-------------|---------------|
| Director | 100 NSN | 30 days | 50 NSN/violation |
| SuperNode | 50 NSN | 14 days | 20 ICN/violation |
| Validator | 10 NSN | 7 days | 5 NSN/violation |
| Relay | 5 NSN | 3 days | 1 ICN/violation |

---

## 13. P2P Network Layer

### 13.1 NAT Traversal Stack

Connection strategies (tried in order):
1. **Direct** - No NAT / port forwarded
2. **STUN** - UDP hole punching
3. **UPnP** - Automatic port forwarding
4. **Circuit Relay** - libp2p relay (incentivized)
5. **TURN** - Fallback (expensive)

Circuit relay rewarded: 0.01 ICN/hour

### 13.2 Hierarchical Swarm

```
TIER 0: Directors (100+ ICN, RTX 3060+, 100 Mbps)
    │
    ▼
TIER 1: Super-Nodes (50+ ICN, 10TB, 500 Mbps) - 7 regions
    │
    ▼
TIER 2: Regional Relays (10+ ICN, 100 Mbps) - Auto-scaled
    │
    ▼
TIER 3: Edge Viewers (Permissionless) - Optional seeding
```

### 13.3 GossipSub Configuration

**Topics:**
- `/icn/recipes/1.0.0` - Recipe JSON
- `/icn/video/1.0.0` - Video chunks (16MB max)
- `/icn/bft/1.0.0` - BFT signals (critical)
- `/icn/attestations/1.0.0` - Validator attestations
- `/icn/challenges/1.0.0` - Challenges

**Reputation-Integrated Peer Scoring:**
- On-chain reputation cached locally (sync every 60s)
- Score 0-1000 → 0-50 GossipSub boost
- Topics weighted: BFT (3.0) > Video (2.0) > Recipes (1.0)

**Thresholds:**
- gossip_threshold: -10
- publish_threshold: -50
- graylist_threshold: -100

### 13.4 Bootstrap Protocol

**Multi-layer discovery:**
1. Hardcoded peers (trusted)
2. DNS seeds (signed TXT records)
3. HTTP endpoints (signed JSON)
4. DHT walk (after connecting)

All manifests require Ed25519 signatures from trusted signers.

---

## 14. Recipe Schema (v2.0)

```json
{
  "recipe_id": "UUID",
  "version": "2.0.0",
  "slot_params": {
    "slot_number": 12345,
    "duration_sec": 45,
    "resolution": "512x512",
    "fps": 24
  },
  "audio_track": {
    "script": "...",
    "voice_id": "rick_c137",
    "speed": 1.1,
    "emotion": "manic"
  },
  "visual_track": {
    "prompt": "...",
    "negative_prompt": "...",
    "motion_preset": "excited_nodding",
    "expression_sequence": ["neutral", "excited"],
    "camera_motion": "slight_zoom_in"
  },
  "semantic_constraints": {
    "min_clip_score": 0.75,
    "banned_concepts": ["violence", "nsfw"],
    "required_concepts": ["scientist"]
  },
  "security": {
    "director_id": "PeerId",
    "ed25519_signature": "...",
    "timestamp": 1734983955
  }
}
```

---

## 15. Development Roadmap

### Sprint 1-2 (Weeks 1-4)
- [ ] Bootstrap NSN Chain (Polkadot SDK template)
- [ ] Define chain spec and genesis configuration
- [ ] Implement pallet-nsn-stake
- [ ] Implement pallet-nsn-reputation
- [ ] Local multi-node testnet

### Sprint 3-4 (Weeks 5-8)
- [ ] Implement pallet-nsn-director
- [ ] Implement pallet-nsn-bft
- [ ] Off-chain node integration
- [ ] Deploy ICN Public Testnet

### Sprint 5-6 (Weeks 9-12)
- [ ] Implement pallet-nsn-pinning
- [ ] Implement pallet-nsn-treasury
- [ ] Security audit
- [ ] Validator onboarding

### Sprint 7-8 (Weeks 13-16)
- [ ] Genesis configuration finalization
- [ ] Token launch
- [ ] Mainnet deployment
- [ ] Community onboarding

---

## 16. Appendix A: Pallet Interface Summary

| Pallet | Key Extrinsics | Key Storage |
|--------|----------------|-------------|
| icn-stake | deposit_stake, delegate, slash, withdraw | Stakes, Delegations, RegionStakes |
| icn-reputation | record_event | ReputationScores, MerkleRoots, Checkpoints |
| icn-director | submit_bft_result, challenge_bft_result | ElectedDirectors, BftResults, PendingChallenges |
| icn-pinning | create_deal, initiate_audit, submit_audit_proof | PinningDeals, ShardAssignments, PendingAudits |
| icn-treasury | distribute_rewards, fund_treasury | TreasuryBalance, RewardSchedule |
| icn-bft | submit_embeddings_hash | EmbeddingsHashes, ConsensusRounds |

---

## 17. Appendix B: Glossary

| Term | Definition |
|------|------------|
| **BFT** | Byzantine Fault Tolerance – 3-of-5 consensus |
| **CLIP** | Contrastive Language-Image Pretraining |
| **Coretime** | Polkadot's execution time allocation model |
| **Cumulus** | Framework for building Polkadot parachains |
| **Director** | High-stake node that generates video |
| **Erasure Coding** | Reed-Solomon (10+4) redundancy |
| **FRAME** | Substrate's modular runtime framework |
| **Frontier** | Substrate EVM compatibility layer |
| **NSN Chain** | ICN's Polkadot SDK-based blockchain |
| **Pallet** | Substrate runtime module |
| **Recipe** | JSON instruction for AI generation |
| **Slot** | 45-90 second generation window |
| **Snowbridge** | Trustless Polkadot ↔ Ethereum bridge |
| **Solochain** | Standalone blockchain (not parachain) |
| **SuperNode** | Regional storage/relay infrastructure |
| **Vortex** | ICN's AI generation engine |
| **VRF** | Verifiable Random Function |

---

## 18. Appendix C: External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Polkadot SDK | polkadot-stable2409 | Runtime framework |
| Cumulus | Latest stable | Parachain support (Phase B) |
| Frontier | Latest stable | Optional EVM (Phase D) |
| libp2p | 0.53.0 | P2P networking |
| Flux-Schnell | NF4 | Image generation |
| LivePortrait | v1.0 | Video warping |
| Kokoro-82M | v1.0 | Text-to-speech |
| CLIP-ViT | B-32, L-14 | Semantic verification |
| PyTorch | 2.1+ | ML runtime |

---

**Document Status:** APPROVED FOR DEVELOPMENT
**Architecture:** NSN Dual-Lane Architecture (v10.0)
**Target MVP:** Q2 2026
**Estimated Cost:** $105k-$235k
**Timeline:** 3-6 months
**Last Updated:** 2025-12-29

*End of Document*
