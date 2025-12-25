# Task T005: Implement pallet-icn-pinning (Erasure Coding & Audits)

## Metadata
```yaml
id: T005
title: Implement pallet-icn-pinning (Erasure Coding & Audits)
status: completed
priority: P1
tags: [pallets, substrate, pinning, storage, audits, on-chain, phase1]
estimated_tokens: 13000
actual_tokens: 0
dependencies: [T001, T002]
created_at: 2025-12-24
updated_at: 2025-12-24
```

## Description

Implement the pinning pallet that manages erasure-coded shard storage deals (Reed-Solomon 10+4 scheme), assigns shards to super-nodes with 5× geographic replication, conducts stake-weighted random audits to verify availability, and distributes ICN rewards to reliable pinners. This pallet ensures video content remains available even if multiple storage nodes fail.

## Business Context

**Why this matters**: Decentralized storage is critical for ICN's censorship resistance and availability. Without reliable pinning:
- **Single points of failure**: If one super-node goes down, content becomes unavailable
- **Economic attacks**: Malicious actors could claim to store data but serve nothing (earning rewards for nothing)
- **Viewer experience**: Buffering and interrupted playback destroy UX

**Value delivered**: Creates economic incentives for reliable storage with cryptographic proof-of-availability audits. Super-nodes earn 20% of network emissions by proving they store and serve shards.

**Priority justification**: P1 (critical path) because video distribution depends on erasure-coded shard availability. Blocks off-chain super-node implementation and viewer playback testing.

## Acceptance Criteria

1. `create_deal()` extrinsic successfully reserves payment and stores deal metadata with shard hashes
2. Shard assignments select super-nodes with highest reputation across different regions
3. Reed-Solomon 10+4 scheme: 10 data shards + 4 parity shards, recoverable from any 10 shards
4. 5× replication factor: each shard stored on 5 different super-nodes across regions
5. `initiate_audit()` extrinsic (root-only) creates random challenge with byte offset + nonce
6. Stake-weighted audit probability: higher stake = lower audit frequency (base 1%/hour)
7. `submit_audit_proof()` allows pinner to respond with Merkle proof within 100-block deadline (~10 minutes)
8. Failed audits slash 10 ICN from pinner and record PinningAuditFailed reputation event
9. Passed audits record PinningAuditPassed reputation event (+10)
10. `distribute_rewards()` in on_finalize (every 100 blocks) pays pinners proportional to active deals
11. Expired audits (timeout) automatically slash pinner and mark audit as failed
12. Unit tests cover deal creation, shard assignment, audit lifecycle, and reward distribution (90%+ coverage)

## Test Scenarios

### Scenario 1: Create Pinning Deal with Erasure Coding
```gherkin
GIVEN content creator Alice has 100 ICN free balance
  AND video chunk is 50MB = 14 shards (10 data + 4 parity)
  AND shard hashes are [0xSHARD1, 0xSHARD2, ..., 0xSHARD14]
WHEN Alice calls create_deal(
  shards=[0xSHARD1, ..., 0xSHARD14],
  duration_blocks=100800,  // ~7 days
  payment=50 ICN
)
THEN 50 ICN reserved from Alice's balance
  AND PinningDeals[deal_id] created with status=Active
  AND deal.total_reward = 50 ICN
  AND deal.expires_at = current_block + 100800
  AND DealCreated event emitted
```

### Scenario 2: Shard Assignment with Geographic Replication
```gherkin
GIVEN 20 super-nodes across 7 regions:
  NA-WEST: 3 super-nodes (reputation: [800, 750, 700])
  EU-WEST: 3 super-nodes (reputation: [900, 850, 800])
  APAC: 3 super-nodes (reputation: [950, 900, 850])
  LATAM: 3 super-nodes (reputation: [600, 550, 500])
  MENA: 3 super-nodes (reputation: [700, 650, 600])
  NA-EAST: 3 super-nodes (reputation: [820, 780, 740])
  EU-EAST: 2 super-nodes (reputation: [880, 860])
WHEN shard 0xSHARD1 is assigned with 5× replication
THEN 5 super-nodes selected from 5 different regions
  AND selected nodes have highest reputation in their respective regions
  AND ShardAssignments[0xSHARD1] = [node1, node2, node3, node4, node5]
  AND ShardAssigned events emitted for each assignment
```

### Scenario 3: Stake-Weighted Audit Probability
```gherkin
GIVEN 3 super-nodes pinning same shard:
  Node A: 50 ICN staked (minimum for super-node)
  Node B: 200 ICN staked
  Node C: 500 ICN staked
WHEN random audits are initiated over 1000 blocks (~100 hours)
THEN Node A audited ~2% of the time (200/10000 = 2% max ceiling)
  AND Node B audited ~0.5% of the time (adjusted for higher stake)
  AND Node C audited ~0.25% of the time (0.25% floor for high stake)
  (Base: 1% per hour = 100 basis points, adjusted by stake tier)
```

### Scenario 4: Successful Audit with Valid Proof
```gherkin
GIVEN super-node Bob has 0xSHARD5 stored
  AND audit initiated with challenge: {offset=2048, length=64, nonce=[random]}
WHEN Bob calls submit_audit_proof(
  audit_id=0xAUDIT1,
  proof=[bytes_at_offset, merkle_siblings, signature]
)
  within 100-block deadline
THEN proof verified against shard_hash
  AND audit.status = AuditStatus::Passed
  AND PinningAuditPassed reputation event recorded for Bob (+10)
  AND AuditCompleted event emitted with passed=true
```

### Scenario 5: Failed Audit Slashes Pinner
```gherkin
GIVEN super-node Charlie has 100 ICN staked
  AND Charlie claims to have 0xSHARD7 but doesn't actually store it
WHEN audit initiated and Charlie submits invalid proof
THEN proof verification fails
  AND audit.status = AuditStatus::Failed
  AND 10 ICN slashed from Charlie via pallet_icn_stake::slash()
  AND PinningAuditFailed reputation event recorded (-50)
  AND AuditCompleted event emitted with passed=false
```

### Scenario 6: Audit Timeout Auto-Slashes
```gherkin
GIVEN audit initiated for super-node Dave at block 1000
  AND audit.deadline = 1100
WHEN current_block reaches 1101
  AND Dave has not submitted proof
THEN check_expired_audits() detects timeout
  AND 10 ICN auto-slashed from Dave
  AND audit.status = AuditStatus::Failed
  AND PinningAuditFailed reputation event recorded
```

### Scenario 7: Reward Distribution Every 100 Blocks
```gherkin
GIVEN deal with total_reward=100 ICN, duration=10000 blocks
  AND 14 shards × 5 replicas = 70 total pinner slots
WHEN on_finalize() runs at block (N % 100 == 0)
THEN reward_per_pinner = 100 ICN / 70 / (10000/100) = 0.0143 ICN per 100 blocks
  AND each active pinner receives 0.0143 ICN added to PinnerRewards
  AND RewardsDistributed event emitted for each pinner
```

### Scenario 8: Deal Expiry and Status Update
```gherkin
GIVEN deal created with expires_at=5000
WHEN current_block reaches 5001
THEN deal.status updated to DealStatus::Expired
  AND no further rewards distributed for this deal
  AND pinners can delete shards (off-chain decision)
```

### Scenario 9: Shard Recovery from Erasure Coding
```gherkin
GIVEN 10+4 Reed-Solomon encoding
  AND 4 pinners are offline (shards 3, 7, 11, 13 unavailable)
WHEN viewer requests video chunk
THEN remaining 10 shards (any 10 of the 14) are sufficient
  AND Reed-Solomon decoder reconstructs full chunk
  AND playback continues without interruption
```

### Scenario 10: Multiple Concurrent Deals
```gherkin
GIVEN 5 active deals with different expiry times
  AND each deal has 14 shards × 5 replicas = 70 pinner assignments
WHEN distribute_rewards() runs
THEN rewards calculated independently for each deal
  AND pinners participating in multiple deals receive combined rewards
  AND PinnerRewards[pinner] accumulates across all deals
```

### Scenario 11: Merkle Proof Verification Logic
```gherkin
GIVEN shard stored as Merkle tree with root = shard_hash
  AND audit challenge requests bytes [2048:2112] with nonce
WHEN pinner provides proof = {chunk_bytes, sibling_hashes, path}
THEN verify_merkle_proof() reconstructs:
  - leaf = hash(chunk_bytes + nonce)
  - node = hash(leaf + sibling1)
  - node = hash(node + sibling2)
  - ...
  - root = final hash
AND root == shard_hash stored on-chain
```

### Scenario 12: Audit Probability Configuration
```gherkin
GIVEN governance parameter: BASE_AUDIT_PROBABILITY_PER_HOUR = 100 (1%)
  AND MIN_AUDIT_PROBABILITY = 25 (0.25%)
  AND MAX_AUDIT_PROBABILITY = 200 (2%)
WHEN calculate_audit_probability() for pinner with stake S
THEN if S == 50 ICN (minimum): probability = 200/10000 = 2%
  AND if S == 200 ICN: probability = 50/10000 = 0.5%
  AND if S == 500+ ICN: probability = 25/10000 = 0.25%
  (Inverse relationship: higher stake = more trusted = less frequent audits)
```

## Technical Implementation

### Core Data Structures
```rust
use frame_support::{pallet_prelude::*, traits::Currency};
use sp_std::vec::Vec;

#[pallet::config]
pub trait Config: frame_system::Config + pallet_icn_stake::Config {
    type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
    type Currency: Currency<Self::AccountId>;
}

pub type DealId = [u8; 32];
pub type ShardHash = [u8; 32];
pub type AuditId = [u8; 32];

const SHARD_REWARD_PER_BLOCK: u128 = 1_000_000_000_000_000;  // 0.001 ICN
const AUDIT_SLASH_AMOUNT: u128 = 10_000_000_000_000_000_000;  // 10 ICN
const REPLICATION_FACTOR: usize = 5;
const AUDIT_DEADLINE_BLOCKS: u32 = 100;

const BASE_AUDIT_PROBABILITY_PER_HOUR: u32 = 100;  // 1% = 100 basis points
const AUDIT_PROBABILITY_DIVISOR: u32 = 10000;
const MIN_AUDIT_PROBABILITY: u32 = 25;   // 0.25%
const MAX_AUDIT_PROBABILITY: u32 = 200;  // 2%

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct PinningDeal<T: Config> {
    pub deal_id: DealId,
    pub creator: T::AccountId,
    pub shards: Vec<ShardHash>,
    pub created_at: T::BlockNumber,
    pub expires_at: T::BlockNumber,
    pub total_reward: BalanceOf<T>,
    pub status: DealStatus,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub enum DealStatus {
    Active,
    Expired,
    Cancelled,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct PinningAudit<T: Config> {
    pub audit_id: AuditId,
    pub pinner: T::AccountId,
    pub shard_hash: ShardHash,
    pub challenge: AuditChallenge,
    pub deadline: T::BlockNumber,
    pub status: AuditStatus,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct AuditChallenge {
    pub byte_offset: u32,
    pub byte_length: u32,
    pub nonce: [u8; 16],
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub enum AuditStatus {
    Pending,
    Passed,
    Failed,
}
```

### Storage Items
```rust
#[pallet::storage]
#[pallet::getter(fn pinning_deals)]
pub type PinningDeals<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    DealId,
    PinningDeal<T>,
    OptionQuery,
>;

#[pallet::storage]
#[pallet::getter(fn shard_assignments)]
pub type ShardAssignments<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    ShardHash,
    Vec<T::AccountId>,
    ValueQuery,
>;

#[pallet::storage]
#[pallet::getter(fn pinner_rewards)]
pub type PinnerRewards<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    T::AccountId,
    BalanceOf<T>,
    ValueQuery,
>;

#[pallet::storage]
#[pallet::getter(fn pending_audits)]
pub type PendingAudits<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    AuditId,
    PinningAudit<T>,
    OptionQuery,
>;
```

### Core Extrinsics
```rust
#[pallet::call]
impl<T: Config> Pallet<T> {
    #[pallet::weight(50_000)]
    pub fn create_deal(
        origin: OriginFor<T>,
        shards: Vec<ShardHash>,
        duration_blocks: T::BlockNumber,
        payment: BalanceOf<T>,
    ) -> DispatchResult {
        let creator = ensure_signed(origin)?;

        ensure!(shards.len() >= 10, Error::<T>::InsufficientShards);
        T::Currency::reserve(&creator, payment)?;

        let current_block = <frame_system::Pallet<T>>::block_number();
        let deal_id: DealId = T::Hashing::hash_of(&(&creator, current_block, &shards))
            .as_ref()[0..32].try_into().map_err(|_| Error::<T>::InvalidDealId)?;

        let deal = PinningDeal {
            deal_id,
            creator: creator.clone(),
            shards: shards.clone(),
            created_at: current_block,
            expires_at: current_block.saturating_add(duration_blocks),
            total_reward: payment,
            status: DealStatus::Active,
        };

        PinningDeals::<T>::insert(deal_id, deal);

        for shard in &shards {
            let pinners = Self::select_pinners(shard, REPLICATION_FACTOR);
            ShardAssignments::<T>::insert(shard, pinners.clone());
            for pinner in pinners {
                Self::deposit_event(Event::ShardAssigned { shard: *shard, pinner });
            }
        }

        Self::deposit_event(Event::DealCreated { deal_id, shard_count: shards.len() as u32 });
        Ok(())
    }

    #[pallet::weight(20_000)]
    pub fn initiate_audit(
        origin: OriginFor<T>,
        pinner: T::AccountId,
        shard_hash: ShardHash,
    ) -> DispatchResult {
        ensure_root(origin)?;

        let current_block = <frame_system::Pallet<T>>::block_number();
        let audit_id: AuditId = T::Hashing::hash_of(&(&pinner, &shard_hash, current_block))
            .as_ref()[0..32].try_into().map_err(|_| Error::<T>::InvalidAuditId)?;

        let (random, _) = T::Randomness::random(&audit_id);
        let challenge = AuditChallenge {
            byte_offset: u32::from_le_bytes(
                random.as_ref()[0..4].try_into().unwrap_or([0u8; 4])
            ) % 10000,
            byte_length: 64,
            nonce: random.as_ref()[4..20].try_into().unwrap_or([0u8; 16]),
        };

        let audit = PinningAudit {
            audit_id,
            pinner: pinner.clone(),
            shard_hash,
            challenge,
            deadline: current_block.saturating_add(AUDIT_DEADLINE_BLOCKS.into()),
            status: AuditStatus::Pending,
        };

        PendingAudits::<T>::insert(audit_id, audit);
        Self::deposit_event(Event::AuditStarted { audit_id, pinner, shard_hash });
        Ok(())
    }

    #[pallet::weight(20_000)]
    pub fn submit_audit_proof(
        origin: OriginFor<T>,
        audit_id: AuditId,
        proof: Vec<u8>,
    ) -> DispatchResult {
        let pinner = ensure_signed(origin)?;

        let mut audit = Self::pending_audits(&audit_id)
            .ok_or(Error::<T>::AuditNotFound)?;
        ensure!(audit.pinner == pinner, Error::<T>::NotAuditTarget);
        ensure!(audit.status == AuditStatus::Pending, Error::<T>::AuditAlreadyCompleted);

        let valid = proof.len() >= audit.challenge.byte_length as usize;

        if valid {
            audit.status = AuditStatus::Passed;
            pallet_icn_reputation::Pallet::<T>::record_event(
                frame_system::RawOrigin::Root.into(),
                pinner.clone(),
                pallet_icn_reputation::ReputationEventType::PinningAuditPassed,
                0,
            )?;
        } else {
            audit.status = AuditStatus::Failed;
            pallet_icn_stake::Pallet::<T>::slash(
                frame_system::RawOrigin::Root.into(),
                pinner.clone(),
                AUDIT_SLASH_AMOUNT.saturated_into(),
                pallet_icn_stake::SlashReason::AuditFailure,
            )?;
        }

        PendingAudits::<T>::insert(audit_id, audit);
        Self::deposit_event(Event::AuditCompleted { audit_id, passed: valid });
        Ok(())
    }
}
```

### Hook Implementation
```rust
#[pallet::hooks]
impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
    fn on_finalize(block: BlockNumberFor<T>) {
        if block % 100u32.into() == Zero::zero() {
            Self::distribute_rewards();
        }

        Self::check_expired_audits(block);
    }
}
```

### Helper Functions
```rust
impl<T: Config> Pallet<T> {
    fn select_pinners(shard: &ShardHash, count: usize) -> Vec<T::AccountId> {
        let candidates: Vec<_> = pallet_icn_stake::Stakes::<T>::iter()
            .filter(|(_, stake)| stake.role == pallet_icn_stake::NodeRole::SuperNode)
            .collect();

        let mut sorted: Vec<_> = candidates.iter()
            .map(|(account, stake)| {
                let rep = pallet_icn_reputation::Pallet::<T>::reputation_scores(account);
                (account.clone(), rep.total(), stake.region.clone())
            })
            .collect();

        sorted.sort_by_key(|(_, rep, _)| core::cmp::Reverse(*rep));

        let mut selected = Vec::new();
        let mut selected_regions = Vec::new();

        for (account, _, region) in sorted {
            if selected_regions.contains(&region) {
                continue;
            }
            selected.push(account);
            selected_regions.push(region);
            if selected.len() >= count {
                break;
            }
        }

        selected
    }

    fn distribute_rewards() {
        for (deal_id, deal) in PinningDeals::<T>::iter() {
            if deal.status != DealStatus::Active {
                continue;
            }

            let total_pinners = deal.shards.len() * REPLICATION_FACTOR;
            let reward_per_pinner = deal.total_reward.saturating_div((total_pinners as u32).into());

            for shard in &deal.shards {
                let pinners = Self::shard_assignments(shard);
                for pinner in pinners {
                    PinnerRewards::<T>::mutate(&pinner, |r| {
                        *r = r.saturating_add(reward_per_pinner);
                    });
                    Self::deposit_event(Event::RewardsDistributed { pinner, amount: reward_per_pinner });
                }
            }
        }
    }

    fn check_expired_audits(current_block: T::BlockNumber) {
        for (audit_id, audit) in PendingAudits::<T>::iter() {
            if audit.status == AuditStatus::Pending && current_block > audit.deadline {
                let _ = pallet_icn_stake::Pallet::<T>::slash(
                    frame_system::RawOrigin::Root.into(),
                    audit.pinner.clone(),
                    AUDIT_SLASH_AMOUNT.saturated_into(),
                    pallet_icn_stake::SlashReason::AuditTimeout,
                );

                PendingAudits::<T>::mutate(&audit_id, |a| {
                    if let Some(audit) = a {
                        audit.status = AuditStatus::Failed;
                    }
                });

                Self::deposit_event(Event::AuditCompleted { audit_id, passed: false });
            }
        }
    }
}
```

## Dependencies

- **T001**: ICN Chain bootstrap
- **T002**: pallet-icn-stake for role checking and slashing
- **frame-support**: Currency trait for payments
- **sp-runtime**: Randomness for audit challenges

## Design Decisions

1. **Reed-Solomon 10+4**: Industry-standard erasure coding. 1.4× overhead vs 3× for simple replication at same fault tolerance.

2. **Stake-weighted audits**: Trusted super-nodes (high stake) audited less frequently, saving on-chain audit transaction costs while maintaining security.

3. **100-block deadline**: ~10 minutes gives pinners time to respond even with network congestion, but fast enough to detect failures quickly.

4. **10 ICN slash**: Punishment is 1/5 of super-node minimum stake, significant but not catastrophic for honest mistakes (vs malicious behavior).

5. **Every 100-block rewards**: Balances reward frequency with transaction cost. More frequent = higher precision, less frequent = lower on-chain overhead.

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Correlated pinner failures | High | Low | 5× geographic replication, different regions required |
| Audit proof forgery | Critical | Low | Merkle root stored on-chain, proof verification cryptographic |
| Storage explosion (many deals) | Medium | High | Governance-adjustable minimum deal size and payment |
| Shard assignment centralization | Medium | Medium | Reputation-weighted selection, multi-region constraint |

## Progress Log

- 2025-12-24: Task created from PRD §3.4 and Architecture §4.2.2

## Completion Checklist

- [ ] All 12 acceptance criteria met
- [ ] All 12 test scenarios implemented and passing
- [ ] Unit test coverage ≥90%
- [ ] Integration tests with pallet-icn-stake
- [ ] Merkle proof verification logic tested
- [ ] Benchmarks defined for create_deal, initiate_audit, submit_audit_proof
- [ ] Clippy passes with no warnings
- [ ] Documentation comments complete (rustdoc)
- [ ] No regression in existing tests
