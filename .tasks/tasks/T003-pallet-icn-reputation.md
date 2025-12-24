# Task T003: Implement pallet-icn-reputation (Reputation Scoring & Merkle Proofs)

## Metadata
```yaml
id: T003
title: Implement pallet-icn-reputation (Reputation Scoring & Merkle Proofs)
status: pending
priority: P1
tags: [pallets, substrate, reputation, on-chain, phase1, merkle-tree]
estimated_tokens: 14000
actual_tokens: 0
dependencies: [T001, T002]
created_at: 2025-12-24
updated_at: 2025-12-24
```

## Description

Implement the reputation pallet that tracks director/validator/seeder performance through weighted scoring, batches events into Merkle trees for off-chain proof generation, and manages checkpointing and pruning. This pallet provides verifiable reputation data that influences director election probability and peer scoring.

## Business Context

**Why this matters**: Reputation is the quality signal for the ICN network. It determines:
- **Director election probability**: Higher reputation = higher weight in VRF-based selection
- **P2P peer scoring**: GossipSub integrates on-chain reputation to prioritize reliable nodes
- **Slashing severity**: Repeated failures compound reputation loss
- **Trust minimization**: Merkle proofs allow off-chain verification without full chain sync

**Value delivered**: Creates a trustless quality metric that rewards consistent performance and punishes bad actors. Without reputation, the network cannot distinguish good from malicious nodes.

**Priority justification**: P1 because director election (T004) depends on reputation scores for weighted selection. Must be completed before BFT coordination can be tested.

## Acceptance Criteria

1. `ReputationScore` storage correctly tracks three weighted components: director_score (50%), validator_score (30%), seeder_score (20%)
2. `record_event()` extrinsic (root-only) successfully applies delta based on event type
3. Event deltas match specification:
   - DirectorSlotAccepted: +100
   - DirectorSlotRejected: -200
   - DirectorSlotMissed: -150
   - ValidatorVoteCorrect: +5
   - ValidatorVoteIncorrect: -10
   - SeederChunkServed: +1
   - PinningAuditPassed: +10
   - PinningAuditFailed: -50
4. `apply_decay()` function correctly applies 5% weekly decay to inactive accounts
5. Merkle tree construction produces deterministic roots for event batches
6. `on_finalize()` hook publishes Merkle root for each block's events
7. Checkpoint created every 1000 blocks with aggregated reputation snapshot
8. Governance-adjustable retention period (default 6 months, ~2.6M blocks)
9. `prune_old_events()` removes Merkle roots beyond retention period
10. Off-chain reputation aggregation batches multiple events per account before on-chain submission
11. Unit tests cover weighted scoring, decay, Merkle proofs, and pruning (90%+ coverage)
12. Integration tests verify inter-pallet calls from director/pinning pallets

## Test Scenarios

### Scenario 1: Weighted Reputation Scoring
```gherkin
GIVEN account Alice has reputation: director=0, validator=0, seeder=0
WHEN the following events are recorded:
  - DirectorSlotAccepted (+100 director)
  - DirectorSlotAccepted (+100 director)
  - ValidatorVoteCorrect (+5 validator)
  - SeederChunkServed (+1 seeder)
THEN Alice's scores are: director=200, validator=5, seeder=1
  AND total() = (200*50 + 5*30 + 1*20) / 100 = 100.7
```

### Scenario 2: Negative Delta and Score Floor
```gherkin
GIVEN account Bob has reputation: director=50, validator=10, seeder=5
WHEN DirectorSlotRejected event is recorded (-200 director)
THEN Bob's director_score = 0 (floor, not -150)
  AND total() = (0*50 + 10*30 + 5*20) / 100 = 4
```

### Scenario 3: Decay Over Time
```gherkin
GIVEN account Charlie has reputation: director=1000, validator=500, seeder=100
  AND Charlie's last_activity = block 10000
  AND current_block = 11008800 (12 weeks later, ~7 days/week * 600 blocks/hour * 24 hours)
WHEN apply_decay() is called with decay_rate=5% per week
THEN weeks_inactive = 12
  AND decay_factor = 100 - (5 * 12) = 40%
  AND Charlie's scores = director=400, validator=200, seeder=40
```

### Scenario 4: Merkle Root Publication
```gherkin
GIVEN 5 reputation events recorded in block N:
  - Event1: Alice, DirectorSlotAccepted
  - Event2: Bob, ValidatorVoteCorrect
  - Event3: Charlie, SeederChunkServed
  - Event4: Alice, DirectorSlotAccepted (second event)
  - Event5: Dave, PinningAuditPassed
WHEN on_finalize(N) is called
THEN PendingEvents is cleared
  AND MerkleRoots[N] = hash of Merkle tree with 5 leaves
  AND MerkleRootPublished event emitted
  AND each leaf = hash(AccountId, EventType, Slot, BlockNumber)
```

### Scenario 5: Checkpoint Creation
```gherkin
GIVEN current_block = 5000 (5000 % 1000 == 0)
  AND 10 accounts have reputation scores
WHEN on_finalize(5000) is called
THEN Checkpoints[5000] is created
  AND checkpoint contains: block=5000, score_count=10, merkle_root=<scores_merkle>
  AND CheckpointCreated event emitted
```

### Scenario 6: Event Pruning Beyond Retention
```gherkin
GIVEN RetentionPeriod = 2592000 blocks (~6 months)
  AND current_block = 3000000
  AND MerkleRoots contains entries for blocks [100, 500, 10000, 400000, 500000]
WHEN prune_old_events() is called
THEN MerkleRoots[100] is removed (3000000 - 100 > 2592000)
  AND MerkleRoots[500] is removed
  AND MerkleRoots[10000] is removed
  AND MerkleRoots[400000] is removed
  AND MerkleRoots[500000] is kept (3000000 - 500000 < 2592000)
  AND EventsPruned event emitted with count=4
```

### Scenario 7: Aggregated Event Batching (TPS Optimization)
```gherkin
GIVEN off-chain aggregator has pending events for Alice:
  - DirectorSlotAccepted (+100)
  - DirectorSlotAccepted (+100)
  - DirectorSlotRejected (-200)
  - ValidatorVoteCorrect (+5)
WHEN flush_to_chain() batches these into AggregatedEvent
THEN AggregatedEvent contains:
  - net_director_delta = 100 + 100 - 200 = 0
  - net_validator_delta = 5
  - net_seeder_delta = 0
  - event_count = 4
AND single on-chain transaction applies all deltas atomically
```

### Scenario 8: Merkle Proof Verification (Off-Chain)
```gherkin
GIVEN MerkleRoots[1000] = 0xABCD1234
  AND Event at index 2 in block 1000: (Charlie, SeederChunkServed, slot=500, block=1000)
WHEN off-chain validator requests proof for this event
THEN proof contains sibling hashes: [sibling1, sibling2, sibling3]
  AND verify_merkle_proof(leaf=hash(event), proof, root=0xABCD1234) returns true
  AND tampering with event invalidates proof
```

### Scenario 9: Governance Adjusts Retention Period
```gherkin
GIVEN current RetentionPeriod = 2592000 blocks
WHEN governance proposes and approves update to 1296000 blocks (~3 months)
THEN RetentionPeriod storage updated via pallet-democracy or OpenGov
  AND subsequent pruning uses new period
  AND old events beyond new period are pruned on next on_finalize
```

### Scenario 10: Multiple Events Per Block Per Account
```gherkin
GIVEN Alice is both a Director and Validator
  AND in block 2000, the following occur:
    - Alice's slot accepted (+100 director)
    - Alice validates 3 slots correctly (+5 validator each)
WHEN record_event() is called 4 times in same block
THEN PendingEvents contains 4 distinct entries
  AND all 4 are included in Merkle tree for block 2000
  AND Alice's final scores: director=100, validator=15
```

## Technical Implementation

### Core Data Structures
```rust
use frame_support::{pallet_prelude::*, traits::Get};
use frame_system::pallet_prelude::*;
use sp_std::vec::Vec;

#[pallet::config]
pub trait Config: frame_system::Config {
    type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
    type MaxEventsPerBlock: Get<u32>;  // Default: 50
    type DefaultRetentionPeriod: Get<Self::BlockNumber>;  // Default: 2592000 blocks
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, Default, MaxEncodedLen)]
pub struct ReputationScore {
    pub director_score: u64,
    pub validator_score: u64,
    pub seeder_score: u64,
    pub last_activity: u64,  // Block number
}

impl ReputationScore {
    pub fn total(&self) -> u64 {
        // Weighted: 50% director, 30% validator, 20% seeder
        (self.director_score.saturating_mul(50)
            .saturating_add(self.validator_score.saturating_mul(30))
            .saturating_add(self.seeder_score.saturating_mul(20)))
            .saturating_div(100)
    }

    pub fn apply_decay(&mut self, current_block: u64, decay_rate: u64) {
        let blocks_inactive = current_block.saturating_sub(self.last_activity);
        // Assume ~600 blocks/hour, 24 hours/day, 7 days/week
        let weeks_inactive = blocks_inactive.saturating_div(7 * 24 * 600);

        if weeks_inactive > 0 {
            let decay_factor = 100u64.saturating_sub(decay_rate.saturating_mul(weeks_inactive));
            self.director_score = self.director_score.saturating_mul(decay_factor).saturating_div(100);
            self.validator_score = self.validator_score.saturating_mul(decay_factor).saturating_div(100);
            self.seeder_score = self.seeder_score.saturating_mul(decay_factor).saturating_div(100);
        }
    }
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub enum ReputationEventType {
    DirectorSlotAccepted,      // +100
    DirectorSlotRejected,      // -200
    DirectorSlotMissed,        // -150
    ValidatorVoteCorrect,      // +5
    ValidatorVoteIncorrect,    // -10
    SeederChunkServed,         // +1
    PinningAuditPassed,        // +10
    PinningAuditFailed,        // -50
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct ReputationEvent<T: Config> {
    pub account: T::AccountId,
    pub event_type: ReputationEventType,
    pub slot: u64,
    pub block: T::BlockNumber,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
pub struct CheckpointData<T: Config> {
    pub block: T::BlockNumber,
    pub score_count: u32,
    pub merkle_root: T::Hash,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, Default)]
pub struct AggregatedReputation {
    pub net_director_delta: i64,
    pub net_validator_delta: i64,
    pub net_seeder_delta: i64,
    pub event_count: u32,
    pub last_aggregation_block: u64,
}
```

### Storage Items
```rust
#[pallet::storage]
#[pallet::getter(fn reputation_scores)]
pub type ReputationScores<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    T::AccountId,
    ReputationScore,
    ValueQuery,
>;

#[pallet::storage]
#[pallet::getter(fn pending_events)]
pub type PendingEvents<T: Config> = StorageValue<_, Vec<ReputationEvent<T>>, ValueQuery>;

#[pallet::storage]
#[pallet::getter(fn merkle_roots)]
pub type MerkleRoots<T: Config> = StorageMap<
    _,
    Twox64Concat,
    T::BlockNumber,
    T::Hash,
    OptionQuery,
>;

#[pallet::storage]
#[pallet::getter(fn checkpoints)]
pub type Checkpoints<T: Config> = StorageMap<
    _,
    Twox64Concat,
    T::BlockNumber,
    CheckpointData<T>,
    OptionQuery,
>;

#[pallet::storage]
#[pallet::getter(fn retention_period)]
pub type RetentionPeriod<T: Config> = StorageValue<_, T::BlockNumber, ValueQuery, T::DefaultRetentionPeriod>;

#[pallet::storage]
#[pallet::getter(fn aggregated_events)]
pub type AggregatedEvents<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    T::AccountId,
    AggregatedReputation,
    ValueQuery,
>;
```

### Core Extrinsics
```rust
#[pallet::call]
impl<T: Config> Pallet<T> {
    /// Record a reputation event (called by other ICN pallets via root)
    #[pallet::weight(5_000)]
    pub fn record_event(
        origin: OriginFor<T>,
        account: T::AccountId,
        event_type: ReputationEventType,
        slot: u64,
    ) -> DispatchResult {
        ensure_root(origin)?;

        let current_block = <frame_system::Pallet<T>>::block_number();

        // Apply score change
        ReputationScores::<T>::mutate(&account, |score| {
            let delta = Self::event_delta(&event_type);
            match event_type {
                ReputationEventType::DirectorSlotAccepted |
                ReputationEventType::DirectorSlotRejected |
                ReputationEventType::DirectorSlotMissed => {
                    score.director_score = score.director_score.saturating_add_signed(delta);
                },
                ReputationEventType::ValidatorVoteCorrect |
                ReputationEventType::ValidatorVoteIncorrect => {
                    score.validator_score = score.validator_score.saturating_add_signed(delta);
                },
                ReputationEventType::SeederChunkServed |
                ReputationEventType::PinningAuditPassed |
                ReputationEventType::PinningAuditFailed => {
                    score.seeder_score = score.seeder_score.saturating_add_signed(delta);
                },
            }
            score.last_activity = current_block.saturated_into();
        });

        // Add to pending events for Merkle tree
        PendingEvents::<T>::mutate(|events| {
            if events.len() < T::MaxEventsPerBlock::get() as usize {
                events.push(ReputationEvent {
                    account: account.clone(),
                    event_type: event_type.clone(),
                    slot,
                    block: current_block,
                });
            }
        });

        Self::deposit_event(Event::ReputationRecorded { account, event_type, slot });
        Ok(())
    }
}
```

### Hook Implementation
```rust
#[pallet::hooks]
impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
    fn on_finalize(block: BlockNumberFor<T>) {
        // Finalize Merkle root for this block
        let events = PendingEvents::<T>::take();
        if !events.is_empty() {
            let root = Self::compute_merkle_root(&events);
            MerkleRoots::<T>::insert(block, root);
            Self::deposit_event(Event::MerkleRootPublished { block, root });
        }

        // Create checkpoint every 1000 blocks
        if block % 1000u32.into() == Zero::zero() {
            Self::create_checkpoint(block);
        }

        // Prune old events beyond retention period
        let prune_before = block.saturating_sub(RetentionPeriod::<T>::get());
        Self::prune_old_events(prune_before);
    }
}
```

### Helper Functions
```rust
impl<T: Config> Pallet<T> {
    fn event_delta(event_type: &ReputationEventType) -> i64 {
        match event_type {
            ReputationEventType::DirectorSlotAccepted => 100,
            ReputationEventType::DirectorSlotRejected => -200,
            ReputationEventType::DirectorSlotMissed => -150,
            ReputationEventType::ValidatorVoteCorrect => 5,
            ReputationEventType::ValidatorVoteIncorrect => -10,
            ReputationEventType::SeederChunkServed => 1,
            ReputationEventType::PinningAuditPassed => 10,
            ReputationEventType::PinningAuditFailed => -50,
        }
    }

    fn compute_merkle_root(events: &[ReputationEvent<T>]) -> T::Hash {
        if events.is_empty() {
            return T::Hash::default();
        }

        let leaves: Vec<T::Hash> = events.iter()
            .map(|e| T::Hashing::hash_of(e))
            .collect();

        Self::build_merkle_tree(&leaves)
    }

    fn build_merkle_tree(leaves: &[T::Hash]) -> T::Hash {
        if leaves.is_empty() {
            return T::Hash::default();
        }
        if leaves.len() == 1 {
            return leaves[0];
        }

        let mut current = leaves.to_vec();
        while current.len() > 1 {
            let mut next = Vec::new();
            for chunk in current.chunks(2) {
                let combined = if chunk.len() == 2 {
                    T::Hashing::hash_of(&(chunk[0], chunk[1]))
                } else {
                    chunk[0]
                };
                next.push(combined);
            }
            current = next;
        }
        current[0]
    }

    fn create_checkpoint(block: T::BlockNumber) {
        let scores: Vec<_> = ReputationScores::<T>::iter().collect();
        let checkpoint = CheckpointData {
            block,
            score_count: scores.len() as u32,
            merkle_root: Self::compute_scores_merkle(&scores),
        };
        Checkpoints::<T>::insert(block, checkpoint);
        Self::deposit_event(Event::CheckpointCreated { block, count: scores.len() as u32 });
    }

    fn compute_scores_merkle(scores: &[(T::AccountId, ReputationScore)]) -> T::Hash {
        let leaves: Vec<T::Hash> = scores.iter()
            .map(|(account, score)| T::Hashing::hash_of(&(account, score)))
            .collect();
        Self::build_merkle_tree(&leaves)
    }

    fn prune_old_events(before_block: T::BlockNumber) {
        let mut pruned = 0u32;
        for (block, _) in MerkleRoots::<T>::iter() {
            if block < before_block {
                MerkleRoots::<T>::remove(block);
                pruned = pruned.saturating_add(1);
            }
        }
        if pruned > 0 {
            Self::deposit_event(Event::EventsPruned { before_block, count: pruned });
        }
    }
}
```

## Dependencies

- **T001**: Moonbeam fork and dev environment
- **T002**: pallet-icn-stake for role definitions and account references
- **frame-support**: For Hashing trait and storage macros
- **sp-runtime**: For saturating arithmetic

## Design Decisions

1. **Weighted scoring (50/30/20)**: Directors have highest weight since they directly impact content quality. Validators second (quality control), seeders third (infrastructure support).

2. **Merkle trees for event batching**: Allows off-chain nodes to prove specific events without downloading full chain state. Critical for light clients and mobile viewers.

3. **Checkpointing every 1000 blocks**: Balances storage overhead (~2.6k checkpoints/6 months) with recovery speed (don't need to replay 2.6M blocks).

4. **Governance-adjustable retention**: Allows network to adapt storage costs vs historical audit depth via on-chain voting.

5. **Aggregated events (v8.0.1)**: Batching multiple events per account into single transaction reduces TPS pressure on Moonbeam (target <50 TPS).

6. **Saturating arithmetic**: All score updates use saturating_add/sub to prevent overflow/underflow panics on chain.

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Merkle tree computation expensive at scale | Medium | Medium | Limit MaxEventsPerBlock to 50, optimize with BTreeMap |
| Checkpointing storage bloat | Medium | High | Prune checkpoints beyond retention period alongside events |
| Decay calculation overflow | Critical | Low | Use saturating arithmetic, add overflow tests |
| Reputation manipulation via Sybil | High | Medium | Combine with stake weighting in director election |
| Event ordering non-determinism | High | Low | Events ordered by insertion in Vec, deterministic across nodes |

## Progress Log

- 2025-12-24: Task created from PRD §3.2 and Architecture §4.2.2

## Completion Checklist

- [ ] All 12 acceptance criteria met
- [ ] All 10 test scenarios implemented and passing
- [ ] Unit test coverage ≥90%
- [ ] Integration tests with pallet-icn-director passing
- [ ] Benchmarks defined for record_event, on_finalize
- [ ] Clippy passes with no warnings
- [ ] Documentation comments complete (rustdoc)
- [ ] Merkle proof verification tested off-chain
- [ ] Storage migration plan for RetentionPeriod updates
- [ ] No regression in existing tests
