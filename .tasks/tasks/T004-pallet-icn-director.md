# Task T004: Implement pallet-icn-director (Director Election & BFT Coordination)

## Metadata
```yaml
id: T004
title: Implement pallet-icn-director (Director Election & BFT Coordination)
status: pending
priority: P1
tags: [pallets, substrate, director, bft, vrf, on-chain, phase1, consensus]
estimated_tokens: 16000
actual_tokens: 0
dependencies: [T001, T002, T003]
created_at: 2025-12-24
updated_at: 2025-12-24
```

## Description

Implement the director election pallet that uses VRF-based randomness to select 5 directors per slot with multi-region distribution, coordinates off-chain BFT consensus submission, implements a 50-block challenge period for dispute resolution, and manages cooldowns and slashing for failed consensus rounds.

## Business Context

**Why this matters**: The director election mechanism is the core decentralization primitive of ICN. It ensures:
- **Unpredictable selection**: VRF prevents pre-computation attacks where bad actors coordinate before election results
- **Geographic diversity**: Multi-region requirement (max 2 per region) prevents single-region capture
- **Economic accountability**: Challenge period + stake slashing punishes collusion and fraud
- **Fairness**: Reputation-weighted + jitter prevents permanent director cartels

**Value delivered**: Enables trustless coordination of 5 independent directors to reach 3-of-5 BFT consensus on video generation quality, while providing on-chain dispute resolution for fraudulent submissions.

**Priority justification**: P1 (critical path) because all video generation workflow depends on director election. Blocks T005 (pinning), T007 (BFT storage), and off-chain Vortex integration.

## Acceptance Criteria

1. `on_initialize()` hook triggers director election at correct slot boundaries (~8 blocks/slot)
2. `elect_directors()` uses ICN Chain's `T::Randomness` for cryptographically secure randomness
3. Exactly 5 directors elected per slot from eligible candidates (Director role + not in cooldown)
4. Multi-region constraint enforced: maximum 2 directors from same region per slot
5. Reputation-weighted selection with sublinear scaling (sqrt) and ±20% deterministic jitter
6. 20-slot cooldown period enforced between director selections for same account
7. `submit_bft_result()` extrinsic stores BFT result in PENDING state with 50-block challenge window
8. `challenge_bft_result()` allows any staker with 25 ICN bond to dispute result
9. `resolve_challenge()` with validator attestations either slashes 100 ICN from fraudulent directors or slashes 25 ICN challenger bond
10. `on_finalize()` auto-finalizes unchallenged results after 50 blocks and records reputation events
11. `DirectorsElected`, `BftResultPending`, `BftChallenged`, `BftConsensusFinalized` events emitted
12. Unit tests cover VRF election, multi-region distribution, challenge/resolution, and edge cases (90%+ coverage)

## Test Scenarios

### Scenario 1: VRF-Based Director Election
```gherkin
GIVEN 15 staked Directors across 5 regions:
  NA-WEST: [D1, D2, D3] with reputation [800, 750, 700]
  EU-WEST: [D4, D5, D6] with reputation [900, 850, 800]
  APAC: [D7, D8, D9] with reputation [950, 900, 850]
  LATAM: [D10, D11, D12] with reputation [600, 550, 500]
  MENA: [D13, D14, D15] with reputation [700, 650, 600]
  AND current_slot = 100
WHEN on_initialize() triggers election for slot 102 (2-slot lookahead)
THEN exactly 5 Directors are elected
  AND no region has >2 Directors elected
  AND all elected Directors have stake ≥100 ICN
  AND all elected Directors have cooldown + 20 < 102
  AND selection is deterministic given same VRF seed
  AND DirectorsElected event emitted with (slot=102, directors=[...])
```

### Scenario 2: Multi-Region Distribution Enforcement
```gherkin
GIVEN 10 Directors all from EU-WEST region
  AND all have reputation >800 and valid stakes
WHEN elect_directors() runs for slot 50
THEN only 2 Directors from EU-WEST are selected
  AND remaining 3 slots are not filled (insufficient regions)
  OR election waits for Directors from other regions to stake
```

### Scenario 3: Cooldown Period Enforcement
```gherkin
GIVEN Director Alice was elected for slot 80
  AND cooldown period is 20 slots
  AND current_slot = 95
WHEN elect_directors() runs for slot 96
THEN Alice is excluded from candidates (96 < 80 + 20)
  AND Alice becomes eligible again at slot 100
```

### Scenario 4: Reputation-Weighted Selection with Jitter
```gherkin
GIVEN 3 eligible Directors from different regions:
  Alice: reputation total = 1000
  Bob: reputation total = 500
  Charlie: reputation total = 100
WHEN elect_directors() runs 1000 times with different VRF seeds
THEN Alice is selected ~70% of the time
  AND Bob is selected ~23% of the time
  AND Charlie is selected ~7% of the time
  (weights: sqrt(1000)≈31.6, sqrt(500)≈22.4, sqrt(100)=10 with ±20% jitter)
```

### Scenario 5: BFT Result Submission with Challenge Period
```gherkin
GIVEN 5 Directors elected for slot 100: [D1, D2, D3, D4, D5]
  AND D1 is designated canonical submitter (off-chain coordination)
  AND 3 Directors agree on CLIP embedding hash 0xABCD1234
WHEN D1 calls submit_bft_result(
  slot=100,
  canonical_director=D1,
  agreeing_directors=[D1, D2, D3],
  embeddings_hash=0xABCD1234
)
THEN BftResults[100] is stored with success=true, canonical_hash=0xABCD1234
  AND FinalizedSlots[100] = false (PENDING)
  AND challenge deadline = current_block + 50
  AND cooldowns updated: Cooldowns[D1] = Cooldowns[D2] = Cooldowns[D3] = 100
  AND BftResultPending event emitted
```

### Scenario 6: Successful Challenge with Director Slashing
```gherkin
GIVEN BFT result for slot 100 submitted at block 1000 (deadline 1050)
  AND Validator Eve has proof that directors colluded (forged CLIP embeddings)
  AND Eve has 30 ICN staked
WHEN Eve calls challenge_bft_result(slot=100, evidence_hash=0xEVIDENCE)
  at block 1020
THEN 25 ICN reserved from Eve's balance
  AND PendingChallenges[100] created with deadline=1070
  AND BftChallenged event emitted

WHEN resolve_challenge() called with validator attestations:
  [V1: agrees_with_challenge=true, V2: true, V3: true, V4: false]
THEN challenge upheld (3/4 validators agree)
  AND each fraudulent director slashed 100 ICN via pallet_icn_stake::slash()
  AND Eve's 25 ICN bond refunded + 10 ICN reward
  AND ChallengeUpheld event emitted
```

### Scenario 7: Failed Challenge Slashes Challenger
```gherkin
GIVEN BFT result for slot 150 with valid consensus
  AND Malicious challenger Frank submits false evidence
WHEN resolve_challenge() called with validator attestations rejecting challenge
THEN challenge rejected (validators confirm BFT result is valid)
  AND Frank's 25 ICN bond slashed
  AND original BFT result finalized
  AND ChallengeRejected event emitted
  AND BftConsensusFinalized event emitted for slot 150
```

### Scenario 8: Auto-Finalization After Challenge Period
```gherkin
GIVEN BFT result submitted for slot 200 at block 2000
  AND 50 blocks pass with no challenge submitted
WHEN on_finalize(2050) executes
THEN FinalizedSlots[200] = true
  AND reputation events recorded:
    - DirectorSlotAccepted for each agreeing director (+100 each)
  AND BftConsensusFinalized event emitted
```

### Scenario 9: Slot Transition and Lookahead
```gherkin
GIVEN slots are ~8 blocks each (45s ÷ 6s/block ≈ 7.5)
  AND current_block = 800
WHEN on_initialize(800) is called
THEN current_slot = 800 ÷ 8 = 100
  AND election triggered for slot 102 (2-slot lookahead)
  AND ElectedDirectors storage updated for slot 102
  AND SlotStarted event emitted
```

### Scenario 10: Insufficient Directors Edge Case
```gherkin
GIVEN only 3 Directors staked across 2 regions
  AND BFT threshold is 3-of-5
WHEN elect_directors() runs
THEN all 3 Directors are elected
  AND BFT submission requires all 3 to agree (3-of-3 threshold adjusted)
  OR election fails and slot skipped (governance decision)
```

### Scenario 11: VRF Randomness Verification
```gherkin
GIVEN two election rounds with same candidates and different block numbers
WHEN elect_directors() uses T::Randomness::random(&slot.to_le_bytes())
THEN VRF outputs are different for each slot
  AND outputs are cryptographically unpredictable
  AND same slot + same candidates + same block = deterministic result (for forks)
```

### Scenario 12: Challenge Deadline Expiry
```gherkin
GIVEN challenge submitted at block 1000 with deadline 1050
WHEN current_block reaches 1051
  AND challenge not yet resolved
THEN challenge considered abandoned
  AND original BFT result auto-finalizes
  AND challenger bond forfeited (slashed for griefing)
```

## Technical Implementation

### Core Data Structures
```rust
use frame_support::{pallet_prelude::*, traits::Randomness};
use sp_runtime::traits::Hash;
use sp_std::vec::Vec;

#[pallet::config]
pub trait Config: frame_system::Config + pallet_icn_stake::Config + pallet_icn_reputation::Config {
    type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
    type Randomness: Randomness<Self::Hash, BlockNumberFor<Self>>;
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct BftConsensusResult<T: Config> {
    pub slot: u64,
    pub success: bool,
    pub canonical_hash: T::Hash,
    pub attestations: Vec<(T::AccountId, bool)>,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct BftChallenge<T: Config> {
    pub slot: u64,
    pub challenger: T::AccountId,
    pub challenge_block: T::BlockNumber,
    pub deadline: T::BlockNumber,
    pub evidence_hash: T::Hash,
    pub resolved: bool,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct SlotResult<T: Config> {
    pub slot: u64,
    pub canonical_director: Option<T::AccountId>,
    pub agreeing_directors: Vec<T::AccountId>,
    pub failed_directors: Vec<T::AccountId>,
    pub timestamp: u64,
}

const DIRECTORS_PER_SLOT: usize = 5;
const BFT_THRESHOLD: usize = 3;
const COOLDOWN_SLOTS: u64 = 20;
const JITTER_FACTOR: u32 = 20;  // ±20%
const CHALLENGE_PERIOD_BLOCKS: u32 = 50;
const CHALLENGE_STAKE: u128 = 25_000_000_000_000_000_000; // 25 ICN
const SLASH_AMOUNT_FRAUD: u128 = 100_000_000_000_000_000_000; // 100 ICN
```

### Storage Items
```rust
#[pallet::storage]
#[pallet::getter(fn current_slot)]
pub type CurrentSlot<T: Config> = StorageValue<_, u64, ValueQuery>;

#[pallet::storage]
#[pallet::getter(fn elected_directors)]
pub type ElectedDirectors<T: Config> = StorageValue<_, Vec<T::AccountId>, ValueQuery>;

#[pallet::storage]
#[pallet::getter(fn cooldowns)]
pub type Cooldowns<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    T::AccountId,
    u64,
    ValueQuery,
>;

#[pallet::storage]
#[pallet::getter(fn bft_results)]
pub type BftResults<T: Config> = StorageMap<
    _,
    Twox64Concat,
    u64,  // slot
    BftConsensusResult<T>,
    OptionQuery,
>;

#[pallet::storage]
#[pallet::getter(fn pending_challenges)]
pub type PendingChallenges<T: Config> = StorageMap<
    _,
    Twox64Concat,
    u64,  // slot
    BftChallenge<T>,
    OptionQuery,
>;

#[pallet::storage]
#[pallet::getter(fn finalized_slots)]
pub type FinalizedSlots<T: Config> = StorageMap<
    _,
    Twox64Concat,
    u64,  // slot
    bool,
    ValueQuery,
>;

#[pallet::storage]
#[pallet::getter(fn slot_results)]
pub type SlotResults<T: Config> = StorageMap<
    _,
    Twox64Concat,
    u64,  // slot
    SlotResult<T>,
    OptionQuery,
>;
```

### Hooks Implementation
```rust
#[pallet::hooks]
impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
    fn on_initialize(block: BlockNumberFor<T>) -> Weight {
        let block_num: u64 = block.saturated_into();
        let slot = block_num / 8;  // ~8 blocks per 45s slot

        if slot > Self::current_slot() {
            Self::start_new_slot(slot);
        }

        Weight::from_parts(10_000, 0)
    }

    fn on_finalize(block: BlockNumberFor<T>) {
        // Auto-finalize unchallenged BFT results past deadline
        for (slot, result) in BftResults::<T>::iter() {
            if Self::finalized_slots(slot) {
                continue;
            }

            if Self::pending_challenges(slot).is_some() {
                continue;
            }

            // Check if challenge period expired
            let _ = Self::finalize_slot(slot);
        }
    }
}
```

### Core Extrinsics
```rust
#[pallet::call]
impl<T: Config> Pallet<T> {
    #[pallet::weight(50_000)]
    pub fn submit_bft_result(
        origin: OriginFor<T>,
        slot: u64,
        canonical_director: T::AccountId,
        agreeing_directors: Vec<T::AccountId>,
        embeddings_hash: T::Hash,
    ) -> DispatchResult {
        let submitter = ensure_signed(origin)?;

        let elected = Self::elected_directors();
        ensure!(elected.contains(&submitter), Error::<T>::NotElectedDirector);
        ensure!(agreeing_directors.len() >= BFT_THRESHOLD, Error::<T>::InsufficientAgreement);

        let result = BftConsensusResult {
            slot,
            success: true,
            canonical_hash: embeddings_hash,
            attestations: agreeing_directors.iter()
                .map(|d| (d.clone(), true))
                .collect(),
        };
        BftResults::<T>::insert(slot, result);

        let current_block = <frame_system::Pallet<T>>::block_number();
        let deadline = current_block.saturating_add(CHALLENGE_PERIOD_BLOCKS.into());

        FinalizedSlots::<T>::insert(slot, false);

        for director in &agreeing_directors {
            Cooldowns::<T>::insert(director, slot);
        }

        Self::deposit_event(Event::BftResultPending { slot, canonical_director, deadline });
        Ok(())
    }

    #[pallet::weight(75_000)]
    pub fn challenge_bft_result(
        origin: OriginFor<T>,
        slot: u64,
        evidence_hash: T::Hash,
    ) -> DispatchResult {
        let challenger = ensure_signed(origin)?;

        ensure!(BftResults::<T>::contains_key(slot), Error::<T>::ResultNotFound);
        ensure!(!Self::finalized_slots(slot), Error::<T>::AlreadyFinalized);
        ensure!(Self::pending_challenges(slot).is_none(), Error::<T>::ChallengeExists);

        let challenger_stake = pallet_icn_stake::Pallet::<T>::stakes(&challenger);
        ensure!(
            challenger_stake.amount >= CHALLENGE_STAKE.saturated_into(),
            Error::<T>::InsufficientChallengeStake
        );

        T::Currency::reserve(&challenger, CHALLENGE_STAKE.saturated_into())?;

        let current_block = <frame_system::Pallet<T>>::block_number();
        let challenge = BftChallenge {
            slot,
            challenger: challenger.clone(),
            challenge_block: current_block,
            deadline: current_block.saturating_add(CHALLENGE_PERIOD_BLOCKS.into()),
            evidence_hash,
            resolved: false,
        };

        PendingChallenges::<T>::insert(slot, challenge);
        Self::deposit_event(Event::BftChallenged { slot, challenger });
        Ok(())
    }

    #[pallet::weight(100_000)]
    pub fn resolve_challenge(
        origin: OriginFor<T>,
        slot: u64,
        validator_attestations: Vec<(T::AccountId, bool, T::Hash)>,
    ) -> DispatchResult {
        ensure_root(origin)?;

        let mut challenge = Self::pending_challenges(slot)
            .ok_or(Error::<T>::NoChallengeExists)?;
        ensure!(!challenge.resolved, Error::<T>::ChallengeAlreadyResolved);

        let agree_count = validator_attestations.iter()
            .filter(|(_, agrees, _)| *agrees)
            .count();
        let challenge_upheld = agree_count > validator_attestations.len() / 2;

        if challenge_upheld {
            // Slash directors
            let result = Self::bft_results(slot).ok_or(Error::<T>::ResultNotFound)?;
            for (director, _) in &result.attestations {
                pallet_icn_stake::Pallet::<T>::slash(
                    frame_system::RawOrigin::Root.into(),
                    director.clone(),
                    SLASH_AMOUNT_FRAUD.saturated_into(),
                    pallet_icn_stake::SlashReason::BftFraud,
                )?;
            }

            T::Currency::unreserve(&challenge.challenger, CHALLENGE_STAKE.saturated_into());
            let reward = 10_000_000_000_000_000_000u128.saturated_into();
            let _ = T::Currency::deposit_into_existing(&challenge.challenger, reward);

            Self::deposit_event(Event::ChallengeUpheld { slot, challenger: challenge.challenger.clone() });
        } else {
            let (_, _slashed) = T::Currency::slash_reserved(
                &challenge.challenger,
                CHALLENGE_STAKE.saturated_into()
            );

            let _ = Self::finalize_slot(slot);

            Self::deposit_event(Event::ChallengeRejected { slot, challenger: challenge.challenger.clone() });
        }

        challenge.resolved = true;
        PendingChallenges::<T>::insert(slot, challenge);
        Ok(())
    }
}
```

### Helper Functions
```rust
impl<T: Config> Pallet<T> {
    fn start_new_slot(slot: u64) {
        CurrentSlot::<T>::put(slot);
        Self::deposit_event(Event::SlotStarted { slot });

        let election_slot = slot.saturating_add(2);
        let directors = Self::elect_directors(election_slot);
        ElectedDirectors::<T>::put(directors.clone());

        Self::deposit_event(Event::DirectorsElected { slot: election_slot, directors });
    }

    fn elect_directors(slot: u64) -> Vec<T::AccountId> {
        let candidates: Vec<_> = pallet_icn_stake::Stakes::<T>::iter()
            .filter(|(account, stake)| {
                stake.role == pallet_icn_stake::NodeRole::Director &&
                Self::cooldowns(account).saturating_add(COOLDOWN_SLOTS) < slot
            })
            .collect();

        if candidates.len() < DIRECTORS_PER_SLOT {
            return candidates.into_iter().map(|(a, _)| a).collect();
        }

        let mut weighted: Vec<(T::AccountId, u64, pallet_icn_stake::Region)> = candidates.iter()
            .map(|(account, stake)| {
                let rep = pallet_icn_reputation::Pallet::<T>::reputation_scores(account);
                let base_weight = rep.total().saturating_add(1);
                let scaled = Self::isqrt(base_weight);

                let jitter_seed = T::Hashing::hash_of(&(slot, account));
                let jitter_raw = u32::from_le_bytes(jitter_seed.as_ref()[0..4].try_into().unwrap_or([0u8; 4]));
                let jitter_pct = (jitter_raw % (JITTER_FACTOR * 2)) as i64 - JITTER_FACTOR as i64;
                let jittered = (scaled as i64).saturating_mul(100 + jitter_pct).saturating_div(100);

                (account.clone(), jittered.max(1) as u64, stake.region.clone())
            })
            .collect();

        let mut selected = Vec::new();
        let mut selected_regions: Vec<pallet_icn_stake::Region> = Vec::new();

        let (vrf_output, _) = T::Randomness::random(&slot.to_le_bytes());
        let mut rng_state: u64 = u64::from_le_bytes(
            vrf_output.as_ref()[0..8].try_into().unwrap_or([0u8; 8])
        );

        for selection_round in 0..DIRECTORS_PER_SLOT {
            if weighted.is_empty() {
                break;
            }

            let total_weight: u64 = weighted.iter()
                .map(|(_, w, region)| {
                    let region_count = selected_regions.iter().filter(|r| *r == region).count();
                    if region_count >= 2 {
                        0  // Exclude if region already has 2
                    } else if region_count == 1 {
                        w.saturating_div(2)
                    } else {
                        w.saturating_mul(2)
                    }
                })
                .sum();

            if total_weight == 0 {
                break;
            }

            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(selection_round as u64);
            let pick = rng_state % total_weight;

            let mut cumulative = 0u64;
            let mut chosen_idx = 0;
            for (i, (_, weight, region)) in weighted.iter().enumerate() {
                let region_count = selected_regions.iter().filter(|r| *r == region).count();
                let adjusted_weight = if region_count >= 2 {
                    0
                } else if region_count == 1 {
                    weight.saturating_div(2)
                } else {
                    weight.saturating_mul(2)
                };
                cumulative = cumulative.saturating_add(adjusted_weight);
                if pick < cumulative {
                    chosen_idx = i;
                    break;
                }
            }

            let (account, _, region) = weighted.remove(chosen_idx);
            selected_regions.push(region);
            selected.push(account);
        }

        selected
    }

    fn finalize_slot(slot: u64) -> DispatchResult {
        let result = Self::bft_results(slot).ok_or(Error::<T>::ResultNotFound)?;

        for (director, agreed) in &result.attestations {
            if *agreed {
                pallet_icn_reputation::Pallet::<T>::record_event(
                    frame_system::RawOrigin::Root.into(),
                    director.clone(),
                    pallet_icn_reputation::ReputationEventType::DirectorSlotAccepted,
                    slot,
                )?;
            }
        }

        FinalizedSlots::<T>::insert(slot, true);
        Self::deposit_event(Event::BftConsensusFinalized { slot });
        Ok(())
    }

    fn isqrt(n: u64) -> u64 {
        if n == 0 { return 0; }
        let mut x = n;
        let mut y = (x + 1) / 2;
        while y < x {
            x = y;
            y = (x + n / x) / 2;
        }
        x
    }
}
```

## Dependencies

- **T001**: ICN Chain bootstrap
- **T002**: pallet-icn-stake for role checking and slashing
- **T003**: pallet-icn-reputation for weighted selection
- **frame-support**: Randomness trait for VRF
- **sp-runtime**: Hashing, saturating arithmetic

## Design Decisions

1. **VRF over block hash**: ICN Chain's randomness source is cryptographically unpredictable, whereas block hashes can be influenced by block producers.

2. **2-slot lookahead**: Electing directors 2 slots ahead gives them ~90 seconds to coordinate off-chain BFT before their slot starts.

3. **50-block challenge period**: ~5 minutes balances dispute resolution time with reputation update latency. Too short = legitimate challenges missed, too long = slow reputation feedback.

4. **100 ICN director slashing**: 2× the challenge bond incentivizes honest behavior. Directors lose 100 ICN vs challenger's 25 ICN bond, making fraud economically irrational.

5. **Sublinear reputation scaling (sqrt)**: Prevents runaway dominance by high-reputation directors. Director with 10× reputation only gets ~3.16× weight.

6. **±20% jitter**: Breaks deterministic patterns that could be exploited for coordination. Even with known reputation, selection has controlled randomness.

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| VRF bias attacks | High | Low | Multi-region requirement limits single-actor impact |
| Challenge griefing | Medium | Medium | 25 ICN bond + slashing for false challenges |
| Insufficient directors | Critical | Medium | Governance can lower stake requirements or adjust BFT threshold |
| Region enumeration mismatch | High | Low | Use pallet_icn_stake::Region enum, validate in extrinsics |
| Cooldown bypass via multiple accounts | Medium | Medium | Document as known limitation, social layer mitigation |

## Progress Log

- 2025-12-24: Task created from PRD §3.3 and Architecture §4.2.2

## Completion Checklist

- [ ] All 12 acceptance criteria met
- [ ] All 12 test scenarios implemented and passing
- [ ] Unit test coverage ≥90%
- [ ] Integration tests with pallet-icn-stake and pallet-icn-reputation
- [ ] VRF randomness verified on testnet
- [ ] Benchmarks defined for submit_bft_result, challenge_bft_result, resolve_challenge
- [ ] Clippy passes with no warnings
- [ ] Documentation comments complete (rustdoc)
- [ ] No regression in existing tests
