# Task T006: Implement pallet-icn-treasury (Reward Distribution & Emissions)

## Metadata
```yaml
id: T006
title: Implement pallet-icn-treasury (Reward Distribution & Emissions)
status: pending
priority: P2
tags: [pallets, substrate, treasury, tokenomics, on-chain, phase1]
estimated_tokens: 10000
actual_tokens: 0
dependencies: [T001, T002]
created_at: 2025-12-24
updated_at: 2025-12-24
```

## Description

Implement the treasury pallet that manages ICN token emissions according to the annual decay schedule (100M Year 1, 15% decay), distributes rewards across four categories (40% directors, 25% validators, 20% pinners, 15% treasury), and funds governance-approved proposals for ecosystem growth.

## Business Context

**Why this matters**: The treasury is the economic engine of ICN. It determines:
- **Emission schedule**: Controls token supply inflation (starting 100M/year, decaying 15% annually)
- **Incentive alignment**: 40% to directors ensures quality content generation incentives
- **Sustainability**: 15% treasury allocation funds long-term development and governance initiatives
- **Predictability**: Automated emissions remove human discretion from reward distribution

**Value delivered**: Creates a self-sustaining economic model where network participants are rewarded proportionally to their contribution type (generation > validation > storage > governance).

**Priority justification**: P2 because reward distribution can be simulated off-chain initially. Required for mainnet economic security but not blocking Phase 1 Moonriver testing.

## Acceptance Criteria

1. `TreasuryBalance` storage correctly tracks total ICN available for distribution
2. `RewardSchedule` storage implements annual emission with 15% decay formula
3. `distribute_rewards()` extrinsic (automated or governance) splits emissions:
   - 40% to directors (proportional to successful slots)
   - 25% to validators (proportional to correct votes)
   - 20% to pinners (proportional to storage deals served)
   - 15% to treasury fund
4. Annual emission calculation: `emission_year_N = 100M × (1 - 0.15)^(N-1)`
5. `fund_treasury()` extrinsic allows governance to deposit additional funds
6. `approve_proposal()` extrinsic (governance-only) releases funds for approved initiatives
7. Emissions distributed via `on_finalize()` hook every 14400 blocks (~1 day at 6s/block)
8. Integration with pallet-icn-stake to identify reward recipients
9. Events emitted: `RewardsDistributed`, `TreasuryFunded`, `ProposalApproved`
10. Unit tests cover emission schedule, reward splitting, and treasury funding (90%+ coverage)

## Test Scenarios

### Scenario 1: Year 1 Annual Emission
```gherkin
GIVEN current year = 1 (first year of network)
WHEN calculate_annual_emission(year=1) is called
THEN emission = 100,000,000 ICN
  AND base_emission = 100M
  AND decay_rate = 0.15
  AND emission = 100M × (1 - 0.15)^0 = 100M
```

### Scenario 2: Year 5 Emission with Decay
```gherkin
GIVEN current year = 5
WHEN calculate_annual_emission(year=5) is called
THEN emission = 100M × (1 - 0.15)^4
  = 100M × (0.85)^4
  = 100M × 0.52200625
  ≈ 52,200,625 ICN
```

### Scenario 3: Daily Reward Distribution Split
```gherkin
GIVEN annual emission for current year = 100M ICN
  AND distribution frequency = 1 day = 14400 blocks
WHEN on_finalize(block % 14400 == 0) triggers distribute_rewards()
THEN daily_emission = 100M / 365 ≈ 273,973 ICN
  AND director_pool = 273,973 × 0.40 = 109,589 ICN
  AND validator_pool = 273,973 × 0.25 = 68,493 ICN
  AND pinner_pool = 273,973 × 0.20 = 54,795 ICN
  AND treasury_allocation = 273,973 × 0.15 = 41,096 ICN
```

### Scenario 4: Director Reward Distribution
```gherkin
GIVEN director_pool for today = 109,589 ICN
  AND 3 directors completed successful slots in last 24 hours:
    Alice: 20 slots accepted
    Bob: 15 slots accepted
    Charlie: 10 slots accepted
  Total: 45 slots
WHEN distribute_director_rewards() is called
THEN Alice receives: 109,589 × (20/45) = 48,706 ICN
  AND Bob receives: 109,589 × (15/45) = 36,530 ICN
  AND Charlie receives: 109,589 × (10/45) = 24,353 ICN
  AND RewardsDistributed events emitted for each
```

### Scenario 5: Validator Reward Distribution
```gherkin
GIVEN validator_pool for today = 68,493 ICN
  AND validator votes in last 24 hours:
    Validator1: 100 correct votes
    Validator2: 80 correct votes
    Validator3: 60 correct votes
  Total: 240 votes
WHEN distribute_validator_rewards() is called
THEN Validator1 receives: 68,493 × (100/240) = 28,539 ICN
  AND Validator2 receives: 68,493 × (80/240) = 22,831 ICN
  AND Validator3 receives: 68,493 × (60/240) = 17,123 ICN
```

### Scenario 6: Treasury Funding from Governance
```gherkin
GIVEN treasury currently has 1M ICN
  AND community proposes additional 500k ICN funding
WHEN governance approves fund_treasury(amount=500k ICN)
THEN TreasuryBalance increases to 1.5M ICN
  AND funds transferred from community multisig or root
  AND TreasuryFunded event emitted
```

### Scenario 7: Governance Proposal Approval
```gherkin
GIVEN treasury has 1M ICN available
  AND proposal P001 requests 100k ICN for developer grants
WHEN governance calls approve_proposal(proposal_id=P001, amount=100k)
THEN 100k ICN transferred to proposal beneficiary
  AND TreasuryBalance decreases to 900k ICN
  AND ProposalApproved event emitted
```

### Scenario 8: Emission Schedule Over 10 Years
```gherkin
GIVEN base emission = 100M ICN
  AND decay rate = 15% annually
WHEN calculate emissions for years 1-10
THEN:
  Year 1: 100.0M
  Year 2: 85.0M
  Year 3: 72.25M
  Year 4: 61.41M
  Year 5: 52.20M
  Year 6: 44.37M
  Year 7: 37.71M
  Year 8: 32.06M
  Year 9: 27.25M
  Year 10: 23.16M
  Total 10-year emission: ~635M ICN
```

### Scenario 9: Zero Participants Edge Case
```gherkin
GIVEN no directors completed slots in last 24 hours
  AND no validators voted
  AND no pinners served shards
WHEN distribute_rewards() is called
THEN director_pool, validator_pool, pinner_pool all have 0 recipients
  AND undistributed rewards roll over to next distribution period
  OR added to treasury allocation (governance decision)
```

### Scenario 10: Reward Accumulation Between Distributions
```gherkin
GIVEN last distribution was at block 14400
  AND current block is 20000 (not yet next 14400 boundary)
WHEN participant completes work (director slot, validation, pinning)
THEN work is tracked in AccumulatedContributions storage
  AND rewards calculated and distributed at next block 28800 boundary
```

## Technical Implementation

### Core Data Structures
```rust
use frame_support::{pallet_prelude::*, traits::Currency};
use sp_runtime::Perbill;

#[pallet::config]
pub trait Config: frame_system::Config + pallet_icn_stake::Config {
    type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
    type Currency: Currency<Self::AccountId>;
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct RewardDistribution {
    pub director_percent: Perbill,    // 40%
    pub validator_percent: Perbill,   // 25%
    pub pinner_percent: Perbill,      // 20%
    pub treasury_percent: Perbill,    // 15%
}

impl Default for RewardDistribution {
    fn default() -> Self {
        Self {
            director_percent: Perbill::from_percent(40),
            validator_percent: Perbill::from_percent(25),
            pinner_percent: Perbill::from_percent(20),
            treasury_percent: Perbill::from_percent(15),
        }
    }
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
pub struct EmissionSchedule {
    pub base_emission: u128,      // 100M ICN for year 1
    pub decay_rate: Perbill,      // 15% annual decay
    pub current_year: u32,
}

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, Default)]
pub struct AccumulatedContributions {
    pub director_slots: u64,
    pub validator_votes: u64,
    pub pinner_shards_served: u64,
}
```

### Storage Items
```rust
#[pallet::storage]
#[pallet::getter(fn treasury_balance)]
pub type TreasuryBalance<T: Config> = StorageValue<_, BalanceOf<T>, ValueQuery>;

#[pallet::storage]
#[pallet::getter(fn reward_distribution)]
pub type RewardDistributionConfig<T: Config> = StorageValue<_, RewardDistribution, ValueQuery>;

#[pallet::storage]
#[pallet::getter(fn emission_schedule)]
pub type EmissionScheduleStorage<T: Config> = StorageValue<_, EmissionSchedule, ValueQuery>;

#[pallet::storage]
#[pallet::getter(fn last_distribution_block)]
pub type LastDistributionBlock<T: Config> = StorageValue<_, T::BlockNumber, ValueQuery>;

#[pallet::storage]
#[pallet::getter(fn accumulated_contributions)]
pub type AccumulatedContributionsMap<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    T::AccountId,
    AccumulatedContributions,
    ValueQuery,
>;
```

### Core Extrinsics
```rust
#[pallet::call]
impl<T: Config> Pallet<T> {
    #[pallet::weight(30_000)]
    pub fn fund_treasury(
        origin: OriginFor<T>,
        amount: BalanceOf<T>,
    ) -> DispatchResult {
        let funder = ensure_signed(origin)?;

        T::Currency::transfer(
            &funder,
            &Self::account_id(),
            amount,
            ExistenceRequirement::KeepAlive,
        )?;

        TreasuryBalance::<T>::mutate(|balance| {
            *balance = balance.saturating_add(amount);
        });

        Self::deposit_event(Event::TreasuryFunded { funder, amount });
        Ok(())
    }

    #[pallet::weight(50_000)]
    pub fn approve_proposal(
        origin: OriginFor<T>,
        beneficiary: T::AccountId,
        amount: BalanceOf<T>,
        proposal_id: u32,
    ) -> DispatchResult {
        ensure_root(origin)?;

        ensure!(
            TreasuryBalance::<T>::get() >= amount,
            Error::<T>::InsufficientTreasuryFunds
        );

        T::Currency::transfer(
            &Self::account_id(),
            &beneficiary,
            amount,
            ExistenceRequirement::AllowDeath,
        )?;

        TreasuryBalance::<T>::mutate(|balance| {
            *balance = balance.saturating_sub(amount);
        });

        Self::deposit_event(Event::ProposalApproved { proposal_id, beneficiary, amount });
        Ok(())
    }
}
```

### Hook Implementation
```rust
#[pallet::hooks]
impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
    fn on_finalize(block: BlockNumberFor<T>) {
        const DISTRIBUTION_FREQUENCY: u32 = 14400;  // ~1 day

        if block % DISTRIBUTION_FREQUENCY.into() == Zero::zero() {
            let _ = Self::distribute_rewards(block);
        }
    }
}
```

### Helper Functions
```rust
impl<T: Config> Pallet<T> {
    pub fn account_id() -> T::AccountId {
        // Treasury pallet account
        T::PalletId::get().into_account_truncating()
    }

    fn calculate_annual_emission(year: u32) -> u128 {
        let schedule = EmissionScheduleStorage::<T>::get();
        let base = schedule.base_emission;  // 100M ICN
        let decay = schedule.decay_rate.deconstruct() as u128;  // 15% = 150000000

        // emission = base × (1 - decay_rate)^(year - 1)
        let decay_factor = 1_000_000_000u128 - decay;  // 0.85 as fixed point
        let mut result = base;

        for _ in 1..year {
            result = result.saturating_mul(decay_factor).saturating_div(1_000_000_000);
        }

        result
    }

    fn distribute_rewards(block: T::BlockNumber) -> DispatchResult {
        let schedule = EmissionScheduleStorage::<T>::get();
        let annual_emission = Self::calculate_annual_emission(schedule.current_year);
        let daily_emission = annual_emission.saturating_div(365);

        let distribution = RewardDistributionConfig::<T>::get();

        // Calculate pools
        let director_pool = distribution.director_percent * daily_emission;
        let validator_pool = distribution.validator_percent * daily_emission;
        let pinner_pool = distribution.pinner_percent * daily_emission;
        let treasury_allocation = distribution.treasury_percent * daily_emission;

        // Distribute to participants
        Self::distribute_director_rewards(director_pool.saturated_into())?;
        Self::distribute_validator_rewards(validator_pool.saturated_into())?;
        Self::distribute_pinner_rewards(pinner_pool.saturated_into())?;

        // Add to treasury
        TreasuryBalance::<T>::mutate(|balance| {
            *balance = balance.saturating_add(treasury_allocation.saturated_into());
        });

        LastDistributionBlock::<T>::put(block);
        Self::deposit_event(Event::RewardsDistributed { block, total: daily_emission.saturated_into() });

        Ok(())
    }

    fn distribute_director_rewards(pool: BalanceOf<T>) -> DispatchResult {
        let mut total_slots = 0u64;
        let contributions: Vec<_> = AccumulatedContributionsMap::<T>::iter()
            .filter(|(_, contrib)| contrib.director_slots > 0)
            .collect();

        for (_, contrib) in &contributions {
            total_slots = total_slots.saturating_add(contrib.director_slots);
        }

        if total_slots == 0 {
            return Ok(());
        }

        for (account, contrib) in contributions {
            let reward = pool.saturating_mul(contrib.director_slots.into())
                .saturating_div(total_slots.into());

            let _ = T::Currency::deposit_into_existing(&account, reward);
            Self::deposit_event(Event::DirectorRewarded { account: account.clone(), amount: reward });

            AccumulatedContributionsMap::<T>::mutate(&account, |c| {
                c.director_slots = 0;
            });
        }

        Ok(())
    }

    fn distribute_validator_rewards(pool: BalanceOf<T>) -> DispatchResult {
        let mut total_votes = 0u64;
        let contributions: Vec<_> = AccumulatedContributionsMap::<T>::iter()
            .filter(|(_, contrib)| contrib.validator_votes > 0)
            .collect();

        for (_, contrib) in &contributions {
            total_votes = total_votes.saturating_add(contrib.validator_votes);
        }

        if total_votes == 0 {
            return Ok(());
        }

        for (account, contrib) in contributions {
            let reward = pool.saturating_mul(contrib.validator_votes.into())
                .saturating_div(total_votes.into());

            let _ = T::Currency::deposit_into_existing(&account, reward);
            Self::deposit_event(Event::ValidatorRewarded { account: account.clone(), amount: reward });

            AccumulatedContributionsMap::<T>::mutate(&account, |c| {
                c.validator_votes = 0;
            });
        }

        Ok(())
    }

    fn distribute_pinner_rewards(pool: BalanceOf<T>) -> DispatchResult {
        // Pinning rewards handled by pallet-icn-pinning directly
        // This is a pass-through for treasury accounting
        Ok(())
    }
}
```

## Dependencies

- **T001**: Moonbeam fork
- **T002**: pallet-icn-stake for role identification
- **frame-support**: Currency trait for transfers

## Design Decisions

1. **15% annual decay**: Balances token supply growth with long-term sustainability. Total supply approaches ~667M ICN asymptotically.

2. **40/25/20/15 split**: Directors get highest share (40%) since they perform most resource-intensive work (GPU generation). Treasury 15% funds development.

3. **Daily distribution**: Balances reward frequency (participants want faster payouts) with on-chain cost (more frequent = more transactions).

4. **Proportional rewards**: Directors/validators/pinners rewarded based on actual work completed, not just stake amount (stake required but work earns rewards).

5. **Accumulated contributions**: Tracks work between distribution periods to ensure fairness even if distribution is delayed.

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Emission calculation overflow | Critical | Low | Use saturating arithmetic, test with extreme years |
| Treasury depletion | High | Medium | Governance approval required for proposals, monitor balance |
| Unequal work distribution | Medium | High | Proportional rewards ensure fairness, not fixed splits |
| Year transition logic | Medium | Low | Automated year increment based on block number |

## Progress Log

- 2025-12-24: Task created from PRD §14.3 and tokenomics specification

## Completion Checklist

- [ ] All 10 acceptance criteria met
- [ ] All 10 test scenarios implemented and passing
- [ ] Unit test coverage ≥90%
- [ ] Integration tests with pallet-icn-stake
- [ ] Emission schedule verified for 10 years
- [ ] Benchmarks defined for distribute_rewards, approve_proposal
- [ ] Clippy passes with no warnings
- [ ] Documentation comments complete (rustdoc)
- [ ] No regression in existing tests
