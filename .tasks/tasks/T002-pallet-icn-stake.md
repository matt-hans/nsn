# Task T002: Implement pallet-icn-stake (Staking & Role Eligibility)

## Metadata
```yaml
id: T002
title: Implement pallet-icn-stake (Staking & Role Eligibility)
status: pending
priority: P1
tags: [pallets, substrate, staking, on-chain, phase1]
estimated_tokens: 12000
actual_tokens: 0
dependencies: [T001]
created_at: 2025-12-24
updated_at: 2025-12-24
```

## Description

Implement the foundational staking pallet that manages ICN token deposits, role assignment (Director/SuperNode/Validator/Relay), delegation, slashing, and anti-centralization caps. This pallet is the basis for all network participation and economic security.

## Business Context

**Why this matters**: The staking mechanism is the economic foundation of ICN. It determines:
- Who can become a Director and generate video content (100+ ICN stake)
- Anti-Sybil protection through stake requirements
- Economic punishment for protocol violations (slashing)
- Geographic decentralization through per-region caps (max 20% stake per region)

**Value delivered**: Enables the core economic incentive structure that makes ICN decentralized and trustless. Without proper staking, the network is vulnerable to Sybil attacks and centralization.

## Acceptance Criteria

1. `StakeInfo` storage map correctly tracks account stakes with fields: amount, locked_until, role, region, delegated_to_me
2. `deposit_stake()` extrinsic successfully reserves tokens and assigns role based on amount
3. Per-node stake cap enforced: maximum 1000 ICN per account
4. Per-region stake cap enforced: maximum 20% of total network stake
5. `delegate()` extrinsic allows delegation up to 5× validator's own stake
6. `slash()` extrinsic (root-only) correctly reduces stake and updates role
7. `withdraw_stake()` extrinsic respects lock period and unreserves tokens
8. Role determination logic matches specification:
   - Director: ≥100 ICN
   - SuperNode: ≥50 ICN
   - Validator: ≥10 ICN
   - Relay: ≥5 ICN
9. Events emitted for all state transitions: StakeDeposited, StakeWithdrawn, StakeSlashed, Delegated
10. Comprehensive unit tests cover success cases, error cases, and edge cases (90%+ coverage)
11. Integration tests verify interaction with frame_system and pallet_balances
12. Benchmarks defined for all extrinsics to calculate proper weights

## Test Scenarios

### Scenario 1: Successful Director Stake Deposit
```gherkin
GIVEN account Alice has 200 ICN free balance
  AND no existing stake for Alice
  AND region NA-WEST has 10% of total network stake
WHEN Alice calls deposit_stake(amount=150 ICN, lock_blocks=1000, region=NA-WEST)
THEN 150 ICN is reserved from Alice's balance
  AND Stakes[Alice].amount == 150 ICN
  AND Stakes[Alice].role == NodeRole::Director
  AND Stakes[Alice].region == Region::NA_WEST
  AND Stakes[Alice].locked_until == current_block + 1000
  AND TotalStaked increases by 150 ICN
  AND RegionStakes[NA_WEST] increases by 150 ICN
  AND StakeDeposited event emitted
```

### Scenario 2: Per-Region Cap Enforcement
```gherkin
GIVEN total network stake is 1000 ICN
  AND RegionStakes[EU-WEST] == 150 ICN (15% of total)
  AND account Bob wants to stake 100 ICN in EU-WEST
WHEN Bob calls deposit_stake(amount=100, lock_blocks=1000, region=EU-WEST)
THEN stake is rejected with Error::RegionCapExceeded
  AND Bob's balance is unchanged
  AND RegionStakes[EU-WEST] remains 150 ICN

GIVEN the same setup but Bob stakes 50 ICN instead
WHEN Bob calls deposit_stake(amount=50, lock_blocks=1000, region=EU-WEST)
THEN stake succeeds (150 + 50 = 200, which is 20% of 1000)
```

### Scenario 3: Delegation with Cap
```gherkin
GIVEN validator Charlie has 100 ICN staked (role: Director)
  AND Charlie has 200 ICN delegated to him already
  AND account Dave wants to delegate 400 ICN to Charlie
WHEN Dave calls delegate(validator=Charlie, amount=400)
THEN delegation fails with Error::DelegationCapExceeded
  (200 + 400 > 5 * 100)

GIVEN the same setup but Dave delegates 100 ICN
WHEN Dave calls delegate(validator=Charlie, amount=100)
THEN delegation succeeds
  AND Delegations[Dave][Charlie] == 100 ICN
  AND Stakes[Charlie].delegated_to_me == 300 ICN
```

### Scenario 4: Slashing Reduces Role
```gherkin
GIVEN account Eve has 110 ICN staked (role: Director)
WHEN pallet-icn-director calls slash(offender=Eve, amount=20 ICN, reason=BftFailure)
THEN 20 ICN is slashed from Eve's reserved balance
  AND Stakes[Eve].amount == 90 ICN
  AND Stakes[Eve].role == NodeRole::SuperNode (downgraded from Director)
  AND StakeSlashed event emitted
```

### Scenario 5: Premature Withdrawal Blocked
```gherkin
GIVEN account Frank has 50 ICN staked
  AND Stakes[Frank].locked_until == current_block + 500
WHEN Frank calls withdraw_stake(amount=50) at current_block + 100
THEN withdrawal fails with Error::StakeLocked
  AND Frank's stake remains 50 ICN

GIVEN the same setup but current_block advances to locked_until + 1
WHEN Frank calls withdraw_stake(amount=50)
THEN withdrawal succeeds
  AND 50 ICN is unreserved to Frank's free balance
  AND Stakes[Frank].amount == 0
  AND Stakes[Frank].role == NodeRole::None
```

### Scenario 6: Multi-Region Balance Enforcement
```gherkin
GIVEN 7 regions each have the following stakes:
  NA-WEST: 200 ICN
  NA-EAST: 180 ICN
  EU-WEST: 190 ICN
  EU-EAST: 170 ICN
  APAC: 160 ICN
  LATAM: 150 ICN
  MENA: 150 ICN
  Total: 1200 ICN
WHEN account George tries to stake 100 ICN in NA-WEST
THEN stake fails (200 + 100 = 300, which is 25% of 1200, exceeds 20% cap)

WHEN George stakes 50 ICN in MENA instead
THEN stake succeeds (150 + 50 = 200, which is 16.7% of 1200)
```

### Scenario 7: Role Determination Boundary Conditions
```gherkin
GIVEN the following stake amounts:
  99.99 ICN → Expected role: SuperNode
  100 ICN → Expected role: Director
  49.99 ICN → Expected role: Validator
  50 ICN → Expected role: SuperNode
  9.99 ICN → Expected role: Relay
  10 ICN → Expected role: Validator
  4.99 ICN → Expected role: None
  5 ICN → Expected role: Relay
WHEN deposit_stake() is called for each amount
THEN Stakes[account].role matches expected role
```

### Scenario 8: Per-Node Cap at Maximum
```gherkin
GIVEN account Helen has 900 ICN staked
WHEN Helen calls deposit_stake(amount=100, lock_blocks=1000, region=APAC)
THEN stake succeeds (900 + 100 = 1000, exactly at cap)
  AND Stakes[Helen].amount == 1000 ICN

WHEN Helen tries to stake additional 1 ICN
THEN stake fails with Error::PerNodeCapExceeded
```

## Technical Implementation

### Storage Items
```rust
#[pallet::storage]
#[pallet::getter(fn stakes)]
pub type Stakes<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    T::AccountId,
    StakeInfo<T>,
    ValueQuery,
>;

#[pallet::storage]
#[pallet::getter(fn total_staked)]
pub type TotalStaked<T: Config> = StorageValue<_, BalanceOf<T>, ValueQuery>;

#[pallet::storage]
#[pallet::getter(fn region_stakes)]
pub type RegionStakes<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    Region,
    BalanceOf<T>,
    ValueQuery,
>;

#[pallet::storage]
#[pallet::getter(fn delegations)]
pub type Delegations<T: Config> = StorageDoubleMap<
    _,
    Blake2_128Concat, T::AccountId,  // delegator
    Blake2_128Concat, T::AccountId,  // validator
    BalanceOf<T>,
    ValueQuery,
>;
```

### Core Extrinsics
```rust
#[pallet::call]
impl<T: Config> Pallet<T> {
    #[pallet::weight(10_000)]
    pub fn deposit_stake(
        origin: OriginFor<T>,
        amount: BalanceOf<T>,
        lock_blocks: T::BlockNumber,
        region: Region,
    ) -> DispatchResult {
        let who = ensure_signed(origin)?;

        // Verify per-node cap
        let current = Self::stakes(&who).amount;
        ensure!(
            current.saturating_add(amount) <= T::MaxStakePerNode::get(),
            Error::<T>::PerNodeCapExceeded
        );

        // Verify per-region cap (20%)
        let region_total = Self::region_stakes(&region).saturating_add(amount);
        let network_total = Self::total_staked().saturating_add(amount);
        ensure!(
            region_total.saturating_mul(100u32.into()) <= network_total.saturating_mul(20u32.into()),
            Error::<T>::RegionCapExceeded
        );

        // Reserve tokens
        T::Currency::reserve(&who, amount)?;

        // Update storage
        let new_total = current.saturating_add(amount);
        let role = Self::determine_role(new_total);
        let unlock_at = <frame_system::Pallet<T>>::block_number().saturating_add(lock_blocks);

        Stakes::<T>::mutate(&who, |info| {
            info.amount = new_total;
            info.locked_until = unlock_at;
            info.role = role.clone();
            info.region = region.clone();
        });

        TotalStaked::<T>::mutate(|t| *t = t.saturating_add(amount));
        RegionStakes::<T>::mutate(&region, |r| *r = r.saturating_add(amount));

        Self::deposit_event(Event::StakeDeposited { who, amount, role });
        Ok(())
    }

    #[pallet::weight(10_000)]
    pub fn slash(
        origin: OriginFor<T>,
        offender: T::AccountId,
        amount: BalanceOf<T>,
        reason: SlashReason,
    ) -> DispatchResult {
        ensure_root(origin)?;  // Only callable by other pallets via root

        let stake = Self::stakes(&offender);
        let slash_amount = amount.min(stake.amount);

        let (_, slashed) = T::Currency::slash_reserved(&offender, slash_amount);

        Stakes::<T>::mutate(&offender, |info| {
            info.amount = info.amount.saturating_sub(slashed);
            info.role = Self::determine_role(info.amount);
        });

        TotalStaked::<T>::mutate(|t| *t = t.saturating_sub(slashed));
        RegionStakes::<T>::mutate(&stake.region, |r| *r = r.saturating_sub(slashed));

        Self::deposit_event(Event::StakeSlashed { offender, amount: slashed, reason });
        Ok(())
    }
}
```

### Helper Functions
```rust
impl<T: Config> Pallet<T> {
    fn determine_role(amount: BalanceOf<T>) -> NodeRole {
        if amount >= T::MinStakeDirector::get() {
            NodeRole::Director
        } else if amount >= T::MinStakeSuperNode::get() {
            NodeRole::SuperNode
        } else if amount >= T::MinStakeValidator::get() {
            NodeRole::Validator
        } else if amount >= T::MinStakeRelay::get() {
            NodeRole::Relay
        } else {
            NodeRole::None
        }
    }
}
```

## Dependencies

- **T001**: Moonbeam fork and development environment must be ready
- **frame-support**: For Currency, ReservableCurrency traits
- **pallet-balances**: For actual token reserve/unreserve operations

## Design Decisions

1. **StorageMap vs StorageValue**: Using `StorageMap<AccountId, StakeInfo>` instead of separate maps for each field reduces storage reads/writes during stake updates.

2. **Percentage-based region cap**: 20% cap prevents any single region from dominating the network, ensuring geographic decentralization even if one region has more participants.

3. **Root-only slash**: Only other ICN pallets (via root origin) can call `slash()`, preventing user griefing while allowing automated protocol enforcement.

4. **Saturation arithmetic**: All calculations use `saturating_add/sub/mul` to prevent overflow panics, which would halt the blockchain.

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Integer overflow in stake calculations | Critical | Low | Use saturating arithmetic throughout, add overflow tests |
| Region enumeration mismatch | High | Medium | Use strongly-typed `Region` enum, validate in extrinsics |
| Delegation cap bypass via multiple accounts | Medium | Medium | Document as known limitation, address in Phase 3 via social layer |
| Lock period too short/long | Medium | Medium | Make configurable via governance, default to 30 days |

## Progress Log

- 2025-12-24: Task created from PRD §3.1 specification

## Completion Checklist

- [ ] All 12 acceptance criteria met
- [ ] All 8 test scenarios implemented and passing
- [ ] Unit test coverage ≥90%
- [ ] Integration tests with pallet-balances passing
- [ ] Benchmarks defined for deposit_stake, delegate, slash, withdraw_stake
- [ ] Clippy passes with no warnings
- [ ] Documentation comments complete (rustdoc renders correctly)
- [ ] Code reviewed by senior Substrate developer
- [ ] No regression in existing Moonbeam tests
