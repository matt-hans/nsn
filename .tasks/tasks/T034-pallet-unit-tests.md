---
id: T034
title: Comprehensive Pallet Unit Tests (85%+ Coverage)
status: pending
priority: 1
agent: backend
dependencies: [T002, T003, T004, T005, T006, T007]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [testing, pallets, unit-tests, quality, phase1]

context_refs:
  - context/acceptance-templates.md

docs_refs:
  - PRD Section 20 (Testing & Simulation Rules)

est_tokens: 9500
actual_tokens: null
---

## Description

Implement comprehensive unit tests for all 8 NSN pallets with 85%+ code coverage target. Tests must cover all extrinsics (success and failure paths), storage mutations, event emissions, error conditions, slashing scenarios, election edge cases, and BFT challenge/resolution flows.

**Technical Approach:**
- Substrate `mock.rs` runtime for isolated testing
- Test all extrinsic permutations
- Property-based testing for invariants (e.g., total stake always positive)
- Fuzz testing for edge cases
- Coverage measured with tarpaulin

**Coverage Targets:**
- pallet-nsn-stake: 90%
- pallet-nsn-reputation: 85%
- pallet-nsn-epochs: 88%
- pallet-nsn-bft: 85%
- pallet-nsn-storage: 87%
- pallet-nsn-treasury: 85%
- pallet-nsn-task-market: 85%
- pallet-nsn-model-registry: 85%

## Acceptance Criteria

- [ ] Each pallet has `tests.rs` module with 20+ test functions
- [ ] All extrinsics tested (success path + error paths)
- [ ] Storage mutations verified (before/after state checks)
- [ ] Events emitted verified with `assert_last_event!()`
- [ ] Error cases tested (insufficient stake, not elected, already finalized, etc.)
- [ ] Slashing scenarios tested (BFT fraud, audit failure, missed slot)
- [ ] Election edge cases tested (cooldowns, region limits, reputation weighting)
- [ ] BFT challenge resolution tested (challenger wins, challenger loses, timeout)
- [ ] Reputation decay tested (weekly decay, pruning after retention period)
- [ ] Pinning audit tested (proof submission, timeout, stake slashing)
- [ ] Coverage report generated: `cargo tarpaulin --all-features --out Xml`
- [ ] Overall coverage ≥85% (measured via Codecov)

## Test Scenarios

**Test Case: Staking - Deposit Success**
```rust
#[test]
fn deposit_stake_success() {
    new_test_ext().execute_with(|| {
        let alice = 1;
        let amount = 100 * UNIT;

        assert_ok!(NsnStake::deposit_stake(Origin::signed(alice), amount, 1000, Region::NaWest));

        assert_eq!(NsnStake::stakes(alice).amount, amount);
        assert_eq!(NsnStake::stakes(alice).role, NodeRole::Director);
        assert_last_event!(Event::StakeDeposited(alice, amount, NodeRole::Director));
    });
}
```

**Test Case: Staking - Per-Region Cap Exceeded**
```rust
#[test]
fn deposit_stake_region_cap_exceeded() {
    new_test_ext().execute_with(|| {
        // Stake 20% of network in NA-WEST
        stake_region(Region::NaWest, 2000 * UNIT);

        // Attempt to stake more in NA-WEST (would exceed 20%)
        assert_noop!(
            NsnStake::deposit_stake(Origin::signed(alice), 100 * UNIT, 1000, Region::NaWest),
            Error::<Test>::RegionCapExceeded
        );
    });
}
```

**Test Case: Director Election - Multi-Region Distribution**
```rust
#[test]
fn elect_directors_multi_region() {
    new_test_ext().execute_with(|| {
        // Setup 10 directors across 3 regions
        setup_directors_multi_region();

        let directors = NsnEpochs::elect_directors(100);

        assert_eq!(directors.len(), 5);
        // Assert no more than 2 from same region
        let na_west_count = directors.iter().filter(|d| stakes(d).region == Region::NaWest).count();
        assert!(na_west_count <= 2);
    });
}
```

**Test Case: BFT Challenge - Upheld**
```rust
#[test]
fn bft_challenge_upheld_slashes_directors() {
    new_test_ext().execute_with(|| {
        // Submit fraudulent BFT result
        submit_bft_result(slot, directors, fraudulent_hash);

        // Challenge with validator attestations
        assert_ok!(NsnBft::challenge_bft_result(Origin::signed(challenger), slot, evidence));

        // Advance past challenge period
        run_to_block(System::block_number() + 51);

        // Resolve challenge (upheld)
        assert_ok!(NsnBft::resolve_challenge(Origin::root(), slot, validator_attestations));

        // Verify directors slashed
        for director in &directors {
            assert_eq!(NsnStake::stakes(director).amount, initial_stake - 100 * UNIT);
        }

        // Verify challenger refunded + rewarded
        assert_eq!(Balances::free_balance(challenger), initial_balance + 10 * UNIT);
    });
}
```

**Test Case: Reputation - Decay After Inactivity**
```rust
#[test]
fn reputation_decays_weekly() {
    new_test_ext().execute_with(|| {
        let alice = 1;

        // Record initial reputation
        NsnReputation::record_event(Origin::root(), alice, DirectorSlotAccepted, 1)?;
        assert_eq!(NsnReputation::reputation_scores(alice).director_score, 100);

        // Advance 4 weeks with no activity
        run_to_block(4 * WEEKS_IN_BLOCKS);

        // Apply decay (10% per week compounding)
        let mut score = NsnReputation::reputation_scores(alice);
        score.apply_decay(System::block_number(), 10);

        // 100 * 0.9^4 = 65.61
        assert_eq!(score.director_score, 65);
    });
}
```

## Technical Implementation

**File:** `pallets/nsn-stake/src/tests.rs`

```rust
use super::*;
use frame_support::{assert_noop, assert_ok};
use sp_runtime::testing::Header;

type UncheckedExtrinsic = frame_system::mocking::MockUncheckedExtrinsic<Test>;
type Block = frame_system::mocking::MockBlock<Test>;

frame_support::construct_runtime!(
    pub enum Test where
        Block = Block,
        NodeBlock = Block,
        UncheckedExtrinsic = UncheckedExtrinsic,
    {
        System: frame_system,
        Balances: pallet_balances,
        NsnStake: pallet_nsn_stake,
    }
);

// ... parameter_types and Config implementations ...

pub fn new_test_ext() -> sp_io::TestExternalities {
    let mut t = frame_system::GenesisConfig::default()
        .build_storage::<Test>()
        .unwrap();

    pallet_balances::GenesisConfig::<Test> {
        balances: vec![(1, 10000 * UNIT), (2, 10000 * UNIT)],
    }
    .assimilate_storage(&mut t)
    .unwrap();

    t.into()
}

#[test]
fn deposit_stake_basic() {
    new_test_ext().execute_with(|| {
        assert_ok!(NsnStake::deposit_stake(
            Origin::signed(1),
            100 * UNIT,
            1000,
            Region::NaWest
        ));
        assert_eq!(NsnStake::stakes(1).amount, 100 * UNIT);
    });
}

// ... 20+ more test functions covering all scenarios ...
```

**File:** `.github/workflows/coverage.yml`

```yaml
name: Code Coverage

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-action@nightly

      - name: Run Tarpaulin
        uses: actions-rs/tarpaulin@v0.1
        with:
          version: '0.22.0'
          args: '--all-features --workspace --out Xml'

      - name: Upload to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./cobertura.xml
          fail_ci_if_error: true
```

### Validation Commands

```bash
# Run all unit tests
cargo test --all-features -- --nocapture

# Run tests for specific pallet
cargo test -p pallet-nsn-stake --all-features

# Generate coverage report
cargo tarpaulin --all-features --workspace --out Html --output-dir coverage/

# View coverage
open coverage/index.html

# Check coverage meets threshold
cargo tarpaulin --all-features --workspace | grep "^Coverage:"
# Should show ≥85%
```

## Dependencies

**Hard Dependencies:**
- [T002-T007] All pallet implementations

## Design Decisions

**Decision 1: Mock Runtime vs. Full Substrate Node**
- **Rationale:** Mock runtime is 100× faster, isolated, deterministic
- **Trade-offs:** (+) Fast, isolated. (-) Doesn't test runtime integration

**Decision 2: Tarpaulin vs. grcov**
- **Rationale:** Tarpaulin easier to use, better GitHub Actions integration
- **Trade-offs:** (+) Simple. (-) Slightly less accurate than grcov

## Progress Log

### [2025-12-24] - Task Created
**Dependencies:** T002-T007

## Completion Checklist

- [ ] tests.rs for all 8 pallets
- [ ] 85%+ coverage achieved
- [ ] All extrinsics tested (success + errors)
- [ ] Slashing, elections, BFT, reputation tests passing
- [ ] Coverage report in CI/CD

**Definition of Done:**
All pallets have comprehensive unit tests with ≥85% coverage, all edge cases tested, coverage report published to Codecov.
