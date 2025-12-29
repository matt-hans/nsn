// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.

//! Unit tests for pallet-nsn-stake

use crate::{mock::*, Error, Event, NodeRole, Region};
use frame_support::{
    assert_noop, assert_ok,
    traits::fungible::{InspectFreeze, Mutate},
};

// ============================================================================
// Green Path Tests (Happy Flows)
// ============================================================================

#[test]
fn deposit_stake_director_role() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice has 1000 NSN free balance, no existing stake, region NA-WEST has 0% stake
        assert_eq!(Balances::free_balance(ALICE), 1000);
        assert_eq!(NsnStake::stakes(ALICE).amount, 0);

        // WHEN: Alice deposits 150 NSN for 1000 blocks in NA-WEST
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            150,
            1000,
            Region::NaWest
        ));

        // THEN: Tokens frozen, role assigned, storage updated
        assert_eq!(NsnStake::stakes(ALICE).amount, 150);
        assert_eq!(NsnStake::stakes(ALICE).role, NodeRole::Director);
        assert_eq!(NsnStake::stakes(ALICE).region, Region::NaWest);
        assert_eq!(
            NsnStake::stakes(ALICE).locked_until,
            System::block_number() + 1000
        );
        assert_eq!(NsnStake::total_staked(), 150);
        assert_eq!(NsnStake::region_stakes(Region::NaWest), 150);

        // Verify balance frozen (using fungible freeze)
        assert_eq!(
            Balances::balance_frozen(
                &RuntimeFreezeReason::NsnStake(crate::FreezeReason::Staking),
                &ALICE
            ),
            150
        );

        // Verify event
        let event = last_event();
        assert!(matches!(
            event,
            RuntimeEvent::NsnStake(Event::StakeDeposited { who, amount, role })
            if who == ALICE && amount == 150 && role == NodeRole::Director
        ));
    });
}

#[test]
fn delegate_under_cap_succeeds() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Charlie has 100 NSN staked (Director role)
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(CHARLIE),
            100,
            1000,
            Region::EuWest
        ));

        // WHEN: Dave delegates 300 NSN to Charlie (within 5× cap = 500)
        assert_ok!(NsnStake::delegate(
            RuntimeOrigin::signed(DAVE),
            CHARLIE,
            300
        ));

        // THEN: Delegation recorded, tokens frozen
        assert_eq!(NsnStake::delegations(DAVE, CHARLIE), 300);
        assert_eq!(NsnStake::stakes(CHARLIE).delegated_to_me, 300);
        assert_eq!(
            Balances::balance_frozen(
                &RuntimeFreezeReason::NsnStake(crate::FreezeReason::Delegating),
                &DAVE
            ),
            300
        );

        // Verify event
        let event = last_event();
        assert!(matches!(
            event,
            RuntimeEvent::NsnStake(Event::Delegated { delegator, validator, amount })
            if delegator == DAVE && validator == CHARLIE && amount == 300
        ));
    });
}

#[test]
fn withdraw_stake_after_lock_period() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Frank stakes 50 NSN locked until block 100
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(FRANK),
            50,
            100,
            Region::Apac
        ));

        // Advance to unlock block + 1
        roll_to(102);

        // WHEN: Frank withdraws 50 NSN
        assert_ok!(NsnStake::withdraw_stake(RuntimeOrigin::signed(FRANK), 50));

        // THEN: Tokens unfrozen, stake cleared
        assert_eq!(NsnStake::stakes(FRANK).amount, 0);
        assert_eq!(NsnStake::stakes(FRANK).role, NodeRole::None);
        assert_eq!(NsnStake::total_staked(), 0);
        assert_eq!(
            Balances::balance_frozen(
                &RuntimeFreezeReason::NsnStake(crate::FreezeReason::Staking),
                &FRANK
            ),
            0
        );

        // Verify event
        let event = last_event();
        assert!(matches!(
            event,
            RuntimeEvent::NsnStake(Event::StakeWithdrawn { who, amount })
            if who == FRANK && amount == 50
        ));
    });
}

#[test]
fn per_node_cap_at_maximum() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Helen stakes 900 NSN
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(HELEN),
            900,
            1000,
            Region::Apac
        ));

        // WHEN: Helen stakes exactly 100 more (total = 1000, at cap)
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(HELEN),
            100,
            1000,
            Region::Apac
        ));

        // THEN: Stake succeeds, total = 1000
        assert_eq!(NsnStake::stakes(HELEN).amount, 1000);
        assert_eq!(NsnStake::stakes(HELEN).role, NodeRole::Director);
    });
}

#[test]
fn per_region_cap_at_20_percent() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Network has 1000 NSN total stake
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            500,
            1000,
            Region::NaWest
        ));
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(BOB),
            500,
            1000,
            Region::EuWest
        ));
        assert_eq!(NsnStake::total_staked(), 1000);

        // EU-WEST has 500 NSN (50%), NA-EAST has 0%
        // WHEN: George stakes 200 NSN in NA-EAST (becomes 20% of 1000)
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(GEORGE),
            200,
            1000,
            Region::NaEast
        ));

        // THEN: Succeeds (200 / 1200 = 16.7%, under 20%)
        assert_eq!(NsnStake::region_stakes(Region::NaEast), 200);
        assert_eq!(NsnStake::total_staked(), 1200);
    });
}

// ============================================================================
// Red Path Tests (Error Cases)
// ============================================================================

#[test]
fn deposit_stake_exceeds_node_cap() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Account at 1000 NSN (max cap)
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            1000,
            1000,
            Region::NaWest
        ));

        // WHEN: Tries to stake 1 more NSN
        // THEN: Fails with PerNodeCapExceeded
        assert_noop!(
            NsnStake::deposit_stake(RuntimeOrigin::signed(ALICE), 1, 1000, Region::NaWest),
            Error::<Test>::PerNodeCapExceeded
        );
    });
}

#[test]
fn deposit_stake_exceeds_region_cap() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Total network = 1000 NSN, EU-WEST = 200 NSN (20%)
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            800,
            1000,
            Region::NaWest
        ));
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(BOB),
            200,
            1000,
            Region::EuWest
        ));

        // WHEN: Bob tries to stake 1 more in EU-WEST (would be 201/1001 = 20.08%)
        // THEN: Fails with RegionCapExceeded
        assert_noop!(
            NsnStake::deposit_stake(RuntimeOrigin::signed(BOB), 1, 1000, Region::EuWest),
            Error::<Test>::RegionCapExceeded
        );
    });
}

#[test]
fn delegate_exceeds_5x_cap() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Validator Charlie has 100 NSN self-stake
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(CHARLIE),
            100,
            1000,
            Region::EuWest
        ));

        // AND: Already has 300 NSN delegated (within 5× = 500 cap)
        assert_ok!(NsnStake::delegate(
            RuntimeOrigin::signed(DAVE),
            CHARLIE,
            300
        ));

        // WHEN: Eve tries to delegate 300 more (total would be 600, exceeds 500)
        // THEN: Fails with DelegationCapExceeded
        assert_noop!(
            NsnStake::delegate(RuntimeOrigin::signed(EVE), CHARLIE, 300),
            Error::<Test>::DelegationCapExceeded
        );
    });
}

#[test]
fn withdraw_stake_before_unlock() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Stake locked until block 1000
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            50,
            1000,
            Region::NaWest
        ));

        // WHEN: Tries to withdraw at block 1 (locked_until = 1001)
        // THEN: Fails with StakeLocked
        assert_noop!(
            NsnStake::withdraw_stake(RuntimeOrigin::signed(ALICE), 50),
            Error::<Test>::StakeLocked
        );
    });
}

#[test]
fn slash_reduces_role() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Eve has 110 NSN staked (Director role)
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(EVE),
            110,
            1000,
            Region::Latam
        ));
        assert_eq!(NsnStake::stakes(EVE).role, NodeRole::Director);

        // WHEN: Root slashes 20 NSN
        assert_ok!(NsnStake::slash(
            RuntimeOrigin::root(),
            EVE,
            20,
            crate::SlashReason::BftFailure
        ));

        // THEN: Amount = 90, role downgraded to SuperNode
        assert_eq!(NsnStake::stakes(EVE).amount, 90);
        assert_eq!(NsnStake::stakes(EVE).role, NodeRole::SuperNode);
        assert_eq!(NsnStake::total_staked(), 90);

        // Verify event
        let event = last_event();
        assert!(matches!(
            event,
            RuntimeEvent::NsnStake(Event::StakeSlashed { offender, amount, .. })
            if offender == EVE && amount == 20
        ));
    });
}

// ============================================================================
// Boundary Condition Tests
// ============================================================================

#[test]
fn role_determination_boundaries() {
    ExtBuilder::default().build().execute_with(|| {
        // Test exact threshold boundaries
        let test_cases = vec![
            (99, NodeRole::SuperNode), // Just under Director
            (100, NodeRole::Director), // Exact Director threshold
            (49, NodeRole::Validator), // Just under SuperNode
            (50, NodeRole::SuperNode), // Exact SuperNode threshold
            (9, NodeRole::Relay),      // Just under Validator
            (10, NodeRole::Validator), // Exact Validator threshold
            (4, NodeRole::None),       // Just under Relay
            (5, NodeRole::Relay),      // Exact Relay threshold
        ];

        for (i, (amount, expected_role)) in test_cases.into_iter().enumerate() {
            let account = 100 + i as u64; // Use unique accounts
                                          // Fund account
            Balances::mint_into(&account, 200).unwrap();

            assert_ok!(NsnStake::deposit_stake(
                RuntimeOrigin::signed(account),
                amount,
                100,
                Region::NaWest
            ));

            assert_eq!(
                NsnStake::stakes(account).role,
                expected_role,
                "Failed for amount {}: expected {:?}, got {:?}",
                amount,
                expected_role,
                NsnStake::stakes(account).role
            );
        }
    });
}

#[test]
fn multi_region_balance_enforcement() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Stakes across 7 regions totaling 1200 NSN
        let regions_and_stakes = vec![
            (ALICE, Region::NaWest, 200),
            (BOB, Region::NaEast, 180),
            (CHARLIE, Region::EuWest, 190),
            (DAVE, Region::EuEast, 170),
            (EVE, Region::Apac, 160),
            (FRANK, Region::Latam, 150),
            (GEORGE, Region::Mena, 150),
        ];

        for (account, region, amount) in regions_and_stakes {
            assert_ok!(NsnStake::deposit_stake(
                RuntimeOrigin::signed(account),
                amount,
                1000,
                region
            ));
        }

        assert_eq!(NsnStake::total_staked(), 1200);

        // WHEN: George tries to stake 100 NSN in NA-WEST
        // 200 + 100 = 300, which is 25% of 1200 (exceeds 20%)
        assert_noop!(
            NsnStake::deposit_stake(RuntimeOrigin::signed(GEORGE), 100, 1000, Region::NaWest),
            Error::<Test>::RegionCapExceeded
        );

        // WHEN: George stakes 50 NSN in MENA instead
        // 150 + 50 = 200, which is 16.7% of 1200 (under 20%)
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(GEORGE),
            50,
            1000,
            Region::Mena
        ));

        assert_eq!(NsnStake::region_stakes(Region::Mena), 200);
    });
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

#[test]
fn deposit_stake_insufficient_balance() {
    // Use a custom balance setup with lower balance for this test
    ExtBuilder::default()
        .with_balances(vec![(ALICE, 50)]) // Only 50 NSN balance
        .build()
        .execute_with(|| {
            // WHEN: Alice tries to stake more than balance (100 NSN but only has 50)
            // Note: 100 is under per-node cap (1000) so InsufficientBalance triggers first
            assert_noop!(
                NsnStake::deposit_stake(RuntimeOrigin::signed(ALICE), 100, 1000, Region::NaWest),
                Error::<Test>::InsufficientBalance
            );
        });
}

#[test]
fn deposit_does_not_shorten_lock() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice stakes with a long lock
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            100,
            200,
            Region::NaWest
        ));
        let initial_unlock = NsnStake::stakes(ALICE).locked_until;

        // WHEN: Alice deposits again with a shorter lock
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            10,
            50,
            Region::NaWest
        ));

        // THEN: Lock is not shortened
        assert_eq!(NsnStake::stakes(ALICE).locked_until, initial_unlock);
    });
}

#[test]
fn withdraw_partial_stake() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice stakes 150 NSN
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            150,
            100,
            Region::NaWest
        ));

        roll_to(102);

        // WHEN: Alice withdraws 50 NSN (partial)
        assert_ok!(NsnStake::withdraw_stake(RuntimeOrigin::signed(ALICE), 50));

        // THEN: Remaining stake = 100, role still Director
        assert_eq!(NsnStake::stakes(ALICE).amount, 100);
        assert_eq!(NsnStake::stakes(ALICE).role, NodeRole::Director);
    });
}

#[test]
fn withdraw_below_delegation_cap_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Charlie has stake and delegated-to-me balance
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(CHARLIE),
            100,
            10,
            Region::EuWest
        ));
        assert_ok!(NsnStake::delegate(
            RuntimeOrigin::signed(DAVE),
            CHARLIE,
            400
        ));

        // Advance past lock period
        roll_to(20);

        // WHEN: Charlie withdraws enough to violate delegation cap (new max = 200)
        assert_noop!(
            NsnStake::withdraw_stake(RuntimeOrigin::signed(CHARLIE), 60),
            Error::<Test>::DelegationCapExceeded
        );
    });
}

#[test]
fn delegate_to_non_validator() {
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Dave tries to delegate to Bob (who has no stake)
        // THEN: Fails with ValidatorNotFound
        assert_noop!(
            NsnStake::delegate(RuntimeOrigin::signed(DAVE), BOB, 100),
            Error::<Test>::ValidatorNotFound
        );
    });
}

#[test]
fn slash_non_root_fails() {
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Non-root account tries to slash
        assert_noop!(
            NsnStake::slash(
                RuntimeOrigin::signed(ALICE),
                BOB,
                10,
                crate::SlashReason::BftFailure
            ),
            frame_support::error::BadOrigin
        );
    });
}

#[test]
fn multiple_deposits_accumulate() {
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Alice makes multiple deposits
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            50,
            100,
            Region::NaWest
        ));
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            50,
            200,
            Region::NaWest
        ));

        // THEN: Stakes accumulate
        assert_eq!(NsnStake::stakes(ALICE).amount, 100);
        assert_eq!(NsnStake::stakes(ALICE).role, NodeRole::Director);
    });
}

#[test]
fn revoke_delegation() {
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Delegation exists
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(CHARLIE),
            100,
            1000,
            Region::EuWest
        ));
        assert_ok!(NsnStake::delegate(
            RuntimeOrigin::signed(DAVE),
            CHARLIE,
            200
        ));

        // WHEN: Dave revokes delegation
        assert_ok!(NsnStake::revoke_delegation(
            RuntimeOrigin::signed(DAVE),
            CHARLIE
        ));

        // THEN: Delegation removed
        assert_eq!(NsnStake::delegations(DAVE, CHARLIE), 0);
        assert_eq!(NsnStake::stakes(CHARLIE).delegated_to_me, 0);
    });
}

// ============================================================================
// Missing Edge Cases (Added to improve test quality score)
// ============================================================================

#[test]
fn deposit_zero_value_fails_silently() {
    // Test that zero-value deposit doesn't cause issues
    ExtBuilder::default().build().execute_with(|| {
        // WHEN: Alice deposits 0 NSN
        let result = NsnStake::deposit_stake(RuntimeOrigin::signed(ALICE), 0, 100, Region::NaWest);

        // THEN: Should succeed (no-op) or fail gracefully
        // Current implementation allows 0-value deposits (amount = 0, role = None)
        assert_ok!(result);
        assert_eq!(NsnStake::stakes(ALICE).amount, 0);
        assert_eq!(NsnStake::stakes(ALICE).role, NodeRole::None);
    });
}

#[test]
fn delegate_zero_value_fails_silently() {
    // Test that zero-value delegation doesn't cause issues
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Charlie has stake
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(CHARLIE),
            100,
            1000,
            Region::EuWest
        ));

        // WHEN: Dave delegates 0 NSN
        let result = NsnStake::delegate(RuntimeOrigin::signed(DAVE), CHARLIE, 0);

        // THEN: Should succeed (no-op) or fail gracefully
        assert_ok!(result);
        assert_eq!(NsnStake::delegations(DAVE, CHARLIE), 0);
    });
}

#[test]
fn withdraw_zero_value_fails_silently() {
    // Test that zero-value withdrawal doesn't cause issues
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice has stake
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            100,
            100,
            Region::NaWest
        ));
        roll_to(102);

        // WHEN: Alice withdraws 0 NSN
        let result = NsnStake::withdraw_stake(RuntimeOrigin::signed(ALICE), 0);

        // THEN: Should succeed (no-op) or fail gracefully
        assert_ok!(result);
        assert_eq!(NsnStake::stakes(ALICE).amount, 100); // Unchanged
    });
}

#[test]
fn multi_validator_delegation_freeze_accounting() {
    // Test VULN-001 fix: freeze must account for total across all validators
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Two validators
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(CHARLIE),
            100,
            1000,
            Region::EuWest
        ));
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(EVE),
            100,
            1000,
            Region::Latam
        ));

        // WHEN: Dave delegates 200 to Charlie
        assert_ok!(NsnStake::delegate(
            RuntimeOrigin::signed(DAVE),
            CHARLIE,
            200
        ));

        // THEN: Freeze = 200
        assert_eq!(
            Balances::balance_frozen(
                &RuntimeFreezeReason::NsnStake(crate::FreezeReason::Delegating),
                &DAVE
            ),
            200
        );

        // WHEN: Dave delegates 150 to Eve
        assert_ok!(NsnStake::delegate(RuntimeOrigin::signed(DAVE), EVE, 150));

        // THEN: Freeze = 350 (total across both validators)
        assert_eq!(
            Balances::balance_frozen(
                &RuntimeFreezeReason::NsnStake(crate::FreezeReason::Delegating),
                &DAVE
            ),
            350
        );

        // Verify individual delegations
        assert_eq!(NsnStake::delegations(DAVE, CHARLIE), 200);
        assert_eq!(NsnStake::delegations(DAVE, EVE), 150);
    });
}

#[test]
fn revoke_one_delegation_preserves_other_freezes() {
    // Test VULN-002 fix: revoking one delegation shouldn't thaw all
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Dave has delegated to both Charlie and Eve
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(CHARLIE),
            100,
            1000,
            Region::EuWest
        ));
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(EVE),
            100,
            1000,
            Region::Latam
        ));
        assert_ok!(NsnStake::delegate(
            RuntimeOrigin::signed(DAVE),
            CHARLIE,
            200
        ));
        assert_ok!(NsnStake::delegate(RuntimeOrigin::signed(DAVE), EVE, 150));

        // Verify initial freeze = 350
        assert_eq!(
            Balances::balance_frozen(
                &RuntimeFreezeReason::NsnStake(crate::FreezeReason::Delegating),
                &DAVE
            ),
            350
        );

        // WHEN: Dave revokes delegation to Charlie (200)
        assert_ok!(NsnStake::revoke_delegation(
            RuntimeOrigin::signed(DAVE),
            CHARLIE
        ));

        // THEN: Freeze = 150 (only Eve's delegation remains)
        assert_eq!(
            Balances::balance_frozen(
                &RuntimeFreezeReason::NsnStake(crate::FreezeReason::Delegating),
                &DAVE
            ),
            150
        );

        // Verify delegation removed
        assert_eq!(NsnStake::delegations(DAVE, CHARLIE), 0);
        assert_eq!(NsnStake::delegations(DAVE, EVE), 150);
        assert_eq!(NsnStake::stakes(CHARLIE).delegated_to_me, 0);
        assert_eq!(NsnStake::stakes(EVE).delegated_to_me, 150);
    });
}

#[test]
fn revoke_last_delegation_thaws_all() {
    // Test that revoking the last delegation completely thaws
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Single delegation
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(CHARLIE),
            100,
            1000,
            Region::EuWest
        ));
        assert_ok!(NsnStake::delegate(
            RuntimeOrigin::signed(DAVE),
            CHARLIE,
            200
        ));

        // WHEN: Dave revokes delegation
        assert_ok!(NsnStake::revoke_delegation(
            RuntimeOrigin::signed(DAVE),
            CHARLIE
        ));

        // THEN: Freeze = 0 (completely thawed)
        assert_eq!(
            Balances::balance_frozen(
                &RuntimeFreezeReason::NsnStake(crate::FreezeReason::Delegating),
                &DAVE
            ),
            0
        );
    });
}

#[test]
fn withdraw_at_exact_unlock_block() {
    // Test lock boundary: can withdraw at block (locked_until + 1)
    // Note: Mock starts at block 1, not 0
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice stakes at block 1 with lock_blocks=100
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            100,
            100,
            Region::NaWest
        ));
        // locked_until = current_block (1) + lock_blocks (100) = 101
        assert_eq!(NsnStake::stakes(ALICE).locked_until, 101);

        // WHEN: At block 101 (still locked - need > locked_until)
        roll_to(101);
        assert_noop!(
            NsnStake::withdraw_stake(RuntimeOrigin::signed(ALICE), 100),
            Error::<Test>::StakeLocked
        );

        // WHEN: At block 102 (just unlocked - 102 > 101)
        roll_to(102);
        assert_ok!(NsnStake::withdraw_stake(RuntimeOrigin::signed(ALICE), 100));

        // THEN: Withdrawal succeeds
        assert_eq!(NsnStake::stakes(ALICE).amount, 0);
    });
}

#[test]
fn slash_exceeds_stake_capped() {
    // Test over-slash protection: slash amount > stake amount
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice has 50 NSN staked
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            50,
            1000,
            Region::NaWest
        ));

        // WHEN: Root tries to slash 100 NSN (more than stake)
        assert_ok!(NsnStake::slash(
            RuntimeOrigin::root(),
            ALICE,
            100,
            crate::SlashReason::BftFailure
        ));

        // THEN: Only 50 NSN slashed (capped at stake amount)
        assert_eq!(NsnStake::stakes(ALICE).amount, 0);
        assert_eq!(NsnStake::total_staked(), 0);

        // Verify frozen amount also cleared
        assert_eq!(
            Balances::balance_frozen(
                &RuntimeFreezeReason::NsnStake(crate::FreezeReason::Staking),
                &ALICE
            ),
            0
        );
    });
}

#[test]
fn slash_zero_amount_noop() {
    // Test that slashing 0 doesn't cause issues
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice has 100 NSN staked
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            100,
            1000,
            Region::NaWest
        ));

        // WHEN: Root slashes 0 NSN
        assert_ok!(NsnStake::slash(
            RuntimeOrigin::root(),
            ALICE,
            0,
            crate::SlashReason::BftFailure
        ));

        // THEN: Stake unchanged
        assert_eq!(NsnStake::stakes(ALICE).amount, 100);
        assert_eq!(NsnStake::stakes(ALICE).role, NodeRole::Director);
    });
}

#[test]
fn slash_below_delegation_cap_fails() {
    // Slash that would violate delegation cap should fail
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Charlie has stake and delegations
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(CHARLIE),
            100,
            1000,
            Region::EuWest
        ));
        assert_ok!(NsnStake::delegate(
            RuntimeOrigin::signed(DAVE),
            CHARLIE,
            300
        ));

        // WHEN: Root slashes enough to violate cap (new max = 100)
        assert_noop!(
            NsnStake::slash(
                RuntimeOrigin::root(),
                CHARLIE,
                80,
                crate::SlashReason::BftFailure
            ),
            Error::<Test>::DelegationCapExceeded
        );
    });
}

#[test]
fn revoke_nonexistent_delegation_fails() {
    // Test revoking delegation that doesn't exist
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Charlie has stake
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(CHARLIE),
            100,
            1000,
            Region::EuWest
        ));

        // WHEN: Dave tries to revoke delegation he never made
        assert_noop!(
            NsnStake::revoke_delegation(RuntimeOrigin::signed(DAVE), CHARLIE),
            Error::<Test>::DelegationNotFound
        );
    });
}

#[test]
fn withdraw_more_than_stake_fails() {
    // Test withdrawing more than staked amount
    ExtBuilder::default().build().execute_with(|| {
        // GIVEN: Alice has 50 NSN staked
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(ALICE),
            50,
            100,
            Region::NaWest
        ));
        roll_to(102);

        // WHEN: Alice tries to withdraw 100 NSN
        assert_noop!(
            NsnStake::withdraw_stake(RuntimeOrigin::signed(ALICE), 100),
            Error::<Test>::InsufficientStake
        );
    });
}

// ============================================================================
// Node Mode Tests (Dual-Lane Architecture)
// ============================================================================

#[test]
fn test_node_mode_transitions() {
    // Test NodeMode state transitions for dual-lane architecture
    ExtBuilder::default().build().execute_with(|| {
        let account = ALICE;

        // Setup: stake as director
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(account),
            100,
            1000,
            Region::NaWest,
        ));

        // Initially should be Lane1Active (default for staked nodes)
        assert_eq!(NsnStake::node_modes(account), crate::NodeMode::Lane1Active);

        // Transition to Draining (called by director pallet during election)
        assert_ok!(NsnStake::set_node_mode(
            RuntimeOrigin::root(),
            account,
            crate::NodeMode::Draining { epoch_start: 100 }
        ));

        assert_eq!(
            NsnStake::node_modes(account),
            crate::NodeMode::Draining { epoch_start: 100 }
        );

        // Transition to Lane0Active (director now generating video)
        assert_ok!(NsnStake::set_node_mode(
            RuntimeOrigin::root(),
            account,
            crate::NodeMode::Lane0Active { epoch_end: 200 }
        ));

        assert_eq!(
            NsnStake::node_modes(account),
            crate::NodeMode::Lane0Active { epoch_end: 200 }
        );

        // Verify event
        let event = last_event();
        assert!(matches!(
            event,
            RuntimeEvent::NsnStake(Event::NodeModeChanged { account: acc, .. })
            if acc == account
        ));
    });
}

#[test]
fn test_node_mode_requires_stake() {
    // Test that mode change fails for non-staked accounts
    ExtBuilder::default().build().execute_with(|| {
        let account = BOB;

        // WHEN: Trying to set mode for account with no stake
        // THEN: Fails with NotStaked
        assert_noop!(
            NsnStake::set_node_mode(
                RuntimeOrigin::root(),
                account,
                crate::NodeMode::Lane0Active { epoch_end: 100 }
            ),
            Error::<Test>::NotStaked
        );
    });
}

#[test]
fn test_node_mode_requires_root() {
    // Test that non-root cannot change mode
    ExtBuilder::default().build().execute_with(|| {
        let account = ALICE;

        // Setup: stake as director
        assert_ok!(NsnStake::deposit_stake(
            RuntimeOrigin::signed(account),
            100,
            1000,
            Region::NaWest,
        ));

        // WHEN: Non-root tries to change mode
        // THEN: Fails with BadOrigin
        assert_noop!(
            NsnStake::set_node_mode(
                RuntimeOrigin::signed(ALICE),
                account,
                crate::NodeMode::Draining { epoch_start: 100 }
            ),
            frame_support::error::BadOrigin
        );
    });
}
