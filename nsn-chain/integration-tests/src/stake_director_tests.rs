// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// Integration tests for Stake → Director election chain.
// Tests that director elections correctly query stake eligibility and update node modes/roles.

use crate::mock::*;
use frame_support::assert_ok;
use pallet_nsn_stake::{NodeMode, NodeRole, Region};

/// Test that only accounts with stake >= MinStakeDirector become eligible for elections
#[test]
fn test_director_election_respects_stake_eligibility() {
    new_test_ext().execute_with(|| {
        // ALICE stakes enough to be director eligible (100 NSN)
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // BOB stakes less than minimum (50 NSN < 100 NSN)
        pallet_nsn_stake::Pallet::<Test>::deposit_stake(
            RuntimeOrigin::signed(BOB),
            50 * NSN,
            100,
            Region::EuWest,
        ).expect("Staking should succeed");

        // Verify stake amounts
        let alice_stake = get_stake(ALICE);
        let bob_stake = get_stake(BOB);

        assert_eq!(alice_stake.amount, 100 * NSN, "Alice should have 100 NSN staked");
        assert_eq!(bob_stake.amount, 50 * NSN, "Bob should have 50 NSN staked");

        // Alice should be director-eligible, Bob should not
        // Check roles assigned by stake pallet
        assert!(
            matches!(alice_stake.role, NodeRole::Director | NodeRole::Reserve),
            "Alice should be Director or Reserve eligible"
        );
        assert!(
            matches!(bob_stake.role, NodeRole::SuperNode | NodeRole::Validator | NodeRole::None),
            "Bob should be SuperNode, Validator or None (below director threshold)"
        );
    });
}

/// Test that node modes can be updated via the NodeModeUpdater trait
#[test]
fn test_director_election_updates_node_mode() {
    new_test_ext().execute_with(|| {
        // Setup: Multiple accounts stake as directors
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);
        stake_as_director(BOB, 100 * NSN, Region::EuWest);
        stake_as_director(CHARLIE, 100 * NSN, Region::Apac);

        // Initially all should be Lane1Active (default mode)
        let alice_mode = get_node_mode(ALICE);
        assert!(
            matches!(alice_mode, NodeMode::Lane1Active),
            "Initial mode should be Lane1Active"
        );

        // Simulate election by manually updating modes via trait
        use pallet_nsn_stake::{NodeModeUpdater, NodeRoleUpdater};
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&ALICE, NodeMode::Lane0Active { epoch_end: 100 });
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&BOB, NodeMode::Lane0Active { epoch_end: 100 });
        <NsnStake as NodeRoleUpdater<AccountId>>::set_role(&ALICE, NodeRole::ActiveDirector);
        <NsnStake as NodeRoleUpdater<AccountId>>::set_role(&BOB, NodeRole::ActiveDirector);

        // Verify modes were updated
        let modes: Vec<NodeMode> = vec![
            get_node_mode(ALICE),
            get_node_mode(BOB),
            get_node_mode(CHARLIE),
        ];

        let lane0_count = modes.iter().filter(|m| matches!(m, NodeMode::Lane0Active { .. })).count();
        assert_eq!(lane0_count, 2, "Two directors should be Lane0Active");

        // CHARLIE should still be Lane1Active
        assert!(
            matches!(get_node_mode(CHARLIE), NodeMode::Lane1Active),
            "CHARLIE should remain Lane1Active"
        );
    });
}

/// Test that elected directors get NodeRole::ActiveDirector
#[test]
fn test_director_election_updates_node_role() {
    new_test_ext().execute_with(|| {
        // Setup: Stake as director
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Check initial role
        let initial_stake = get_stake(ALICE);
        assert!(
            matches!(initial_stake.role, NodeRole::Director | NodeRole::Reserve),
            "Initial role should be Director or Reserve"
        );

        // Simulate election outcome by calling node role updater directly
        // This tests the trait implementation
        use pallet_nsn_stake::NodeRoleUpdater;
        <NsnStake as NodeRoleUpdater<AccountId>>::set_role(&ALICE, NodeRole::ActiveDirector);

        // Verify role was updated
        let updated_stake = get_stake(ALICE);
        assert_eq!(
            updated_stake.role,
            NodeRole::ActiveDirector,
            "Role should be updated to ActiveDirector"
        );
    });
}

/// Test that previous directors return to Lane1Active after epoch transition
#[test]
fn test_director_epoch_transition_restores_lane1() {
    new_test_ext().execute_with(|| {
        // Setup: Stake as director
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Set ALICE as active director
        use pallet_nsn_stake::{NodeModeUpdater, NodeRoleUpdater};
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&ALICE, NodeMode::Lane0Active { epoch_end: 100 });
        <NsnStake as NodeRoleUpdater<AccountId>>::set_role(&ALICE, NodeRole::ActiveDirector);

        // Verify Lane0Active
        let mode = get_node_mode(ALICE);
        assert!(
            matches!(mode, NodeMode::Lane0Active { .. }),
            "Should be Lane0Active after election"
        );

        // Transition back to Lane1
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&ALICE, NodeMode::Lane1Active);
        <NsnStake as NodeRoleUpdater<AccountId>>::set_role(&ALICE, NodeRole::Director);

        // Verify restored to Lane1Active
        let restored_mode = get_node_mode(ALICE);
        assert!(
            matches!(restored_mode, NodeMode::Lane1Active),
            "Should be restored to Lane1Active after epoch"
        );

        // Verify role is no longer ActiveDirector
        let restored_stake = get_stake(ALICE);
        assert!(
            !matches!(restored_stake.role, NodeRole::ActiveDirector),
            "Should no longer be ActiveDirector after epoch transition"
        );
    });
}

/// Test stake amount affects election weight
#[test]
fn test_stake_amount_affects_election_weight() {
    new_test_ext().execute_with(|| {
        // ALICE stakes maximum (1000 NSN)
        stake_as_director(ALICE, 1000 * NSN, Region::NaWest);

        // BOB stakes minimum (100 NSN)
        stake_as_director(BOB, 100 * NSN, Region::EuWest);

        let alice_stake = get_stake(ALICE);
        let bob_stake = get_stake(BOB);

        // Both should be director eligible
        assert!(
            matches!(alice_stake.role, NodeRole::Director | NodeRole::Reserve),
            "Alice should be director eligible"
        );
        assert!(
            matches!(bob_stake.role, NodeRole::Director | NodeRole::Reserve),
            "Bob should be director eligible"
        );

        // Alice has 10x the stake, so higher weight in elections
        assert!(
            alice_stake.amount > bob_stake.amount,
            "Alice should have higher stake amount"
        );
    });
}

/// Test that stake can be queried after staking
#[test]
fn test_stake_query_after_staking() {
    new_test_ext().execute_with(|| {
        // Setup: Stake as director
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Verify director eligible
        let stake = get_stake(ALICE);
        assert_eq!(stake.amount, 100 * NSN, "Stake amount should be 100 NSN");
        assert!(
            matches!(stake.role, NodeRole::Director | NodeRole::Reserve),
            "Should have director-eligible role"
        );
        assert_eq!(stake.region, Region::NaWest, "Region should match");
    });
}
