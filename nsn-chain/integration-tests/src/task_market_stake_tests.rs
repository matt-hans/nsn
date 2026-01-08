// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// Integration tests for Task Market → Stake chain.
// Tests that the task market correctly queries node eligibility and handles slashing.

use crate::mock::*;
use frame_support::assert_ok;
use pallet_nsn_stake::{NodeMode, NodeRole, Region};
use pallet_nsn_task_market::{LaneNodeProvider, TaskLane, ValidatorProvider};

/// Test that only Lane0Active/ActiveDirector nodes are eligible for Lane 0
#[test]
fn test_lane0_nodes_filtered_correctly() {
    new_test_ext().execute_with(|| {
        // Setup: Stake directors
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);
        stake_as_director(BOB, 100 * NSN, Region::EuWest);

        // Set ALICE as Lane0Active (active director)
        use pallet_nsn_stake::{NodeModeUpdater, NodeRoleUpdater};
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&ALICE, NodeMode::Lane0Active { epoch_end: 100 });
        <NsnStake as NodeRoleUpdater<AccountId>>::set_role(&ALICE, NodeRole::ActiveDirector);

        // BOB remains Lane1Active
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&BOB, NodeMode::Lane1Active);

        // Query Lane 0 eligible nodes
        let lane0_nodes = StakeLaneNodeProvider::eligible_nodes(TaskLane::Lane0, 10);

        // Only ALICE should be eligible for Lane 0
        assert_eq!(lane0_nodes.len(), 1, "Only one node should be Lane0 eligible");
        assert_eq!(lane0_nodes[0].0, ALICE, "ALICE should be the Lane0 eligible node");
    });
}

/// Test that only Lane1Active/Reserve nodes are eligible for Lane 1
#[test]
fn test_lane1_nodes_filtered_correctly() {
    new_test_ext().execute_with(|| {
        // Setup: Stake directors
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);
        stake_as_director(BOB, 100 * NSN, Region::EuWest);
        stake_as_director(CHARLIE, 100 * NSN, Region::Apac);

        // ALICE is Lane0Active (should NOT be Lane1 eligible)
        use pallet_nsn_stake::{NodeModeUpdater, NodeRoleUpdater};
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&ALICE, NodeMode::Lane0Active { epoch_end: 100 });
        <NsnStake as NodeRoleUpdater<AccountId>>::set_role(&ALICE, NodeRole::ActiveDirector);

        // BOB and CHARLIE are Lane1Active
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&BOB, NodeMode::Lane1Active);
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&CHARLIE, NodeMode::Lane1Active);

        // Query Lane 1 eligible nodes
        let lane1_nodes = StakeLaneNodeProvider::eligible_nodes(TaskLane::Lane1, 10);

        // BOB and CHARLIE should be eligible for Lane 1
        let lane1_accounts: Vec<AccountId> = lane1_nodes.iter().map(|(a, _)| *a).collect();
        assert!(!lane1_accounts.contains(&ALICE), "ALICE should NOT be Lane1 eligible");
        assert!(lane1_accounts.contains(&BOB), "BOB should be Lane1 eligible");
        assert!(lane1_accounts.contains(&CHARLIE), "CHARLIE should be Lane1 eligible");
    });
}

/// Test is_eligible function
#[test]
fn test_is_eligible_function() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);
        stake_as_director(BOB, 100 * NSN, Region::EuWest);

        // Set modes
        use pallet_nsn_stake::{NodeModeUpdater, NodeRoleUpdater};
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&ALICE, NodeMode::Lane0Active { epoch_end: 100 });
        <NsnStake as NodeRoleUpdater<AccountId>>::set_role(&ALICE, NodeRole::ActiveDirector);
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&BOB, NodeMode::Lane1Active);

        // Check eligibility
        assert!(
            StakeLaneNodeProvider::is_eligible(&ALICE, TaskLane::Lane0),
            "ALICE should be Lane0 eligible"
        );
        assert!(
            !StakeLaneNodeProvider::is_eligible(&ALICE, TaskLane::Lane1),
            "ALICE should NOT be Lane1 eligible"
        );
        assert!(
            !StakeLaneNodeProvider::is_eligible(&BOB, TaskLane::Lane0),
            "BOB should NOT be Lane0 eligible"
        );
        assert!(
            StakeLaneNodeProvider::is_eligible(&BOB, TaskLane::Lane1),
            "BOB should be Lane1 eligible"
        );
    });
}

/// Test that unstaked accounts are not eligible
#[test]
fn test_unstaked_not_eligible() {
    new_test_ext().execute_with(|| {
        // DAVE has balance but no stake
        assert!(!StakeLaneNodeProvider::is_eligible(&DAVE, TaskLane::Lane0));
        assert!(!StakeLaneNodeProvider::is_eligible(&DAVE, TaskLane::Lane1));

        let lane0_nodes = StakeLaneNodeProvider::eligible_nodes(TaskLane::Lane0, 10);
        let lane1_nodes = StakeLaneNodeProvider::eligible_nodes(TaskLane::Lane1, 10);

        let all_accounts: Vec<AccountId> = lane0_nodes.iter()
            .chain(lane1_nodes.iter())
            .map(|(a, _)| *a)
            .collect();

        assert!(!all_accounts.contains(&DAVE), "Unstaked DAVE should not be eligible");
    });
}

/// Test validator provider
#[test]
fn test_validator_provider() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Stake BOB as validator (less than director threshold)
        pallet_nsn_stake::Pallet::<Test>::deposit_stake(
            RuntimeOrigin::signed(BOB),
            10 * NSN, // Validator minimum
            100,
            Region::EuWest,
        ).expect("Staking should succeed");

        // ALICE is a director (also counts as validator)
        assert!(
            StakeValidatorProvider::is_validator(&ALICE),
            "Director ALICE should count as validator"
        );

        // BOB is a validator
        assert!(
            StakeValidatorProvider::is_validator(&BOB),
            "Staked BOB should be validator"
        );

        // CHARLIE has no stake
        assert!(
            !StakeValidatorProvider::is_validator(&CHARLIE),
            "Unstaked CHARLIE should not be validator"
        );
    });
}

/// Test that task abandonment slashes stake
#[test]
fn test_task_abandonment_slashes_stake() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 200 * NSN, Region::NaWest);

        let stake_before = get_stake(ALICE);
        assert_eq!(stake_before.amount, 200 * NSN);

        // Slash for abandonment (5 NSN)
        let slash_amount = 5 * NSN;
        use pallet_nsn_task_market::TaskSlashHandler;
        let result = StakeSlashHandler::slash_for_abandonment(&ALICE, slash_amount);
        assert_ok!(result);

        let stake_after = get_stake(ALICE);
        assert_eq!(
            stake_after.amount,
            200 * NSN - slash_amount,
            "Stake should be reduced by slash amount"
        );
    });
}

/// Test max limit on eligible nodes
#[test]
fn test_eligible_nodes_max_limit() {
    new_test_ext().execute_with(|| {
        // Stake many accounts
        for (i, account) in [ALICE, BOB, CHARLIE, DAVE, EVE, FRANK, GRACE, HENRY, IVAN, JULIA].iter().enumerate() {
            stake_as_director(*account, 100 * NSN, Region::NaWest);

            // Set all as Lane1Active
            use pallet_nsn_stake::NodeModeUpdater;
            <NsnStake as NodeModeUpdater<AccountId>>::set_mode(account, NodeMode::Lane1Active);
        }

        // Request only 3 nodes
        let limited_nodes = StakeLaneNodeProvider::eligible_nodes(TaskLane::Lane1, 3);
        assert!(
            limited_nodes.len() <= 3,
            "Should respect max limit"
        );

        // Request more than available
        let all_nodes = StakeLaneNodeProvider::eligible_nodes(TaskLane::Lane1, 100);
        assert!(
            all_nodes.len() <= 10,
            "Should return at most the number of eligible nodes"
        );
    });
}

/// Test eligibility changes when mode changes
#[test]
fn test_eligibility_updates_on_mode_change() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        use pallet_nsn_stake::{NodeModeUpdater, NodeRoleUpdater};

        // Initially set as Lane1Active
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&ALICE, NodeMode::Lane1Active);

        assert!(StakeLaneNodeProvider::is_eligible(&ALICE, TaskLane::Lane1));
        assert!(!StakeLaneNodeProvider::is_eligible(&ALICE, TaskLane::Lane0));

        // Change to Lane0Active
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&ALICE, NodeMode::Lane0Active { epoch_end: 200 });
        <NsnStake as NodeRoleUpdater<AccountId>>::set_role(&ALICE, NodeRole::ActiveDirector);

        assert!(StakeLaneNodeProvider::is_eligible(&ALICE, TaskLane::Lane0));
        assert!(!StakeLaneNodeProvider::is_eligible(&ALICE, TaskLane::Lane1));

        // Change back to Lane1Active
        <NsnStake as NodeModeUpdater<AccountId>>::set_mode(&ALICE, NodeMode::Lane1Active);
        <NsnStake as NodeRoleUpdater<AccountId>>::set_role(&ALICE, NodeRole::Director);

        assert!(!StakeLaneNodeProvider::is_eligible(&ALICE, TaskLane::Lane0));
        assert!(StakeLaneNodeProvider::is_eligible(&ALICE, TaskLane::Lane1));
    });
}

/// Test reputation is updated on task outcomes via ReputationUpdater
#[test]
fn test_reputation_updated_on_task_outcomes() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Record task success via ReputationUpdater
        use pallet_nsn_task_market::ReputationUpdater;
        ReputationUpdaterImpl::record_task_result(&ALICE, true);

        let rep_after_success = get_reputation(ALICE);
        // TaskCompleted adds to seeder score
        assert!(rep_after_success.seeder_score > 0, "Success should increase seeder score");

        // Record task failure
        let score_before_fail = rep_after_success.seeder_score;
        ReputationUpdaterImpl::record_task_result(&ALICE, false);

        let rep_after_fail = get_reputation(ALICE);
        // TaskFailed decreases seeder score (but won't go below 0)
        assert!(
            rep_after_fail.seeder_score <= score_before_fail,
            "Failure should decrease or maintain seeder score"
        );
    });
}
