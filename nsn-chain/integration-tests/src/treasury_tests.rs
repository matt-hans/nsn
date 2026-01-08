// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// Integration tests for Treasury work recording and reward distribution.
// Tests that treasury correctly accumulates work and distributes rewards.

use crate::mock::*;
use frame_support::assert_ok;
use pallet_nsn_stake::Region;

/// Test that record_director_work increments slot count
#[test]
fn test_director_slots_accumulate_for_rewards() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Record director work
        record_director_work(ALICE, 5);

        let contributions = get_accumulated_contributions(ALICE);
        assert_eq!(contributions.director_slots, 5, "Should have 5 director slots");

        // Record more work
        record_director_work(ALICE, 3);

        let updated = get_accumulated_contributions(ALICE);
        assert_eq!(updated.director_slots, 8, "Should accumulate to 8 slots");
    });
}

/// Test that record_validator_work increments vote count
#[test]
fn test_validator_votes_accumulate_for_rewards() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Record validator work
        record_validator_work(ALICE, 10);

        let contributions = get_accumulated_contributions(ALICE);
        assert_eq!(contributions.validator_votes, 10, "Should have 10 validator votes");

        // Record more votes
        record_validator_work(ALICE, 15);

        let updated = get_accumulated_contributions(ALICE);
        assert_eq!(updated.validator_votes, 25, "Should accumulate to 25 votes");
    });
}

/// Test multiple accounts accumulate work independently
#[test]
fn test_multiple_accounts_accumulate_independently() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);
        stake_as_director(BOB, 100 * NSN, Region::EuWest);

        // ALICE: 10 slots, 5 votes
        record_director_work(ALICE, 10);
        record_validator_work(ALICE, 5);

        // BOB: 3 slots, 20 votes
        record_director_work(BOB, 3);
        record_validator_work(BOB, 20);

        let alice_contrib = get_accumulated_contributions(ALICE);
        let bob_contrib = get_accumulated_contributions(BOB);

        assert_eq!(alice_contrib.director_slots, 10);
        assert_eq!(alice_contrib.validator_votes, 5);
        assert_eq!(bob_contrib.director_slots, 3);
        assert_eq!(bob_contrib.validator_votes, 20);
    });
}

/// Test treasury balance
#[test]
fn test_treasury_balance() {
    new_test_ext().execute_with(|| {
        // Initial treasury balance
        let initial_balance = get_treasury_balance();
        // Treasury starts at 0 (emission schedule creates rewards)
        assert_eq!(initial_balance, 0);

        // Fund treasury
        let fund_amount = 1000 * NSN;
        assert_ok!(pallet_nsn_treasury::Pallet::<Test>::fund_treasury(
            RuntimeOrigin::signed(ALICE),
            fund_amount,
        ));

        let after_funding = get_treasury_balance();
        assert_eq!(after_funding, fund_amount);
    });
}

/// Test emission schedule configuration
#[test]
fn test_emission_schedule() {
    new_test_ext().execute_with(|| {
        let schedule = pallet_nsn_treasury::Pallet::<Test>::emission_schedule();

        // Base emission: 100M NSN year 1
        assert_eq!(schedule.base_emission, 100_000_000 * NSN);
        // Decay rate: 15%
        assert_eq!(schedule.decay_rate, sp_runtime::Perbill::from_percent(15));
        // Current year: 1
        assert_eq!(schedule.current_year, 1);
    });
}

/// Test annual emission calculation
#[test]
fn test_annual_emission_calculation() {
    new_test_ext().execute_with(|| {
        // Year 1: 100M
        let year1 = pallet_nsn_treasury::Pallet::<Test>::calculate_annual_emission(1);
        assert!(year1.is_ok(), "Year 1 calculation should succeed");
        assert_eq!(year1.unwrap(), 100_000_000 * NSN);

        // Year 2: 85M (100M * 0.85)
        let year2 = pallet_nsn_treasury::Pallet::<Test>::calculate_annual_emission(2);
        assert!(year2.is_ok(), "Year 2 calculation should succeed");
        assert_eq!(year2.unwrap(), 85_000_000 * NSN);

        // Year 3: ~72.25M (85M * 0.85)
        let year3 = pallet_nsn_treasury::Pallet::<Test>::calculate_annual_emission(3);
        assert!(year3.is_ok(), "Year 3 calculation should succeed");
        // Due to Perbill precision, check approximate value
        let expected_year3 = 72_250_000 * NSN;
        let actual_year3 = year3.unwrap();
        assert!(
            actual_year3 >= expected_year3 - NSN && actual_year3 <= expected_year3 + NSN,
            "Year 3 emission should be approximately 72.25M"
        );
    });
}

/// Test reward distribution configuration
#[test]
fn test_reward_distribution_config() {
    new_test_ext().execute_with(|| {
        let distribution = pallet_nsn_treasury::Pallet::<Test>::reward_distribution();

        // 40% directors
        assert_eq!(distribution.director_percent, sp_runtime::Perbill::from_percent(40));
        // 25% validators
        assert_eq!(distribution.validator_percent, sp_runtime::Perbill::from_percent(25));
        // 20% pinners
        assert_eq!(distribution.pinner_percent, sp_runtime::Perbill::from_percent(20));
        // 15% treasury
        assert_eq!(distribution.treasury_percent, sp_runtime::Perbill::from_percent(15));
    });
}

/// Test work recording events are emitted correctly
#[test]
fn test_distribution_events() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);
        stake_as_director(BOB, 100 * NSN, Region::EuWest);

        // Record work - these emit events
        record_director_work(ALICE, 10);
        record_validator_work(BOB, 20);

        // Check for work recording events
        let events = events();
        let director_event = events.iter().any(|e| {
            matches!(e, RuntimeEvent::NsnTreasury(
                pallet_nsn_treasury::Event::DirectorWorkRecorded { .. }
            ))
        });
        let validator_event = events.iter().any(|e| {
            matches!(e, RuntimeEvent::NsnTreasury(
                pallet_nsn_treasury::Event::ValidatorWorkRecorded { .. }
            ))
        });

        assert!(director_event, "DirectorWorkRecorded event should be emitted");
        assert!(validator_event, "ValidatorWorkRecorded event should be emitted");
    });
}

/// Test DirectorWorkRecorded event
#[test]
fn test_director_work_event() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);
        record_director_work(ALICE, 5);

        let events = events();
        let work_recorded = events.iter().any(|e| {
            matches!(e, RuntimeEvent::NsnTreasury(
                pallet_nsn_treasury::Event::DirectorWorkRecorded { account, slots }
            ) if *account == ALICE && *slots == 5)
        });

        assert!(work_recorded, "DirectorWorkRecorded event should be emitted");
    });
}

/// Test ValidatorWorkRecorded event
#[test]
fn test_validator_work_event() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);
        record_validator_work(ALICE, 10);

        let events = events();
        let work_recorded = events.iter().any(|e| {
            matches!(e, RuntimeEvent::NsnTreasury(
                pallet_nsn_treasury::Event::ValidatorWorkRecorded { account, votes }
            ) if *account == ALICE && *votes == 10)
        });

        assert!(work_recorded, "ValidatorWorkRecorded event should be emitted");
    });
}

/// Test treasury funding event
#[test]
fn test_treasury_funding_event() {
    new_test_ext().execute_with(|| {
        let fund_amount = 100 * NSN;
        assert_ok!(pallet_nsn_treasury::Pallet::<Test>::fund_treasury(
            RuntimeOrigin::signed(ALICE),
            fund_amount,
        ));

        let events = events();
        let funding_event = events.iter().any(|e| {
            matches!(e, RuntimeEvent::NsnTreasury(
                pallet_nsn_treasury::Event::TreasuryFunded { funder, amount }
            ) if *funder == ALICE && *amount == fund_amount)
        });

        assert!(funding_event, "TreasuryFunded event should be emitted");
    });
}

/// Test that contributions accumulate and persist correctly
#[test]
fn test_contributions_persistence() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Record work across multiple blocks
        record_director_work(ALICE, 2);
        roll_to(10);
        record_director_work(ALICE, 3);
        roll_to(20);
        record_validator_work(ALICE, 5);

        let contributions = get_accumulated_contributions(ALICE);
        assert_eq!(contributions.director_slots, 5);
        assert_eq!(contributions.validator_votes, 5);
    });
}

/// Test work accumulation persists correctly
#[test]
fn test_distribution_frequency() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);
        record_director_work(ALICE, 10);

        // Record work and verify it accumulates
        let contrib = get_accumulated_contributions(ALICE);
        assert_eq!(contrib.director_slots, 10, "Work should be accumulated");

        // Verify last distribution starts at 0
        let last_dist = pallet_nsn_treasury::Pallet::<Test>::last_distribution_block();
        assert_eq!(last_dist, 0, "Last distribution should start at 0");

        // Record more work
        record_director_work(ALICE, 5);
        let updated_contrib = get_accumulated_contributions(ALICE);
        assert_eq!(updated_contrib.director_slots, 15, "Work should continue to accumulate");
    });
}
