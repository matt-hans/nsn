// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.

//! Tests for pallet-nsn-treasury

use crate::{mock::*, Error, Event};
use frame_support::{assert_noop, assert_ok, traits::{fungible::Inspect, Hooks}};
use sp_runtime::Perbill;

const ALICE: u64 = 1;
const BOB: u64 = 2;
const CHARLIE: u64 = 3;

// Helper to get treasury pallet account
#[allow(dead_code)]
fn treasury_account() -> u64 {
	Treasury::account_id()
}

#[test]
fn test_emission_year_1() {
	new_test_ext().execute_with(|| {
		// Year 1 should return base emission exactly (100M with 18 decimals)
		let emission = Treasury::calculate_annual_emission(1).unwrap();
		assert_eq!(emission, 100_000_000_000_000_000_000_000_000u128);
	});
}

#[test]
fn test_emission_year_2() {
	new_test_ext().execute_with(|| {
		// Year 2 should have 15% decay: 100M * 0.85 = 85M
		let emission = Treasury::calculate_annual_emission(2).unwrap();
		// Allow small rounding error due to Perbill precision
		let expected = 85_000_000_000_000_000_000_000_000u128;
		let tolerance = expected / 1000; // 0.1% tolerance
		assert!((emission as i128 - expected as i128).abs() < tolerance as i128);
	});
}

#[test]
fn test_emission_year_5() {
	new_test_ext().execute_with(|| {
		// Year 5: 100M * (0.85)^4 ≈ 52.2M
		let emission = Treasury::calculate_annual_emission(5).unwrap();
		let expected = 52_200_625_000_000_000_000_000_000u128;
		let tolerance = expected / 100; // 1% tolerance for accumulated rounding
		assert!((emission as i128 - expected as i128).abs() < tolerance as i128);
	});
}

#[test]
fn test_emission_year_10() {
	new_test_ext().execute_with(|| {
		// Year 10: 100M * (0.85)^9 ≈ 23.16M
		let emission = Treasury::calculate_annual_emission(10).unwrap();
		let expected = 23_160_000_000_000_000_000_000_000u128;
		let tolerance = expected / 50; // 2% tolerance
		assert!((emission as i128 - expected as i128).abs() < tolerance as i128);
	});
}

#[test]
fn test_emission_year_zero() {
	new_test_ext().execute_with(|| {
		let emission = Treasury::calculate_annual_emission(0).unwrap();
		assert_eq!(emission, 0);
	});
}

#[test]
fn test_reward_split_percentages() {
	new_test_ext().execute_with(|| {
		let distribution = Treasury::reward_distribution();

		// Verify 40/25/20/15 split
		assert_eq!(distribution.director_percent, Perbill::from_percent(40));
		assert_eq!(distribution.validator_percent, Perbill::from_percent(25));
		assert_eq!(distribution.pinner_percent, Perbill::from_percent(20));
		assert_eq!(distribution.treasury_percent, Perbill::from_percent(15));

		// Verify total = 100%
		let total = distribution.director_percent
			+ distribution.validator_percent
			+ distribution.pinner_percent
			+ distribution.treasury_percent;
		assert_eq!(total, Perbill::from_percent(100));
	});
}

#[test]
fn test_fund_treasury() {
	new_test_ext().execute_with(|| {
		let amount = 500_000_000_000_000_000_000_000_000u128; // 500M ICN
		let initial_balance = Balances::balance(&ALICE);

		// Fund treasury
		assert_ok!(Treasury::fund_treasury(RuntimeOrigin::signed(ALICE), amount));

		// Check treasury balance increased
		assert_eq!(Treasury::treasury_balance(), amount);

		// Check Alice's balance decreased
		assert_eq!(Balances::balance(&ALICE), initial_balance - amount);

		// Check event
		System::assert_last_event(
			Event::TreasuryFunded { funder: ALICE, amount }.into()
		);
	});
}

#[test]
fn test_approve_proposal_success() {
	new_test_ext().execute_with(|| {
		let amount = 100_000_000_000_000_000_000_000_000u128; // 100M ICN

		// Fund treasury first
		assert_ok!(Treasury::fund_treasury(RuntimeOrigin::signed(ALICE), amount));

		let bob_initial = Balances::balance(&BOB);

		// Approve proposal
		assert_ok!(Treasury::approve_proposal(
			RuntimeOrigin::root(),
			BOB,
			amount,
			1 // proposal_id
		));

		// Treasury balance should be 0
		assert_eq!(Treasury::treasury_balance(), 0);

		// Bob received funds
		assert_eq!(Balances::balance(&BOB), bob_initial + amount);

		// Check event
		System::assert_last_event(
			Event::ProposalApproved { proposal_id: 1, beneficiary: BOB, amount }.into()
		);
	});
}

#[test]
fn test_approve_proposal_insufficient_funds() {
	new_test_ext().execute_with(|| {
		let amount = 100_000_000_000_000_000_000_000_000u128;

		// Fund treasury with less than proposal amount
		assert_ok!(Treasury::fund_treasury(RuntimeOrigin::signed(ALICE), amount / 2));

		// Try to approve larger proposal
		assert_noop!(
			Treasury::approve_proposal(RuntimeOrigin::root(), BOB, amount, 1),
			Error::<Test>::InsufficientTreasuryFunds
		);
	});
}

#[test]
fn test_approve_proposal_requires_root() {
	new_test_ext().execute_with(|| {
		let amount = 100_000_000_000_000_000_000_000_000u128;
		assert_ok!(Treasury::fund_treasury(RuntimeOrigin::signed(ALICE), amount));

		// Non-root origin should fail
		assert_noop!(
			Treasury::approve_proposal(RuntimeOrigin::signed(ALICE), BOB, amount, 1),
			sp_runtime::DispatchError::BadOrigin
		);
	});
}

#[test]
fn test_director_work_recording() {
	new_test_ext().execute_with(|| {
		// Record 5 slots for Alice
		assert_ok!(Treasury::record_director_work(RuntimeOrigin::root(), ALICE, 5));

		let contrib = Treasury::accumulated_contributions(ALICE);
		assert_eq!(contrib.director_slots, 5);
		assert_eq!(contrib.validator_votes, 0);

		// Record 3 more slots
		assert_ok!(Treasury::record_director_work(RuntimeOrigin::root(), ALICE, 3));

		let contrib = Treasury::accumulated_contributions(ALICE);
		assert_eq!(contrib.director_slots, 8);

		// Check event
		System::assert_last_event(
			Event::DirectorWorkRecorded { account: ALICE, slots: 3 }.into()
		);
	});
}

#[test]
fn test_validator_work_recording() {
	new_test_ext().execute_with(|| {
		// Record 10 votes for Bob
		assert_ok!(Treasury::record_validator_work(RuntimeOrigin::root(), BOB, 10));

		let contrib = Treasury::accumulated_contributions(BOB);
		assert_eq!(contrib.validator_votes, 10);
		assert_eq!(contrib.director_slots, 0);

		// Record 5 more votes
		assert_ok!(Treasury::record_validator_work(RuntimeOrigin::root(), BOB, 5));

		let contrib = Treasury::accumulated_contributions(BOB);
		assert_eq!(contrib.validator_votes, 15);
	});
}

#[test]
fn test_director_rewards_proportional() {
	new_test_ext().execute_with(|| {
		// Alice: 20 slots, Bob: 15 slots, Charlie: 10 slots
		assert_ok!(Treasury::record_director_work(RuntimeOrigin::root(), ALICE, 20));
		assert_ok!(Treasury::record_director_work(RuntimeOrigin::root(), BOB, 15));
		assert_ok!(Treasury::record_director_work(RuntimeOrigin::root(), CHARLIE, 10));

		let pool = 109_589_000_000_000_000_000_000_000u128; // 109.589M ICN
		let initial_alice = Balances::balance(&ALICE);
		let initial_bob = Balances::balance(&BOB);
		let initial_charlie = Balances::balance(&CHARLIE);

		// Distribute rewards
		assert_ok!(Treasury::distribute_director_rewards(pool));

		// Total slots = 45
		// Alice should get: pool * 20/45 ≈ 48.706M
		// Bob should get: pool * 15/45 ≈ 36.530M
		// Charlie should get: pool * 10/45 ≈ 24.353M

		let alice_reward = Balances::balance(&ALICE) - initial_alice;
		let bob_reward = Balances::balance(&BOB) - initial_bob;
		let charlie_reward = Balances::balance(&CHARLIE) - initial_charlie;

		// Check proportions (allowing rounding)
		assert!(alice_reward > bob_reward);
		assert!(bob_reward > charlie_reward);

		// Check ratios approximately match slots
		let alice_expected = pool * 20 / 45;
		let bob_expected = pool * 15 / 45;
		let charlie_expected = pool * 10 / 45;

		assert!((alice_reward as i128 - alice_expected as i128).abs() < 1000);
		assert!((bob_reward as i128 - bob_expected as i128).abs() < 1000);
		assert!((charlie_reward as i128 - charlie_expected as i128).abs() < 1000);

		// Check contributions reset
		assert_eq!(Treasury::accumulated_contributions(ALICE).director_slots, 0);
		assert_eq!(Treasury::accumulated_contributions(BOB).director_slots, 0);
		assert_eq!(Treasury::accumulated_contributions(CHARLIE).director_slots, 0);
	});
}

#[test]
fn test_validator_rewards_proportional() {
	new_test_ext().execute_with(|| {
		// Alice: 100 votes, Bob: 80 votes, Charlie: 60 votes
		assert_ok!(Treasury::record_validator_work(RuntimeOrigin::root(), ALICE, 100));
		assert_ok!(Treasury::record_validator_work(RuntimeOrigin::root(), BOB, 80));
		assert_ok!(Treasury::record_validator_work(RuntimeOrigin::root(), CHARLIE, 60));

		let pool = 68_493_000_000_000_000_000_000_000u128; // 68.493M ICN
		let initial_alice = Balances::balance(&ALICE);
		let initial_bob = Balances::balance(&BOB);
		let initial_charlie = Balances::balance(&CHARLIE);

		// Distribute rewards
		assert_ok!(Treasury::distribute_validator_rewards(pool));

		// Total votes = 240
		let alice_reward = Balances::balance(&ALICE) - initial_alice;
		let bob_reward = Balances::balance(&BOB) - initial_bob;
		let charlie_reward = Balances::balance(&CHARLIE) - initial_charlie;

		// Check proportions
		assert!(alice_reward > bob_reward);
		assert!(bob_reward > charlie_reward);

		// Check approximate expected values
		let alice_expected = pool * 100 / 240;
		let bob_expected = pool * 80 / 240;
		let charlie_expected = pool * 60 / 240;

		assert!((alice_reward as i128 - alice_expected as i128).abs() < 1000);
		assert!((bob_reward as i128 - bob_expected as i128).abs() < 1000);
		assert!((charlie_reward as i128 - charlie_expected as i128).abs() < 1000);

		// Check contributions reset
		assert_eq!(Treasury::accumulated_contributions(ALICE).validator_votes, 0);
		assert_eq!(Treasury::accumulated_contributions(BOB).validator_votes, 0);
		assert_eq!(Treasury::accumulated_contributions(CHARLIE).validator_votes, 0);
	});
}

#[test]
fn test_zero_participants_directors() {
	new_test_ext().execute_with(|| {
		let pool = 100_000_000_000_000_000_000_000_000u128;

		// Distribute with no contributors
		assert_ok!(Treasury::distribute_director_rewards(pool));

		// No events should be emitted except the function completing
		// No panics or errors
	});
}

#[test]
fn test_zero_participants_validators() {
	new_test_ext().execute_with(|| {
		let pool = 100_000_000_000_000_000_000_000_000u128;

		// Distribute with no contributors
		assert_ok!(Treasury::distribute_validator_rewards(pool));

		// No events, no panics
	});
}

#[test]
fn test_distribution_frequency_trigger() {
	new_test_ext().execute_with(|| {
		// Record some work
		assert_ok!(Treasury::record_director_work(RuntimeOrigin::root(), ALICE, 10));

		// Distribution should not trigger at block 1000
		System::set_block_number(1000);
		Treasury::on_finalize(1000);
		assert_eq!(Treasury::last_distribution_block(), 0);

		// Distribution should trigger at block 14400
		System::set_block_number(14400);
		Treasury::on_finalize(14400);
		assert_eq!(Treasury::last_distribution_block(), 14400);

		// Next distribution at block 28800
		System::set_block_number(28800);
		Treasury::on_finalize(28800);
		assert_eq!(Treasury::last_distribution_block(), 28800);
	});
}

#[test]
fn test_year_auto_increment() {
	new_test_ext().execute_with(|| {
		// Initial year = 1
		let schedule = Treasury::emission_schedule();
		assert_eq!(schedule.current_year, 1);

		// After 1 year of blocks (365 * 14400 = 5,256,000)
		let blocks_per_year = 365 * 14400;
		System::set_block_number(blocks_per_year);
		Treasury::on_finalize(blocks_per_year);

		let schedule = Treasury::emission_schedule();
		assert_eq!(schedule.current_year, 2);

		// After 2 years
		System::set_block_number(blocks_per_year * 2);
		Treasury::on_finalize(blocks_per_year * 2);

		let schedule = Treasury::emission_schedule();
		assert_eq!(schedule.current_year, 3);
	});
}

#[test]
fn test_full_distribution_cycle() {
	new_test_ext().execute_with(|| {
		// Setup: Record work for multiple participants
		assert_ok!(Treasury::record_director_work(RuntimeOrigin::root(), ALICE, 20));
		assert_ok!(Treasury::record_director_work(RuntimeOrigin::root(), BOB, 10));
		assert_ok!(Treasury::record_validator_work(RuntimeOrigin::root(), CHARLIE, 50));

		let initial_treasury = Treasury::treasury_balance();
		let initial_alice = Balances::balance(&ALICE);
		let initial_bob = Balances::balance(&BOB);
		let initial_charlie = Balances::balance(&CHARLIE);

		// Trigger distribution at block 14400
		System::set_block_number(14400);
		Treasury::on_finalize(14400);

		// Check that rewards were distributed
		assert!(Balances::balance(&ALICE) > initial_alice, "Alice should receive director rewards");
		assert!(Balances::balance(&BOB) > initial_bob, "Bob should receive director rewards");
		assert!(Balances::balance(&CHARLIE) > initial_charlie, "Charlie should receive validator rewards");

		// Check treasury balance increased (15% allocation)
		assert!(Treasury::treasury_balance() > initial_treasury, "Treasury should receive 15% allocation");

		// Check contributions were reset
		assert_eq!(Treasury::accumulated_contributions(ALICE).director_slots, 0);
		assert_eq!(Treasury::accumulated_contributions(BOB).director_slots, 0);
		assert_eq!(Treasury::accumulated_contributions(CHARLIE).validator_votes, 0);

		// Check distribution event emitted
		let events = System::events();
		assert!(events.iter().any(|e| matches!(
			e.event,
			RuntimeEvent::Treasury(Event::RewardsDistributed { .. })
		)));
	});
}

#[test]
fn test_overflow_protection_emission() {
	new_test_ext().execute_with(|| {
		// Test emission calculation doesn't panic on large years
		for year in 1..=50 {
			let emission = Treasury::calculate_annual_emission(year);
			assert!(emission.is_ok(), "Emission calculation should not overflow for year {}", year);
		}
	});
}

#[test]
fn test_overflow_protection_rewards() {
	new_test_ext().execute_with(|| {
		// Record maximum u64 work
		assert_ok!(Treasury::record_director_work(RuntimeOrigin::root(), ALICE, u64::MAX));

		let pool = u128::MAX / 1000; // Large pool

		// Should not panic with saturating arithmetic
		assert_ok!(Treasury::distribute_director_rewards(pool));
	});
}
