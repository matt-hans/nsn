// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Unit tests for pallet-nsn-reputation

use super::*;
use crate::mock::*;
use frame_support::{assert_err, assert_ok, traits::Hooks, BoundedVec};
use sp_core::H256;
use sp_runtime::traits::Hash;

// Scenario 1: Weighted Reputation Scoring
#[test]
fn test_weighted_reputation_scoring() {
	new_test_ext().execute_with(|| {
		// GIVEN: Alice has zero reputation
		assert_eq!(NsnReputation::reputation_scores(ALICE).total(), 0);

		// WHEN: Record multiple events
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			ALICE,
			ReputationEventType::DirectorSlotAccepted,
			100u64,
		));
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			ALICE,
			ReputationEventType::DirectorSlotAccepted,
			101u64,
		));
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			ALICE,
			ReputationEventType::ValidatorVoteCorrect,
			102u64,
		));
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			ALICE,
			ReputationEventType::SeederChunkServed,
			103u64,
		));

		// THEN: Alice's scores are director=200, validator=5, seeder=1
		let score = NsnReputation::reputation_scores(ALICE);
		assert_eq!(score.director_score, 200);
		assert_eq!(score.validator_score, 5);
		assert_eq!(score.seeder_score, 1);

		// AND total() = (200*50 + 5*30 + 1*20) / 100 = 10170 / 100 = 101
		assert_eq!(score.total(), 101);
	});
}

// Scenario 2: Negative Delta and Score Floor
#[test]
fn test_negative_delta_score_floor() {
	new_test_ext().execute_with(|| {
		// GIVEN: Bob has reputation: director=50, validator=10, seeder=5
		let score = ReputationScore {
			director_score: 50,
			validator_score: 10,
			seeder_score: 5,
			last_activity: 1000,
		};
		ReputationScores::<Test>::insert(BOB, score);

		// WHEN: DirectorSlotRejected event is recorded (-200 director)
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			BOB,
			ReputationEventType::DirectorSlotRejected,
			200u64,
		));

		// THEN: Bob's director_score = 0 (floor, not -150)
		let score = NsnReputation::reputation_scores(BOB);
		assert_eq!(score.director_score, 0);
		assert_eq!(score.validator_score, 10);
		assert_eq!(score.seeder_score, 5);

		// AND total() = (0*50 + 10*30 + 5*20) / 100 = 400 / 100 = 4
		assert_eq!(score.total(), 4);
	});
}

// Scenario 3: Decay Over Time
#[test]
fn test_decay_over_time() {
	new_test_ext().execute_with(|| {
		// GIVEN: Charlie has reputation: director=1000, validator=500, seeder=100
		// AND last_activity = block 10000
		let score = ReputationScore {
			director_score: 1000,
			validator_score: 500,
			seeder_score: 100,
			last_activity: 10000,
		};
		ReputationScores::<Test>::insert(CHARLIE, score);

		// WHEN: Current block = 10000 + (12 weeks * ~100,800 blocks/week)
		// 12 weeks later, 5% decay per week
		let weeks = 12u64;
		let blocks_per_week = 7 * 24 * 600; // 100,800 blocks/week
		let current_block = 10000 + (weeks * blocks_per_week);
		System::set_block_number(current_block as u32);

		// Apply decay
		NsnReputation::apply_decay(&CHARLIE, current_block);

		// THEN: weeks_inactive = 12
		// AND decay_factor = 100 - (5 * 12) = 40%
		// AND Charlie's scores = director=400, validator=200, seeder=40
		let score = NsnReputation::reputation_scores(CHARLIE);
		assert_eq!(score.director_score, 400);
		assert_eq!(score.validator_score, 200);
		assert_eq!(score.seeder_score, 40);
	});
}

// Scenario 4: Merkle Root Publication
#[test]
fn test_merkle_root_publication() {
	new_test_ext().execute_with(|| {
		// GIVEN: 5 reputation events recorded in block 100
		let block = 100u32;
		System::set_block_number(block);

		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			ALICE,
			ReputationEventType::DirectorSlotAccepted,
			100u64,
		));
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			BOB,
			ReputationEventType::ValidatorVoteCorrect,
			101u64,
		));
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			CHARLIE,
			ReputationEventType::SeederChunkServed,
			102u64,
		));
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			ALICE,
			ReputationEventType::DirectorSlotAccepted,
			103u64,
		));
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			DAVE,
			ReputationEventType::PinningAuditPassed,
			104u64,
		));

		// WHEN: on_finalize(100) is called
		<NsnReputation as Hooks<u32>>::on_finalize(block);

		// THEN: PendingEvents is cleared
		assert!(NsnReputation::pending_events().is_empty());

		// AND MerkleRoots[100] exists (hash of Merkle tree with 5 leaves)
		assert!(NsnReputation::merkle_roots(block).is_some());

		// AND MerkleRootPublished event emitted
		let events = events();
		let merkle_event = events
			.iter()
			.find(|e| matches!(e, RuntimeEvent::NsnReputation(crate::Event::MerkleRootPublished { .. })));

		assert!(merkle_event.is_some());
	});
}

// Scenario 8: Merkle Proof Verification
#[test]
fn test_merkle_proof_verification() {
	new_test_ext().execute_with(|| {
		let block = 1000u32;
		System::set_block_number(block);

		let events = vec![
			ReputationEvent {
				account: ALICE,
				event_type: ReputationEventType::DirectorSlotAccepted,
				slot: 500,
				block,
			},
			ReputationEvent {
				account: BOB,
				event_type: ReputationEventType::ValidatorVoteCorrect,
				slot: 501,
				block,
			},
			ReputationEvent {
				account: CHARLIE,
				event_type: ReputationEventType::SeederChunkServed,
				slot: 502,
				block,
			},
			ReputationEvent {
				account: DAVE,
				event_type: ReputationEventType::DirectorSlotAccepted,
				slot: 503,
				block,
			},
			ReputationEvent {
				account: EVE,
				event_type: ReputationEventType::PinningAuditPassed,
				slot: 504,
				block,
			},
		];

		let leaves: Vec<H256> = events
			.iter()
			.map(|event| <Test as frame_system::Config>::Hashing::hash_of(event))
			.collect();
		let root = NsnReputation::compute_merkle_root(&events);

		let leaf_index = 2usize;
		let leaf = leaves[leaf_index];
		let proof = build_merkle_proof(&leaves, leaf_index);

		assert!(NsnReputation::verify_merkle_proof(
			leaf,
			leaf_index as u32,
			leaves.len() as u32,
			&proof,
			root
		));

		// Tamper with the leaf to ensure proof fails
		let tampered_leaf = H256::random();
		assert!(!NsnReputation::verify_merkle_proof(
			tampered_leaf,
			leaf_index as u32,
			leaves.len() as u32,
			&proof,
			root
		));
	});
}

fn build_merkle_proof(leaves: &[H256], leaf_index: usize) -> Vec<H256> {
	let mut proof = Vec::new();
	let mut index = leaf_index;
	let mut current = leaves.to_vec();

	while current.len() > 1 {
		let mut next = Vec::new();
		let mut i = 0usize;

		while i < current.len() {
			let left = current[i];
			if i + 1 < current.len() {
				let right = current[i + 1];
				let parent = <Test as frame_system::Config>::Hashing::hash_of(&(left, right));
				next.push(parent);

				if index == i {
					proof.push(right);
					index = next.len() - 1;
				} else if index == i + 1 {
					proof.push(left);
					index = next.len() - 1;
				}
			} else {
				next.push(left);
				if index == i {
					index = next.len() - 1;
				}
			}
			i += 2;
		}

		current = next;
	}

	proof
}

// Scenario 5: Checkpoint Creation
#[test]
fn test_checkpoint_creation() {
	new_test_ext().execute_with(|| {
		// GIVEN: Current block = 5000 (5000 % 1000 == 0)
		// AND 10 accounts have reputation scores
		let block = 5000u32;
		System::set_block_number(block);

		for i in 1..=10u64 {
			let score = ReputationScore {
				director_score: i * 100,
				validator_score: i * 50,
				seeder_score: i * 20,
				last_activity: 100,
			};
			ReputationScores::<Test>::insert(i, score);
		}

		// WHEN: on_finalize(5000) is called
		<NsnReputation as Hooks<u32>>::on_finalize(block);

		// THEN: Checkpoints[5000] is created
		assert!(NsnReputation::checkpoints(block).is_some());

		// AND checkpoint contains: block=5000, score_count=10, merkle_root
		let checkpoint = NsnReputation::checkpoints(block).unwrap();
		assert_eq!(checkpoint.block, block);
		assert_eq!(checkpoint.score_count, 10);

		// AND CheckpointCreated event emitted
		let events = events();
		let checkpoint_event = events
			.iter()
			.find(|e| matches!(e, RuntimeEvent::NsnReputation(crate::Event::CheckpointCreated { .. })));

		assert!(checkpoint_event.is_some());
	});
}

// Scenario 6: Event Pruning Beyond Retention
#[test]
fn test_event_pruning_beyond_retention() {
	new_test_ext().execute_with(|| {
		// GIVEN: RetentionPeriod = 2592000 blocks
		// AND current_block = 3000000
		// AND MerkleRoots contains entries for blocks [100, 500, 10000, 400000, 500000]
		let current_block = 3_000_000u32;
		System::set_block_number(current_block);

		let old_blocks = [100u32, 500, 10_000, 400_000, 500_000];
		for &block in &old_blocks {
			MerkleRoots::<Test>::insert(block, H256::random());
		}

		// WHEN: prune_old_events() is called via on_finalize
		<NsnReputation as Hooks<u32>>::on_finalize(current_block);

		// THEN: MerkleRoots[100], [500], [10000], [400000] are removed
		// (3000000 - block > 2592000)
		assert!(NsnReputation::merkle_roots(100).is_none());
		assert!(NsnReputation::merkle_roots(500).is_none());
		assert!(NsnReputation::merkle_roots(10_000).is_none());
		assert!(NsnReputation::merkle_roots(400_000).is_none());

		// AND MerkleRoots[500000] is kept (3000000 - 500000 < 2592000)
		assert!(NsnReputation::merkle_roots(500_000).is_some());

		// AND EventsPruned event emitted with count=4
		let events = events();
		let prune_event = events
			.iter()
			.find(|e| matches!(e, RuntimeEvent::NsnReputation(crate::Event::EventsPruned { .. })));

		assert!(prune_event.is_some());
	});
}

// Scenario 7: Aggregated Event Batching (TPS Optimization)
#[test]
fn test_aggregated_event_batching() {
	new_test_ext().execute_with(|| {
		// GIVEN: Off-chain aggregator has pending events for Alice
		// - DirectorSlotAccepted (+100)
		// - DirectorSlotAccepted (+100)
		// - DirectorSlotRejected (-200)
		// - ValidatorVoteCorrect (+5)

		// WHEN: Applying these events as a single aggregated call
		let events = BoundedVec::try_from(vec![
			AggregatedEvent {
				event_type: ReputationEventType::DirectorSlotAccepted,
				slot: 100u64,
			},
			AggregatedEvent {
				event_type: ReputationEventType::DirectorSlotAccepted,
				slot: 101u64,
			},
			AggregatedEvent {
				event_type: ReputationEventType::DirectorSlotRejected,
				slot: 102u64,
			},
			AggregatedEvent {
				event_type: ReputationEventType::ValidatorVoteCorrect,
				slot: 103u64,
			},
		])
		.unwrap();

		assert_ok!(NsnReputation::record_aggregated_events(
			RuntimeOrigin::root(),
			ALICE,
			events,
		));

		// THEN: Alice's scores are:
		// director: 100 + 100 - 200 = 0
		// validator: 0 + 5 = 5
		// seeder: 0
		let score = NsnReputation::reputation_scores(ALICE);
		assert_eq!(score.director_score, 0);
		assert_eq!(score.validator_score, 5);
		assert_eq!(score.seeder_score, 0);

		// AND PendingEvents contains 4 entries
		assert_eq!(NsnReputation::pending_events().len(), 4);

		// AND AggregatedEvents storage reflects the batch
		let agg = NsnReputation::aggregated_events(ALICE);
		assert_eq!(agg.net_director_delta, 0);
		assert_eq!(agg.net_validator_delta, 5);
		assert_eq!(agg.net_seeder_delta, 0);
		assert_eq!(agg.event_count, 4);
	});
}

// Scenario 8: Multiple Events Per Block Per Account
#[test]
fn test_multiple_events_per_block_per_account() {
	new_test_ext().execute_with(|| {
		// GIVEN: Alice is both a Director and Validator
		// AND in block 2000, the following occur:
		let block = 2000u32;
		System::set_block_number(block);

		// WHEN: Alice's slot accepted (+100 director)
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			ALICE,
			ReputationEventType::DirectorSlotAccepted,
			200u64,
		));

		// AND Alice validates 3 slots correctly (+5 validator each)
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			ALICE,
			ReputationEventType::ValidatorVoteCorrect,
			201u64,
		));
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			ALICE,
			ReputationEventType::ValidatorVoteCorrect,
			202u64,
		));
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			ALICE,
			ReputationEventType::ValidatorVoteCorrect,
			203u64,
		));

		// THEN: PendingEvents contains 4 distinct entries
		assert_eq!(NsnReputation::pending_events().len(), 4);

		// AND all 4 are included in Merkle tree for block 2000
		<NsnReputation as Hooks<u32>>::on_finalize(block);
		assert!(NsnReputation::merkle_roots(block).is_some());

		// AND Alice's final scores: director=100, validator=15
		let score = NsnReputation::reputation_scores(ALICE);
		assert_eq!(score.director_score, 100);
		assert_eq!(score.validator_score, 15);
	});
}

// Scenario 9: Max Events Per Block Exceeded
#[test]
fn test_max_events_per_block_exceeded() {
	new_test_ext().execute_with(|| {
		// GIVEN: MaxEventsPerBlock = 50
		// AND 50 events already recorded this block
		for i in 0..50u64 {
			assert_ok!(NsnReputation::record_event(
				RuntimeOrigin::root(),
				i,
				ReputationEventType::SeederChunkServed,
				0u64,
			));
		}

		// WHEN: Attempting to record 51st event
		let result = NsnReputation::record_event(
			RuntimeOrigin::root(),
			100u64,
			ReputationEventType::SeederChunkServed,
			0u64,
		);

		// THEN: Call fails with MaxEventsExceeded
		assert_err!(result, Error::<Test>::MaxEventsExceeded);

		// AND 50 events remain recorded
		assert_eq!(NsnReputation::pending_events().len(), 50);
	});
}

// Scenario 10: Governance Adjusts Retention Period
#[test]
fn test_governance_adjusts_retention_period() {
	new_test_ext().execute_with(|| {
		// GIVEN: Current RetentionPeriod = 2592000 blocks
		let initial_period = NsnReputation::retention_period();
		assert_eq!(initial_period, 2_592_000u32);

		// WHEN: Governance proposes and approves update to 1296000 blocks
		assert_ok!(NsnReputation::update_retention(
			RuntimeOrigin::root(),
			1_296_000u32,
		));

		// THEN: RetentionPeriod storage updated
		assert_eq!(NsnReputation::retention_period(), 1_296_000u32);

		// AND RetentionPeriodUpdated event emitted
		let events = events();
		let update_event = events
			.iter()
			.find(|e| {
				matches!(e, RuntimeEvent::NsnReputation(crate::Event::RetentionPeriodUpdated { .. }))
			});

		assert!(update_event.is_some());
	});
}

// Additional Test: Unauthorized Call Fails
#[test]
fn test_unauthorized_call_fails() {
	new_test_ext().execute_with(|| {
		// GIVEN: Regular user (not root)
		// WHEN: Attempting to record event
		let result = NsnReputation::record_event(
			RuntimeOrigin::signed(ALICE),
			BOB,
			ReputationEventType::DirectorSlotAccepted,
			100u64,
		);

		// THEN: Call fails with BadOrigin
		assert_err!(result, sp_runtime::DispatchError::BadOrigin);
	});
}

// Additional Test: Zero Slot Allowed
#[test]
fn test_zero_slot_allowed() {
	new_test_ext().execute_with(|| {
		// WHEN: Recording event with slot=0
		assert_ok!(NsnReputation::record_event(
			RuntimeOrigin::root(),
			ALICE,
			ReputationEventType::SeederChunkServed,
			0u64,
		));

		// THEN: Event recorded successfully
		let score = NsnReputation::reputation_scores(ALICE);
		assert_eq!(score.seeder_score, 1);
	});
}

// Additional Test: Checkpoint Truncation Warning (Best Practice Fix)
#[test]
fn test_checkpoint_truncation_warning() {
	new_test_ext().execute_with(|| {
		// This test documents the behavior when accounts exceed MaxCheckpointAccounts
		// Current implementation truncates at 10,000 accounts

		// GIVEN: 15,000 accounts with reputation
		for i in 1..=15_000u64 {
			let score = ReputationScore {
				director_score: 100,
				validator_score: 50,
				seeder_score: 20,
				last_activity: 100,
			};
			ReputationScores::<Test>::insert(i, score);
		}

		// WHEN: Checkpoint created at block 1000
		let block = 1000u32;
		System::set_block_number(block);
		<NsnReputation as Hooks<u32>>::on_finalize(block);

		// THEN: Checkpoint created with truncated data
		let checkpoint = NsnReputation::checkpoints(block);
		assert!(checkpoint.is_some());

		let checkpoint = checkpoint.unwrap();
		assert_eq!(checkpoint.score_count, 10_000); // Truncated to MaxCheckpointAccounts

		// AND CheckpointTruncated event emitted
		let events = events();
		let truncated_event = events
			.iter()
			.find(|e| matches!(e, RuntimeEvent::NsnReputation(crate::Event::CheckpointTruncated { .. })));

		assert!(truncated_event.is_some());
	});
}
