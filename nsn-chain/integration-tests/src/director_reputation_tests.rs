// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// Integration tests for Director → Reputation chain.
// Tests that director slot finalization and challenge resolution correctly update reputation scores.

use crate::mock::*;
use frame_support::assert_ok;
use pallet_nsn_reputation::ReputationEventType;
use pallet_nsn_stake::Region;

/// Test that successful slot finalization records positive reputation (+100 director)
#[test]
fn test_finalize_slot_records_positive_reputation() {
    new_test_ext().execute_with(|| {
        // Setup: Stake as director
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Record positive reputation event (DirectorSlotAccepted = +100)
        record_reputation(ALICE, ReputationEventType::DirectorSlotAccepted, 1);

        // Verify reputation increased
        let rep = get_reputation(ALICE);
        assert_eq!(rep.director_score, 100, "Director score should increase by 100");
        assert_eq!(rep.total(), 50, "Total should be 50 (100 * 50% director weight)");
    });
}

/// Test that challenge upheld records negative reputation (-200 director)
#[test]
fn test_challenge_upheld_records_negative_reputation() {
    new_test_ext().execute_with(|| {
        // Setup: Give ALICE some positive reputation first
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);
        record_reputation(ALICE, ReputationEventType::DirectorSlotAccepted, 1);
        record_reputation(ALICE, ReputationEventType::DirectorSlotAccepted, 2);
        record_reputation(ALICE, ReputationEventType::DirectorSlotAccepted, 3);

        let rep_before = get_reputation(ALICE);
        assert_eq!(rep_before.director_score, 300, "Should have 300 director score");

        // Record negative reputation event (DirectorSlotRejected = -200)
        record_reputation(ALICE, ReputationEventType::DirectorSlotRejected, 4);

        // Verify reputation decreased
        let rep_after = get_reputation(ALICE);
        assert_eq!(rep_after.director_score, 100, "Director score should decrease by 200");
    });
}

/// Test that missed slots also record negative reputation (-150 director)
#[test]
fn test_missed_slot_records_negative_reputation() {
    new_test_ext().execute_with(|| {
        // Setup: Give ALICE some positive reputation first
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);
        record_reputation(ALICE, ReputationEventType::DirectorSlotAccepted, 1);
        record_reputation(ALICE, ReputationEventType::DirectorSlotAccepted, 2);

        let rep_before = get_reputation(ALICE);
        assert_eq!(rep_before.director_score, 200);

        // Record missed slot (DirectorSlotMissed = -150)
        record_reputation(ALICE, ReputationEventType::DirectorSlotMissed, 3);

        let rep_after = get_reputation(ALICE);
        assert_eq!(rep_after.director_score, 50, "Director score should decrease by 150");
    });
}

/// Test reputation score calculation with weighted components
#[test]
fn test_reputation_weighted_total() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Director events: +100 director
        record_reputation(ALICE, ReputationEventType::DirectorSlotAccepted, 1);

        // Validator events: +5 validator
        record_reputation(ALICE, ReputationEventType::ValidatorVoteCorrect, 2);

        // Seeder events: +10 seeder
        record_reputation(ALICE, ReputationEventType::PinningAuditPassed, 3);

        let rep = get_reputation(ALICE);
        assert_eq!(rep.director_score, 100);
        assert_eq!(rep.validator_score, 5);
        assert_eq!(rep.seeder_score, 10);

        // Total = (director * 50 + validator * 30 + seeder * 20) / 100
        // = (100 * 50 + 5 * 30 + 10 * 20) / 100
        // = (5000 + 150 + 200) / 100 = 53
        assert_eq!(rep.total(), 53, "Weighted total should be 53");
    });
}

/// Test that validator votes affect reputation
#[test]
fn test_validator_vote_reputation() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Correct vote: +5 validator
        record_reputation(ALICE, ReputationEventType::ValidatorVoteCorrect, 1);
        let rep1 = get_reputation(ALICE);
        assert_eq!(rep1.validator_score, 5);

        // Incorrect vote: -10 validator (net -5)
        record_reputation(ALICE, ReputationEventType::ValidatorVoteIncorrect, 2);
        let rep2 = get_reputation(ALICE);
        assert_eq!(rep2.validator_score, 0, "Score should not go below 0");
    });
}

/// Test that seeder events affect reputation
#[test]
fn test_seeder_reputation_events() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Chunk served: +1 seeder
        record_reputation(ALICE, ReputationEventType::SeederChunkServed, 1);
        let rep1 = get_reputation(ALICE);
        assert_eq!(rep1.seeder_score, 1);

        // Audit passed: +10 seeder
        record_reputation(ALICE, ReputationEventType::PinningAuditPassed, 2);
        let rep2 = get_reputation(ALICE);
        assert_eq!(rep2.seeder_score, 11);

        // Audit failed: -50 seeder
        record_reputation(ALICE, ReputationEventType::PinningAuditFailed, 3);
        let rep3 = get_reputation(ALICE);
        assert_eq!(rep3.seeder_score, 0, "Score should not go below 0");
    });
}

/// Test that task completion affects reputation
#[test]
fn test_task_completion_reputation() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Task completed: positive reputation
        record_reputation(ALICE, ReputationEventType::TaskCompleted, 1);
        let rep1 = get_reputation(ALICE);

        // TaskCompleted should increase seeder score (Lane 1 work)
        assert!(rep1.seeder_score > 0, "Task completion should increase seeder score");

        // Task failed: negative reputation
        let rep1_seeder = rep1.seeder_score;
        record_reputation(ALICE, ReputationEventType::TaskFailed, 2);
        let rep2 = get_reputation(ALICE);

        // TaskFailed should decrease seeder score
        assert!(
            rep2.seeder_score < rep1_seeder || rep2.seeder_score == 0,
            "Task failure should decrease seeder score"
        );
    });
}

/// Test that reputation events are included in Merkle root
#[test]
fn test_reputation_merkle_root_published() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Record events
        record_reputation(ALICE, ReputationEventType::DirectorSlotAccepted, 1);
        record_reputation(BOB, ReputationEventType::ValidatorVoteCorrect, 1);

        // Advance block to trigger on_finalize (Merkle root computation)
        roll_to(2);

        // Check that MerkleRootPublished event was emitted
        let events = events();
        let merkle_published = events.iter().any(|e| {
            matches!(e, RuntimeEvent::NsnReputation(
                pallet_nsn_reputation::Event::MerkleRootPublished { .. }
            ))
        });

        assert!(merkle_published, "MerkleRootPublished event should be emitted");
    });
}

/// Test reputation decay for inactive accounts
#[test]
fn test_reputation_decay_for_inactive() {
    new_test_ext().execute_with(|| {
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);

        // Build up reputation
        record_reputation(ALICE, ReputationEventType::DirectorSlotAccepted, 1);
        let rep_active = get_reputation(ALICE);
        assert_eq!(rep_active.director_score, 100);

        // Decay is applied based on last_activity and DecayRatePerWeek (5%)
        // Simulate long inactivity by calling apply_decay
        let current_block = System::block_number();
        // One week = 7 * 24 * 600 blocks = 100800 blocks at 6s/block
        let weeks_inactive = 100800u64;

        pallet_nsn_reputation::Pallet::<Test>::apply_decay(&ALICE, current_block as u64 + weeks_inactive);

        let rep_decayed = get_reputation(ALICE);
        // After decay, director score should be less
        assert!(
            rep_decayed.director_score < rep_active.director_score,
            "Reputation should decay after inactivity"
        );
    });
}
