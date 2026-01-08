// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// Integration tests for Director → BFT storage chain.
// Tests that director BFT consensus results are correctly stored in the BFT pallet.

use crate::mock::*;
use frame_support::assert_ok;
use pallet_nsn_stake::Region;
use sp_core::H256;

/// Test that finalized slot creates BFT record
#[test]
fn test_director_finalization_stores_bft_result() {
    new_test_ext().execute_with(|| {
        // Setup: Stake directors
        stake_as_director(ALICE, 100 * NSN, Region::NaWest);
        stake_as_director(BOB, 100 * NSN, Region::EuWest);
        stake_as_director(CHARLIE, 100 * NSN, Region::Apac);

        // Store BFT result for slot 1
        let embeddings_hash = test_hash(b"clip_embeddings_slot_1");
        let directors = vec![ALICE, BOB, CHARLIE];

        store_bft_result(1, embeddings_hash, directors.clone(), true);

        // Verify result was stored
        let result = pallet_nsn_bft::Pallet::<Test>::get_slot_result(1);
        assert!(result.is_some(), "BFT result should be stored");

        let round = result.unwrap();
        assert_eq!(round.slot, 1);
        assert_eq!(round.embeddings_hash, embeddings_hash);
        assert!(round.success);
        assert_eq!(round.directors.len(), 3);
    });
}

/// Test that consensus stats are updated on success
#[test]
fn test_bft_stats_updated_on_success() {
    new_test_ext().execute_with(|| {
        // Initial stats should be zero
        let initial_stats = get_bft_stats();
        assert_eq!(initial_stats.total_rounds, 0);
        assert_eq!(initial_stats.successful_rounds, 0);

        // Store successful BFT result
        let hash1 = test_hash(b"slot_1");
        store_bft_result(1, hash1, vec![ALICE, BOB], true);

        let stats1 = get_bft_stats();
        assert_eq!(stats1.total_rounds, 1);
        assert_eq!(stats1.successful_rounds, 1);
        assert_eq!(stats1.failed_rounds, 0);

        // Store another successful result
        let hash2 = test_hash(b"slot_2");
        store_bft_result(2, hash2, vec![ALICE, BOB, CHARLIE], true);

        let stats2 = get_bft_stats();
        assert_eq!(stats2.total_rounds, 2);
        assert_eq!(stats2.successful_rounds, 2);
    });
}

/// Test that consensus stats are updated on failure
#[test]
fn test_bft_stats_updated_on_failure() {
    new_test_ext().execute_with(|| {
        // Store failed BFT result (zero hash indicates failure)
        let zero_hash = H256::zero();
        store_bft_result(1, zero_hash, vec![ALICE], false);

        let stats = get_bft_stats();
        assert_eq!(stats.total_rounds, 1);
        assert_eq!(stats.successful_rounds, 0);
        assert_eq!(stats.failed_rounds, 1);
    });
}

/// Test mixed success and failure stats
#[test]
fn test_bft_mixed_results() {
    new_test_ext().execute_with(|| {
        // 3 successes
        store_bft_result(1, test_hash(b"slot_1"), vec![ALICE, BOB], true);
        store_bft_result(2, test_hash(b"slot_2"), vec![ALICE, BOB], true);
        store_bft_result(3, test_hash(b"slot_3"), vec![ALICE, BOB, CHARLIE], true);

        // 2 failures
        store_bft_result(4, H256::zero(), vec![ALICE], false);
        store_bft_result(5, H256::zero(), vec![BOB], false);

        let stats = get_bft_stats();
        assert_eq!(stats.total_rounds, 5);
        assert_eq!(stats.successful_rounds, 3);
        assert_eq!(stats.failed_rounds, 2);
    });
}

/// Test that embeddings hash can be queried
#[test]
fn test_embeddings_hash_query() {
    new_test_ext().execute_with(|| {
        let hash1 = test_hash(b"embeddings_slot_1");
        let hash2 = test_hash(b"embeddings_slot_2");

        store_bft_result(1, hash1, vec![ALICE], true);
        store_bft_result(2, hash2, vec![BOB], true);

        // Query by slot
        let retrieved1 = pallet_nsn_bft::Pallet::<Test>::get_embeddings_hash(1);
        let retrieved2 = pallet_nsn_bft::Pallet::<Test>::get_embeddings_hash(2);
        let retrieved_none = pallet_nsn_bft::Pallet::<Test>::get_embeddings_hash(99);

        assert_eq!(retrieved1, Some(hash1));
        assert_eq!(retrieved2, Some(hash2));
        assert_eq!(retrieved_none, None);
    });
}

/// Test slot range query
#[test]
fn test_slot_range_query() {
    new_test_ext().execute_with(|| {
        // Store results for slots 5-10
        for slot in 5..=10 {
            let hash = test_hash(&format!("slot_{}", slot).into_bytes());
            store_bft_result(slot, hash, vec![ALICE], true);
        }

        // Query range
        let range = pallet_nsn_bft::Pallet::<Test>::get_slot_range(5, 10);
        assert_eq!(range.len(), 6, "Should return 6 rounds");

        // Query partial range
        let partial = pallet_nsn_bft::Pallet::<Test>::get_slot_range(7, 9);
        assert_eq!(partial.len(), 3);

        // Query empty range (no results stored)
        let empty = pallet_nsn_bft::Pallet::<Test>::get_slot_range(100, 105);
        assert_eq!(empty.len(), 0);
    });
}

/// Test average directors agreeing calculation
#[test]
fn test_average_directors_agreeing() {
    new_test_ext().execute_with(|| {
        // Round 1: 3 directors
        store_bft_result(1, test_hash(b"1"), vec![ALICE, BOB, CHARLIE], true);

        let stats1 = get_bft_stats();
        // First round: average = 3 * 100 = 300
        assert_eq!(stats1.average_directors_agreeing, 300);

        // Round 2: 5 directors
        store_bft_result(2, test_hash(b"2"), vec![ALICE, BOB, CHARLIE, DAVE, EVE], true);

        let stats2 = get_bft_stats();
        // Moving average: ((300 * 1) + (5 * 100)) / 2 = 400
        assert_eq!(stats2.average_directors_agreeing, 400);
    });
}

/// Test that duplicate slot storage is rejected
#[test]
fn test_duplicate_slot_rejected() {
    new_test_ext().execute_with(|| {
        let hash = test_hash(b"slot_1");
        store_bft_result(1, hash, vec![ALICE], true);

        // Try to store again for same slot
        let result = pallet_nsn_bft::Pallet::<Test>::store_embeddings_hash(
            RuntimeOrigin::root(),
            1,
            test_hash(b"different"),
            vec![BOB],
            true,
        );

        assert!(result.is_err(), "Duplicate slot should be rejected");
    });
}

/// Test BFT event emission
#[test]
fn test_bft_event_emission() {
    new_test_ext().execute_with(|| {
        let hash = test_hash(b"slot_1");
        store_bft_result(1, hash, vec![ALICE, BOB], true);

        // Check ConsensusStored event was emitted
        let events = events();
        let consensus_stored = events.iter().any(|e| {
            matches!(e, RuntimeEvent::NsnBft(
                pallet_nsn_bft::Event::ConsensusStored { slot: 1, success: true, .. }
            ))
        });

        assert!(consensus_stored, "ConsensusStored event should be emitted");
    });
}

/// Test retention period configuration
#[test]
fn test_retention_period() {
    new_test_ext().execute_with(|| {
        // Get current retention period
        let retention = pallet_nsn_bft::Pallet::<Test>::retention_period();

        // Default is BftDefaultRetentionPeriod = 2592000 blocks
        assert_eq!(retention, 2592000);
    });
}

/// Test pruning old consensus data
#[test]
fn test_bft_pruning() {
    new_test_ext().execute_with(|| {
        // Store results for slots 1-5
        for slot in 1..=5 {
            store_bft_result(slot, test_hash(&[slot as u8]), vec![ALICE], true);
        }

        // Verify all stored
        assert!(pallet_nsn_bft::Pallet::<Test>::get_slot_result(1).is_some());
        assert!(pallet_nsn_bft::Pallet::<Test>::get_slot_result(5).is_some());

        // Prune slots before 3
        assert_ok!(pallet_nsn_bft::Pallet::<Test>::prune_old_consensus(
            RuntimeOrigin::root(),
            3,
        ));

        // Slots 1-2 should be pruned
        assert!(pallet_nsn_bft::Pallet::<Test>::get_slot_result(1).is_none());
        assert!(pallet_nsn_bft::Pallet::<Test>::get_slot_result(2).is_none());

        // Slots 3-5 should still exist
        assert!(pallet_nsn_bft::Pallet::<Test>::get_slot_result(3).is_some());
        assert!(pallet_nsn_bft::Pallet::<Test>::get_slot_result(5).is_some());
    });
}
