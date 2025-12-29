// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.

//! Tests for pallet-nsn-bft
//!
//! Comprehensive test suite covering all 10 acceptance criteria from T007 task spec.

use crate::{mock::*, Error, Event};
use frame_support::{assert_noop, assert_ok, traits::Hooks};
use sp_core::H256;

// =============================================================================
// Test Scenario 1: Store Finalized BFT Result
// =============================================================================

#[test]
fn test_store_finalized_bft_result() {
    new_test_ext().execute_with(|| {
        // Setup
        let slot = 100u64;
        let embeddings_hash = H256::from_low_u64_be(0xABCD1234);
        let directors = vec![1u64, 2u64, 3u64];

        // Execute
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            slot,
            embeddings_hash,
            directors.clone(),
            true, // success
        ));

        // Verify storage
        assert_eq!(IcnBft::embeddings_hashes(slot), Some(embeddings_hash));

        let round = IcnBft::consensus_rounds(slot).unwrap();
        assert_eq!(round.slot, slot);
        assert_eq!(round.embeddings_hash, embeddings_hash);
        assert_eq!(round.directors, directors);
        assert_eq!(round.timestamp, 1); // Block 1
        assert!(round.success);

        // Verify event
        System::assert_last_event(
            Event::ConsensusStored {
                slot,
                embeddings_hash,
                success: true,
            }
            .into(),
        );

        // Verify statistics updated
        let stats = IcnBft::consensus_stats();
        assert_eq!(stats.total_rounds, 1);
        assert_eq!(stats.successful_rounds, 1);
        assert_eq!(stats.failed_rounds, 0);
        assert_eq!(stats.average_directors_agreeing, 300); // 3 directors × 100
    });
}

// =============================================================================
// Test Scenario 2: Query Historical Slot Result
// =============================================================================

#[test]
fn test_query_historical_slot_result() {
    new_test_ext().execute_with(|| {
        // Store results for slots 50, 51, 52
        let slots = vec![50u64, 51u64, 52u64];
        for slot in &slots {
            let hash = H256::from_low_u64_be(*slot);
            assert_ok!(IcnBft::store_embeddings_hash(
                RuntimeOrigin::root(),
                *slot,
                hash,
                vec![1, 2, 3],
                true,
            ));
        }

        // Query slot 51
        let result = IcnBft::get_slot_result(51);
        assert!(result.is_some());

        let round = result.unwrap();
        assert_eq!(round.slot, 51);
        assert_eq!(round.embeddings_hash, H256::from_low_u64_be(51));
        assert_eq!(round.directors, vec![1, 2, 3]);
        assert!(round.success);

        // Query non-existent slot
        assert!(IcnBft::get_slot_result(999).is_none());
    });
}

// =============================================================================
// Test Scenario 3: Consensus Statistics Tracking
// =============================================================================

#[test]
fn test_consensus_statistics_tracking() {
    new_test_ext().execute_with(|| {
        // Simulate 100 rounds: 95 successful, 5 failed
        for i in 0..95 {
            assert_ok!(IcnBft::store_embeddings_hash(
                RuntimeOrigin::root(),
                i,
                H256::from_low_u64_be(i),
                vec![1, 2, 3, 4], // 4 directors agreeing
                true,
            ));
        }

        for i in 95..100 {
            assert_ok!(IcnBft::store_embeddings_hash(
                RuntimeOrigin::root(),
                i,
                H256::zero(), // ZERO_HASH for failure
                vec![],       // no directors
                false,        // failed consensus
            ));
        }

        // Verify statistics
        let stats = IcnBft::get_stats();
        assert_eq!(stats.total_rounds, 100);
        assert_eq!(stats.successful_rounds, 95);
        assert_eq!(stats.failed_rounds, 5);
        assert_eq!(stats.success_rate(), 95);
        assert_eq!(stats.average_directors_agreeing, 400); // 4.00 directors average
    });
}

// =============================================================================
// Test Scenario 4: Failed Consensus Recording
// =============================================================================

#[test]
fn test_failed_consensus_recording() {
    new_test_ext().execute_with(|| {
        let slot = 200u64;

        // Store failed consensus
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            slot,
            H256::zero(), // ZERO_HASH indicates failure
            vec![],       // empty directors
            false,        // success = false
        ));

        // Verify storage
        let round = IcnBft::consensus_rounds(slot).unwrap();
        assert!(!round.success);
        assert_eq!(round.embeddings_hash, H256::zero());
        assert_eq!(round.directors.len(), 0);

        // Verify statistics
        let stats = IcnBft::consensus_stats();
        assert_eq!(stats.total_rounds, 1);
        assert_eq!(stats.successful_rounds, 0);
        assert_eq!(stats.failed_rounds, 1);
    });
}

// =============================================================================
// Test Scenario 5: Pruning Old Consensus Data
// =============================================================================

#[test]
fn test_pruning_old_consensus_data() {
    new_test_ext().execute_with(|| {
        // Store consensus for slots at different blocks
        // Slot 12 at block 100 (old)
        System::set_block_number(100);
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            12,
            H256::from_low_u64_be(12),
            vec![1, 2, 3],
            true,
        ));

        // Slot 62500 at block 500000 (middle)
        System::set_block_number(500_000);
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            62_500,
            H256::from_low_u64_be(62_500),
            vec![1, 2, 3],
            true,
        ));

        // Slot 312500 at block 2500000 (recent)
        System::set_block_number(2_500_000);
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            312_500,
            H256::from_low_u64_be(312_500),
            vec![1, 2, 3],
            true,
        ));

        // Current block: 3000000
        // Retention: 2592000 blocks
        // Cutoff: 3000000 - 2592000 = 408000 blocks
        // Cutoff slot: 408000 / 8 = 51000
        System::set_block_number(3_000_000);

        // Prune before slot 51000
        assert_ok!(IcnBft::prune_old_consensus(RuntimeOrigin::root(), 51_000));

        // Verify: slot 12 removed, others kept
        assert!(IcnBft::consensus_rounds(12).is_none());
        assert!(IcnBft::consensus_rounds(62_500).is_some());
        assert!(IcnBft::consensus_rounds(312_500).is_some());

        // Verify event
        System::assert_last_event(
            Event::ConsensusPruned {
                before_slot: 51_000,
                count: 1,
            }
            .into(),
        );
    });
}

// =============================================================================
// Test Scenario 6: Batch Query for Range
// =============================================================================

#[test]
fn test_batch_query_for_range() {
    new_test_ext().execute_with(|| {
        // Store consensus for slots 100-200
        for slot in 100..=200 {
            assert_ok!(IcnBft::store_embeddings_hash(
                RuntimeOrigin::root(),
                slot,
                H256::from_low_u64_be(slot),
                vec![1, 2, 3],
                true,
            ));
        }

        // Query range 150-160 (11 slots)
        let results = IcnBft::get_slot_range(150, 160);
        assert_eq!(results.len(), 11);

        // Verify ordered by slot ascending
        for (i, round) in results.iter().enumerate() {
            assert_eq!(round.slot, 150 + i as u64);
        }
    });
}

// =============================================================================
// Test Scenario 7: Challenge Evidence Verification (Query Support)
// =============================================================================

#[test]
fn test_challenge_evidence_verification_support() {
    new_test_ext().execute_with(|| {
        let slot = 300u64;
        let stored_hash = H256::from_low_u64_be(0xABCDEF); // Fraudulent hash
        let real_hash = H256::from_low_u64_be(0x123456); // Real hash (off-chain)

        // Store fraudulent consensus
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            slot,
            stored_hash,
            vec![1, 2, 3],
            true,
        ));

        // Validator queries on-chain hash for comparison
        let on_chain_hash = IcnBft::get_embeddings_hash(slot).unwrap();
        assert_eq!(on_chain_hash, stored_hash);

        // Off-chain: Validator compares on_chain_hash vs real_hash
        // If different, proves fraud (comparison done off-chain or in pallet-nsn-director)
        assert_ne!(on_chain_hash, real_hash);
    });
}

// =============================================================================
// Test Scenario 8: Statistics Update on Each Store
// =============================================================================

#[test]
fn test_statistics_update_on_each_store() {
    new_test_ext().execute_with(|| {
        // Initial state: 50 rounds, 47 successful
        for i in 0..47 {
            assert_ok!(IcnBft::store_embeddings_hash(
                RuntimeOrigin::root(),
                i,
                H256::from_low_u64_be(i),
                vec![1, 2, 3],
                true,
            ));
        }
        for i in 47..50 {
            assert_ok!(IcnBft::store_embeddings_hash(
                RuntimeOrigin::root(),
                i,
                H256::zero(),
                vec![],
                false,
            ));
        }

        let stats_before = IcnBft::consensus_stats();
        assert_eq!(stats_before.total_rounds, 50);
        assert_eq!(stats_before.successful_rounds, 47);

        // Store new successful consensus for slot 51
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            51,
            H256::from_low_u64_be(51),
            vec![1, 2, 3],
            true,
        ));

        // Verify statistics updated atomically
        let stats_after = IcnBft::consensus_stats();
        assert_eq!(stats_after.total_rounds, 51);
        assert_eq!(stats_after.successful_rounds, 48);

        // Success rate: 48/51 * 100 = 94.11... truncates to 94
        assert_eq!(stats_after.success_rate(), 94);
    });
}

// =============================================================================
// Test Scenario 9: Empty Slot Handling
// =============================================================================

#[test]
fn test_empty_slot_handling() {
    new_test_ext().execute_with(|| {
        let slot = 600u64;

        // Attempt to store empty consensus (no directors elected)
        // This represents a slot where insufficient stake prevented election
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            slot,
            H256::zero(),
            vec![], // empty directors
            false,  // failed
        ));

        // Verify stored as failed consensus
        let round = IcnBft::consensus_rounds(slot).unwrap();
        assert!(!round.success);
        assert_eq!(round.directors.len(), 0);
        assert_eq!(round.embeddings_hash, H256::zero());
    });
}

// =============================================================================
// Test Scenario 10: Auto-Pruning on_finalize
// =============================================================================

#[test]
fn test_auto_pruning_on_finalize() {
    new_test_ext().execute_with(|| {
        // Store old consensus at slot 10 (block 1)
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            10,
            H256::from_low_u64_be(10),
            vec![1, 2, 3],
            true,
        ));

        // Store recent consensus at slot 500000 (block 10000)
        System::set_block_number(10_000);
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            500_000,
            H256::from_low_u64_be(500_000),
            vec![1, 2, 3],
            true,
        ));

        // Run to block 20000 (triggers auto-prune at block 10000 and 20000)
        run_to_block(20_000);

        // At block 20000:
        // Retention: 2592000 blocks
        // Cutoff: 20000 - 2592000 = negative, so cutoff_slot = 0
        // Slot 10 should not be pruned yet (retention period not exceeded)

        // Verify slot 10 still exists
        assert!(IcnBft::consensus_rounds(10).is_some());

        // Run to block past retention period
        // Need block > 2,592,000 + 80 (slot 10 stored at approximate block 80)
        System::set_block_number(2_600_000);

        // Manually trigger finalize to test auto-prune
        <IcnBft as Hooks<u64>>::on_finalize(2_600_000);

        // Now slot 10 should be pruned (cutoff_slot = (2600000 - 2592000) / 8 = 1000)
        assert!(IcnBft::consensus_rounds(10).is_none());
        assert!(IcnBft::consensus_rounds(500_000).is_some());
    });
}

// =============================================================================
// Error Cases
// =============================================================================

#[test]
fn test_too_many_directors_error() {
    new_test_ext().execute_with(|| {
        // Attempt to store with 6 directors (max is 5)
        assert_noop!(
            IcnBft::store_embeddings_hash(
                RuntimeOrigin::root(),
                1,
                H256::from_low_u64_be(1),
                vec![1, 2, 3, 4, 5, 6],
                true,
            ),
            Error::<Test>::TooManyDirectors
        );
    });
}

#[test]
fn test_slot_already_stored_error() {
    new_test_ext().execute_with(|| {
        let slot = 100u64;

        // Store consensus for slot 100
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            slot,
            H256::from_low_u64_be(100),
            vec![1, 2, 3],
            true,
        ));

        // Attempt to store again for same slot
        assert_noop!(
            IcnBft::store_embeddings_hash(
                RuntimeOrigin::root(),
                slot,
                H256::from_low_u64_be(200),
                vec![1, 2, 3],
                true,
            ),
            Error::<Test>::SlotAlreadyStored
        );
    });
}

#[test]
fn test_non_root_origin_fails() {
    new_test_ext().execute_with(|| {
        // Attempt to call store_embeddings_hash from signed origin
        assert_noop!(
            IcnBft::store_embeddings_hash(
                RuntimeOrigin::signed(1),
                1,
                H256::from_low_u64_be(1),
                vec![1, 2, 3],
                true,
            ),
            sp_runtime::DispatchError::BadOrigin
        );
    });
}

// =============================================================================
// Edge Cases & Additional Coverage
// =============================================================================

#[test]
fn test_moving_average_calculation_first_round() {
    new_test_ext().execute_with(|| {
        // First successful round with 5 directors
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            1,
            H256::from_low_u64_be(1),
            vec![1, 2, 3, 4, 5],
            true,
        ));

        let stats = IcnBft::consensus_stats();
        assert_eq!(stats.average_directors_agreeing, 500); // 5 × 100
    });
}

#[test]
fn test_moving_average_calculation_multiple_rounds() {
    new_test_ext().execute_with(|| {
        // Round 1: 5 directors
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            1,
            H256::from_low_u64_be(1),
            vec![1, 2, 3, 4, 5],
            true,
        ));

        // Round 2: 3 directors
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            2,
            H256::from_low_u64_be(2),
            vec![1, 2, 3],
            true,
        ));

        let stats = IcnBft::consensus_stats();
        // Average: (500 × 1 + 300 × 1) / 2 = 800 / 2 = 400
        assert_eq!(stats.average_directors_agreeing, 400); // 4.00 directors
    });
}

#[test]
fn test_get_embeddings_hash_query() {
    new_test_ext().execute_with(|| {
        let slot = 42u64;
        let hash = H256::from_low_u64_be(0xDEADBEEF);

        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            slot,
            hash,
            vec![1, 2, 3],
            true,
        ));

        assert_eq!(IcnBft::get_embeddings_hash(slot), Some(hash));
        assert_eq!(IcnBft::get_embeddings_hash(999), None);
    });
}

#[test]
fn test_prune_empty_range() {
    new_test_ext().execute_with(|| {
        // Store some consensus
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            100,
            H256::from_low_u64_be(100),
            vec![1, 2, 3],
            true,
        ));

        // Prune before slot 50 (nothing to prune)
        assert_ok!(IcnBft::prune_old_consensus(RuntimeOrigin::root(), 50));

        // Verify consensus still exists
        assert!(IcnBft::consensus_rounds(100).is_some());
    });
}

#[test]
fn test_success_rate_edge_cases() {
    new_test_ext().execute_with(|| {
        // Test 0% success rate
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            1,
            H256::zero(),
            vec![],
            false,
        ));

        let stats = IcnBft::consensus_stats();
        assert_eq!(stats.success_rate(), 0);

        // Test 100% success rate
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            2,
            H256::from_low_u64_be(2),
            vec![1, 2, 3],
            true,
        ));
        assert_ok!(IcnBft::store_embeddings_hash(
            RuntimeOrigin::root(),
            3,
            H256::from_low_u64_be(3),
            vec![1, 2, 3],
            true,
        ));

        let stats2 = IcnBft::consensus_stats();
        // 2 successful out of 3 total = 66.66... truncates to 66
        assert_eq!(stats2.success_rate(), 66);
    });
}
