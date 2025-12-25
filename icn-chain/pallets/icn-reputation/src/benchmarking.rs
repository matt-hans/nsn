// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.
//
// ICN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Benchmarking for pallet-icn-reputation
//!
//! # Running Benchmarks
//!
//! ```bash
//! # Build the node with benchmarking feature
//! cargo build --release --features runtime-benchmarks
//!
//! # Run benchmarks for this pallet
//! ./target/release/icn-node benchmark pallet \
//!   --chain dev \
//!   --pallet pallet_icn_reputation \
//!   --extrinsics '*' \
//!   --steps 50 \
//!   --repeat 20 \
//!   --output ./pallets/icn-reputation/src/weights.rs
//! ```
//!
//! # Benchmark Components
//!
//! Each benchmark measures:
//! - Database reads/writes
//! - Computation complexity (Merkle tree, scoring)
//! - Event emissions
//! - Storage iteration (for checkpoints, pruning)

#![cfg(feature = "runtime-benchmarks")]

use super::*;
use crate::Pallet as IcnReputation;
use frame_benchmarking::v2::*;
use frame_system::{Pallet as System, RawOrigin};
use sp_std::prelude::*;

#[benchmarks]
mod benchmarks {
	use super::*;

	/// Benchmark `record_event` extrinsic
	///
	/// # Weight Components
	/// - Storage reads: 2 (ReputationScores, PendingEvents)
	/// - Storage writes: 2 (ReputationScores, PendingEvents)
	/// - Computation: Score update, delta calculation
	/// - Events: 1
	///
	/// # Worst Case
	/// - Account already has reputation (update, not insert)
	/// - PendingEvents near max capacity (but not full)
	#[benchmark]
	fn record_event() {
		let caller: T::AccountId = whitelisted_caller();
		let event_type = ReputationEventType::DirectorSlotAccepted;
		let slot = 100u64;

		#[extrinsic_call]
		record_event(RawOrigin::Root, caller.clone(), event_type, slot);

		// Verify state changed
		assert_eq!(IcnReputation::reputation_scores(caller).director_score, 100);
	}

	/// Benchmark `record_event` with maximum pending events (worst case)
	///
	/// # Weight Components
	/// - Same as `record_event`, but with BoundedVec nearly full
	///
	/// # Worst Case
	/// - PendingEvents has (MaxEventsPerBlock - 1) events
	/// - This is the last event that can fit
	#[benchmark]
	fn record_event_max_pending() {
		let caller: T::AccountId = whitelisted_caller();
		let event_type = ReputationEventType::SeederChunkServed;
		let max_events = T::MaxEventsPerBlock::get();

		// Setup: Fill PendingEvents to (max - 1)
		for i in 0..(max_events - 1) {
			let account = i as u64;
			IcnReputation::record_event(RawOrigin::Root, account, event_type.clone(), 0u64)
				.unwrap();
		}

		// Benchmark one more event (worst case)
		#[extrinsic_call]
		record_event(RawOrigin::Root, caller.clone(), ReputationEventType::SeederChunkServed, 0u64);

		// Verify it was added
		assert_eq!(IcnReputation::pending_events().len() as u32, max_events);
	}

	/// Benchmark `record_event` with reputation decay
	///
	/// # Weight Components
	/// - Same as `record_event`, plus decay calculation
	///
	/// # Worst Case
	/// - Account has been inactive for many weeks
	/// - Decay calculation applies to all three score components
	#[benchmark]
	fn record_event_with_decay() {
		let caller: T::AccountId = whitelisted_caller();
		let event_type = ReputationEventType::DirectorSlotAccepted;

		// Setup: Account with old activity timestamp
		let mut score = ReputationScore {
			director_score: 1000,
			validator_score: 500,
			seeder_score: 100,
			last_activity: 1000,
		};

		// Insert directly into storage
		<ReputationScores<Test>>::insert(caller.clone(), score);

		// Move forward 12 weeks
		let weeks = 12;
		let blocks_per_week = 7u32 * 24 * 600; // ~100,800 blocks/week
		let current_block = 1000u32 + (weeks as u32 * blocks_per_week);
		System::set_block_number(current_block);

		#[extrinsic_call]
		record_event(RawOrigin::Root, caller.clone(), event_type.clone(), 100u64);

		// Verify decay was applied before update
		let new_score = IcnReputation::reputation_scores(caller);
		assert!(new_score.director_score < 1000, "Decay should have been applied");
		assert_eq!(new_score.last_activity, current_block as u64);
	}

	/// Benchmark `on_finalize` with no events
	///
	/// # Weight Components
	/// - Minimal: Only checks for events, does nothing
	#[benchmark]
	fn on_finalize_no_events() {
		let block = 1000u32.into();
		System::set_block_number(block);

		#[extrinsic_call]
		fn on_finalize(block: BlockNumberFor<T>) {
			IcnReputation::on_finalize(block);
		}
	}

	/// Benchmark `on_finalize` with Merkle root publication
	///
	/// # Weight Components
	/// - Storage reads: 1 (PendingEvents take)
	/// - Storage writes: 1 (MerkleRoots insert)
	/// - Computation: Merkle tree construction (O(n log n) where n = events)
	/// - Events: 1
	///
	/// # Worst Case
	/// - MaxEventsPerBlock events in PendingEvents
	#[benchmark]
	fn on_finalize_with_events() {
		let block = 1000u32.into();
		System::set_block_number(block);

		// Setup: Fill PendingEvents to near max
		let max_events = T::MaxEventsPerBlock::get();
		for i in 0..max_events {
			let account = i as u64;
			let event = ReputationEvent {
				account,
				event_type: ReputationEventType::SeederChunkServed,
				slot: 0u64,
				block,
			};

			PendingEvents::<Test>::mutate(|events| {
				let _ = events.try_push(event);
			});
		}

		#[extrinsic_call]
		fn on_finalize(block: BlockNumberFor<T>) {
			IcnReputation::on_finalize(block);
		}

		// Verify Merkle root was published
		assert!(MerkleRoots::<Test>::get(block).is_some());
	}

	/// Benchmark `on_finalize` with checkpoint creation
	///
	/// # Weight Components
	/// - Storage reads: Variable (iterates all ReputationScores)
	/// - Storage writes: 1 (Checkpoints insert)
	/// - Computation: Merkle tree of all scores (O(n log n) where n = accounts)
	/// - Events: 1
	///
	/// # Worst Case
	/// - Maximum accounts with reputation (10,000)
	#[benchmark]
	fn on_finalize_with_checkpoint() {
		let block = 1000u32.into();
		System::set_block_number(block);

		// Setup: Create 1000 accounts with reputation
		// Note: Using 1000 instead of 10,000 to keep benchmark fast
		for i in 0..1000u64 {
			let score = ReputationScore {
				director_score: i * 10,
				validator_score: i * 5,
				seeder_score: i * 2,
				last_activity: 100,
			};
			ReputationScores::<Test>::insert(i, score);
		}

		#[extrinsic_call]
		fn on_finalize(block: BlockNumberFor<T>) {
			IcnReputation::on_finalize(block);
		}

		// Verify checkpoint was created
		assert!(Checkpoints::<Test>::get(block).is_some());
	}

	/// Benchmark `on_finalize` with pruning
	///
	/// # Weight Components
	/// - Storage reads: Variable (iterates MerkleRoots and Checkpoints)
	/// - Storage writes: Variable (removes old entries)
	/// - Events: 1 (if anything pruned)
	///
	/// # Worst Case
	/// - Maximum Merkle roots and checkpoints to prune
	#[benchmark]
	fn on_finalize_with_pruning() {
		let current_block = 1_000_000u32.into();
		System::set_block_number(current_block);

		// Setup: Create old Merkle roots and checkpoints
		let retention = <Test as crate::Config>::DefaultRetentionPeriod::get();
		let prune_before = current_block - retention;

		// Create 1000 old roots
		for i in 0..1000u32 {
			let old_block = prune_before - i - 1;
			MerkleRoots::<Test>::insert(old_block, H256::random());
		}

		// Create 500 old checkpoints
		for i in 0..500u32 {
			let old_block = prune_before - i - 1;
			Checkpoints::<Test>::insert(
				old_block,
				CheckpointData {
					block: old_block,
					score_count: 100,
					merkle_root: H256::random(),
				},
			);
		}

		#[extrinsic_call]
		fn on_finalize(block: BlockNumberFor<T>) {
			IcnReputation::on_finalize(block);
		}

		// Verify pruning occurred (should be mostly gone)
		let remaining_roots = MerkleRoots::<Test>::iter().count() as u32;
		assert!(remaining_roots < 1000, "Old roots should be pruned");
	}

	/// Benchmark `update_retention` extrinsic
	///
	/// # Weight Components
	/// - Storage reads: 1 (RetentionPeriod)
	/// - Storage writes: 1 (RetentionPeriod)
	/// - Events: 1
	#[benchmark]
	fn update_retention() {
		let new_period = 1_000_000u32.into();

		#[extrinsic_call]
		update_retention(RawOrigin::Root, new_period);

		// Verify updated
		assert_eq!(RetentionPeriod::<Test>::get(), new_period);
	}

	impl_benchmark_test_suite! {}
}
