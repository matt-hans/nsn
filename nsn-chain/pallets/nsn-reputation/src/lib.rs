// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! # NSN Reputation Pallet
//!
//! Verifiable reputation events with Merkle proofs and pruning for the Neural Sovereign Network.
//!
//! ## Overview
//!
//! This pallet implements:
//! - Reputation score tracking (director, validator, seeder)
//! - Weighted scoring: 50% director, 30% validator, 20% seeder
//! - Merkle tree proofs for reputation events
//! - Automatic pruning after retention period (governance-adjustable)
//! - Checkpoint system every 1000 blocks
//! - TPS optimization via aggregated events
//!
//! ## Interface
//!
//! ### Dispatchable Functions
//!
//! - `record_event`: Record a reputation event and update scores (root only)
//!
//! ## Storage
//!
//! - `ReputationScores`: Account → ReputationScore
//! - `PendingEvents`: BoundedVec of events for current block
//! - `MerkleRoots`: Block → Hash (Merkle root for that block's events)
//! - `Checkpoints`: Block → CheckpointData (every 1000 blocks)
//! - `RetentionPeriod`: Configurable retention period in blocks
//! - `AggregatedEvents`: Account → AggregatedReputation (off-chain batching)
//!
//! ## Weighted Scoring
//!
//! Total reputation = (director_score * 50 + validator_score * 30 + seeder_score * 20) / 100
//!
//! ## Event Deltas
//!
//! - DirectorSlotAccepted: +100 director
//! - DirectorSlotRejected: -200 director
//! - DirectorSlotMissed: -150 director
//! - ValidatorVoteCorrect: +5 validator
//! - ValidatorVoteIncorrect: -10 validator
//! - SeederChunkServed: +1 seeder
//! - PinningAuditPassed: +10 seeder
//! - PinningAuditFailed: -50 seeder
//!
//! ## Decay
//!
//! Inactive accounts decay at 5% per week (configurable).
//! Decay calculation: blocks / (7 * 24 * 600) = weeks inactive

#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;
pub use weights::WeightInfo;

mod types;
pub use types::{
	AggregatedEvent, AggregatedReputation, CheckpointData, ReputationEvent, ReputationEventType,
	ReputationScore,
};

#[cfg(test)]
mod mock;
#[cfg(test)]
mod tests;

#[cfg(feature = "runtime-benchmarks")]
mod benchmarking;

pub mod weights;

#[frame_support::pallet]
pub mod pallet {
	use super::*;
	use frame_support::pallet_prelude::*;
	use frame_support::traits::StorageVersion;
	use frame_system::pallet_prelude::*;
	use sp_runtime::traits::{Hash, SaturatedConversion, Zero};
	use sp_runtime::Saturating;
	use sp_std::vec::Vec;

	/// The in-code storage version.
	const STORAGE_VERSION: StorageVersion = StorageVersion::new(0);

	/// Pallet for NSN reputation tracking
	#[pallet::pallet]
	#[pallet::storage_version(STORAGE_VERSION)]
	pub struct Pallet<T>(_);

	/// Configuration trait for the NSN Reputation pallet
	#[pallet::config]
	pub trait Config: frame_system::Config {
		/// The overarching event type.
		#[allow(deprecated)]
		type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
		/// Maximum events per block (L0: bounded storage)
		///
		/// Prevents unbounded growth of PendingEvents and limits
		/// Merkle tree computation cost.
		#[pallet::constant]
		type MaxEventsPerBlock: Get<u32>;

		/// Default retention period in blocks (~6 months = 2,592,000 blocks)
		///
		/// Can be adjusted by governance via storage update.
		#[pallet::constant]
		type DefaultRetentionPeriod: Get<BlockNumberFor<Self>>;

		/// Checkpoint interval in blocks (default: 1000)
		///
		/// Every N blocks, a snapshot of all reputation scores is taken.
		#[pallet::constant]
		type CheckpointInterval: Get<BlockNumberFor<Self>>;

		/// Decay rate per week in percent (default: 5)
		///
		/// Applied to inactive accounts based on last_activity.
		#[pallet::constant]
		type DecayRatePerWeek: Get<u64>;

		/// Maximum accounts to include in checkpoint (L0: bounded iteration)
		///
		/// Prevents unbounded iteration in create_checkpoint().
		/// Realistic bound: < 10,000 accounts (MVP phase).
		#[pallet::constant]
		type MaxCheckpointAccounts: Get<u32>;

		/// Maximum Merkle roots/checkpoints to prune per block.
		///
		/// Bounds on_finalize pruning cost.
		#[pallet::constant]
		type MaxPrunePerBlock: Get<u32>;

		/// Weight information for extrinsics
		type WeightInfo: WeightInfo;
	}

	/// Reputation scores for each account
	///
	/// Maps an account to their three-component reputation score with
	/// weighted total and last activity timestamp for decay.
	///
	/// # Storage Key
	/// Blake2_128Concat(AccountId) - safe for user-controlled keys
	///
	/// # L2: MaxEncodedLen
	/// ReputationScore derives MaxEncodedLen for accurate weight calculation.
	#[pallet::storage]
	#[pallet::getter(fn reputation_scores)]
	pub type ReputationScores<T: Config> =
		StorageMap<_, Blake2_128Concat, T::AccountId, ReputationScore, ValueQuery>;

	/// Pending events for the current block
	///
	/// Accumulated during block processing, then finalized into Merkle root
	/// in on_finalize. Cleared after Merkle root computation.
	///
	/// # L0 Compliance
	/// BoundedVec with MaxEventsPerBlock ensures no unbounded storage growth.
	///
	/// # Behavior
	/// - Events accumulate during extrinsic execution
	/// - on_finalize() computes Merkle root and clears
	/// - If limit reached, new events are rejected with error
	#[pallet::storage]
	#[pallet::getter(fn pending_events)]
	pub type PendingEvents<T: Config> = StorageValue<
		_,
		BoundedVec<ReputationEvent<T::AccountId, BlockNumberFor<T>>, T::MaxEventsPerBlock>,
		ValueQuery,
	>;

	/// Merkle roots for each block's events
	///
	/// Stores the Merkle root hash of all reputation events recorded in each block.
	/// Used for off-chain proof verification without full chain sync.
	///
	/// # Storage Key
	/// Twox64Concat(BlockNumber) - fast sequential access
	///
	/// # L0 Compliance
	/// Pruned in on_finalize() beyond RetentionPeriod to prevent unbounded growth.
	#[pallet::storage]
	#[pallet::getter(fn merkle_roots)]
	pub type MerkleRoots<T: Config> =
		StorageMap<_, Twox64Concat, BlockNumberFor<T>, T::Hash, OptionQuery>;

	/// Checkpoints created at regular intervals
	///
	/// Every CheckpointInterval blocks, a snapshot of all reputation scores
	/// is taken with its own Merkle root. Enables efficient state recovery.
	///
	/// # Storage Key
	/// Twox64Concat(BlockNumber) - sequential checkpoint access
	///
	/// # L0 Compliance
	/// Pruned alongside MerkleRoots beyond RetentionPeriod.
	#[pallet::storage]
	#[pallet::getter(fn checkpoints)]
	pub type Checkpoints<T: Config> = StorageMap<
		_,
		Twox64Concat,
		BlockNumberFor<T>,
		CheckpointData<T::Hash, BlockNumberFor<T>>,
		OptionQuery,
	>;

	/// Retention period for Merkle roots and checkpoints
	///
	/// Configurable retention period in blocks. Data older than this
	/// is pruned to prevent unbounded storage growth.
	///
	/// # Default
	/// DefaultRetentionPeriod (2592000 blocks = ~6 months)
	///
	/// # Governance
	/// Can be updated via root origin (governance/pallets).
	#[pallet::storage]
	#[pallet::getter(fn retention_period)]
	pub type RetentionPeriod<T: Config> =
		StorageValue<_, BlockNumberFor<T>, ValueQuery, T::DefaultRetentionPeriod>;

	/// Aggregated reputation events for TPS optimization
	///
	/// Off-chain aggregators can batch multiple events for the same account
	/// into a single on-chain transaction. Reduces TPS load for high-activity accounts.
	///
	/// # Usage
	/// 1. Off-chain aggregator accumulates events for an account
	/// 2. Computes net deltas for each component
	/// 3. Submits single transaction with aggregated result
	///
	/// # L2: MaxEncodedLen
	/// AggregatedReputation derives MaxEncodedLen for accurate weight calculation.
	#[pallet::storage]
	#[pallet::getter(fn aggregated_events)]
	pub type AggregatedEvents<T: Config> =
		StorageMap<_, Blake2_128Concat, T::AccountId, AggregatedReputation, ValueQuery>;

	/// Events emitted by the pallet
	#[pallet::event]
	#[pallet::generate_deposit(pub(super) fn deposit_event)]
	pub enum Event<T: Config> {
		/// Reputation event recorded
		ReputationRecorded {
			account: T::AccountId,
			event_type: ReputationEventType,
			slot: u64,
		},
		/// Aggregated reputation events recorded
		AggregatedReputationRecorded {
			account: T::AccountId,
			net_director_delta: i64,
			net_validator_delta: i64,
			net_seeder_delta: i64,
			event_count: u32,
		},
		/// Merkle root published for block
		MerkleRootPublished { block: BlockNumberFor<T>, root: T::Hash, event_count: u32 },
		/// Checkpoint created
		CheckpointCreated { block: BlockNumberFor<T>, score_count: u32 },
		/// Checkpoint truncated (exceeded MaxCheckpointAccounts)
		CheckpointTruncated {
			block: BlockNumberFor<T>,
			total: u32,
			included: u32,
		},
		/// Old events pruned
		EventsPruned { before_block: BlockNumberFor<T>, count: u32 },
		/// Retention period updated
		RetentionPeriodUpdated { old_period: BlockNumberFor<T>, new_period: BlockNumberFor<T> },
	}

	/// Errors returned by the pallet
	#[pallet::error]
	pub enum Error<T> {
		/// Maximum events per block exceeded
		MaxEventsExceeded,
		/// Aggregated submission contained no events
		EmptyAggregation,
	}

	/// Hooks for block finalization
	#[pallet::hooks]
	impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
		/// Validate configuration constraints at compile time.
		fn integrity_test() {
			assert!(
				T::MaxEventsPerBlock::get() > 0,
				"MaxEventsPerBlock must be greater than 0"
			);
			assert!(
				T::CheckpointInterval::get() > Zero::zero(),
				"CheckpointInterval must be greater than 0"
			);
			assert!(
				T::DecayRatePerWeek::get() <= 100,
				"DecayRatePerWeek cannot exceed 100%"
			);
		}

		/// Block finalization hook
		///
		/// # Operations
		/// 1. Finalize Merkle root for pending events
		/// 2. Create checkpoint if at interval boundary
		/// 3. Prune old events beyond retention period
		fn on_finalize(block: BlockNumberFor<T>) {
			// Step 1: Finalize Merkle root for this block
			let events = PendingEvents::<T>::take();
			let event_count = events.len() as u32;

			if !events.is_empty() {
				let root = Self::compute_merkle_root(&events);
				MerkleRoots::<T>::insert(block, root);
				Self::deposit_event(Event::MerkleRootPublished { block, root, event_count });
			}

			// Step 2: Create checkpoint if at interval
			if block % T::CheckpointInterval::get() == Zero::zero() {
				Self::create_checkpoint(block);
			}

			// Step 3: Prune old events beyond retention period
			let retention = RetentionPeriod::<T>::get();
			let prune_before = block.saturating_sub(retention);
			Self::prune_old_events(prune_before);
		}
	}

	/// Extrinsic calls
	#[pallet::call]
	impl<T: Config> Pallet<T> {
		/// Record a reputation event
		///
		/// Updates the reputation score for an account based on the event type.
		/// Only callable by root origin (other pallets, governance).
		///
		/// # Arguments
		/// * `account` - Account to update reputation for
		/// * `event_type` - Type of event (determines delta)
		/// * `slot` - Slot number (for director events)
		///
		/// # Errors
		/// * `MaxEventsExceeded` - Too many events this block
		///
		/// # Events
		/// * `ReputationRecorded` - Event successfully recorded
		///
		/// # Weight
		/// Database reads: ReputationScores, PendingEvents
		/// Database writes: ReputationScores, PendingEvents
		#[pallet::call_index(0)]
		#[pallet::weight(T::WeightInfo::record_event())]
		pub fn record_event(
			origin: OriginFor<T>,
			account: T::AccountId,
			event_type: ReputationEventType,
			slot: u64,
		) -> DispatchResult {
			ensure_root(origin)?;

			Self::record_event_internal(&account, event_type, slot)
		}

		/// Record a batch of reputation events for a single account (root only).
		///
		/// Applies all deltas atomically and records each event into PendingEvents
		/// for inclusion in the Merkle root.
		#[pallet::call_index(1)]
		#[pallet::weight(T::WeightInfo::record_aggregated_events(events.len() as u32))]
		pub fn record_aggregated_events(
			origin: OriginFor<T>,
			account: T::AccountId,
			events: BoundedVec<AggregatedEvent, T::MaxEventsPerBlock>,
		) -> DispatchResult {
			ensure_root(origin)?;
			ensure!(!events.is_empty(), Error::<T>::EmptyAggregation);

			let current_block = <frame_system::Pallet<T>>::block_number();
			let current_block_u64 = current_block.saturated_into::<u64>();

			// Pre-compute net deltas
			let mut net_director_delta: i64 = 0;
			let mut net_validator_delta: i64 = 0;
			let mut net_seeder_delta: i64 = 0;

			for event in events.iter() {
				let delta = event.event_type.delta();
				if event.event_type.is_director_event() {
					net_director_delta = net_director_delta.saturating_add(delta);
				} else if event.event_type.is_validator_event() {
					net_validator_delta = net_validator_delta.saturating_add(delta);
				} else if event.event_type.is_seeder_event() {
					net_seeder_delta = net_seeder_delta.saturating_add(delta);
				}
			}

			// Add all events to pending events (L0: bounded)
			PendingEvents::<T>::try_mutate(|pending| -> DispatchResult {
				let new_len = pending.len().saturating_add(events.len());
				ensure!(
					new_len <= T::MaxEventsPerBlock::get() as usize,
					Error::<T>::MaxEventsExceeded
				);

				for event in events.iter() {
					let pending_event = ReputationEvent {
						account: account.clone(),
						event_type: event.event_type.clone(),
						slot: event.slot,
						block: current_block,
					};
					pending
						.try_push(pending_event)
						.map_err(|_| Error::<T>::MaxEventsExceeded)?;
				}

				Ok(())
			})?;

			// Apply aggregated deltas atomically
			ReputationScores::<T>::mutate(&account, |score| {
				score.apply_delta(net_director_delta, 0);
				score.apply_delta(net_validator_delta, 1);
				score.apply_delta(net_seeder_delta, 2);
				score.update_activity(current_block_u64);
			});

			AggregatedEvents::<T>::insert(
				&account,
				AggregatedReputation {
					net_director_delta,
					net_validator_delta,
					net_seeder_delta,
					event_count: events.len() as u32,
					last_aggregation_block: current_block_u64,
				},
			);

			Self::deposit_event(Event::AggregatedReputationRecorded {
				account,
				net_director_delta,
				net_validator_delta,
				net_seeder_delta,
				event_count: events.len() as u32,
			});

			Ok(())
		}

		/// Update retention period (root only)
		///
		/// # Arguments
		/// * `new_period` - New retention period in blocks
		///
		/// # Events
		/// * `RetentionPeriodUpdated` - Period successfully updated
		#[pallet::call_index(2)]
		#[pallet::weight(T::WeightInfo::update_retention())]
		pub fn update_retention(
			origin: OriginFor<T>,
			new_period: BlockNumberFor<T>,
		) -> DispatchResult {
			ensure_root(origin)?;

			let old_period = RetentionPeriod::<T>::get();
			RetentionPeriod::<T>::put(new_period);

			Self::deposit_event(Event::RetentionPeriodUpdated { old_period, new_period });
			Ok(())
		}
	}

	// Helper functions
	impl<T: Config> Pallet<T> {
		/// Record a task outcome as a reputation event.
		///
		/// Used by the task market via a loose-coupled trait implementation.
		pub fn record_task_outcome(account: &T::AccountId, success: bool) -> DispatchResult {
			let event_type = if success {
				ReputationEventType::TaskCompleted
			} else {
				ReputationEventType::TaskFailed
			};
			let current_block = <frame_system::Pallet<T>>::block_number();
			let slot = current_block.saturated_into::<u64>();
			Self::record_event_internal(account, event_type, slot)
		}

		fn record_event_internal(
			account: &T::AccountId,
			event_type: ReputationEventType,
			slot: u64,
		) -> DispatchResult {
			let current_block = <frame_system::Pallet<T>>::block_number();
			let current_block_u64 = current_block.saturated_into::<u64>();

			// Apply score change using helper
			let delta = event_type.delta();

			// Determine which component to update
			let component = if event_type.is_director_event() {
				0
			} else if event_type.is_validator_event() {
				1
			} else {
				2
			};

			// Add to pending events for Merkle tree (L0: bounded)
			let event = ReputationEvent {
				account: account.clone(),
				event_type: event_type.clone(),
				slot,
				block: current_block,
			};

			PendingEvents::<T>::try_mutate(|events| -> DispatchResult {
				events
					.try_push(event)
					.map_err(|_| Error::<T>::MaxEventsExceeded)?;
				Ok(())
			})?;

			// Update reputation score (L2: saturating arithmetic)
			ReputationScores::<T>::mutate(account, |score| {
				score.apply_delta(delta, component);
				score.update_activity(current_block_u64);
			});

			Self::deposit_event(Event::ReputationRecorded {
				account: account.clone(),
				event_type,
				slot,
			});

			Ok(())
		}
	}

	// Helper functions
	impl<T: Config> Pallet<T> {
		/// Compute Merkle root for a list of events
		///
		/// # Arguments
		/// * `events` - Events to compute root for
		///
		/// # Returns
		/// Merkle root hash, or default if empty
		///
		/// # Algorithm
		/// Binary Merkle tree:
		/// 1. Hash each event as a leaf
		/// 2. Pair up leaves and hash their concatenation
		/// 3. Repeat until single root hash remains
		///
		/// # L2: Saturating arithmetic
		/// No arithmetic operations, just hashing.
		pub fn compute_merkle_root(
			events: &[ReputationEvent<T::AccountId, BlockNumberFor<T>>],
		) -> T::Hash {
			if events.is_empty() {
				return T::Hash::default();
			}

			// Hash each event as a leaf
			let leaves: Vec<T::Hash> =
				events.iter().map(|e| T::Hashing::hash_of(e)).collect();

			Self::build_merkle_tree(&leaves)
		}

		/// Build Merkle tree from leaves
		///
		/// # Arguments
		/// * `leaves` - Leaf hashes to build tree from
		///
		/// # Returns
		/// Root hash of Merkle tree
		///
		/// # Algorithm
		/// Iteratively pair and hash until single root remains.
		/// Odd leaf hashes propagate to next level unchanged.
		fn build_merkle_tree(leaves: &[T::Hash]) -> T::Hash {
			if leaves.is_empty() {
				return T::Hash::default();
			}
			if leaves.len() == 1 {
				return leaves[0];
			}

			let mut current = leaves.to_vec();

			while current.len() > 1 {
				let mut next = Vec::new();

				for chunk in current.chunks(2) {
					let combined = if chunk.len() == 2 {
						T::Hashing::hash_of(&(chunk[0], chunk[1]))
					} else {
						// Odd leaf, propagate as-is
						chunk[0]
					};
					next.push(combined);
				}

				current = next;
			}

			current[0]
		}

		/// Verify a Merkle proof for a leaf.
		///
		/// # Arguments
		/// * `leaf` - Hash of the leaf
		/// * `leaf_index` - Index of the leaf in the original list
		/// * `leaf_count` - Total number of leaves in the original list
		/// * `proof` - Sibling hashes from leaf to root
		/// * `root` - Expected Merkle root
		pub fn verify_merkle_proof(
			leaf: T::Hash,
			leaf_index: u32,
			leaf_count: u32,
			proof: &[T::Hash],
			root: T::Hash,
		) -> bool {
			if leaf_count == 0 || leaf_index >= leaf_count {
				return false;
			}

			let mut hash = leaf;
			let mut index = leaf_index;
			let mut count = leaf_count;
			let mut proof_index = 0usize;

			while count > 1 {
				let is_last = index == count - 1;
				let has_sibling = !(is_last && (count % 2 == 1));

				if has_sibling {
					if proof_index >= proof.len() {
						return false;
					}

					let sibling = proof[proof_index];
					proof_index = proof_index.saturating_add(1);

					hash = if index % 2 == 0 {
						T::Hashing::hash_of(&(hash, sibling))
					} else {
						T::Hashing::hash_of(&(sibling, hash))
					};
				}

				index /= 2;
				count = (count + 1) / 2;
			}

			proof_index == proof.len() && hash == root
		}

		/// Create checkpoint for current block
		///
		/// # Arguments
		/// * `block` - Current block number
		///
		/// # Behavior
		/// 1. Iterate all reputation scores (bounded by MaxCheckpointAccounts)
		/// 2. Compute Merkle root of (account, score) pairs
		/// 3. Store checkpoint with count and root
		///
		/// # L0 Compliance
		/// Iteration bounded by MaxCheckpointAccounts constant.
		/// If account count exceeds limit, checkpoint is truncated and warning emitted.
		fn create_checkpoint(block: BlockNumberFor<T>) {
			// L0: Bounded iteration to prevent unbounded storage reads
			let max_accounts = T::MaxCheckpointAccounts::get() as usize;
			let mut truncated = false;

			let mut scores: Vec<(T::AccountId, ReputationScore)> = ReputationScores::<T>::iter()
				.take(max_accounts + 1) // Peek one past limit
				.collect();

			if scores.len() > max_accounts {
				truncated = true;
				scores.truncate(max_accounts);
			}

			let score_count = scores.len() as u32;
			let merkle_root = Self::compute_scores_merkle(&scores);

			let checkpoint = CheckpointData { block, score_count, merkle_root };

			Checkpoints::<T>::insert(block, checkpoint);

			// Emit warning if checkpoint was truncated
			if truncated {
				Self::deposit_event(Event::CheckpointTruncated {
					block,
					total: score_count.saturating_add(1),
					included: score_count,
				});
			} else {
				Self::deposit_event(Event::CheckpointCreated { block, score_count });
			}
		}

		/// Compute Merkle root of all reputation scores
		///
		/// # Arguments
		/// * `scores` - List of (account, score) pairs
		///
		/// # Returns
		/// Merkle root of all scores
		///
		/// # Purpose
		/// Provides snapshot verification for off-chain queries.
		fn compute_scores_merkle(scores: &[(T::AccountId, ReputationScore)]) -> T::Hash {
			if scores.is_empty() {
				return T::Hash::default();
			}

			// Hash each (account, score) pair as a leaf
			let leaves: Vec<T::Hash> =
				scores.iter().map(|(account, score)| T::Hashing::hash_of(&(account, score))).collect();

			Self::build_merkle_tree(&leaves)
		}

		/// Prune old Merkle roots and checkpoints
		///
		/// # Arguments
		/// * `before_block` - Remove all data before this block
		///
		/// # Behavior
		/// 1. Remove Merkle roots older than retention period
		/// 2. Remove checkpoints older than retention period
		/// 3. Emit event with count of pruned entries
		///
		/// # L0 Compliance
		/// Bounded iteration over MerkleRoots and Checkpoints.
		/// Pruning ensures storage doesn't grow unbounded.
		fn prune_old_events(before_block: BlockNumberFor<T>) {
			let mut pruned_roots = 0u32;
			let max_prune = T::MaxPrunePerBlock::get() as usize;

			// Prune Merkle roots (L0: bounded iteration)
			for (block, _) in MerkleRoots::<T>::iter().take(max_prune) {
				if block < before_block {
					MerkleRoots::<T>::remove(block);
					pruned_roots = pruned_roots.saturating_add(1);
				}
			}

			// Prune checkpoints (L0: bounded iteration)
			let mut pruned_checkpoints = 0u32;
			for (block, _) in Checkpoints::<T>::iter().take(max_prune) {
				if block < before_block {
					Checkpoints::<T>::remove(block);
					pruned_checkpoints = pruned_checkpoints.saturating_add(1);
				}
			}

			let total_pruned = pruned_roots.saturating_add(pruned_checkpoints);

			if total_pruned > 0 {
				Self::deposit_event(Event::EventsPruned { before_block, count: total_pruned });
			}
		}

		/// Apply decay to an account's reputation score
		///
		/// # Arguments
		/// * `account` - Account to apply decay to
		/// * `current_block` - Current block number
		///
		/// # Behavior
		/// If account has been inactive for > 1 week, apply decay rate.
		/// Decay is cumulative (5% per week).
		///
		/// # Public Interface
		/// Called by other pallets (e.g., director election) before queries.
		pub fn apply_decay(account: &T::AccountId, current_block: u64) {
			let decay_rate = T::DecayRatePerWeek::get();

			ReputationScores::<T>::mutate(account, |score| {
				score.apply_decay(current_block, decay_rate);
			});
		}

		/// Get total reputation score for an account
		///
		/// # Arguments
		/// * `account` - Account to query
		///
		/// # Returns
		/// Weighted total score (50% director, 30% validator, 20% seeder)
		///
		/// # Note
		/// This does NOT apply decay. Call `apply_decay` first if needed.
		pub fn get_reputation_total(account: &T::AccountId) -> u64 {
			Self::reputation_scores(account).total()
		}

		/// Get reputation score for an account
		///
		/// # Arguments
		/// * `account` - Account to query
		///
		/// # Returns
		/// Full ReputationScore struct
		pub fn get_reputation(account: &T::AccountId) -> ReputationScore {
			Self::reputation_scores(account)
		}
	}
}
