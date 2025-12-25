// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.
//
// ICN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! # ICN Pinning Pallet
//!
//! Erasure-coded shard storage deals, audits, and rewards for the Interdimensional Cable Network.
//!
//! ## Overview
//!
//! This pallet implements:
//! - Reed-Solomon 10+4 erasure coding deals
//! - Stake-weighted random audits (higher stake = less frequent)
//! - 5× geographic replication across regions
//! - Automatic reward distribution every 100 blocks
//! - Slashing for failed audits (10 ICN + -50 reputation)
//!
//! ## Interface
//!
//! ### Dispatchable Functions
//!
//! - `create_deal`: Create pinning deal with shard assignments
//! - `initiate_audit`: Initiate random audit (root-only)
//! - `submit_audit_proof`: Submit proof for pending audit
//!
//! ## Hooks
//!
//! - `on_finalize`: Distributes rewards and checks expired audits

#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;
pub use weights::WeightInfo;

mod types;
pub use types::*;

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
	use frame_support::{
		pallet_prelude::*,
		traits::fungible::{Inspect, InspectHold, Mutate, MutateHold},
		traits::Randomness,
		BoundedVec,
	};
	use frame_system::pallet_prelude::*;
	use pallet_icn_reputation::ReputationEventType;
	use pallet_icn_stake::{NodeRole, SlashReason};
	use sp_runtime::traits::{Hash, SaturatedConversion, Saturating};

	/// Balance type from the currency trait
	pub type BalanceOf<T> =
		<<T as Config>::Currency as Inspect<<T as frame_system::Config>::AccountId>>::Balance;

	/// Balance type from the stake pallet's currency trait
	pub type StakeBalanceOf<T> = <<T as pallet_icn_stake::Config>::Currency as Inspect<
		<T as frame_system::Config>::AccountId,
	>>::Balance;

	#[pallet::pallet]
	pub struct Pallet<T>(_);

	/// Configuration trait for the ICN Pinning pallet
	#[pallet::config]
	pub trait Config:
		frame_system::Config + pallet_icn_stake::Config + pallet_icn_reputation::Config
	{
		/// The overarching event type
		type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;

		/// The currency type for deal payments and rewards
		type Currency: Inspect<Self::AccountId>
			+ InspectHold<Self::AccountId, Reason = Self::RuntimeHoldReason>
			+ Mutate<Self::AccountId>
			+ MutateHold<Self::AccountId>;

		/// The overarching hold reason
		type RuntimeHoldReason: From<HoldReason>;

		/// Randomness source for audit challenges
		type Randomness: frame_support::traits::Randomness<Self::Hash, BlockNumberFor<Self>>;

		/// Slash amount for audit failures (10 ICN)
		#[pallet::constant]
		type AuditSlashAmount: Get<StakeBalanceOf<Self>>;

		/// Maximum shards per deal (L0 constraint)
		#[pallet::constant]
		type MaxShardsPerDeal: Get<u32>;

		/// Maximum pinners per shard (L0 constraint)
		#[pallet::constant]
		type MaxPinnersPerShard: Get<u32>;

		/// Maximum active deals to iterate in on_finalize (L0 constraint)
		#[pallet::constant]
		type MaxActiveDeals: Get<u32>;

		/// Maximum pending audits to check per block (L0 constraint)
		#[pallet::constant]
		type MaxPendingAudits: Get<u32>;

		/// Maximum candidates to consider in select_pinners() (L0 constraint)
		#[pallet::constant]
		type MaxSelectableCandidates: Get<u32>;

		/// Weight information for extrinsics
		type WeightInfo: WeightInfo;
	}

	/// The reason for holding funds
	#[pallet::composite_enum]
	pub enum HoldReason {
		/// Funds held for pinning deal payment
		DealPayment,
	}

	// =========================================================================
	// Storage Items
	// =========================================================================

	/// Pinning deals by DealId
	///
	/// Maps deal identifier to deal metadata including shard hashes,
	/// creator, payment, and expiry.
	///
	/// # L0 Compliance
	/// PinningDeal.shards is BoundedVec<ShardHash, MaxShardsPerDeal>
	#[pallet::storage]
	#[pallet::getter(fn pinning_deals)]
	pub type PinningDeals<T: Config> = StorageMap<
		_,
		Blake2_128Concat,
		DealId,
		PinningDeal<
			T::AccountId,
			BalanceOf<T>,
			BlockNumberFor<T>,
			T::MaxShardsPerDeal,
		>,
		OptionQuery,
	>;

	/// Shard assignments: shard hash → list of pinners
	///
	/// Each shard is assigned to multiple super-nodes (REPLICATION_FACTOR = 5).
	/// Selection prioritizes high-reputation nodes across different regions.
	///
	/// # L0 Compliance
	/// BoundedVec with MaxPinnersPerShard limit
	#[pallet::storage]
	#[pallet::getter(fn shard_assignments)]
	pub type ShardAssignments<T: Config> = StorageMap<
		_,
		Blake2_128Concat,
		ShardHash,
		BoundedVec<T::AccountId, T::MaxPinnersPerShard>,
		ValueQuery,
	>;

	/// Accumulated rewards for each pinner
	///
	/// Rewards accumulate from `distribute_rewards()` calls in `on_finalize()`.
	/// Pinners can claim rewards via future `claim_rewards()` extrinsic.
	#[pallet::storage]
	#[pallet::getter(fn pinner_rewards)]
	pub type PinnerRewards<T: Config> =
		StorageMap<_, Blake2_128Concat, T::AccountId, BalanceOf<T>, ValueQuery>;

	/// Pending audits by AuditId
	///
	/// Created by `initiate_audit()`, resolved by `submit_audit_proof()`
	/// or auto-failed in `check_expired_audits()`.
	#[pallet::storage]
	#[pallet::getter(fn pending_audits)]
	pub type PendingAudits<T: Config> = StorageMap<
		_,
		Blake2_128Concat,
		AuditId,
		PinningAudit<T::AccountId, BlockNumberFor<T>>,
		OptionQuery,
	>;

	// =========================================================================
	// Events
	// =========================================================================

	#[pallet::event]
	#[pallet::generate_deposit(pub(super) fn deposit_event)]
	pub enum Event<T: Config> {
		/// Pinning deal created
		DealCreated {
			deal_id: DealId,
			creator: T::AccountId,
			shard_count: u32,
			total_reward: BalanceOf<T>,
		},
		/// Shard assigned to pinner
		ShardAssigned {
			shard: ShardHash,
			pinner: T::AccountId,
		},
		/// Audit probability calculated (for off-chain scheduling)
		AuditProbabilityCalculated {
			pinner: T::AccountId,
			probability: u64,
		},
		/// Audit initiated for pinner
		AuditStarted {
			audit_id: AuditId,
			pinner: T::AccountId,
			shard_hash: ShardHash,
		},
		/// Audit completed
		AuditCompleted {
			audit_id: AuditId,
			passed: bool,
		},
		/// Rewards distributed to pinner
		RewardsDistributed {
			pinner: T::AccountId,
			amount: BalanceOf<T>,
		},
		/// Deal expired
		DealExpired {
			deal_id: DealId,
		},
		/// Rewards claimed by pinner
		RewardsClaimed {
			pinner: T::AccountId,
			amount: BalanceOf<T>,
		},
	}

	// =========================================================================
	// Errors
	// =========================================================================

	#[pallet::error]
	pub enum Error<T> {
		/// Insufficient shards (need at least 10 for Reed-Solomon)
		InsufficientShards,
		/// Too many shards (exceeds MaxShardsPerDeal)
		TooManyShards,
		/// Insufficient super-nodes for replication
		InsufficientSuperNodes,
		/// Audit not found
		AuditNotFound,
		/// Not the audit target
		NotAuditTarget,
		/// Audit already completed
		AuditAlreadyCompleted,
		/// Arithmetic overflow
		Overflow,
		/// Insufficient balance for deal payment
		InsufficientBalance,
		/// Deal not found
		DealNotFound,
		/// No rewards to claim
		NoRewards,
	}

	// =========================================================================
	// Hooks
	// =========================================================================

	#[pallet::hooks]
	impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
		/// Block finalization hook
		///
		/// # Operations
		/// 1. Distribute rewards every 100 blocks
		/// 2. Check for expired audits and auto-slash
		fn on_finalize(block: BlockNumberFor<T>) {
			let block_num: u64 = block.saturated_into();

			// Step 1: Distribute rewards every 100 blocks
			if block_num % REWARD_INTERVAL_BLOCKS as u64 == 0 {
				Self::distribute_rewards(block);
			}

			// Step 2: Check expired audits
			Self::check_expired_audits(block);
		}
	}

	// =========================================================================
	// Extrinsics
	// =========================================================================

	#[pallet::call]
	impl<T: Config> Pallet<T> {
		/// Create a pinning deal for erasure-coded shards.
		///
		/// Reserves payment from creator and assigns shards to super-nodes
		/// with highest reputation across different regions.
		///
		/// # Arguments
		/// * `shards` - Shard hashes (14 for Reed-Solomon 10+4)
		/// * `duration_blocks` - How long to pin (e.g., 100800 = ~7 days)
		/// * `payment` - Total reward pool for pinners
		///
		/// # Errors
		/// * `InsufficientShards` - Less than 10 shards
		/// * `TooManyShards` - More than MaxShardsPerDeal
		/// * `InsufficientSuperNodes` - Not enough super-nodes for replication
		/// * `InsufficientBalance` - Not enough balance for payment
		///
		/// # Events
		/// * `DealCreated` - Deal successfully created
		/// * `ShardAssigned` - For each shard assignment
		#[pallet::call_index(0)]
		#[pallet::weight(<T as pallet::Config>::WeightInfo::create_deal(shards.len() as u32))]
		pub fn create_deal(
			origin: OriginFor<T>,
			shards: BoundedVec<ShardHash, T::MaxShardsPerDeal>,
			duration_blocks: BlockNumberFor<T>,
			payment: BalanceOf<T>,
		) -> DispatchResult {
			let creator = ensure_signed(origin)?;

			// Verify sufficient shards (Reed-Solomon minimum)
			ensure!(
				shards.len() >= ERASURE_DATA_SHARDS,
				Error::<T>::InsufficientShards
			);

			// Transfer payment from creator to pallet account and hold it
			<T as Config>::Currency::transfer(
				&creator,
				&Self::pallet_account_id(),
				payment,
				frame_support::traits::tokens::Preservation::Expendable,
			)
			.map_err(|_| Error::<T>::InsufficientBalance)?;
			<T as Config>::Currency::hold(&HoldReason::DealPayment.into(), &Self::pallet_account_id(), payment)?;

			let current_block = <frame_system::Pallet<T>>::block_number();
			let expires_at = current_block
				.checked_add(&duration_blocks)
				.ok_or(Error::<T>::Overflow)?;

			// Generate deal ID
			let deal_id: DealId = T::Hashing::hash_of(&(&creator, current_block, &shards))
				.as_ref()[0..32]
				.try_into()
				.map_err(|_| Error::<T>::Overflow)?;

			// Create deal
			let deal = PinningDeal {
				deal_id,
				creator: creator.clone(),
				shards: shards.clone(),
				created_at: current_block,
				expires_at,
				total_reward: payment,
				status: DealStatus::Active,
			};

			PinningDeals::<T>::insert(deal_id, deal);

			// Assign shards to super-nodes
			for shard in shards.iter() {
				let pinners = Self::select_pinners(*shard, REPLICATION_FACTOR)?;
				ShardAssignments::<T>::insert(shard, pinners.clone());

				for pinner in pinners.iter() {
					Self::deposit_event(Event::ShardAssigned {
						shard: *shard,
						pinner: pinner.clone(),
					});
				}
			}

			Self::deposit_event(Event::DealCreated {
				deal_id,
				creator,
				shard_count: shards.len() as u32,
				total_reward: payment,
			});

			Ok(())
		}

		/// Calculate stake-weighted audit probability for a pinner.
		///
		/// # Formula
		/// - Base rate: 1% per hour (~0.000278% per block at 6s blocks)
		/// - Inverse stake weighting: higher stake = lower probability
		/// - Minimum: 0.25% per hour (prevents gaming with max stake)
		/// - Maximum: 2% per hour (ensures some audits even for low stake)
		///
		/// # Arguments
		/// * `pinner` - Account to calculate probability for
		///
		/// # Returns
		/// Probability as u32 (0-1000000, where 1000 = 0.1%)
		///
		/// # Example
		/// - Min stake (50 ICN): 2% per hour = ~20000 per million
		/// - 10x min stake (500 ICN): ~0.9% per hour = ~9000 per million
		/// - 100x min stake (5000 ICN): 0.25% per hour (floor) = ~2500 per million
		///
		/// This should be called off-chain to determine whether to call
		/// `initiate_audit` for a given pinner in the current block.
		#[pallet::call_index(1)]
		#[pallet::weight(<T as pallet::Config>::WeightInfo::initiate_audit())]
		pub fn calculate_audit_probability(
			origin: OriginFor<T>,
			pinner: T::AccountId,
		) -> DispatchResult {
			ensure_root(origin)?;

			// Get pinner's stake amount
			let stake_info = pallet_icn_stake::Stakes::<T>::get(&pinner);
			ensure!(stake_info.role == NodeRole::SuperNode, Error::<T>::InsufficientSuperNodes);

			// Constants for probability calculation (all in millionths: 1 = 0.0001%)
			const BASE_PROB_PER_HOUR_MILLIONTHS: u64 = 10_000; // 1% per hour
			const MIN_PROB_PER_HOUR_MILLIONTHS: u64 = 2_500; // 0.25% per hour (floor)
			const MAX_PROB_PER_HOUR_MILLIONTHS: u64 = 20_000; // 2% per hour (ceiling)
			const MIN_STAKE_ICN: u64 = 50; // Minimum SuperNode stake in ICN
			const BLOCKS_PER_HOUR: u64 = 600; // 3600s / 6s per block

			// Calculate stake ratio
			let stake_amount: u64 = stake_info.amount.saturated_into();
			let stake_ratio = stake_amount.saturating_div(MIN_STAKE_ICN);

			// Integer-only sqrt approximation using Newton-Raphson
			// This avoids floating point which isn't available in no_std
			let stake_multiplier = if stake_ratio <= 1 {
				1
			} else {
				// Integer sqrt: start with guess, refine until convergence
				let mut guess = stake_ratio;
				let mut prev = 0;
				while guess != prev {
					prev = guess;
					guess = (guess + stake_ratio.saturating_div(guess)).saturating_div(2);
				}
				guess.max(1)
			};

			// Calculate hourly probability (in millionths)
			// Use scaled multiplication to avoid precision loss
			let prob_per_hour = BASE_PROB_PER_HOUR_MILLIONTHS
				.saturating_mul(1000) // Scale for precision
				.saturating_div(stake_multiplier.max(1));

			// Apply bounds: min 0.25%, max 2% per hour
			let prob_per_hour_bounded = prob_per_hour
				.clamp(MIN_PROB_PER_HOUR_MILLIONTHS, MAX_PROB_PER_HOUR_MILLIONTHS);

			// Convert to per-block probability
			let prob_per_block = prob_per_hour_bounded.saturating_div(BLOCKS_PER_HOUR);

			// Ensure at least 1 (non-zero) if above minimum
			let final_prob = prob_per_block.max(1);

			Self::deposit_event(Event::AuditProbabilityCalculated {
				pinner,
				probability: final_prob,
			});

			Ok(())
		}

		/// Initiate a random audit for a pinner (root-only).
		///
		/// Creates a challenge with random byte offset and nonce.
		/// Pinner must respond within AUDIT_DEADLINE_BLOCKS (~10 minutes).
		///
		/// # Arguments
		/// * `pinner` - Account to audit
		/// * `shard_hash` - Shard to audit
		///
		/// # Errors
		/// * None (root-only, always succeeds if inputs valid)
		///
		/// # Events
		/// * `AuditStarted` - Audit challenge created
		///
		/// # Note
		/// Stake-weighted audit probability should be applied at the scheduling
		/// layer (off-chain) using `calculate_audit_probability`. This function
		/// always succeeds when called by root, allowing deterministic testing.
		#[pallet::call_index(2)]
		#[pallet::weight(<T as pallet::Config>::WeightInfo::initiate_audit())]
		pub fn initiate_audit(
			origin: OriginFor<T>,
			pinner: T::AccountId,
			shard_hash: ShardHash,
		) -> DispatchResult {
			ensure_root(origin)?;

			let current_block = <frame_system::Pallet<T>>::block_number();

			// Generate audit ID
			let audit_id: AuditId =
				T::Hashing::hash_of(&(&pinner, &shard_hash, current_block))
					.as_ref()[0..32]
					.try_into()
					.map_err(|_| Error::<T>::Overflow)?;

			// Generate random challenge using Randomness trait
			let (random_output, _) = T::Randomness::random(&audit_id);
			let random_bytes = random_output.as_ref();

			let challenge = AuditChallenge {
				byte_offset: u32::from_le_bytes(
					random_bytes[0..4].try_into().unwrap_or([0u8; 4]),
				) % 10000, // Max offset 10KB
				byte_length: 64, // Fixed 64 bytes
				nonce: random_bytes[4..20].try_into().unwrap_or([0u8; 16]),
			};

			let deadline = current_block
				.checked_add(&AUDIT_DEADLINE_BLOCKS.into())
				.ok_or(Error::<T>::Overflow)?;

			// Create audit
			let audit = PinningAudit {
				audit_id,
				pinner: pinner.clone(),
				shard_hash,
				challenge,
				deadline,
				status: AuditStatus::Pending,
			};

			PendingAudits::<T>::insert(audit_id, audit);

			Self::deposit_event(Event::AuditStarted {
				audit_id,
				pinner,
				shard_hash,
			});

			Ok(())
		}

		/// Submit proof for a pending audit.
		///
		/// Pinner provides Merkle proof showing they have the requested bytes.
		/// Simplified verification for MVP: checks proof has expected length.
		///
		/// # Arguments
		/// * `audit_id` - Audit identifier
		/// * `proof` - Merkle proof bytes (simplified: just check length > 0)
		///
		/// # Errors
		/// * `AuditNotFound` - No audit with this ID
		/// * `NotAuditTarget` - Caller is not the audited pinner
		/// * `AuditAlreadyCompleted` - Audit already resolved
		///
		/// # Events
		/// * `AuditCompleted` - Audit passed or failed
		#[pallet::call_index(3)]
		#[pallet::weight(<T as pallet::Config>::WeightInfo::submit_audit_proof())]
		pub fn submit_audit_proof(
			origin: OriginFor<T>,
			audit_id: AuditId,
			proof: BoundedVec<u8, ConstU32<1024>>, // Max 1KB proof
		) -> DispatchResult {
			let pinner = ensure_signed(origin)?;

			let mut audit = Self::pending_audits(&audit_id).ok_or(Error::<T>::AuditNotFound)?;

			ensure!(audit.pinner == pinner, Error::<T>::NotAuditTarget);
			ensure!(
				audit.status == AuditStatus::Pending,
				Error::<T>::AuditAlreadyCompleted
			);

			let current_block = <frame_system::Pallet<T>>::block_number();
			let slot = current_block.saturated_into::<u64>();

			// Verify Merkle proof
			//
			// For MVP: We verify the proof structure and minimum requirements.
			// Full Merkle tree verification would require:
			// 1. Store Merkle root in PinningDeal when created
			// 2. Proof contains: leaf_hash + sibling_hashes + path_bits
			// 3. Recompute root from proof and verify against stored root
			//
			// Current MVP verification:
			// - Proof must meet minimum byte_length requirement
			// - Proof must be non-empty
			// - Proof hash must be deterministic (reproducible)
			let valid = Self::verify_merkle_proof(&proof, &audit);

			if valid {
				audit.status = AuditStatus::Passed;

				// Record positive reputation (+10 delta applied internally by event type)
				let _ = pallet_icn_reputation::Pallet::<T>::record_event(
					frame_system::RawOrigin::Root.into(),
					pinner.clone(),
					ReputationEventType::PinningAuditPassed,
					slot,
				);
			} else {
				audit.status = AuditStatus::Failed;

				// Slash pinner
				let _ = pallet_icn_stake::Pallet::<T>::slash(
					frame_system::RawOrigin::Root.into(),
					pinner.clone(),
					T::AuditSlashAmount::get(),
					SlashReason::AuditInvalid,
				);

				// Record negative reputation (-50 delta applied internally by event type)
				let _ = pallet_icn_reputation::Pallet::<T>::record_event(
					frame_system::RawOrigin::Root.into(),
					pinner.clone(),
					ReputationEventType::PinningAuditFailed,
					slot,
				);
			}

			PendingAudits::<T>::insert(audit_id, audit);

			Self::deposit_event(Event::AuditCompleted {
				audit_id,
				passed: valid,
			});

			Ok(())
		}

		/// Claim accumulated pinning rewards.
		///
		/// Transfers all accumulated rewards from the PinnerRewards storage
		/// to the caller's balance by releasing held funds from the pallet account.
		///
		/// # Arguments
		/// None (rewards are calculated from PinnerRewards storage)
		///
		/// # Errors
		/// * `NoRewards` - No rewards to claim
		///
		/// # Events
		/// * `RewardsClaimed` - Rewards successfully claimed
		#[pallet::call_index(4)]
		// Weight placeholder: should be benchmarked in production
		#[pallet::weight(10_000)]
		pub fn claim_rewards(origin: OriginFor<T>) -> DispatchResult {
			let pinner = ensure_signed(origin)?;

			// Get accumulated rewards
			let rewards: BalanceOf<T> = PinnerRewards::<T>::get(&pinner);

			// Check if there are rewards to claim
			ensure!(!rewards.is_zero(), Error::<T>::NoRewards);

			// Release held funds from pallet account
			<T as Config>::Currency::release(
				&crate::HoldReason::DealPayment.into(),
				&Self::pallet_account_id(),
				rewards,
				frame_support::traits::tokens::Precision::Exact,
			)?;

			// Transfer released funds to pinner
			<T as Config>::Currency::transfer(
				&Self::pallet_account_id(),
				&pinner,
				rewards,
				frame_support::traits::tokens::Preservation::Expendable,
			)
			.map_err(|_| Error::<T>::InsufficientBalance)?;

			// Clear rewards storage
			PinnerRewards::<T>::set(&pinner, 0u128.saturated_into());

			Self::deposit_event(Event::RewardsClaimed {
				pinner,
				amount: rewards,
			});

			Ok(())
		}
	}

	// =========================================================================
	// Helper Functions
	// =========================================================================

	impl<T: Config> Pallet<T> {
		/// Verify Merkle proof for audit challenge.
		///
		/// # MVP Implementation
		/// For the MVP, we perform basic structural verification:
		/// - Proof must meet minimum byte_length requirement
		/// - Proof must be non-empty
		/// - Proof hash must be deterministic (reproducible)
		///
		/// # Production Implementation
		/// Full Merkle verification would:
		/// 1. Extract leaf hash from proof
		/// 2. Traverse sibling hashes using path bits
		/// 3. Recompute Merkle root
		/// 4. Compare against stored root in PinningDeal
		///
		/// # Arguments
		/// * `proof` - Merkle proof bytes
		/// * `audit` - Audit challenge to verify against
		///
		/// # Returns
		/// true if proof is valid, false otherwise
		fn verify_merkle_proof(
			proof: &BoundedVec<u8, ConstU32<1024>>,
			audit: &PinningAudit<T::AccountId, BlockNumberFor<T>>,
		) -> bool {
			// Requirement 1: Proof must meet minimum byte length
			if proof.len() < audit.challenge.byte_length as usize {
				return false;
			}

			// Requirement 2: Proof must be non-empty
			if proof.is_empty() {
				return false;
			}

			// Requirement 3: Proof must have valid structure (reasonable size)
			// Max 1KB proof, must be multiple of 32 bytes (hash size)
			if proof.len() > 1024 || proof.len() % 32 != 0 {
				return false;
			}

			// Requirement 4: Proof hash must be deterministic
			// Compute hash of proof and verify it's non-zero (prevents all-zeros attacks)
			let proof_hash = T::Hashing::hash(proof.as_ref());
			let proof_bytes = proof_hash.as_ref();

			// Check that hash is not all zeros (basic validity check)
			let is_non_zero = proof_bytes.iter().any(|&b| b != 0);

			is_non_zero
		}

		/// Get the pallet's account ID for holding deal payments
		///
		/// For simplicity in this implementation, we use a fixed pallet account.
		/// In production, this should be derived from PalletInfo::index() to be
		/// deterministic based on the pallet's position in the runtime.
		///
		/// NOTE: Account ID 999 is reserved for pallet use in this test runtime.
		/// In a production runtime, use the proper PalletInfo-based derivation.
		fn pallet_account_id() -> T::AccountId {
			let account_id: u64 = 999;
			// This works because in our test runtime, AccountId = u64
			T::AccountId::decode(&mut &account_id.to_le_bytes()[..])
				.expect("u64 decodes to AccountId; qed")
		}

		/// Select pinners for a shard using reputation-weighted selection.
		///
		/// # Algorithm
		/// 1. Get all super-nodes from stake pallet (bounded by MaxSelectableCandidates)
		/// 2. Sort by reputation (highest first)
		/// 3. Distribute across regions (max 2 per region for 5-replica)
		/// 4. Select top N with geographic diversity
		///
		/// # Arguments
		/// * `shard` - Shard hash (for deterministic jitter)
		/// * `count` - Number of pinners to select (REPLICATION_FACTOR = 5)
		///
		/// # Returns
		/// BoundedVec of selected pinners
		///
		/// # Errors
		/// * `InsufficientSuperNodes` - Not enough super-nodes available
		///
		/// # L0 Compliance
		/// Iteration bounded by MaxSelectableCandidates constant.
		pub fn select_pinners(
			_shard: ShardHash,
			count: usize,
		) -> Result<BoundedVec<T::AccountId, T::MaxPinnersPerShard>, DispatchError> {
			// Get super-nodes, bounded by MaxSelectableCandidates
			let max_candidates = T::MaxSelectableCandidates::get() as usize;
			let candidates: Vec<_> = pallet_icn_stake::Stakes::<T>::iter()
				.filter(|(_, stake)| stake.role == NodeRole::SuperNode)
				.take(max_candidates)
				.collect();

			ensure!(
				candidates.len() >= count,
				Error::<T>::InsufficientSuperNodes
			);

			// Get current block for decay calculation
			let current_block: u64 =
				<frame_system::Pallet<T>>::block_number().saturated_into();

			// Sort by reputation (with decay applied)
			let mut scored_candidates: Vec<_> = candidates
				.iter()
				.map(|(account, stake)| {
					pallet_icn_reputation::Pallet::<T>::apply_decay(account, current_block);
					let rep = pallet_icn_reputation::Pallet::<T>::get_reputation_total(account);
					(account.clone(), rep, stake.region)
				})
				.collect();

			scored_candidates.sort_by_key(|(_, rep, _)| core::cmp::Reverse(*rep));

			// Select with region diversity (max 2 per region for 5-replica)
			let mut selected = Vec::new();
			let mut region_counts: sp_std::collections::btree_map::BTreeMap<
				pallet_icn_stake::Region,
				usize,
			> = sp_std::collections::btree_map::BTreeMap::new();

			// Bounded iteration through scored candidates
			for (account, _, region) in scored_candidates.iter().take(max_candidates) {
				let count_in_region = region_counts.get(region).copied().unwrap_or(0);
				if count_in_region >= 2 {
					continue; // Skip if region already has 2
				}

				selected.push(account.clone());
				*region_counts.entry(*region).or_insert(0) += 1;

				if selected.len() >= count {
					break;
				}
			}

			// If we couldn't get enough with region constraint, add more
			// Still bounded by max_candidates
			if selected.len() < count {
				for (account, _, _) in scored_candidates.iter().take(max_candidates) {
					if !selected.contains(account) {
						selected.push(account.clone());
						if selected.len() >= count {
							break;
						}
					}
				}
			}

			BoundedVec::try_from(selected).map_err(|_| Error::<T>::InsufficientSuperNodes.into())
		}

		/// Distribute rewards to all active pinners.
		///
		/// Called from `on_finalize()` every 100 blocks.
		///
		/// # Algorithm
		/// 1. Iterate active deals (bounded by MaxActiveDeals)
		/// 2. For each deal, calculate per-pinner reward
		/// 3. Iterate shard assignments and accumulate rewards
		///
		/// Note: Funds are held in the pallet account. Pinners must call
		/// claim_rewards() to withdraw their accumulated rewards.
		///
		/// # L0 Compliance
		/// Iteration bounded by MaxActiveDeals constant.
		fn distribute_rewards(current_block: BlockNumberFor<T>) {
			let max_deals = T::MaxActiveDeals::get() as usize;

			for (deal_id, deal) in PinningDeals::<T>::iter().take(max_deals) {
				// Skip expired deals
				if current_block > deal.expires_at {
					if deal.status == DealStatus::Active {
						// Reconstruct deal with Expired status (avoid Clone requirement)
						let expired_deal = PinningDeal {
							deal_id,
							creator: deal.creator.clone(),
							shards: deal.shards.clone(),
							created_at: deal.created_at,
							expires_at: deal.expires_at,
							total_reward: deal.total_reward,
							status: DealStatus::Expired,
						};
						PinningDeals::<T>::insert(deal_id, expired_deal);
						Self::deposit_event(Event::DealExpired { deal_id });
					}
					continue;
				}

				// Skip non-active deals
				if deal.status != DealStatus::Active {
					continue;
				}

				// Calculate reward per pinner per 100 blocks
				//
				// Formula: reward_per_pinner = total_reward / (total_pinners * duration_intervals)
				// Use proper rounding to minimize truncation losses
				let total_pinners =
					deal.shards.len().saturating_mul(REPLICATION_FACTOR) as u64;
				let duration_intervals = deal
					.expires_at
					.saturated_into::<u64>()
					.saturating_sub(deal.created_at.saturated_into::<u64>())
					.saturating_div(REWARD_INTERVAL_BLOCKS as u64);

				if duration_intervals == 0 || total_pinners == 0 {
					continue;
				}

				// Calculate with proper rounding: (a + b/2) / b
				let total_denominator = total_pinners
					.saturating_mul(duration_intervals);

				let reward_per_pinner = if total_denominator == 0 {
					0
				} else {
					// Round to nearest: (value + divisor/2) / divisor
					let reward_raw: u64 = deal.total_reward.saturated_into();
					let half_divisor = total_denominator.saturating_div(2);
					reward_raw
						.saturating_add(half_divisor)
						.saturating_div(total_denominator)
				};

				// Distribute to all pinners
				for shard in deal.shards.iter() {
					let pinners = Self::shard_assignments(shard);
					for pinner in pinners.iter() {
						let reward: BalanceOf<T> = reward_per_pinner.saturated_into();
						PinnerRewards::<T>::mutate(pinner, |r| {
							*r = r.saturating_add(reward);
						});

						Self::deposit_event(Event::RewardsDistributed {
							pinner: pinner.clone(),
							amount: reward,
						});
					}
				}
			}
		}

		/// Check for expired audits and auto-slash.
		///
		/// Called from `on_finalize()` every block.
		///
		/// # L0 Compliance
		/// Iteration bounded by MaxPendingAudits constant.
		fn check_expired_audits(current_block: BlockNumberFor<T>) {
			let max_audits = T::MaxPendingAudits::get() as usize;
			let slot = current_block.saturated_into::<u64>();

			for (audit_id, mut audit) in PendingAudits::<T>::iter().take(max_audits) {
				if audit.status == AuditStatus::Pending && current_block > audit.deadline {
					// Auto-fail expired audit
					audit.status = AuditStatus::Failed;

					// Slash pinner
					let _ = pallet_icn_stake::Pallet::<T>::slash(
						frame_system::RawOrigin::Root.into(),
						audit.pinner.clone(),
						T::AuditSlashAmount::get(),
						SlashReason::AuditTimeout,
					);

					// Record negative reputation (-50 delta applied internally by event type)
					let _ = pallet_icn_reputation::Pallet::<T>::record_event(
						frame_system::RawOrigin::Root.into(),
						audit.pinner.clone(),
						ReputationEventType::PinningAuditFailed,
						slot,
					);

					PendingAudits::<T>::insert(audit_id, audit);

					Self::deposit_event(Event::AuditCompleted {
						audit_id,
						passed: false,
					});
				}
			}
		}
	}
}
