// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.
//
// ICN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! # ICN Treasury Pallet
//!
//! Reward distribution and emission management for the Interdimensional Cable Network.
//!
//! ## Overview
//!
//! This pallet implements:
//! - Annual ICN token emission with 15% decay (100M Year 1 → 85M Year 2 → ...)
//! - Daily reward distribution in 40/25/20/15 split (Directors/Validators/Pinners/Treasury)
//! - Treasury balance management for governance proposals
//! - Proportional reward calculation based on actual work completed
//!
//! ## Interface
//!
//! ### Dispatchable Functions
//!
//! - `fund_treasury`: Add funds to treasury (any signed account)
//! - `approve_proposal`: Release funds for governance-approved proposals (root only)
//! - `record_director_work`: Track director slot completion (internal, called by pallet-icn-director)
//! - `record_validator_work`: Track validator votes (internal, called by pallet-icn-bft)
//!
//! ### Hooks
//!
//! - `on_finalize`: Every 14400 blocks (~1 day), distribute accumulated rewards

#![cfg_attr(not(feature = "std"), no_std)]

pub use pallet::*;

mod types;
pub use types::{AccumulatedContributions, EmissionSchedule, RewardDistribution};

#[cfg(test)]
mod mock;
#[cfg(test)]
mod tests;

#[cfg(feature = "runtime-benchmarks")]
mod benchmarking;

pub mod weights;
pub use weights::WeightInfo;

#[frame_support::pallet]
pub mod pallet {
	use super::*;
	use frame_support::{
		pallet_prelude::*,
		traits::{
			fungible::{Inspect, Mutate},
			tokens::Preservation,
			StorageVersion,
		},
		PalletId,
	};
	use frame_system::pallet_prelude::*;
	use sp_runtime::{
		Perbill, SaturatedConversion, Saturating,
		traits::{AccountIdConversion, Zero},
	};
	use sp_std::vec::Vec;

	pub type BalanceOf<T> =
		<<T as Config>::Currency as Inspect<<T as frame_system::Config>::AccountId>>::Balance;

	/// The in-code storage version.
	const STORAGE_VERSION: StorageVersion = StorageVersion::new(0);

	#[pallet::pallet]
	#[pallet::storage_version(STORAGE_VERSION)]
	pub struct Pallet<T>(_);

	/// Configuration trait for the ICN Treasury pallet
	#[pallet::config]
	pub trait Config: frame_system::Config<RuntimeEvent: From<Event<Self>>> {
		/// The currency type for treasury operations
		type Currency: Inspect<Self::AccountId> + Mutate<Self::AccountId>;

		/// The treasury's pallet ID, used for deriving its sovereign account
		#[pallet::constant]
		type PalletId: Get<PalletId>;

		/// Distribution frequency in blocks (~1 day = 14400 blocks at 6s/block)
		#[pallet::constant]
		type DistributionFrequency: Get<BlockNumberFor<Self>>;

		/// Weight information for extrinsics
		type WeightInfo: WeightInfo;
	}

	/// Total ICN available in treasury for governance proposals
	#[pallet::storage]
	#[pallet::getter(fn treasury_balance)]
	pub type TreasuryBalance<T: Config> = StorageValue<_, BalanceOf<T>, ValueQuery>;

	/// Reward distribution percentages (40/25/20/15)
	#[pallet::storage]
	#[pallet::getter(fn reward_distribution)]
	pub type RewardDistributionConfig<T: Config> =
		StorageValue<_, RewardDistribution, ValueQuery>;

	/// Annual emission schedule with decay
	#[pallet::storage]
	#[pallet::getter(fn emission_schedule)]
	pub type EmissionScheduleStorage<T: Config> = StorageValue<_, EmissionSchedule, ValueQuery>;

	/// Last block number when rewards were distributed
	#[pallet::storage]
	#[pallet::getter(fn last_distribution_block)]
	pub type LastDistributionBlock<T: Config> = StorageValue<_, BlockNumberFor<T>, ValueQuery>;

	/// Accumulated contributions since last distribution
	#[pallet::storage]
	#[pallet::getter(fn accumulated_contributions)]
	pub type AccumulatedContributionsMap<T: Config> = StorageMap<
		_,
		Blake2_128Concat,
		T::AccountId,
		AccumulatedContributions,
		ValueQuery,
	>;

	#[pallet::event]
	#[pallet::generate_deposit(pub(super) fn deposit_event)]
	pub enum Event<T: Config> {
		/// Treasury funded by account
		TreasuryFunded { funder: T::AccountId, amount: BalanceOf<T> },
		/// Governance proposal approved and funds released
		ProposalApproved { proposal_id: u32, beneficiary: T::AccountId, amount: BalanceOf<T> },
		/// Daily rewards distributed
		RewardsDistributed { block: BlockNumberFor<T>, total: BalanceOf<T> },
		/// Director rewarded for slots
		DirectorRewarded { account: T::AccountId, amount: BalanceOf<T> },
		/// Validator rewarded for votes
		ValidatorRewarded { account: T::AccountId, amount: BalanceOf<T> },
		/// Director work recorded
		DirectorWorkRecorded { account: T::AccountId, slots: u64 },
		/// Validator work recorded
		ValidatorWorkRecorded { account: T::AccountId, votes: u64 },
	}

	#[pallet::error]
	pub enum Error<T> {
		/// Treasury has insufficient funds for proposal
		InsufficientTreasuryFunds,
		/// Arithmetic overflow in emission calculation
		EmissionOverflow,
		/// Distribution calculation overflow
		DistributionOverflow,
	}

	#[pallet::hooks]
	impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
		fn on_finalize(block: BlockNumberFor<T>) {
			// Trigger distribution every DistributionFrequency blocks
			if block % T::DistributionFrequency::get() == Zero::zero() && !block.is_zero() {
				let _ = Self::distribute_rewards(block);
			}

			// Update current year based on blocks elapsed since launch
			let schedule = EmissionScheduleStorage::<T>::get();
			let blocks_per_year: u32 = 365 * 14400; // ~365 days * 14400 blocks/day
			let launch_block_num: u32 = schedule.launch_block;
			let current_block: u32 = block.saturated_into::<u32>();

			if current_block > launch_block_num {
				let blocks_since_launch = current_block.saturating_sub(launch_block_num);
				let new_year = (blocks_since_launch / blocks_per_year).saturating_add(1);

				if new_year != schedule.current_year {
					EmissionScheduleStorage::<T>::mutate(|s| {
						s.current_year = new_year;
					});
				}
			}
		}
	}

	#[pallet::call]
	impl<T: Config> Pallet<T> {
		/// Add funds to the treasury
		///
		/// Any account can fund the treasury. Funds are transferred from the caller
		/// to the treasury pallet account.
		///
		/// # Parameters
		/// - `origin`: Signed account funding the treasury
		/// - `amount`: Amount of ICN to transfer
		#[pallet::call_index(0)]
		#[pallet::weight(T::WeightInfo::fund_treasury())]
		pub fn fund_treasury(origin: OriginFor<T>, amount: BalanceOf<T>) -> DispatchResult {
			let funder = ensure_signed(origin)?;

			// Transfer from funder to treasury pallet account
			T::Currency::transfer(&funder, &Self::account_id(), amount, Preservation::Preserve)?;

			// Increase treasury balance
			TreasuryBalance::<T>::mutate(|balance| {
				*balance = balance.saturating_add(amount);
			});

			Self::deposit_event(Event::TreasuryFunded { funder, amount });
			Ok(())
		}

		/// Approve a governance proposal and release funds
		///
		/// Only root (governance) can approve proposals. Funds are transferred from
		/// the treasury pallet account to the beneficiary.
		///
		/// # Parameters
		/// - `origin`: Root (governance)
		/// - `beneficiary`: Account receiving the funds
		/// - `amount`: Amount of ICN to release
		/// - `proposal_id`: Unique proposal identifier
		#[pallet::call_index(1)]
		#[pallet::weight(T::WeightInfo::approve_proposal())]
		pub fn approve_proposal(
			origin: OriginFor<T>,
			beneficiary: T::AccountId,
			amount: BalanceOf<T>,
			proposal_id: u32,
		) -> DispatchResult {
			ensure_root(origin)?;

			// Check treasury has sufficient funds
			let treasury_balance = TreasuryBalance::<T>::get();
			ensure!(treasury_balance >= amount, Error::<T>::InsufficientTreasuryFunds);

			// Transfer from treasury pallet account to beneficiary
			T::Currency::transfer(&Self::account_id(), &beneficiary, amount, Preservation::Expendable)?;

			// Decrease treasury balance
			TreasuryBalance::<T>::mutate(|balance| {
				*balance = balance.saturating_sub(amount);
			});

			Self::deposit_event(Event::ProposalApproved { proposal_id, beneficiary, amount });
			Ok(())
		}

		/// Record director work (slots completed)
		///
		/// Internal function called by pallet-icn-director when a director completes a slot.
		///
		/// # Parameters
		/// - `origin`: Root or pallet-icn-director
		/// - `account`: Director account
		/// - `slots`: Number of slots completed
		#[pallet::call_index(2)]
		#[pallet::weight(T::WeightInfo::record_director_work())]
		pub fn record_director_work(
			origin: OriginFor<T>,
			account: T::AccountId,
			slots: u64,
		) -> DispatchResult {
			ensure_root(origin)?;

			AccumulatedContributionsMap::<T>::mutate(&account, |contrib| {
				contrib.director_slots = contrib.director_slots.saturating_add(slots);
			});

			Self::deposit_event(Event::DirectorWorkRecorded { account, slots });
			Ok(())
		}

		/// Record validator work (correct votes)
		///
		/// Internal function called by pallet-icn-bft when a validator submits correct votes.
		///
		/// # Parameters
		/// - `origin`: Root or pallet-icn-bft
		/// - `account`: Validator account
		/// - `votes`: Number of correct votes
		#[pallet::call_index(3)]
		#[pallet::weight(T::WeightInfo::record_validator_work())]
		pub fn record_validator_work(
			origin: OriginFor<T>,
			account: T::AccountId,
			votes: u64,
		) -> DispatchResult {
			ensure_root(origin)?;

			AccumulatedContributionsMap::<T>::mutate(&account, |contrib| {
				contrib.validator_votes = contrib.validator_votes.saturating_add(votes);
			});

			Self::deposit_event(Event::ValidatorWorkRecorded { account, votes });
			Ok(())
		}
	}

	impl<T: Config> Pallet<T> {
		/// Treasury pallet account ID
		pub fn account_id() -> T::AccountId {
			T::PalletId::get().into_account_truncating()
		}

		/// Calculate annual emission for a given year
		///
		/// Formula: emission = base × (1 - decay_rate)^(year - 1)
		///
		/// Year 1: 100M
		/// Year 2: 85M
		/// Year 3: 72.25M
		/// etc.
		pub fn calculate_annual_emission(year: u32) -> Result<u128, Error<T>> {
			let schedule = EmissionScheduleStorage::<T>::get();
			let base = schedule.base_emission;

			if year == 0 {
				return Ok(0);
			}

			if year == 1 {
				return Ok(base);
			}

			// Calculate (1 - decay_rate)^(year - 1)
			// decay_rate = 0.15 → (1 - 0.15) = 0.85
			let one_minus_decay = Perbill::one().saturating_sub(schedule.decay_rate);
			let mut result = base;

			// Apply decay (year - 1) times
			for _ in 1..year {
				result = one_minus_decay
					.mul_floor(result);
			}

			Ok(result)
		}

		/// Distribute accumulated rewards to participants
		fn distribute_rewards(block: BlockNumberFor<T>) -> DispatchResult {
			let schedule = EmissionScheduleStorage::<T>::get();
			let annual_emission = Self::calculate_annual_emission(schedule.current_year)?;

			// Daily emission = annual / 365
			let daily_emission = annual_emission.saturating_div(365);

			let distribution = RewardDistributionConfig::<T>::get();

			// Calculate pools using Perbill (safe percentage multiplication)
			let director_pool = distribution.director_percent.mul_floor(daily_emission);
			let validator_pool = distribution.validator_percent.mul_floor(daily_emission);
			let _pinner_pool = distribution.pinner_percent.mul_floor(daily_emission);
			let treasury_allocation = distribution.treasury_percent.mul_floor(daily_emission);

			// Convert u128 to BalanceOf<T> safely
			let director_pool_balance: BalanceOf<T> = director_pool.saturated_into();
			let validator_pool_balance: BalanceOf<T> = validator_pool.saturated_into();
			let treasury_allocation_balance: BalanceOf<T> = treasury_allocation.saturated_into();

			// Distribute to participants
			Self::distribute_director_rewards(director_pool_balance)?;
			Self::distribute_validator_rewards(validator_pool_balance)?;
			// pinner_pool reserved for pallet-icn-pinning integration

			// Add treasury allocation
			TreasuryBalance::<T>::mutate(|balance| {
				*balance = balance.saturating_add(treasury_allocation_balance);
			});

			LastDistributionBlock::<T>::put(block);
			Self::deposit_event(Event::RewardsDistributed {
				block,
				total: daily_emission.saturated_into(),
			});

			Ok(())
		}

		/// Distribute rewards to directors proportional to slots completed
		pub fn distribute_director_rewards(pool: BalanceOf<T>) -> DispatchResult {
			let mut total_slots = 0u64;
			let contributions: Vec<_> = AccumulatedContributionsMap::<T>::iter()
				.filter(|(_, contrib)| contrib.director_slots > 0)
				.collect();

			for (_, contrib) in &contributions {
				total_slots = total_slots.saturating_add(contrib.director_slots);
			}

			if total_slots == 0 {
				return Ok(());
			}

			for (account, contrib) in contributions {
				// reward = pool * (slots / total_slots)
				// Use checked math to safely convert u64 to BalanceOf<T>
				let slots_balance: BalanceOf<T> = contrib.director_slots.saturated_into();
				let total_slots_balance: BalanceOf<T> = total_slots.saturated_into();
				let reward = pool
					.saturating_mul(slots_balance)
					/ total_slots_balance;

				if !reward.is_zero() {
					// Mint new tokens to director
					T::Currency::mint_into(&account, reward)?;
					Self::deposit_event(Event::DirectorRewarded {
						account: account.clone(),
						amount: reward,
					});
				}

				// Reset accumulated contributions
				AccumulatedContributionsMap::<T>::mutate(&account, |c| {
					c.director_slots = 0;
				});
			}

			Ok(())
		}

		/// Distribute rewards to validators proportional to correct votes
		pub fn distribute_validator_rewards(pool: BalanceOf<T>) -> DispatchResult {
			let mut total_votes = 0u64;
			let contributions: Vec<_> = AccumulatedContributionsMap::<T>::iter()
				.filter(|(_, contrib)| contrib.validator_votes > 0)
				.collect();

			for (_, contrib) in &contributions {
				total_votes = total_votes.saturating_add(contrib.validator_votes);
			}

			if total_votes == 0 {
				return Ok(());
			}

			for (account, contrib) in contributions {
				// reward = pool * (votes / total_votes)
				let votes_balance: BalanceOf<T> = contrib.validator_votes.saturated_into();
				let total_votes_balance: BalanceOf<T> = total_votes.saturated_into();
				let reward = pool
					.saturating_mul(votes_balance)
					/ total_votes_balance;

				if !reward.is_zero() {
					// Mint new tokens to validator
					T::Currency::mint_into(&account, reward)?;
					Self::deposit_event(Event::ValidatorRewarded {
						account: account.clone(),
						amount: reward,
					});
				}

				// Reset accumulated contributions
				AccumulatedContributionsMap::<T>::mutate(&account, |c| {
					c.validator_votes = 0;
				});
			}

			Ok(())
		}
	}
}
