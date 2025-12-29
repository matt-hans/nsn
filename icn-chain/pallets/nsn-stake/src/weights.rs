// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.

//! Weights for pallet-nsn-stake
//!
//! THIS FILE WAS AUTO-GENERATED USING THE SUBSTRATE BENCHMARK CLI VERSION 4.0.0-dev
//! DATE: Placeholder - benchmarks to be run
//! HOSTNAME: Placeholder
//! CPU: Placeholder
//!
//! NOTE: Runtime benchmarking not yet performed. These are placeholder weights
//! with estimated PoV (Proof of Validity) sizes for Cumulus compatibility.
//!
//! PoV Size Estimation:
//! - Storage item size is estimated from MaxEncodedLen
//! - PoV includes: storage key prefix (32 bytes) + key (32 bytes) + value

#![cfg_attr(rustfmt, rustfmt_skip)]
#![allow(unused_parens)]
#![allow(unused_imports)]

use frame_support::{traits::Get, weights::Weight};
use sp_std::marker::PhantomData;

/// Weight functions needed for pallet_nsn_stake.
pub trait WeightInfo {
	fn deposit_stake() -> Weight;
	fn delegate() -> Weight;
	fn withdraw_stake() -> Weight;
	fn revoke_delegation() -> Weight;
	fn slash() -> Weight;
}

/// Weights for pallet_nsn_stake using the Substrate node and recommended hardware.
pub struct SubstrateWeight<T>(PhantomData<T>);
impl<T: frame_system::Config> WeightInfo for SubstrateWeight<T> {
	/// Storage: IcnStake Stakes (r:1 w:1)
	/// Proof: IcnStake Stakes (max_values: None, max_size: Some(128), added: 2603, mode: MaxEncodedLen)
	/// Storage: IcnStake TotalStaked (r:1 w:1)
	/// Proof: IcnStake TotalStaked (max_values: Some(1), max_size: Some(16), added: 511, mode: MaxEncodedLen)
	/// Storage: IcnStake RegionStakes (r:1 w:1)
	/// Proof: IcnStake RegionStakes (max_values: None, max_size: Some(32), added: 2507, mode: MaxEncodedLen)
	fn deposit_stake() -> Weight {
		// PoV size: Stakes(128) + TotalStaked(16) + RegionStakes(32) + overhead(192) = 368 bytes
		Weight::from_parts(50_000_000, 5621)
			.saturating_add(T::DbWeight::get().reads(3))
			.saturating_add(T::DbWeight::get().writes(3))
	}

	/// Storage: IcnStake Stakes (r:1 w:1)
	/// Proof: IcnStake Stakes (max_values: None, max_size: Some(128), added: 2603, mode: MaxEncodedLen)
	/// Storage: IcnStake Delegations (r:1 w:1)
	/// Proof: IcnStake Delegations (max_values: None, max_size: Some(64), added: 2539, mode: MaxEncodedLen)
	fn delegate() -> Weight {
		// PoV size: Stakes(128) + Delegations(64) + overhead(128) = 320 bytes
		Weight::from_parts(40_000_000, 5142)
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(2))
	}

	/// Storage: IcnStake Stakes (r:1 w:1)
	/// Proof: IcnStake Stakes (max_values: None, max_size: Some(128), added: 2603, mode: MaxEncodedLen)
	/// Storage: IcnStake TotalStaked (r:1 w:1)
	/// Proof: IcnStake TotalStaked (max_values: Some(1), max_size: Some(16), added: 511, mode: MaxEncodedLen)
	/// Storage: IcnStake RegionStakes (r:1 w:1)
	/// Proof: IcnStake RegionStakes (max_values: None, max_size: Some(32), added: 2507, mode: MaxEncodedLen)
	fn withdraw_stake() -> Weight {
		// PoV size: Stakes(128) + TotalStaked(16) + RegionStakes(32) + overhead(192) = 368 bytes
		Weight::from_parts(45_000_000, 5621)
			.saturating_add(T::DbWeight::get().reads(3))
			.saturating_add(T::DbWeight::get().writes(3))
	}

	/// Storage: IcnStake Delegations (r:1 w:1)
	/// Proof: IcnStake Delegations (max_values: None, max_size: Some(64), added: 2539, mode: MaxEncodedLen)
	/// Storage: IcnStake Stakes (r:1 w:1)
	/// Proof: IcnStake Stakes (max_values: None, max_size: Some(128), added: 2603, mode: MaxEncodedLen)
	fn revoke_delegation() -> Weight {
		// PoV size: Delegations(64) + Stakes(128) + overhead(128) = 320 bytes
		Weight::from_parts(35_000_000, 5142)
			.saturating_add(T::DbWeight::get().reads(2))
			.saturating_add(T::DbWeight::get().writes(2))
	}

	/// Storage: IcnStake Stakes (r:1 w:1)
	/// Proof: IcnStake Stakes (max_values: None, max_size: Some(128), added: 2603, mode: MaxEncodedLen)
	/// Storage: IcnStake TotalStaked (r:1 w:1)
	/// Proof: IcnStake TotalStaked (max_values: Some(1), max_size: Some(16), added: 511, mode: MaxEncodedLen)
	/// Storage: IcnStake RegionStakes (r:1 w:1)
	/// Proof: IcnStake RegionStakes (max_values: None, max_size: Some(32), added: 2507, mode: MaxEncodedLen)
	fn slash() -> Weight {
		// PoV size: Stakes(128) + TotalStaked(16) + RegionStakes(32) + overhead(192) = 368 bytes
		Weight::from_parts(50_000_000, 5621)
			.saturating_add(T::DbWeight::get().reads(3))
			.saturating_add(T::DbWeight::get().writes(3))
	}
}

// For backwards compatibility and tests
impl WeightInfo for () {
	fn deposit_stake() -> Weight {
		Weight::from_parts(50_000_000, 5621)
	}
	fn delegate() -> Weight {
		Weight::from_parts(40_000_000, 5142)
	}
	fn withdraw_stake() -> Weight {
		Weight::from_parts(45_000_000, 5621)
	}
	fn revoke_delegation() -> Weight {
		Weight::from_parts(35_000_000, 5142)
	}
	fn slash() -> Weight {
		Weight::from_parts(50_000_000, 5621)
	}
}
