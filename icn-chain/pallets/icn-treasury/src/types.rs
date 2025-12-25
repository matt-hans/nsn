// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.

//! Types for pallet-icn-treasury

use frame_support::pallet_prelude::*;
use parity_scale_codec::{Decode, Encode};
use scale_info::TypeInfo;
use sp_runtime::{Perbill, RuntimeDebug};

/// Reward distribution percentages across participant categories
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
pub struct RewardDistribution {
	/// Directors: 40% (GPU generation work)
	pub director_percent: Perbill,
	/// Validators: 25% (semantic verification)
	pub validator_percent: Perbill,
	/// Pinners: 20% (storage provision)
	pub pinner_percent: Perbill,
	/// Treasury: 15% (governance/development)
	pub treasury_percent: Perbill,
}

impl Default for RewardDistribution {
	fn default() -> Self {
		Self {
			director_percent: Perbill::from_percent(40),
			validator_percent: Perbill::from_percent(25),
			pinner_percent: Perbill::from_percent(20),
			treasury_percent: Perbill::from_percent(15),
		}
	}
}

/// Annual emission schedule with decay
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
pub struct EmissionSchedule {
	/// Base emission for year 1 (100M ICN)
	pub base_emission: u128,
	/// Annual decay rate (15% = 0.15)
	pub decay_rate: Perbill,
	/// Current year (starts at 1)
	pub current_year: u32,
	/// Block number when network launched (genesis)
	pub launch_block: u32,
}

impl Default for EmissionSchedule {
	fn default() -> Self {
		Self {
			base_emission: 100_000_000_000_000_000_000_000_000u128, // 100M with 18 decimals
			decay_rate: Perbill::from_percent(15),
			current_year: 1,
			launch_block: 0,
		}
	}
}

/// Accumulated contributions between distribution periods
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen, Default)]
pub struct AccumulatedContributions {
	/// Number of slots successfully completed as director
	pub director_slots: u64,
	/// Number of correct validation votes
	pub validator_votes: u64,
	/// Number of shards served (for future use)
	pub pinner_shards_served: u64,
}
