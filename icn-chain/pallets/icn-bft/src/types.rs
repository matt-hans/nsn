// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.
//
// ICN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Types for pallet-icn-bft
//!
//! ## Core Types
//!
//! - `ConsensusRound`: Metadata for a finalized BFT round
//! - `ConsensusStats`: Aggregate network health metrics
//!
//! ## Constants
//!
//! - `DEFAULT_RETENTION_BLOCKS`: 6 months (~2.59M blocks at 6s/block)
//! - `AUTO_PRUNE_FREQUENCY`: Prune every 10000 blocks (~16.7 hours)
//! - `ZERO_HASH_SENTINEL`: Special value indicating failed consensus

use parity_scale_codec::{Decode, Encode, MaxEncodedLen};
use scale_info::TypeInfo;
use sp_runtime::RuntimeDebug;
use sp_std::vec::Vec;

// =============================================================================
// Constants
// =============================================================================

/// Default retention period in blocks (~6 months at 6s/block)
///
/// 6 months × 30 days × 24 hours × 3600 seconds ÷ 6 seconds/block
/// = 2,592,000 blocks
pub const DEFAULT_RETENTION_BLOCKS: u32 = 2_592_000;

/// Auto-prune frequency in blocks (~16.7 hours at 6s/block)
pub const AUTO_PRUNE_FREQUENCY: u32 = 10_000;

/// Maximum directors stored per consensus round (L0 constraint)
pub const MAX_DIRECTORS_PER_ROUND: u32 = 5;

// =============================================================================
// Consensus Round Types
// =============================================================================

/// BFT consensus round metadata.
///
/// Stores the outcome of a finalized director consensus round for a slot.
/// Provides historical record for auditing, challenge evidence, and analytics.
///
/// # Fields
///
/// * `slot` - Slot number this round applies to
/// * `embeddings_hash` - Canonical CLIP embedding hash (or ZERO_HASH if failed)
/// * `directors` - Directors who participated (up to 5)
/// * `timestamp` - Block number when stored
/// * `success` - Whether consensus was reached (true) or failed (false)
///
/// # Storage
///
/// Stored in `ConsensusRounds<slot>` after director finalization.
/// Retrieved via `get_slot_result(slot)` query.
///
/// # Lifecycle
///
/// 1. Directors reach off-chain BFT consensus
/// 2. pallet-icn-director finalizes slot
/// 3. Calls `store_embeddings_hash()` to create ConsensusRound
/// 4. Stored for ~6 months, then auto-pruned
#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
#[scale_info(skip_type_params(T))]
pub struct ConsensusRound<T: frame_system::Config> {
	/// Slot number this round applies to
	pub slot: u64,
	/// Canonical CLIP embedding hash (ZERO_HASH if failed)
	pub embeddings_hash: T::Hash,
	/// Directors who participated (bounded to MAX_DIRECTORS_PER_ROUND)
	pub directors: Vec<T::AccountId>,
	/// Block number when this round was stored
	pub timestamp: T::BlockNumber,
	/// Whether consensus was successfully reached
	pub success: bool,
}

/// Aggregate consensus statistics.
///
/// Tracks network health metrics across all consensus rounds.
/// Updated atomically with each `store_embeddings_hash()` call.
///
/// # Fields
///
/// * `total_rounds` - Total consensus rounds attempted
/// * `successful_rounds` - Rounds where BFT consensus was reached
/// * `failed_rounds` - Rounds where consensus failed
/// * `average_directors_agreeing` - Moving average × 100 (fixed-point)
///
/// # Metrics
///
/// - Success rate: `successful_rounds / total_rounds × 100`
/// - Avg directors: `average_directors_agreeing / 100` (e.g., 380 → 3.80)
///
/// # Usage
///
/// ```ignore
/// let stats = pallet_icn_bft::Pallet::<T>::get_stats();
/// let success_rate = stats.success_rate(); // Returns percentage
/// println!("Network health: {}% success", success_rate);
/// ```
#[derive(
	Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, Default, MaxEncodedLen,
)]
pub struct ConsensusStats {
	/// Total consensus rounds attempted
	pub total_rounds: u64,
	/// Rounds where consensus was successfully reached
	pub successful_rounds: u64,
	/// Rounds where consensus failed (< BFT_THRESHOLD agreement)
	pub failed_rounds: u64,
	/// Moving average of directors agreeing × 100 (fixed-point)
	///
	/// Example: 380 = 3.80 directors on average
	/// Calculated as: `(prev_avg × prev_count + new_count × 100) / total_count`
	pub average_directors_agreeing: u32,
}

impl ConsensusStats {
	/// Calculate success rate as percentage.
	///
	/// Returns 0 if no rounds have been attempted.
	///
	/// # Example
	///
	/// ```ignore
	/// let stats = ConsensusStats {
	///     total_rounds: 100,
	///     successful_rounds: 95,
	///     ..Default::default()
	/// };
	/// assert_eq!(stats.success_rate(), 95);
	/// ```
	pub fn success_rate(&self) -> u32 {
		if self.total_rounds == 0 {
			return 0;
		}
		((self.successful_rounds * 100) / self.total_rounds) as u32
	}

	/// Get average directors agreeing as floating point.
	///
	/// Converts fixed-point representation back to decimal.
	///
	/// # Example
	///
	/// ```ignore
	/// let stats = ConsensusStats {
	///     average_directors_agreeing: 380,
	///     ..Default::default()
	/// };
	/// assert_eq!(stats.average_directors_float(), 3.80);
	/// ```
	pub fn average_directors_float(&self) -> f64 {
		(self.average_directors_agreeing as f64) / 100.0
	}
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_constants() {
		assert_eq!(DEFAULT_RETENTION_BLOCKS, 2_592_000);
		assert_eq!(AUTO_PRUNE_FREQUENCY, 10_000);
		assert_eq!(MAX_DIRECTORS_PER_ROUND, 5);
	}

	#[test]
	fn test_consensus_stats_default() {
		let stats = ConsensusStats::default();
		assert_eq!(stats.total_rounds, 0);
		assert_eq!(stats.successful_rounds, 0);
		assert_eq!(stats.failed_rounds, 0);
		assert_eq!(stats.average_directors_agreeing, 0);
	}

	#[test]
	fn test_consensus_stats_success_rate_zero_rounds() {
		let stats = ConsensusStats::default();
		assert_eq!(stats.success_rate(), 0);
	}

	#[test]
	fn test_consensus_stats_success_rate_100_percent() {
		let stats = ConsensusStats {
			total_rounds: 100,
			successful_rounds: 100,
			failed_rounds: 0,
			average_directors_agreeing: 500,
		};
		assert_eq!(stats.success_rate(), 100);
	}

	#[test]
	fn test_consensus_stats_success_rate_95_percent() {
		let stats = ConsensusStats {
			total_rounds: 100,
			successful_rounds: 95,
			failed_rounds: 5,
			average_directors_agreeing: 380,
		};
		assert_eq!(stats.success_rate(), 95);
	}

	#[test]
	fn test_consensus_stats_success_rate_partial() {
		let stats = ConsensusStats {
			total_rounds: 37,
			successful_rounds: 30,
			failed_rounds: 7,
			average_directors_agreeing: 320,
		};
		// 30/37 * 100 = 81.08... truncates to 81
		assert_eq!(stats.success_rate(), 81);
	}

	#[test]
	fn test_average_directors_float() {
		let stats = ConsensusStats {
			total_rounds: 10,
			successful_rounds: 10,
			failed_rounds: 0,
			average_directors_agreeing: 380,
		};
		assert_eq!(stats.average_directors_float(), 3.80);
	}

	#[test]
	fn test_average_directors_float_exact_five() {
		let stats = ConsensusStats {
			total_rounds: 5,
			successful_rounds: 5,
			failed_rounds: 0,
			average_directors_agreeing: 500,
		};
		assert_eq!(stats.average_directors_float(), 5.0);
	}
}
