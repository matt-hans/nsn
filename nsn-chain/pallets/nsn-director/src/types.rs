// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Types for the NSN Director pallet.
//!
//! ## Core Types
//!
//! - `BftConsensusResult`: Stores BFT consensus outcome for a slot
//! - `BftChallenge`: Tracks pending challenge against a slot result
//! - `SlotInfo`: Metadata about a slot's election and status
//!
//! ## Constants (from PRD §3.3)
//!
//! - `DIRECTORS_PER_SLOT`: 5 directors elected per slot
//! - `BFT_THRESHOLD`: 3-of-5 agreement required
//! - `COOLDOWN_SLOTS`: 20-slot cooldown between elections
//! - `CHALLENGE_PERIOD_BLOCKS`: 50 blocks (~5 minutes)
//! - `JITTER_PERCENT`: ±20% deterministic jitter

use parity_scale_codec::{Decode, DecodeWithMemTracking, Encode, MaxEncodedLen};
use scale_info::TypeInfo;
use sp_runtime::RuntimeDebug;
use sp_std::vec::Vec;

// =============================================================================
// Constants (PRD §3.3)
// =============================================================================

/// Number of directors elected per slot
pub const DIRECTORS_PER_SLOT: u32 = 5;

/// BFT threshold for consensus (3-of-5)
pub const BFT_THRESHOLD: u32 = 3;

/// Cooldown period in slots between director selections
pub const COOLDOWN_SLOTS: u64 = 20;

/// Challenge period duration in blocks (~5 minutes at 6s/block)
pub const CHALLENGE_PERIOD_BLOCKS: u32 = 50;

/// Jitter factor for election randomization (±20%)
pub const JITTER_PERCENT: u32 = 20;

/// Blocks per slot (~8 blocks = ~48 seconds)
pub const BLOCKS_PER_SLOT: u64 = 8;

/// Lookahead slots for director election
pub const ELECTION_LOOKAHEAD: u64 = 2;

/// Maximum directors to store per slot (L0 constraint)
pub const MAX_DIRECTORS_PER_SLOT: u32 = 5;

/// Maximum attestations in a BFT result (L0 constraint)
pub const MAX_ATTESTATIONS: u32 = 10;

/// Maximum validator attestations per challenge (L0 constraint)
pub const MAX_VALIDATOR_ATTESTATIONS: u32 = 20;

// =============================================================================
// Epoch Management Constants
// =============================================================================

/// Epoch duration in blocks (1 hour at 6s/block = 600 blocks)
pub const EPOCH_DURATION_BLOCKS: u64 = 600;

/// Epoch lookahead for On-Deck notification (2 minutes = 20 blocks)
pub const EPOCH_LOOKAHEAD_BLOCKS: u64 = 20;

/// Maximum directors per epoch
pub const MAX_DIRECTORS_PER_EPOCH: u32 = 5;

// =============================================================================
// BFT Consensus Types
// =============================================================================

/// BFT consensus result for a slot.
///
/// Represents the outcome of off-chain BFT coordination between elected directors.
/// Stored in `BftResults` after submission via `submit_bft_result()`.
///
/// # Fields
///
/// * `slot` - Slot number this result applies to
/// * `success` - Whether BFT consensus was reached
/// * `canonical_hash` - Hash of agreed CLIP embeddings
/// * `submitter` - Director who submitted this result
/// * `attestations` - Directors who attested to this result
///
/// # Lifecycle
///
/// 1. Directors reach off-chain consensus
/// 2. Canonical director calls `submit_bft_result()`
/// 3. Result enters PENDING state with 50-block challenge period
/// 4. After challenge period (no challenge), auto-finalized in `on_finalize()`
#[derive(Encode, Decode, DecodeWithMemTracking, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
#[scale_info(skip_type_params(AccountId, Hash))]
pub struct BftConsensusResult<AccountId, Hash> {
	/// Slot number this result applies to
	pub slot: u64,
	/// Whether BFT consensus was reached
	pub success: bool,
	/// Hash of agreed CLIP embeddings
	pub canonical_hash: Hash,
	/// Director who submitted this result
	pub submitter: AccountId,
	/// Block when this result was submitted (for deadline calculation)
	pub submitted_at_block: u64,
}

/// Challenge against a BFT result.
///
/// Created when a staker disputes a submitted BFT result via `challenge_bft_result()`.
/// Validators then attest to support or reject the challenge.
///
/// # Challenge Flow
///
/// 1. Challenger calls `challenge_bft_result()` with 25 NSN bond
/// 2. Challenge stored in `PendingChallenges` with deadline
/// 3. Validators submit attestations (agree/disagree with challenge)
/// 4. `resolve_challenge()` tallies attestations:
///    - If upheld: Slash directors 100 NSN each, refund + reward challenger
///    - If rejected: Slash challenger's 25 NSN bond
///
/// # Bond Amounts (PRD §3.3)
///
/// * Challenge bond: 25 NSN (forfeited if rejected)
/// * Director slash: 100 NSN per fraudulent director
/// * Challenger reward: 10 NSN (if upheld)
#[derive(Encode, Decode, DecodeWithMemTracking, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
#[scale_info(skip_type_params(AccountId, Hash))]
pub struct BftChallenge<AccountId, Hash> {
	/// Slot number being challenged
	pub slot: u64,
	/// Account that submitted the challenge
	pub challenger: AccountId,
	/// Block when challenge was submitted
	pub challenge_block: u64,
	/// Deadline block for challenge resolution
	pub deadline: u64,
	/// Hash of evidence (off-chain reference)
	pub evidence_hash: Hash,
	/// Whether challenge has been resolved
	pub resolved: bool,
}

/// Validator attestation for a challenge.
///
/// Validators attest whether they agree with the challenger's claim
/// that the BFT result is fraudulent.
///
/// # Fields
///
/// * `validator` - Validator account providing attestation
/// * `agrees_with_challenge` - True if validator confirms fraud
/// * `attestation_hash` - Hash of attestation proof (CLIP embeddings, etc.)
#[derive(Encode, Decode, DecodeWithMemTracking, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
#[scale_info(skip_type_params(AccountId, Hash))]
pub struct ValidatorAttestation<AccountId, Hash> {
	/// Validator account
	pub validator: AccountId,
	/// Whether validator agrees with the challenge
	pub agrees_with_challenge: bool,
	/// Hash of attestation proof
	pub attestation_hash: Hash,
}

// =============================================================================
// Epoch Management Types
// =============================================================================

use frame_support::{pallet_prelude::*, BoundedVec};

/// Epoch identifier
pub type EpochId = u64;

/// Epoch information
///
/// Represents a 1-hour shift during which 5 directors are active.
/// Epoch transitions trigger lane swaps (Lane1Active → Draining → Lane0Active).
///
/// # Lifecycle
///
/// 1. Scheduled: Elected 20 blocks before start (On-Deck notification)
/// 2. Active: Currently running, directors are generating content
/// 3. Completed: Epoch has ended, directors moved to draining/inactive
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
#[scale_info(skip_type_params(MaxDirectors))]
pub struct Epoch<BlockNumber, AccountId, MaxDirectors: Get<u32>> {
	/// Unique epoch identifier
	pub id: EpochId,
	/// Block number when epoch starts
	pub start_block: BlockNumber,
	/// Block number when epoch ends
	pub end_block: BlockNumber,
	/// Elected directors for this epoch
	pub directors: BoundedVec<AccountId, MaxDirectors>,
	/// Current status of the epoch
	pub status: EpochStatus,
}

/// Epoch lifecycle status
#[derive(Clone, Encode, Decode, Eq, PartialEq, RuntimeDebug, TypeInfo, MaxEncodedLen, Default)]
pub enum EpochStatus {
	/// Epoch scheduled but not yet active (On-Deck phase)
	#[default]
	Scheduled,
	/// Epoch is currently active
	Active,
	/// Epoch has completed
	Completed,
}

// =============================================================================
// Slot Management Types
// =============================================================================

/// Information about a slot.
///
/// Tracks the state of a slot through its lifecycle.
///
/// # Slot Lifecycle
///
/// 1. Election: Directors elected 2 slots ahead (lookahead)
/// 2. Active: Directors perform BFT coordination off-chain
/// 3. Submitted: BFT result submitted on-chain
/// 4. Challenged (optional): Challenge submitted during 50-block period
/// 5. Finalized: Result finalized, reputation updated
#[derive(Encode, Decode, DecodeWithMemTracking, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, MaxEncodedLen, Default)]
pub enum SlotStatus {
	/// Slot not yet processed
	#[default]
	Pending,
	/// Directors have been elected for this slot
	Elected,
	/// BFT result submitted, in challenge period
	Submitted,
	/// Result has been challenged
	Challenged,
	/// Result finalized successfully
	Finalized,
	/// Slot failed (no consensus or challenge upheld)
	Failed,
}

/// Election candidate with computed weight.
///
/// Used during director election to track candidates and their
/// reputation-weighted + jittered selection probability.
///
/// # Weight Calculation
///
/// `weight = sqrt(reputation_total + 1) * (100 ± jitter%) / 100`
///
/// Sublinear scaling (sqrt) prevents runaway dominance by high-reputation directors.
/// ±20% jitter breaks deterministic patterns.
#[derive(Clone, PartialEq, Eq, RuntimeDebug)]
pub struct ElectionCandidate<AccountId> {
	/// Candidate account
	pub account: AccountId,
	/// Computed weight for selection
	pub weight: u64,
	/// Region of the candidate
	pub region: pallet_nsn_stake::Region,
}

// =============================================================================
// Result Types
// =============================================================================

/// Election result for a slot.
///
/// Returned by `elect_directors()` containing the selected directors
/// and election metadata.
#[derive(Clone, PartialEq, Eq, RuntimeDebug)]
pub struct ElectionResult<AccountId> {
	/// Slot number
	pub slot: u64,
	/// Elected directors (exactly 5, or fewer if insufficient candidates)
	pub directors: Vec<AccountId>,
	/// Number of eligible candidates
	pub candidate_count: u32,
	/// Whether multi-region constraint was satisfied
	pub multi_region_satisfied: bool,
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_constants() {
		assert_eq!(DIRECTORS_PER_SLOT, 5);
		assert_eq!(BFT_THRESHOLD, 3);
		assert_eq!(COOLDOWN_SLOTS, 20);
		assert_eq!(CHALLENGE_PERIOD_BLOCKS, 50);
		assert_eq!(JITTER_PERCENT, 20);
		assert_eq!(BLOCKS_PER_SLOT, 8);
	}

	#[test]
	fn test_slot_status_default() {
		let status = SlotStatus::default();
		assert_eq!(status, SlotStatus::Pending);
	}
}
