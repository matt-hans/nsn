// Copyright 2024 Interdimensional Cable Network
// This file is part of ICN Chain.
//
// ICN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! Types for the ICN Pinning pallet.

use parity_scale_codec::{Decode, DecodeWithMemTracking, Encode, MaxEncodedLen};
use scale_info::TypeInfo;
use sp_runtime::RuntimeDebug;

/// 32-byte deal identifier (hash of deal parameters)
pub type DealId = [u8; 32];

/// 32-byte shard hash (content hash of erasure-coded shard)
pub type ShardHash = [u8; 32];

/// 32-byte audit identifier (hash of audit parameters)
pub type AuditId = [u8; 32];

/// Reed-Solomon erasure coding scheme: 10 data shards + 4 parity shards
pub const ERASURE_DATA_SHARDS: usize = 10;
pub const ERASURE_PARITY_SHARDS: usize = 4;
pub const TOTAL_SHARDS_PER_CHUNK: usize = ERASURE_DATA_SHARDS + ERASURE_PARITY_SHARDS;

/// Replication factor: each shard stored on 5 super-nodes
pub const REPLICATION_FACTOR: usize = 5;

/// Audit deadline in blocks (~10 minutes at 600 blocks/hour)
pub const AUDIT_DEADLINE_BLOCKS: u32 = 100;

/// Reward distribution interval (every 100 blocks)
pub const REWARD_INTERVAL_BLOCKS: u32 = 100;

/// Status of a pinning deal.
#[derive(
	Encode,
	Decode,
	DecodeWithMemTracking,
	Clone,
	PartialEq,
	Eq,
	RuntimeDebug,
	TypeInfo,
	MaxEncodedLen,
)]
pub enum DealStatus {
	/// Deal is active and rewards are being distributed
	Active,
	/// Deal has expired (past expires_at block)
	Expired,
	/// Deal was cancelled by creator (future feature)
	Cancelled,
}

/// Status of an audit challenge.
#[derive(
	Encode,
	Decode,
	DecodeWithMemTracking,
	Clone,
	PartialEq,
	Eq,
	RuntimeDebug,
	TypeInfo,
	MaxEncodedLen,
)]
pub enum AuditStatus {
	/// Audit pending (waiting for proof submission)
	Pending,
	/// Audit passed (proof valid)
	Passed,
	/// Audit failed (proof invalid or timeout)
	Failed,
}

/// Pinning deal metadata.
///
/// Created when a content creator calls `create_deal()` to store
/// erasure-coded shards across the super-node network.
///
/// # Reed-Solomon 10+4
/// - 10 data shards: original video chunk split into 10 pieces
/// - 4 parity shards: redundancy for recovery
/// - Any 10 of 14 shards can reconstruct original chunk
///
/// # Replication
/// Each of the 14 shards is replicated 5× across different regions.
/// Total pinner slots = 14 shards × 5 replicas = 70 assignments.
#[derive(Encode, Decode, DecodeWithMemTracking, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
#[scale_info(skip_type_params(AccountId, Balance, BlockNumber, MaxShards))]
pub struct PinningDeal<AccountId, Balance, BlockNumber, MaxShards: Get<u32>> {
	/// Unique deal identifier
	pub deal_id: DealId,
	/// Account that created the deal
	pub creator: AccountId,
	/// Hashes of all shards (14 for Reed-Solomon 10+4)
	pub shards: BoundedVec<ShardHash, MaxShards>,
	/// Block when deal was created
	pub created_at: BlockNumber,
	/// Block when deal expires (no more rewards after this)
	pub expires_at: BlockNumber,
	/// Total reward pool for this deal
	pub total_reward: Balance,
	/// Current deal status
	pub status: DealStatus,
}

// Manual MaxEncodedLen for PinningDeal
impl<AccountId: MaxEncodedLen, Balance: MaxEncodedLen, BlockNumber: MaxEncodedLen, MaxShards: Get<u32>>
	MaxEncodedLen for PinningDeal<AccountId, Balance, BlockNumber, MaxShards>
{
	fn max_encoded_len() -> usize {
		32 // deal_id
			+ AccountId::max_encoded_len() // creator
			+ <BoundedVec<ShardHash, MaxShards>>::max_encoded_len() // shards
			+ BlockNumber::max_encoded_len() // created_at
			+ BlockNumber::max_encoded_len() // expires_at
			+ Balance::max_encoded_len() // total_reward
			+ DealStatus::max_encoded_len() // status
	}
}

use frame_support::{pallet_prelude::*, BoundedVec};

/// Audit challenge for a pinner.
///
/// Created when `initiate_audit()` is called (root-only).
/// Pinner must respond with Merkle proof within deadline.
///
/// # Challenge Structure
/// - `byte_offset`: Random offset within shard (e.g., 2048)
/// - `byte_length`: Length of requested data (fixed at 64 bytes)
/// - `nonce`: Random nonce for proof freshness
///
/// Pinner must prove they have bytes [offset:offset+length] by:
/// 1. Providing the raw bytes
/// 2. Providing Merkle siblings to reconstruct root
/// 3. Signature to prevent replay attacks
#[derive(Encode, Decode, DecodeWithMemTracking, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo)]
#[scale_info(skip_type_params(AccountId, BlockNumber))]
pub struct PinningAudit<AccountId, BlockNumber> {
	/// Unique audit identifier
	pub audit_id: AuditId,
	/// Account being audited
	pub pinner: AccountId,
	/// Shard hash being audited
	pub shard_hash: ShardHash,
	/// Challenge parameters
	pub challenge: AuditChallenge,
	/// Block number when response is due
	pub deadline: BlockNumber,
	/// Current audit status
	pub status: AuditStatus,
}

// Manual MaxEncodedLen for PinningAudit
impl<AccountId: MaxEncodedLen, BlockNumber: MaxEncodedLen> MaxEncodedLen
	for PinningAudit<AccountId, BlockNumber>
{
	fn max_encoded_len() -> usize {
		32 // audit_id
			+ AccountId::max_encoded_len() // pinner
			+ 32 // shard_hash
			+ AuditChallenge::max_encoded_len() // challenge
			+ BlockNumber::max_encoded_len() // deadline
			+ AuditStatus::max_encoded_len() // status
	}
}

/// Audit challenge parameters.
///
/// Randomly generated using VRF to prevent prediction.
#[derive(
	Encode,
	Decode,
	DecodeWithMemTracking,
	Clone,
	PartialEq,
	Eq,
	RuntimeDebug,
	TypeInfo,
	MaxEncodedLen,
)]
pub struct AuditChallenge {
	/// Byte offset within shard to request
	pub byte_offset: u32,
	/// Number of bytes to request (fixed at 64)
	pub byte_length: u32,
	/// Random nonce for proof freshness
	pub nonce: [u8; 16],
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_erasure_coding_constants() {
		assert_eq!(ERASURE_DATA_SHARDS, 10);
		assert_eq!(ERASURE_PARITY_SHARDS, 4);
		assert_eq!(TOTAL_SHARDS_PER_CHUNK, 14);
		assert_eq!(REPLICATION_FACTOR, 5);
	}

	#[test]
	fn test_deal_status_encoding() {
		let active = DealStatus::Active;
		let expired = DealStatus::Expired;
		assert_ne!(active, expired);
	}

	#[test]
	fn test_audit_status_encoding() {
		let pending = AuditStatus::Pending;
		let passed = AuditStatus::Passed;
		let failed = AuditStatus::Failed;
		assert_ne!(pending, passed);
		assert_ne!(passed, failed);
	}
}
