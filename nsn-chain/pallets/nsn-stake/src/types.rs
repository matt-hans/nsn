// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.

//! Types for pallet-nsn-stake

use frame_support::pallet_prelude::*;
use parity_scale_codec::{Decode, DecodeWithMemTracking, Encode};
use scale_info::TypeInfo;
use sp_runtime::RuntimeDebug;

/// Node role based on stake amount
#[derive(
	Clone,
	Encode,
	Decode,
	DecodeWithMemTracking,
	Eq,
	PartialEq,
	RuntimeDebug,
	TypeInfo,
	MaxEncodedLen,
	Default,
)]
pub enum NodeRole {
	/// No role (stake < 5 NSN)
	#[default]
	None,
	/// Relay node (5 ≤ stake < 10 NSN)
	Relay,
	/// Validator node (10 ≤ stake < 50 NSN)
	Validator,
	/// SuperNode (50 ≤ stake < 100 NSN)
	SuperNode,
	/// Director node (stake ≥ 100 NSN)
	Director,
}

/// Geographic regions for anti-centralization
#[derive(
	Clone,
	Copy,
	Encode,
	Decode,
	DecodeWithMemTracking,
	Eq,
	PartialEq,
	Ord,
	PartialOrd,
	RuntimeDebug,
	TypeInfo,
	MaxEncodedLen,
)]
pub enum Region {
	NaWest,
	NaEast,
	EuWest,
	EuEast,
	Apac,
	Latam,
	Mena,
}

/// Reason for slashing
#[derive(
	Clone,
	Encode,
	Decode,
	DecodeWithMemTracking,
	Eq,
	PartialEq,
	RuntimeDebug,
	TypeInfo,
	MaxEncodedLen,
)]
pub enum SlashReason {
	BftFailure,
	AuditTimeout,
	AuditInvalid,
	MissedSlot,
	ContentViolation,
}

/// Stake information for an account
/// Generic over Balance and BlockNumber types for flexibility
#[derive(Clone, Encode, Decode, DecodeWithMemTracking, Eq, PartialEq, RuntimeDebug, TypeInfo)]
pub struct StakeInfo<Balance, BlockNumber> {
	/// Total staked amount
	pub amount: Balance,
	/// Block number when stake unlocks
	pub locked_until: BlockNumber,
	/// Current node role
	pub role: NodeRole,
	/// Geographic region
	pub region: Region,
	/// Total amount delegated to this account (if validator)
	pub delegated_to_me: Balance,
}

impl<Balance: Default, BlockNumber: Default> Default for StakeInfo<Balance, BlockNumber> {
	fn default() -> Self {
		Self {
			amount: Balance::default(),
			locked_until: BlockNumber::default(),
			role: NodeRole::None,
			region: Region::NaWest,
			delegated_to_me: Balance::default(),
		}
	}
}

// Manual MaxEncodedLen for StakeInfo
impl<Balance: MaxEncodedLen, BlockNumber: MaxEncodedLen> MaxEncodedLen
	for StakeInfo<Balance, BlockNumber>
{
	fn max_encoded_len() -> usize {
		Balance::max_encoded_len() // amount
			+ BlockNumber::max_encoded_len() // locked_until
			+ NodeRole::max_encoded_len() // role
			+ Region::max_encoded_len() // region
			+ Balance::max_encoded_len() // delegated_to_me
	}
}
