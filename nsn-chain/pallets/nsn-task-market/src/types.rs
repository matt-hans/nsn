// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.

//! Types for pallet-nsn-task-market

use frame_support::pallet_prelude::*;
use parity_scale_codec::{Decode, DecodeWithMemTracking, Encode};
use scale_info::TypeInfo;
use sp_runtime::RuntimeDebug;

/// Status of a task in the task market
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
pub enum TaskStatus {
    /// Task is open and available for assignment
    #[default]
    Open,
    /// Task has been assigned to an executor
    Assigned,
    /// Task has been completed successfully
    Completed,
    /// Task has failed
    Failed,
}

/// Reason for task failure
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
pub enum FailReason {
    /// Executor could not complete the task
    ExecutorFailed,
    /// Task was cancelled by the requester
    Cancelled,
    /// Task deadline was exceeded
    DeadlineExceeded,
    /// Other failure reason
    Other,
}

/// A compute task intent in the task market
///
/// Generic over AccountId, Balance, and BlockNumber types for flexibility.
#[derive(Clone, Encode, Decode, DecodeWithMemTracking, Eq, PartialEq, RuntimeDebug, TypeInfo)]
pub struct TaskIntent<AccountId, Balance, BlockNumber> {
    /// Task identifier (unique)
    pub id: u64,
    /// Account that created the task
    pub requester: AccountId,
    /// Executor assigned to the task (if any)
    pub executor: Option<AccountId>,
    /// Current status of the task
    pub status: TaskStatus,
    /// Amount escrowed for the task
    pub escrow: Balance,
    /// Block number when task was created
    pub created_at: BlockNumber,
    /// Block number deadline for task completion
    pub deadline: BlockNumber,
}

impl<AccountId: Default, Balance: Default, BlockNumber: Default> Default
    for TaskIntent<AccountId, Balance, BlockNumber>
{
    fn default() -> Self {
        Self {
            id: 0,
            requester: AccountId::default(),
            executor: None,
            status: TaskStatus::Open,
            escrow: Balance::default(),
            created_at: BlockNumber::default(),
            deadline: BlockNumber::default(),
        }
    }
}

// Manual MaxEncodedLen for TaskIntent
impl<AccountId: MaxEncodedLen, Balance: MaxEncodedLen, BlockNumber: MaxEncodedLen> MaxEncodedLen
    for TaskIntent<AccountId, Balance, BlockNumber>
{
    fn max_encoded_len() -> usize {
        u64::max_encoded_len() // id
            + AccountId::max_encoded_len() // requester
            + Option::<AccountId>::max_encoded_len() // executor
            + TaskStatus::max_encoded_len() // status
            + Balance::max_encoded_len() // escrow
            + BlockNumber::max_encoded_len() // created_at
            + BlockNumber::max_encoded_len() // deadline
    }
}
