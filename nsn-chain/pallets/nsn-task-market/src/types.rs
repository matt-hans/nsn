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
    /// Task has expired (deadline passed without completion)
    Expired,
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
    /// Task was preempted for Lane 0 video generation
    Preempted,
    /// Bad input data
    InvalidInput,
    /// Other failure reason
    Other,
}

/// A compute task intent in the task market
///
/// Generic over AccountId, Balance, BlockNumber, and bounded length types for flexibility.
#[derive(Clone, Encode, Decode, DecodeWithMemTracking, Eq, PartialEq, RuntimeDebug, TypeInfo)]
#[scale_info(skip_type_params(MaxModelIdLen, MaxCidLen))]
pub struct TaskIntent<AccountId, Balance, BlockNumber, MaxModelIdLen, MaxCidLen>
where
    MaxModelIdLen: Get<u32>,
    MaxCidLen: Get<u32>,
{
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
    /// AI model identifier for task execution
    pub model_id: BoundedVec<u8, MaxModelIdLen>,
    /// Content identifier for input data
    pub input_cid: BoundedVec<u8, MaxCidLen>,
    /// Maximum compute units budget for the task
    pub max_compute_units: u32,
    /// Content identifier for output data (set on completion)
    pub output_cid: Option<BoundedVec<u8, MaxCidLen>>,
}

impl<
        AccountId: Default,
        Balance: Default,
        BlockNumber: Default,
        MaxModelIdLen: Get<u32>,
        MaxCidLen: Get<u32>,
    > Default for TaskIntent<AccountId, Balance, BlockNumber, MaxModelIdLen, MaxCidLen>
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
            model_id: BoundedVec::default(),
            input_cid: BoundedVec::default(),
            max_compute_units: 0,
            output_cid: None,
        }
    }
}

// Manual MaxEncodedLen for TaskIntent
impl<
        AccountId: MaxEncodedLen,
        Balance: MaxEncodedLen,
        BlockNumber: MaxEncodedLen,
        MaxModelIdLen: Get<u32>,
        MaxCidLen: Get<u32>,
    > MaxEncodedLen for TaskIntent<AccountId, Balance, BlockNumber, MaxModelIdLen, MaxCidLen>
{
    fn max_encoded_len() -> usize {
        u64::max_encoded_len() // id
            + AccountId::max_encoded_len() // requester
            + Option::<AccountId>::max_encoded_len() // executor
            + TaskStatus::max_encoded_len() // status
            + Balance::max_encoded_len() // escrow
            + BlockNumber::max_encoded_len() // created_at
            + BlockNumber::max_encoded_len() // deadline
            + BoundedVec::<u8, MaxModelIdLen>::max_encoded_len() // model_id
            + BoundedVec::<u8, MaxCidLen>::max_encoded_len() // input_cid
            + u32::max_encoded_len() // max_compute_units
            + Option::<BoundedVec<u8, MaxCidLen>>::max_encoded_len() // output_cid
    }
}
