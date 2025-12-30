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
    /// Task is executing on an assigned node
    Executing,
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

/// Task lane designation (Lane 0 vs Lane 1)
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
pub enum TaskLane {
    /// Lane 0: Time-triggered video generation
    Lane0,
    /// Lane 1: Demand-triggered general compute
    Lane1,
}

impl TaskLane {
    pub fn as_u8(&self) -> u8 {
        match self {
            TaskLane::Lane0 => 0,
            TaskLane::Lane1 => 1,
        }
    }
}

/// Task priority for queue ordering
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
pub enum TaskPriority {
    High,
    Normal,
    Low,
}

impl TaskPriority {
    /// Numeric priority used for sorting (higher is more important).
    pub fn weight(&self) -> u8 {
        match self {
            TaskPriority::High => 2,
            TaskPriority::Normal => 1,
            TaskPriority::Low => 0,
        }
    }
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
    /// Lane assignment (Lane 0 or Lane 1)
    pub lane: TaskLane,
    /// Task priority (queue ordering within lane)
    pub priority: TaskPriority,
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
    /// AI model requirements for task execution
    pub model_requirements: BoundedVec<u8, MaxModelIdLen>,
    /// Content identifier for input data
    pub input_cid: BoundedVec<u8, MaxCidLen>,
    /// Maximum compute units budget for the task
    pub compute_budget: u64,
    /// Content identifier for output data (set on completion)
    pub output_cid: Option<BoundedVec<u8, MaxCidLen>>,
    /// Optional attestation CID for result verification
    pub attestation_cid: Option<BoundedVec<u8, MaxCidLen>>,
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
            lane: TaskLane::Lane1,
            priority: TaskPriority::Normal,
            requester: AccountId::default(),
            executor: None,
            status: TaskStatus::Open,
            escrow: Balance::default(),
            created_at: BlockNumber::default(),
            deadline: BlockNumber::default(),
            model_requirements: BoundedVec::default(),
            input_cid: BoundedVec::default(),
            compute_budget: 0,
            output_cid: None,
            attestation_cid: None,
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
            + TaskLane::max_encoded_len() // lane
            + TaskPriority::max_encoded_len() // priority
            + AccountId::max_encoded_len() // requester
            + Option::<AccountId>::max_encoded_len() // executor
            + TaskStatus::max_encoded_len() // status
            + Balance::max_encoded_len() // escrow
            + BlockNumber::max_encoded_len() // created_at
            + BlockNumber::max_encoded_len() // deadline
            + BoundedVec::<u8, MaxModelIdLen>::max_encoded_len() // model_requirements
            + BoundedVec::<u8, MaxCidLen>::max_encoded_len() // input_cid
            + u64::max_encoded_len() // compute_budget
            + Option::<BoundedVec<u8, MaxCidLen>>::max_encoded_len() // output_cid
            + Option::<BoundedVec<u8, MaxCidLen>>::max_encoded_len() // attestation_cid
    }
}
