// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.

//! Types for pallet-nsn-task-market

use frame_support::pallet_prelude::*;
/// Task lane designation (Lane 0 vs Lane 1).
pub use nsn_primitives::Lane as TaskLane;
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
    /// Executor submitted output awaiting verification
    Submitted,
    /// Task is in active verification window
    PendingVerification,
    /// Task verified successfully
    Verified,
    /// Task rejected after failed verification or deadline
    Rejected,
}

/// Attestation from a validator for a task result.
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
#[scale_info(skip_type_params(AccountId, MaxCidLen))]
pub struct TaskAttestation<AccountId, MaxCidLen>
where
    AccountId: Encode + Decode + Clone + PartialEq + Eq + core::fmt::Debug + TypeInfo,
    MaxCidLen: Get<u32>,
{
    /// Validator account submitting attestation (signed origin).
    pub validator: AccountId,
    /// Verification score in range [0, 100].
    pub score: u8,
    /// Optional CID for attestation proof.
    pub attestation_cid: Option<BoundedVec<u8, MaxCidLen>>,
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

/// Renderer metadata registered for task execution.
#[derive(Clone, Encode, Decode, DecodeWithMemTracking, Eq, PartialEq, RuntimeDebug, TypeInfo)]
pub struct RendererInfo<BlockNumber> {
    /// Lane supported by this renderer
    pub lane: TaskLane,
    /// Whether renderer is deterministic (required for Lane 0)
    pub deterministic: bool,
    /// Maximum latency budget in milliseconds
    pub max_latency_ms: u32,
    /// VRAM required for execution (in MB)
    pub vram_required_mb: u32,
    /// Block number when registered
    pub registered_at: BlockNumber,
}

impl<BlockNumber: Default> Default for RendererInfo<BlockNumber> {
    fn default() -> Self {
        Self {
            lane: TaskLane::Lane1,
            deterministic: false,
            max_latency_ms: 0,
            vram_required_mb: 0,
            registered_at: BlockNumber::default(),
        }
    }
}

impl<BlockNumber: MaxEncodedLen> MaxEncodedLen for RendererInfo<BlockNumber> {
    fn max_encoded_len() -> usize {
        TaskLane::max_encoded_len()
            + bool::max_encoded_len()
            + u32::max_encoded_len()
            + u32::max_encoded_len()
            + BlockNumber::max_encoded_len()
    }
}

/// A compute task intent in the task market
///
/// Generic over AccountId, Balance, BlockNumber, and bounded length types for flexibility.
#[derive(Encode, Decode, DecodeWithMemTracking, Eq, PartialEq, RuntimeDebug, TypeInfo)]
#[scale_info(skip_type_params(MaxModelIdLen, MaxCidLen))]
pub struct TaskIntent<AccountId, Balance, BlockNumber, MaxModelIdLen, MaxCidLen>
where
    AccountId: Encode + Decode + Clone + PartialEq + Eq + core::fmt::Debug + TypeInfo,
    Balance: Encode + Decode + Clone + PartialEq + Eq + core::fmt::Debug + TypeInfo,
    BlockNumber: Encode + Decode + Clone + PartialEq + Eq + core::fmt::Debug + TypeInfo,
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
        AccountId: Default + Encode + Decode + Clone + PartialEq + Eq + core::fmt::Debug + TypeInfo,
        Balance: Default + Encode + Decode + Clone + PartialEq + Eq + core::fmt::Debug + TypeInfo,
        BlockNumber: Default + Encode + Decode + Clone + PartialEq + Eq + core::fmt::Debug + TypeInfo,
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

// Manual Clone implementation for TaskIntent (no bounds needed on length type params)
impl<
        AccountId: Encode + Decode + Clone + PartialEq + Eq + core::fmt::Debug + TypeInfo,
        Balance: Encode + Decode + Clone + PartialEq + Eq + core::fmt::Debug + TypeInfo,
        BlockNumber: Encode + Decode + Clone + PartialEq + Eq + core::fmt::Debug + TypeInfo,
        MaxModelIdLen: Get<u32>,
        MaxCidLen: Get<u32>,
    > Clone for TaskIntent<AccountId, Balance, BlockNumber, MaxModelIdLen, MaxCidLen>
{
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            lane: self.lane.clone(),
            priority: self.priority.clone(),
            requester: self.requester.clone(),
            executor: self.executor.clone(),
            status: self.status.clone(),
            escrow: self.escrow.clone(),
            created_at: self.created_at.clone(),
            deadline: self.deadline.clone(),
            model_requirements: self.model_requirements.clone(),
            input_cid: self.input_cid.clone(),
            compute_budget: self.compute_budget,
            output_cid: self.output_cid.clone(),
            attestation_cid: self.attestation_cid.clone(),
        }
    }
}

// Manual MaxEncodedLen for TaskIntent
impl<
        AccountId: MaxEncodedLen + Encode + Decode + Clone + PartialEq + Eq + core::fmt::Debug + TypeInfo,
        Balance: MaxEncodedLen + Encode + Decode + Clone + PartialEq + Eq + core::fmt::Debug + TypeInfo,
        BlockNumber: MaxEncodedLen + Encode + Decode + Clone + PartialEq + Eq + core::fmt::Debug + TypeInfo,
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
