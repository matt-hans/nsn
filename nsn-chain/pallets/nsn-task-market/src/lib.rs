// Copyright 2024 Neural Sovereign Network
// This file is part of NSN Chain.
//
// NSN Chain is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! # NSN Task Market Pallet
//!
//! Off-chain compute task scheduling with escrow for the Neural Sovereign Network.
//!
//! ## Overview
//!
//! This pallet implements:
//! - Task intent creation with escrow deposit
//! - Task assignment to executors
//! - Task result submission with verification gating
//! - Task failure handling with escrow return
//!
//! ## Interface
//!
//! ### Dispatchable Functions
//!
//! - `create_task_intent`: Create a new task with escrow deposit
//! - `accept_assignment`: Executor claims an open task
//! - `submit_result`: Submit task output for verification (no payout)
//! - `submit_attestation`: Validators submit verification attestations
//! - `finalize_task`: Finalize task after quorum or deadline
//! - `fail_task`: Mark task as failed and return escrow

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub use pallet::*;

use alloc::vec::Vec;
use frame_support::pallet_prelude::DispatchResult;

mod types;
pub use types::{
    FailReason, RendererInfo, TaskAttestation, TaskIntent, TaskLane, TaskPriority, TaskStatus,
};

#[cfg(test)]
mod mock;
#[cfg(test)]
mod tests;

#[cfg(feature = "runtime-benchmarks")]
mod benchmarking;

pub mod weights;
pub use weights::WeightInfo;

/// Trait for providing eligible nodes for task assignment per lane.
pub trait LaneNodeProvider<AccountId, Balance> {
    /// Returns a bounded list of eligible nodes and their stake weights for a lane.
    fn eligible_nodes(lane: TaskLane, max: u32) -> Vec<(AccountId, Balance)>;
    /// Returns true if the account is eligible for the given lane.
    fn is_eligible(account: &AccountId, lane: TaskLane) -> bool;
}

/// Trait for updating reputation based on task outcomes.
pub trait ReputationUpdater<AccountId> {
    /// Record a task outcome for reputation.
    fn record_task_result(account: &AccountId, success: bool);
}

/// Trait for slashing task abandonment during director transition.
pub trait TaskSlashHandler<AccountId, Balance> {
    fn slash_for_abandonment(account: &AccountId, amount: Balance) -> DispatchResult;
}

/// Trait for verifying validator eligibility for attestations.
pub trait ValidatorProvider<AccountId> {
    /// Returns true if the account is an eligible validator.
    fn is_validator(account: &AccountId) -> bool;
}

#[frame_support::pallet]
pub mod pallet {
    use super::*;
    use frame_support::{
        pallet_prelude::*,
        traits::{
            Currency, EnsureOrigin, ExistenceRequirement, Randomness, ReservableCurrency,
            StorageVersion,
        },
    };
    use frame_system::pallet_prelude::*;
    use sp_runtime::traits::{CheckedAdd, Hash, SaturatedConversion, Saturating, Zero};

    /// Balance type alias using ReservableCurrency
    pub type BalanceOf<T> =
        <<T as Config>::Currency as Currency<<T as frame_system::Config>::AccountId>>::Balance;

    /// Task intent type alias
    pub type TaskIntentOf<T> = TaskIntent<
        <T as frame_system::Config>::AccountId,
        BalanceOf<T>,
        BlockNumberFor<T>,
        <T as Config>::MaxModelIdLen,
        <T as Config>::MaxCidLen,
    >;

    /// Renderer info type alias
    pub type RendererInfoOf<T> = RendererInfo<BlockNumberFor<T>>;

    /// Task attestation type alias
    pub type TaskAttestationOf<T> =
        TaskAttestation<<T as frame_system::Config>::AccountId, <T as Config>::MaxCidLen>;

    /// The in-code storage version.
    const STORAGE_VERSION: StorageVersion = StorageVersion::new(0);

    #[pallet::pallet]
    #[pallet::storage_version(STORAGE_VERSION)]
    pub struct Pallet<T>(_);

    /// Configuration trait for the NSN Task Market pallet
    #[pallet::config]
    pub trait Config: frame_system::Config<RuntimeEvent: From<Event<Self>>> {
        /// The currency type for escrow operations
        type Currency: ReservableCurrency<Self::AccountId>;

        /// Maximum number of pending (open) tasks
        #[pallet::constant]
        type MaxPendingTasks: Get<u32>;

        /// Maximum number of Lane 1 assigned tasks tracked for preemption
        #[pallet::constant]
        type MaxAssignedLane1Tasks: Get<u32>;

        /// Maximum number of candidate reserve nodes to consider per assignment
        #[pallet::constant]
        type MaxAssignmentCandidates: Get<u32>;

        /// Maximum expired tasks to process per block
        #[pallet::constant]
        type MaxExpiredPerBlock: Get<u32>;

        /// Maximum Lane 1 preemptions to process per block
        #[pallet::constant]
        type MaxPreemptionsPerBlock: Get<u32>;

        /// Maximum length of model identifier
        #[pallet::constant]
        type MaxModelIdLen: Get<u32>;

        /// Maximum length of content identifier (CID)
        #[pallet::constant]
        type MaxCidLen: Get<u32>;

        /// Maximum number of registered renderers
        #[pallet::constant]
        type MaxRegisteredRenderers: Get<u32>;

        /// Maximum latency for Lane 0 renderers (ms)
        #[pallet::constant]
        type MaxLane0LatencyMs: Get<u32>;

        /// Maximum latency for Lane 1 renderers (ms)
        #[pallet::constant]
        type MaxLane1LatencyMs: Get<u32>;

        /// Maximum renderer VRAM budget (MB)
        #[pallet::constant]
        type MaxRendererVramMb: Get<u32>;

        /// Minimum escrow amount required for task creation
        #[pallet::constant]
        type MinEscrow: Get<BalanceOf<Self>>;

        /// Maximum number of attestations stored per task
        #[pallet::constant]
        type MaxAttestations: Get<u32>;

        /// Maximum number of pending verification tasks tracked
        #[pallet::constant]
        type MaxPendingVerifications: Get<u32>;

        /// Verification quorum required to finalize a task
        #[pallet::constant]
        type VerificationQuorum: Get<u32>;

        /// Verification period (in blocks) after result submission
        #[pallet::constant]
        type VerificationPeriod: Get<BlockNumberFor<Self>>;

        /// Minimum average attestation score required for verification
        #[pallet::constant]
        type MinAttestationScore: Get<u8>;

        /// Slash amount for failed verification
        #[pallet::constant]
        type VerificationFailureSlash: Get<BalanceOf<Self>>;

        /// Treasury account for fee distribution
        type TreasuryAccount: Get<Self::AccountId>;

        /// Slash amount for task abandonment during director transition
        #[pallet::constant]
        type TaskAbandonmentSlash: Get<BalanceOf<Self>>;

        /// Lane node provider (loose coupling to stake pallet)
        type LaneNodeProvider: LaneNodeProvider<Self::AccountId, BalanceOf<Self>>;

        /// Reputation updater (loose coupling to reputation pallet)
        type ReputationUpdater: ReputationUpdater<Self::AccountId>;

        /// Slash handler for task abandonment (loose coupling to stake pallet)
        type TaskSlashHandler: TaskSlashHandler<Self::AccountId, BalanceOf<Self>>;

        /// Validator provider for result attestations
        type ValidatorProvider: ValidatorProvider<Self::AccountId>;

        /// Weight information
        type WeightInfo: WeightInfo;

        /// Randomness source for assignment selection
        type Randomness: Randomness<Self::Hash, BlockNumberFor<Self>>;

        /// Origin allowed to register/deregister renderers
        type RendererRegistrarOrigin: EnsureOrigin<Self::RuntimeOrigin>;
    }

    /// Next task ID counter
    ///
    /// Monotonically increasing counter for unique task IDs.
    #[pallet::storage]
    #[pallet::getter(fn next_task_id)]
    pub type NextTaskId<T: Config> = StorageValue<_, u64, ValueQuery>;

    /// All tasks by ID
    ///
    /// Maps task ID to TaskIntent struct containing all task details.
    ///
    /// # Storage Key
    /// Twox64Concat(u64) - task IDs are not user-controlled
    #[pallet::storage]
    #[pallet::getter(fn tasks)]
    pub type Tasks<T: Config> = StorageMap<_, Twox64Concat, u64, TaskIntentOf<T>, OptionQuery>;

    /// Attestations submitted for a task
    #[pallet::storage]
    #[pallet::getter(fn attestations)]
    pub type Attestations<T: Config> = StorageMap<
        _,
        Twox64Concat,
        u64,
        BoundedVec<TaskAttestationOf<T>, T::MaxAttestations>,
        ValueQuery,
    >;

    /// Required quorum for task verification
    #[pallet::storage]
    #[pallet::getter(fn required_quorum)]
    pub type RequiredQuorum<T: Config> = StorageMap<_, Twox64Concat, u64, u32, OptionQuery>;

    /// Verification deadline per task
    #[pallet::storage]
    #[pallet::getter(fn verification_deadline)]
    pub type VerificationDeadline<T: Config> =
        StorageMap<_, Twox64Concat, u64, BlockNumberFor<T>, OptionQuery>;

    /// Queue of tasks pending verification
    #[pallet::storage]
    #[pallet::getter(fn pending_verifications)]
    pub type PendingVerificationTasks<T: Config> =
        StorageValue<_, BoundedVec<u64, T::MaxPendingVerifications>, ValueQuery>;

    /// Queue of Lane 0 open task IDs (priority ordered)
    #[pallet::storage]
    #[pallet::getter(fn open_lane0_tasks)]
    pub type OpenLane0Tasks<T: Config> =
        StorageValue<_, BoundedVec<u64, T::MaxPendingTasks>, ValueQuery>;

    /// Queue of Lane 1 open task IDs (priority ordered)
    #[pallet::storage]
    #[pallet::getter(fn open_lane1_tasks)]
    pub type OpenLane1Tasks<T: Config> =
        StorageValue<_, BoundedVec<u64, T::MaxPendingTasks>, ValueQuery>;

    /// Assigned Lane 1 tasks for preemption tracking
    #[pallet::storage]
    #[pallet::getter(fn assigned_lane1_tasks)]
    pub type AssignedLane1Tasks<T: Config> =
        StorageValue<_, BoundedVec<u64, T::MaxAssignedLane1Tasks>, ValueQuery>;

    /// Registered renderers allowed to execute tasks
    #[pallet::storage]
    #[pallet::getter(fn renderer_registry)]
    pub type RendererRegistry<T: Config> = StorageValue<
        _,
        BoundedBTreeMap<
            BoundedVec<u8, T::MaxModelIdLen>,
            RendererInfoOf<T>,
            T::MaxRegisteredRenderers,
        >,
        ValueQuery,
    >;

    /// Events emitted by the pallet
    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        /// A new task intent was created
        TaskCreated {
            task_id: u64,
            requester: T::AccountId,
            escrow: BalanceOf<T>,
            deadline: BlockNumberFor<T>,
            lane: TaskLane,
            priority: TaskPriority,
            model_requirements: BoundedVec<u8, T::MaxModelIdLen>,
        },
        /// A task was assigned to an executor
        TaskAssigned {
            task_id: u64,
            executor: T::AccountId,
        },
        /// A task started execution
        TaskStarted {
            task_id: u64,
            executor: T::AccountId,
        },
        /// A task result was submitted for verification
        TaskSubmitted {
            task_id: u64,
            executor: T::AccountId,
            output_cid: BoundedVec<u8, T::MaxCidLen>,
            attestation_cid: Option<BoundedVec<u8, T::MaxCidLen>>,
            verification_deadline: BlockNumberFor<T>,
            required_quorum: u32,
        },
        /// A validator submitted an attestation
        AttestationSubmitted {
            task_id: u64,
            validator: T::AccountId,
            score: u8,
        },
        /// A task was verified successfully
        TaskVerified {
            task_id: u64,
            executor: T::AccountId,
            payment: BalanceOf<T>,
            average_score: u8,
        },
        /// A task was rejected after verification or deadline
        TaskRejected {
            task_id: u64,
            executor: Option<T::AccountId>,
            average_score: Option<u8>,
            deadline_exceeded: bool,
        },
        /// A task failed
        TaskFailed { task_id: u64, reason: FailReason },
        /// A task expired due to deadline
        TaskExpired { task_id: u64 },
        /// A renderer was registered
        RendererRegistered {
            renderer_id: BoundedVec<u8, T::MaxModelIdLen>,
            lane: TaskLane,
            deterministic: bool,
            max_latency_ms: u32,
            vram_required_mb: u32,
        },
        /// A renderer was deregistered
        RendererDeregistered {
            renderer_id: BoundedVec<u8, T::MaxModelIdLen>,
        },
    }

    /// Errors returned by the pallet
    #[pallet::error]
    pub enum Error<T> {
        /// Task not found
        TaskNotFound,
        /// Task is not in Open status
        TaskNotOpen,
        /// Task is not in Assigned status
        TaskNotAssigned,
        /// Task is not in Executing status
        TaskNotExecuting,
        /// Task is not in Submitted/PendingVerification status
        TaskNotSubmitted,
        /// Only the assigned executor can submit/fail the task
        NotExecutor,
        /// Only validators may submit attestations
        NotValidator,
        /// Duplicate attestation from the same validator
        DuplicateAttestation,
        /// Too many attestations submitted
        TooManyAttestations,
        /// Attestation score must be between 0 and 100
        InvalidAttestationScore,
        /// Verification quorum configuration is invalid
        InvalidVerificationQuorum,
        /// Verification deadline has passed
        VerificationDeadlinePassed,
        /// Verification quorum not yet met
        VerificationPending,
        /// Only the requester can cancel an open task
        NotRequester,
        /// Too many pending tasks in the queue
        TooManyPendingTasks,
        /// Too many assigned Lane 1 tasks
        TooManyAssignedLane1Tasks,
        /// Too many pending verification tasks
        TooManyPendingVerifications,
        /// Arithmetic overflow
        Overflow,
        /// Insufficient balance for escrow
        InsufficientBalance,
        /// Escrow amount is below the minimum required
        InsufficientEscrow,
        /// Invalid deadline (must be in the future)
        InvalidDeadline,
        /// Task already assigned
        TaskAlreadyAssigned,
        /// No eligible reserve nodes for assignment
        NoEligibleExecutors,
        /// Lane 0 tasks pending; Lane 1 assignment is blocked
        Lane0Priority,
        /// Task burn failed (insufficient balance)
        BurnFailed,
        /// Renderer registry is full
        RendererRegistryFull,
        /// Renderer is already registered
        RendererAlreadyRegistered,
        /// Renderer ID is empty
        InvalidRendererId,
        /// Renderer is not registered
        RendererNotRegistered,
        /// Renderer lane does not match task lane
        RendererLaneMismatch,
        /// Renderer latency exceeds lane constraints
        RendererLatencyExceeded,
        /// Renderer must be deterministic for Lane 0
        RendererNotDeterministic,
        /// Renderer VRAM requirement exceeds limit
        RendererVramExceeded,
        /// Missing attestation for Lane 0 completion
        MissingAttestation,
        /// Attestation CID is invalid (empty)
        InvalidAttestationCid,
    }

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
        /// Block initialization - expire tasks and enforce Lane 0 preemption (bounded).
        fn on_initialize(n: BlockNumberFor<T>) -> Weight {
            let mut total_weight = Weight::zero();
            total_weight = total_weight.saturating_add(Self::expire_open_tasks(n));
            total_weight = total_weight.saturating_add(Self::expire_pending_verifications(n));
            total_weight = total_weight.saturating_add(Self::preempt_lane1_if_needed());
            total_weight
        }
    }

    /// Extrinsic calls
    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Create a new task intent with escrow deposit
        ///
        /// # Arguments
        /// * `lane` - Task lane (Lane 0 or Lane 1)
        /// * `priority` - Priority within the lane queue
        /// * `model_requirements` - AI model requirements for task execution
        /// * `input_cid` - Content identifier for input data
        /// * `compute_budget` - Maximum compute budget for the task
        /// * `deadline_blocks` - Number of blocks until deadline (relative)
        /// * `escrow_amount` - Amount to escrow for payment
        ///
        /// # Errors
        /// * `TooManyPendingTasks` - Open tasks queue is full
        /// * `InsufficientBalance` - Not enough balance for escrow
        /// * `InsufficientEscrow` - Escrow amount below minimum
        /// * `InvalidDeadline` - Deadline must be > 0
        #[pallet::call_index(0)]
        #[pallet::weight(T::WeightInfo::create_task_intent())]
        pub fn create_task_intent(
            origin: OriginFor<T>,
            lane: TaskLane,
            priority: TaskPriority,
            model_requirements: BoundedVec<u8, T::MaxModelIdLen>,
            input_cid: BoundedVec<u8, T::MaxCidLen>,
            compute_budget: u64,
            deadline_blocks: BlockNumberFor<T>,
            escrow_amount: BalanceOf<T>,
        ) -> DispatchResult {
            let requester = ensure_signed(origin)?;

            // Validate deadline is in the future
            ensure!(
                deadline_blocks > BlockNumberFor::<T>::zero(),
                Error::<T>::InvalidDeadline
            );

            // Validate renderer/model requirements
            ensure!(
                !model_requirements.is_empty(),
                Error::<T>::InvalidRendererId
            );
            Self::ensure_renderer_allowed(&model_requirements, lane.clone())?;

            // Validate minimum escrow
            ensure!(
                escrow_amount >= T::MinEscrow::get(),
                Error::<T>::InsufficientEscrow
            );

            // Reserve escrow from requester
            T::Currency::reserve(&requester, escrow_amount)
                .map_err(|_| Error::<T>::InsufficientBalance)?;

            // Get next task ID and increment counter
            let task_id = NextTaskId::<T>::get();
            NextTaskId::<T>::put(task_id.checked_add(1).ok_or(Error::<T>::Overflow)?);

            // Calculate absolute deadline
            let current_block = <frame_system::Pallet<T>>::block_number();
            let deadline = current_block
                .checked_add(&deadline_blocks)
                .ok_or(Error::<T>::Overflow)?;

            // Create task intent
            let task = TaskIntent {
                id: task_id,
                lane: lane.clone(),
                priority: priority.clone(),
                requester: requester.clone(),
                executor: None,
                status: TaskStatus::Open,
                escrow: escrow_amount,
                created_at: current_block,
                deadline,
                model_requirements: model_requirements.clone(),
                input_cid,
                compute_budget,
                output_cid: None,
                attestation_cid: None,
            };

            // Add to lane-specific open queue (bounded, priority ordered)
            Self::enqueue_open_task(task_id, &task)?;

            // Store task
            Tasks::<T>::insert(task_id, task);

            Self::deposit_event(Event::TaskCreated {
                task_id,
                requester,
                escrow: escrow_amount,
                deadline,
                lane,
                priority,
                model_requirements,
            });

            Ok(())
        }

        /// Assign an open task to an eligible node (stake-weighted)
        ///
        /// Any signed origin may trigger assignment; executor is chosen
        /// deterministically based on stake weights and lane eligibility.
        ///
        /// # Arguments
        /// * `task_id` - ID of the task to assign
        ///
        /// # Errors
        /// * `TaskNotFound` - Task does not exist
        /// * `TaskNotOpen` - Task is not in Open status
        /// * `NoEligibleExecutors` - No reserve nodes available for lane
        /// * `Lane0Priority` - Lane 1 assignment blocked by pending Lane 0 tasks
        #[pallet::call_index(1)]
        #[pallet::weight(T::WeightInfo::accept_assignment())]
        pub fn accept_assignment(origin: OriginFor<T>, task_id: u64) -> DispatchResult {
            let _caller = ensure_signed(origin)?;

            // Get and validate task
            Tasks::<T>::try_mutate(task_id, |maybe_task| -> DispatchResult {
                let task = maybe_task.as_mut().ok_or(Error::<T>::TaskNotFound)?;

                // Verify task is open
                ensure!(task.status == TaskStatus::Open, Error::<T>::TaskNotOpen);

                // Lane 0 preemption: block Lane 1 assignment if any Lane 0 tasks pending
                if task.lane == TaskLane::Lane1 && !OpenLane0Tasks::<T>::get().is_empty() {
                    return Err(Error::<T>::Lane0Priority.into());
                }

                // Select executor (stake-weighted)
                let executor = Self::select_executor(task.lane.clone())?;

                // Assign executor and update status
                task.executor = Some(executor.clone());
                task.status = TaskStatus::Assigned;

                // Remove from open tasks queue
                Self::remove_from_open_queue(task.lane.clone(), task_id)?;

                // Track assigned Lane 1 tasks for preemption
                if task.lane == TaskLane::Lane1 {
                    AssignedLane1Tasks::<T>::try_mutate(|assigned| {
                        assigned
                            .try_push(task_id)
                            .map_err(|_| Error::<T>::TooManyAssignedLane1Tasks)
                    })?;
                }

                Self::deposit_event(Event::TaskAssigned { task_id, executor });

                Ok(())
            })
        }

        /// Start executing an assigned task
        ///
        /// Only the assigned executor can start a task.
        ///
        /// # Arguments
        /// * `task_id` - ID of the task to start
        #[pallet::call_index(2)]
        #[pallet::weight(T::WeightInfo::start_task())]
        pub fn start_task(origin: OriginFor<T>, task_id: u64) -> DispatchResult {
            let caller = ensure_signed(origin)?;

            Tasks::<T>::try_mutate(task_id, |maybe_task| -> DispatchResult {
                let task = maybe_task.as_mut().ok_or(Error::<T>::TaskNotFound)?;

                ensure!(
                    task.status == TaskStatus::Assigned,
                    Error::<T>::TaskNotAssigned
                );

                let executor = task.executor.as_ref().ok_or(Error::<T>::TaskNotAssigned)?;
                ensure!(&caller == executor, Error::<T>::NotExecutor);

                // Lane 0 preemption: prevent Lane 1 execution if Lane 0 tasks are pending
                if task.lane == TaskLane::Lane1 && !OpenLane0Tasks::<T>::get().is_empty() {
                    return Err(Error::<T>::Lane0Priority.into());
                }

                task.status = TaskStatus::Executing;

                Self::deposit_event(Event::TaskStarted {
                    task_id,
                    executor: caller,
                });

                Ok(())
            })
        }

        /// Submit a task result for verification (escrow remains reserved)
        ///
        /// Only the assigned executor can submit a result.
        ///
        /// # Arguments
        /// * `task_id` - ID of the task to submit
        /// * `output_cid` - Content identifier for the output data
        /// * `attestation_cid` - Optional attestation CID for result verification
        ///
        /// # Errors
        /// * `TaskNotFound` - Task does not exist
        /// * `TaskNotAssigned` - Task is not in Assigned/Executing status
        /// * `NotExecutor` - Caller is not the assigned executor
        #[pallet::call_index(3)]
        #[pallet::weight(T::WeightInfo::submit_result())]
        pub fn submit_result(
            origin: OriginFor<T>,
            task_id: u64,
            output_cid: BoundedVec<u8, T::MaxCidLen>,
            attestation_cid: Option<BoundedVec<u8, T::MaxCidLen>>,
        ) -> DispatchResult {
            let caller = ensure_signed(origin)?;

            // Get and validate task
            Tasks::<T>::try_mutate(task_id, |maybe_task| -> DispatchResult {
                let task = maybe_task.as_mut().ok_or(Error::<T>::TaskNotFound)?;

                // Verify task is assigned
                ensure!(
                    task.status == TaskStatus::Assigned || task.status == TaskStatus::Executing,
                    Error::<T>::TaskNotAssigned
                );

                // Verify caller is the executor
                let executor = task.executor.as_ref().ok_or(Error::<T>::TaskNotAssigned)?;
                ensure!(&caller == executor, Error::<T>::NotExecutor);

                if task.lane == TaskLane::Lane0 && attestation_cid.is_none() {
                    return Err(Error::<T>::MissingAttestation.into());
                }
                if let Some(ref cid) = attestation_cid {
                    ensure!(!cid.is_empty(), Error::<T>::InvalidAttestationCid);
                }

                let required_quorum = T::VerificationQuorum::get();
                ensure!(required_quorum > 0, Error::<T>::InvalidVerificationQuorum);
                ensure!(
                    required_quorum <= T::MaxAttestations::get(),
                    Error::<T>::InvalidVerificationQuorum
                );

                let now = <frame_system::Pallet<T>>::block_number();
                let verification_deadline = now
                    .checked_add(&T::VerificationPeriod::get())
                    .ok_or(Error::<T>::Overflow)?;

                // Store verification metadata
                RequiredQuorum::<T>::insert(task_id, required_quorum);
                VerificationDeadline::<T>::insert(task_id, verification_deadline);
                Attestations::<T>::insert(task_id, BoundedVec::default());

                PendingVerificationTasks::<T>::try_mutate(|queue| {
                    if queue.contains(&task_id) {
                        return Ok::<(), DispatchError>(());
                    }
                    queue
                        .try_push(task_id)
                        .map_err(|_| Error::<T>::TooManyPendingVerifications)?;
                    Ok::<(), DispatchError>(())
                })?;

                // Update status and store output
                task.status = TaskStatus::Submitted;
                task.output_cid = Some(output_cid.clone());
                task.attestation_cid = attestation_cid.clone();

                // Remove from assigned Lane 1 list if present
                if task.lane == TaskLane::Lane1 {
                    AssignedLane1Tasks::<T>::mutate(|assigned| {
                        if let Some(pos) = assigned.iter().position(|&id| id == task_id) {
                            assigned.remove(pos);
                        }
                    });
                }

                Self::deposit_event(Event::TaskSubmitted {
                    task_id,
                    executor: caller,
                    output_cid,
                    attestation_cid,
                    verification_deadline,
                    required_quorum,
                });

                Ok(())
            })
        }

        /// Fail a task and return escrow to requester
        ///
        /// Authorization rules:
        /// - If task is Assigned: both executor AND requester can fail it
        /// - If task is Open: only requester can fail it (cancel)
        ///
        /// # Arguments
        /// * `task_id` - ID of the task to fail
        /// * `reason` - Reason for failure
        ///
        /// # Errors
        /// * `TaskNotFound` - Task does not exist
        /// * `NotExecutor` - Caller is not authorized to fail this task
        /// * `NotRequester` - Caller is not the requester (for Open tasks)
        #[pallet::call_index(4)]
        #[pallet::weight(T::WeightInfo::fail_task())]
        pub fn fail_task(origin: OriginFor<T>, task_id: u64, reason: FailReason) -> DispatchResult {
            let caller = ensure_signed(origin)?;

            // Get and validate task
            Tasks::<T>::try_mutate(task_id, |maybe_task| -> DispatchResult {
                let task = maybe_task.as_mut().ok_or(Error::<T>::TaskNotFound)?;

                match task.status {
                    TaskStatus::Open => {
                        // Only requester can cancel an open task
                        ensure!(caller == task.requester, Error::<T>::NotRequester);

                        // Remove from lane-specific open queue
                        Self::remove_from_open_queue(task.lane.clone(), task_id)?;
                    }
                    TaskStatus::Assigned | TaskStatus::Executing => {
                        // Both executor and requester can fail an assigned task
                        let executor = task.executor.as_ref().ok_or(Error::<T>::TaskNotAssigned)?;
                        ensure!(
                            &caller == executor || caller == task.requester,
                            Error::<T>::NotExecutor
                        );
                    }
                    TaskStatus::Submitted | TaskStatus::PendingVerification => {
                        let executor = task.executor.as_ref().ok_or(Error::<T>::TaskNotAssigned)?;
                        ensure!(
                            &caller == executor || caller == task.requester,
                            Error::<T>::NotExecutor
                        );
                        Self::clear_verification_state(task_id);
                    }
                    TaskStatus::Completed
                    | TaskStatus::Verified
                    | TaskStatus::Rejected
                    | TaskStatus::Failed
                    | TaskStatus::Expired => {
                        // Cannot fail a completed, already failed, or expired task
                        return Err(Error::<T>::TaskNotAssigned.into());
                    }
                }

                // Return escrow to requester
                T::Currency::unreserve(&task.requester, task.escrow);

                // Update status
                task.status = TaskStatus::Failed;

                // Remove from assigned Lane 1 list if present
                if task.lane == TaskLane::Lane1 {
                    AssignedLane1Tasks::<T>::mutate(|assigned| {
                        if let Some(pos) = assigned.iter().position(|&id| id == task_id) {
                            assigned.remove(pos);
                        }
                    });
                }

                Self::deposit_event(Event::TaskFailed {
                    task_id,
                    reason: reason.clone(),
                });

                // Reputation/slash handling for abandonment
                if matches!(reason, FailReason::Preempted | FailReason::ExecutorFailed) {
                    if let Some(executor) = task.executor.as_ref() {
                        let _ = T::TaskSlashHandler::slash_for_abandonment(
                            executor,
                            T::TaskAbandonmentSlash::get(),
                        );
                        T::ReputationUpdater::record_task_result(executor, false);
                    }
                }

                Ok(())
            })
        }

        /// Register a renderer for task execution
        ///
        /// Only the configured registrar origin can register renderers.
        #[pallet::call_index(5)]
        #[pallet::weight(T::WeightInfo::register_renderer())]
        pub fn register_renderer(
            origin: OriginFor<T>,
            renderer_id: BoundedVec<u8, T::MaxModelIdLen>,
            lane: TaskLane,
            deterministic: bool,
            max_latency_ms: u32,
            vram_required_mb: u32,
        ) -> DispatchResult {
            T::RendererRegistrarOrigin::ensure_origin(origin)?;

            ensure!(!renderer_id.is_empty(), Error::<T>::InvalidRendererId);
            ensure!(
                vram_required_mb <= T::MaxRendererVramMb::get(),
                Error::<T>::RendererVramExceeded
            );

            let lane_limit = match lane {
                TaskLane::Lane0 => T::MaxLane0LatencyMs::get(),
                TaskLane::Lane1 => T::MaxLane1LatencyMs::get(),
            };
            ensure!(
                max_latency_ms <= lane_limit,
                Error::<T>::RendererLatencyExceeded
            );

            if lane == TaskLane::Lane0 {
                ensure!(deterministic, Error::<T>::RendererNotDeterministic);
            }

            RendererRegistry::<T>::try_mutate(|registry| -> Result<(), DispatchError> {
                if registry.contains_key(&renderer_id) {
                    return Err(Error::<T>::RendererAlreadyRegistered.into());
                }
                let info = RendererInfo {
                    lane: lane.clone(),
                    deterministic,
                    max_latency_ms,
                    vram_required_mb,
                    registered_at: <frame_system::Pallet<T>>::block_number(),
                };
                registry
                    .try_insert(renderer_id.clone(), info)
                    .map_err(|_| Error::<T>::RendererRegistryFull)?;
                Ok(())
            })?;

            Self::deposit_event(Event::RendererRegistered {
                renderer_id,
                lane,
                deterministic,
                max_latency_ms,
                vram_required_mb,
            });

            Ok(())
        }

        /// Deregister a renderer
        #[pallet::call_index(6)]
        #[pallet::weight(T::WeightInfo::deregister_renderer())]
        pub fn deregister_renderer(
            origin: OriginFor<T>,
            renderer_id: BoundedVec<u8, T::MaxModelIdLen>,
        ) -> DispatchResult {
            T::RendererRegistrarOrigin::ensure_origin(origin)?;

            RendererRegistry::<T>::try_mutate(|registry| -> Result<(), DispatchError> {
                if registry.remove(&renderer_id).is_none() {
                    return Err(Error::<T>::RendererNotRegistered.into());
                }
                Ok(())
            })?;

            Self::deposit_event(Event::RendererDeregistered { renderer_id });
            Ok(())
        }

        /// Submit a verification attestation for a task result
        ///
        /// Only validators can submit attestations.
        #[pallet::call_index(7)]
        #[pallet::weight(T::WeightInfo::submit_attestation())]
        pub fn submit_attestation(
            origin: OriginFor<T>,
            task_id: u64,
            score: u8,
            attestation_cid: Option<BoundedVec<u8, T::MaxCidLen>>,
        ) -> DispatchResult {
            let caller = ensure_signed(origin)?;

            ensure!(
                T::ValidatorProvider::is_validator(&caller),
                Error::<T>::NotValidator
            );
            ensure!(score <= 100, Error::<T>::InvalidAttestationScore);
            if let Some(ref cid) = attestation_cid {
                ensure!(!cid.is_empty(), Error::<T>::InvalidAttestationCid);
            }

            let now = <frame_system::Pallet<T>>::block_number();

            Tasks::<T>::try_mutate(task_id, |maybe_task| -> DispatchResult {
                let task = maybe_task.as_mut().ok_or(Error::<T>::TaskNotFound)?;
                ensure!(
                    matches!(task.status, TaskStatus::Submitted | TaskStatus::PendingVerification),
                    Error::<T>::TaskNotSubmitted
                );

                let deadline =
                    VerificationDeadline::<T>::get(task_id).ok_or(Error::<T>::TaskNotSubmitted)?;
                ensure!(now <= deadline, Error::<T>::VerificationDeadlinePassed);

                Attestations::<T>::try_mutate(task_id, |attestations| -> DispatchResult {
                    if attestations.iter().any(|a| a.validator == caller) {
                        return Err(Error::<T>::DuplicateAttestation.into());
                    }

                    let entry = TaskAttestation {
                        validator: caller.clone(),
                        score,
                        attestation_cid: attestation_cid.clone(),
                    };
                    attestations
                        .try_push(entry)
                        .map_err(|_| Error::<T>::TooManyAttestations)?;
                    Ok(())
                })?;

                if task.status == TaskStatus::Submitted {
                    task.status = TaskStatus::PendingVerification;
                }

                Self::deposit_event(Event::AttestationSubmitted {
                    task_id,
                    validator: caller.clone(),
                    score,
                });

                let required =
                    RequiredQuorum::<T>::get(task_id).ok_or(Error::<T>::TaskNotSubmitted)?;
                let attestations = Attestations::<T>::get(task_id);

                if attestations.len() as u32 >= required {
                    let average = Self::average_attestation_score(&attestations);
                    if average >= T::MinAttestationScore::get() {
                        Self::finalize_verified(task_id, task, average)?;
                    } else {
                        Self::finalize_rejected(task_id, task, Some(average), false)?;
                    }
                }

                Ok(())
            })
        }

        /// Finalize a task once verification quorum is met or deadline exceeded
        #[pallet::call_index(8)]
        #[pallet::weight(T::WeightInfo::finalize_task())]
        pub fn finalize_task(origin: OriginFor<T>, task_id: u64) -> DispatchResult {
            let _caller = ensure_signed(origin)?;
            let now = <frame_system::Pallet<T>>::block_number();

            Tasks::<T>::try_mutate(task_id, |maybe_task| -> DispatchResult {
                let task = maybe_task.as_mut().ok_or(Error::<T>::TaskNotFound)?;
                ensure!(
                    matches!(task.status, TaskStatus::Submitted | TaskStatus::PendingVerification),
                    Error::<T>::TaskNotSubmitted
                );

                let deadline =
                    VerificationDeadline::<T>::get(task_id).ok_or(Error::<T>::TaskNotSubmitted)?;
                let required =
                    RequiredQuorum::<T>::get(task_id).ok_or(Error::<T>::TaskNotSubmitted)?;
                let attestations = Attestations::<T>::get(task_id);

                if now > deadline {
                    Self::finalize_rejected(task_id, task, None, true)?;
                    return Ok(());
                }

                ensure!(
                    attestations.len() as u32 >= required,
                    Error::<T>::VerificationPending
                );

                let average = Self::average_attestation_score(&attestations);
                if average >= T::MinAttestationScore::get() {
                    Self::finalize_verified(task_id, task, average)?;
                } else {
                    Self::finalize_rejected(task_id, task, Some(average), false)?;
                }

                Ok(())
            })
        }
    }

    // Helper functions
    impl<T: Config> Pallet<T> {
        /// Get the number of open tasks across both lanes
        pub fn open_task_count() -> u32 {
            OpenLane0Tasks::<T>::get().len() as u32 + OpenLane1Tasks::<T>::get().len() as u32
        }

        /// Check if a task exists
        pub fn task_exists(task_id: u64) -> bool {
            Tasks::<T>::contains_key(task_id)
        }

        fn average_attestation_score(
            attestations: &BoundedVec<TaskAttestationOf<T>, T::MaxAttestations>,
        ) -> u8 {
            if attestations.is_empty() {
                return 0;
            }
            let total: u32 = attestations.iter().map(|a| a.score as u32).sum();
            (total / attestations.len() as u32) as u8
        }

        fn clear_verification_state(task_id: u64) {
            Attestations::<T>::remove(task_id);
            RequiredQuorum::<T>::remove(task_id);
            VerificationDeadline::<T>::remove(task_id);
            PendingVerificationTasks::<T>::mutate(|queue| {
                if let Some(pos) = queue.iter().position(|&id| id == task_id) {
                    queue.remove(pos);
                }
            });
        }

        fn finalize_verified(
            task_id: u64,
            task: &mut TaskIntentOf<T>,
            average_score: u8,
        ) -> DispatchResult {
            let executor = task.executor.as_ref().ok_or(Error::<T>::TaskNotAssigned)?;

            // Unreserve escrow
            let escrow = task.escrow;
            T::Currency::unreserve(&task.requester, escrow);

            // Fee split: Node 70%, Treasury 20%, Burn 10%
            let node_share = escrow.saturating_mul(70u32.into()) / 100u32.into();
            let treasury_share = escrow.saturating_mul(20u32.into()) / 100u32.into();
            let burn_share = escrow
                .saturating_sub(node_share)
                .saturating_sub(treasury_share);

            // Transfer node and treasury shares
            T::Currency::transfer(
                &task.requester,
                executor,
                node_share,
                ExistenceRequirement::AllowDeath,
            )?;
            T::Currency::transfer(
                &task.requester,
                &T::TreasuryAccount::get(),
                treasury_share,
                ExistenceRequirement::AllowDeath,
            )?;

            // Burn remainder (must fully burn or revert)
            let (_imbalance, remaining) = T::Currency::slash(&task.requester, burn_share);
            ensure!(remaining.is_zero(), Error::<T>::BurnFailed);

            task.status = TaskStatus::Verified;

            Self::clear_verification_state(task_id);

            Self::deposit_event(Event::TaskVerified {
                task_id,
                executor: executor.clone(),
                payment: node_share,
                average_score,
            });

            T::ReputationUpdater::record_task_result(executor, true);

            Ok(())
        }

        fn finalize_rejected(
            task_id: u64,
            task: &mut TaskIntentOf<T>,
            average_score: Option<u8>,
            deadline_exceeded: bool,
        ) -> DispatchResult {
            let executor = task.executor.clone();

            // Return escrow to requester
            T::Currency::unreserve(&task.requester, task.escrow);

            if let Some(ref account) = executor {
                let _ = T::TaskSlashHandler::slash_for_abandonment(
                    account,
                    T::VerificationFailureSlash::get(),
                );
                T::ReputationUpdater::record_task_result(account, false);
            }

            task.status = TaskStatus::Rejected;

            Self::clear_verification_state(task_id);

            Self::deposit_event(Event::TaskRejected {
                task_id,
                executor,
                average_score,
                deadline_exceeded,
            });

            Ok(())
        }

        fn ensure_renderer_allowed(
            renderer_id: &BoundedVec<u8, T::MaxModelIdLen>,
            lane: TaskLane,
        ) -> Result<(), DispatchError> {
            let registry = RendererRegistry::<T>::get();
            let info = registry
                .get(renderer_id)
                .ok_or(Error::<T>::RendererNotRegistered)?;

            ensure!(info.lane == lane, Error::<T>::RendererLaneMismatch);

            let lane_limit = match lane {
                TaskLane::Lane0 => T::MaxLane0LatencyMs::get(),
                TaskLane::Lane1 => T::MaxLane1LatencyMs::get(),
            };
            ensure!(
                info.max_latency_ms <= lane_limit,
                Error::<T>::RendererLatencyExceeded
            );
            ensure!(
                info.vram_required_mb <= T::MaxRendererVramMb::get(),
                Error::<T>::RendererVramExceeded
            );
            if lane == TaskLane::Lane0 {
                ensure!(info.deterministic, Error::<T>::RendererNotDeterministic);
            }
            Ok(())
        }

        fn enqueue_open_task(task_id: u64, task: &TaskIntentOf<T>) -> DispatchResult {
            match task.lane {
                TaskLane::Lane0 => OpenLane0Tasks::<T>::try_mutate(|queue| {
                    Self::insert_by_priority(queue, task_id, task)
                }),
                TaskLane::Lane1 => OpenLane1Tasks::<T>::try_mutate(|queue| {
                    Self::insert_by_priority(queue, task_id, task)
                }),
            }
        }

        fn remove_from_open_queue(lane: TaskLane, task_id: u64) -> DispatchResult {
            match lane {
                TaskLane::Lane0 => OpenLane0Tasks::<T>::mutate(|queue| {
                    if let Some(pos) = queue.iter().position(|&id| id == task_id) {
                        queue.remove(pos);
                    }
                }),
                TaskLane::Lane1 => OpenLane1Tasks::<T>::mutate(|queue| {
                    if let Some(pos) = queue.iter().position(|&id| id == task_id) {
                        queue.remove(pos);
                    }
                }),
            }
            Ok(())
        }

        fn insert_by_priority(
            queue: &mut BoundedVec<u64, T::MaxPendingTasks>,
            task_id: u64,
            task: &TaskIntentOf<T>,
        ) -> DispatchResult {
            let insert_pos = queue
                .iter()
                .position(|existing_id| {
                    if let Some(existing) = Tasks::<T>::get(existing_id) {
                        Self::is_higher_priority(task, &existing)
                    } else {
                        false
                    }
                })
                .unwrap_or(queue.len());

            queue
                .try_insert(insert_pos, task_id)
                .map_err(|_| Error::<T>::TooManyPendingTasks)?;
            Ok(())
        }

        fn is_higher_priority(a: &TaskIntentOf<T>, b: &TaskIntentOf<T>) -> bool {
            let pa = a.priority.weight();
            let pb = b.priority.weight();
            if pa != pb {
                return pa > pb;
            }
            if a.deadline != b.deadline {
                return a.deadline < b.deadline;
            }
            a.id < b.id
        }

        fn select_executor(lane: TaskLane) -> Result<T::AccountId, DispatchError> {
            let candidates =
                T::LaneNodeProvider::eligible_nodes(lane, T::MaxAssignmentCandidates::get());
            ensure!(!candidates.is_empty(), Error::<T>::NoEligibleExecutors);

            // Compute total weight (stake)
            let mut total_weight: u128 = 0;
            let mut weighted: Vec<(T::AccountId, u128)> = Vec::new();
            for (account, stake) in candidates {
                let weight: u128 = stake.saturated_into::<u128>();
                if weight == 0 {
                    continue;
                }
                total_weight = total_weight.saturating_add(weight);
                weighted.push((account, weight));
            }

            ensure!(total_weight > 0, Error::<T>::NoEligibleExecutors);

            // Randomness from configured source + block/lane
            let now = <frame_system::Pallet<T>>::block_number();
            let (rand, _) = T::Randomness::random(&b"task-market-assignment"[..]);
            let seed = T::Hashing::hash_of(&(rand, now, lane.as_u8(), total_weight));
            let mut seed_bytes = [0u8; 16];
            seed_bytes.copy_from_slice(&seed.as_ref()[0..16]);
            let mut pick = u128::from_le_bytes(seed_bytes) % total_weight;

            for (account, weight) in weighted {
                if pick < weight {
                    return Ok(account);
                }
                pick = pick.saturating_sub(weight);
            }

            Err(Error::<T>::NoEligibleExecutors.into())
        }

        fn expire_open_tasks(now: BlockNumberFor<T>) -> Weight {
            let mut total_weight = Weight::zero();
            let max_expired = T::MaxExpiredPerBlock::get() as usize;
            let mut expired = 0usize;

            let mut process_queue = |queue: &mut BoundedVec<u64, T::MaxPendingTasks>| {
                let mut idx = 0usize;
                while idx < queue.len() && expired < max_expired {
                    let task_id = queue[idx];
                    if let Some(mut task) = Tasks::<T>::get(task_id) {
                        if task.deadline <= now {
                            // Expire task
                            task.status = TaskStatus::Expired;
                            T::Currency::unreserve(&task.requester, task.escrow);
                            Tasks::<T>::insert(task_id, task);
                            queue.remove(idx);
                            expired += 1;
                            total_weight =
                                total_weight.saturating_add(T::DbWeight::get().reads_writes(2, 2));
                            Self::deposit_event(Event::TaskExpired { task_id });
                            continue;
                        }
                    }
                    idx += 1;
                }
            };

            OpenLane0Tasks::<T>::mutate(|queue| process_queue(queue));
            OpenLane1Tasks::<T>::mutate(|queue| process_queue(queue));

            total_weight
        }

        fn expire_pending_verifications(now: BlockNumberFor<T>) -> Weight {
            let mut total_weight = Weight::zero();
            let max_expired = T::MaxExpiredPerBlock::get() as usize;
            let mut expired_ids: Vec<u64> = Vec::new();

            PendingVerificationTasks::<T>::mutate(|queue| {
                let mut idx = 0usize;
                while idx < queue.len() && expired_ids.len() < max_expired {
                    let task_id = queue[idx];
                    if let Some(deadline) = VerificationDeadline::<T>::get(task_id) {
                        if deadline <= now {
                            queue.remove(idx);
                            expired_ids.push(task_id);
                            continue;
                        }
                    }
                    idx += 1;
                }
            });

            for task_id in expired_ids {
                if let Some(mut task) = Tasks::<T>::get(task_id) {
                    if matches!(
                        task.status,
                        TaskStatus::Submitted | TaskStatus::PendingVerification
                    ) {
                        let _ = Self::finalize_rejected(task_id, &mut task, None, true);
                        Tasks::<T>::insert(task_id, task);
                        total_weight =
                            total_weight.saturating_add(T::DbWeight::get().reads_writes(2, 2));
                    }
                }
            }

            total_weight
        }

        fn preempt_lane1_if_needed() -> Weight {
            if OpenLane0Tasks::<T>::get().is_empty() {
                return Weight::zero();
            }

            let mut total_weight = Weight::zero();
            let max_preemptions = T::MaxPreemptionsPerBlock::get() as usize;

            AssignedLane1Tasks::<T>::mutate(|assigned| {
                let mut count = 0usize;
                while count < max_preemptions && !assigned.is_empty() {
                    let task_id = assigned.remove(0);
                    if let Some(mut task) = Tasks::<T>::get(task_id) {
                        if matches!(task.status, TaskStatus::Assigned | TaskStatus::Executing) {
                            task.status = TaskStatus::Failed;
                            T::Currency::unreserve(&task.requester, task.escrow);
                            Tasks::<T>::insert(task_id, task.clone());
                            Self::deposit_event(Event::TaskFailed {
                                task_id,
                                reason: FailReason::Preempted,
                            });

                            if let Some(executor) = task.executor.as_ref() {
                                let _ = T::TaskSlashHandler::slash_for_abandonment(
                                    executor,
                                    T::TaskAbandonmentSlash::get(),
                                );
                                T::ReputationUpdater::record_task_result(executor, false);
                            }
                            total_weight =
                                total_weight.saturating_add(T::DbWeight::get().reads_writes(2, 2));
                        }
                    }
                    count += 1;
                }
            });

            total_weight
        }
    }
}
