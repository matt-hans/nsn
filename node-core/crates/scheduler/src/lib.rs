//! Task scheduler for NSN nodes
//!
//! Manages model loading, task queuing, and lane switching between:
//! - **Lane 0**: Video generation (priority, latency-sensitive, uses Flux/LivePortrait/Kokoro)
//! - **Lane 1**: General AI compute (LLM inference, image gen, etc.)
//!
//! ## Architecture
//!
//! The scheduler implements a dual-lane architecture with epoch-based director elections:
//!
//! 1. **On-Deck notification** (2 minutes before epoch): Start draining Lane 1 tasks
//! 2. **Epoch start**: Switch to Lane 0, load video models if needed
//! 3. **Epoch end**: Switch back to Lane 1, resume general compute
//!
//! ## Example
//!
//! ```rust
//! use nsn_scheduler::{SchedulerState, Priority};
//! use nsn_types::NodeState;
//!
//! // Create scheduler
//! let mut scheduler = SchedulerState::new();
//!
//! // Initialize
//! scheduler.transition(NodeState::LoadingModels).unwrap();
//! scheduler.transition(NodeState::Idle).unwrap();
//!
//! // Enqueue tasks
//! let task_id = scheduler.enqueue_lane0(
//!     "flux-schnell".to_string(),
//!     "QmInputCID".to_string()
//! ).unwrap();
//!
//! // Get next task (Lane 0 has priority)
//! if let Some(task) = scheduler.next_task() {
//!     let handle = scheduler.start_task(&task).unwrap();
//!     // ... execute task ...
//! }
//! ```

pub mod epoch;
pub mod redundancy;
pub mod state_machine;
pub mod task_queue;

#[cfg(test)]
mod tests;

// Re-export main types for convenience
pub use epoch::{EpochEvent, EpochTracker, ON_DECK_LEAD_TIME_SECS};
pub use redundancy::{
    AttestationBundle, AttestationError, AttestationSubmitter, ConsensusFailureReason,
    ConsensusMode, ConsensusOutcome, ConsensusPolicy, ConsensusRecord, DualAttestationSubmitter,
    ExecutionResult, ExecutorInfo, ExecutorRegistry, NoopAttestationSubmitter,
    P2pAttestationSubmitter, RedundancyConfig, RedundancyError, RedundancyMetrics,
    RedundantAssignment, RedundantScheduler, RedundantTask, RedundantTaskStatus,
    StaticExecutorRegistry,
};
pub use state_machine::{SchedulerError, SchedulerState, SchedulerStats, TaskHandle};
pub use task_queue::{Lane, Priority, QueueError, Task, TaskId, TaskQueue, TaskResult};
