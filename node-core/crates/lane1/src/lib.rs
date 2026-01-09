//! # Lane 1 Task Marketplace Orchestration
//!
//! This crate implements the off-chain orchestration layer for Lane 1
//! (general AI compute) task execution. It bridges on-chain task marketplace
//! events to off-chain sidecar execution.
//!
//! ## Architecture
//!
//! ```text
//! Chain Events → ChainListener → Scheduler → ExecutionRunner → ResultSubmitter
//! ```
//!
//! ## Components
//!
//! - [`TaskExecutorService`]: Main service coordinating task lifecycle
//! - [`ChainListener`]: Subscribes to chain events, routes to scheduler
//! - [`ExecutionRunner`]: Executes tasks via sidecar gRPC
//! - [`ResultSubmitter`]: Submits results and status updates to chain
//!
//! ## Task Lifecycle
//!
//! 1. **TaskCreated** event received → enqueue to scheduler
//! 2. **TaskAssigned** event (to us) → mark ready for execution
//! 3. Scheduler.next_task() → get highest-priority Lane 1 task
//! 4. Call chain.start_task() → notify execution start
//! 5. Call sidecar.execute_task() → run AI workload
//! 6. Call chain.submit_result() → submit output CID
//! 7. Validators verify → chain finalizes → payment distributed
//!
//! ## Example
//!
//! ```rust,ignore
//! use nsn_lane1::{TaskExecutorService, ExecutorConfig};
//! use nsn_scheduler::state_machine::SchedulerState;
//! use sp_core::{sr25519, Pair};
//! use std::sync::Arc;
//! use tokio::sync::RwLock;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create scheduler
//!     let scheduler = Arc::new(RwLock::new(SchedulerState::new()));
//!
//!     // Create keypair for signing
//!     let keypair = sr25519::Pair::from_string("//Alice", None).unwrap();
//!
//!     // Create executor service
//!     let config = ExecutorConfig::default();
//!     let (mut executor, mut listener) = TaskExecutorService::new(
//!         config,
//!         scheduler,
//!         keypair,
//!         "5GrwvaEF...".to_string(),
//!     );
//!
//!     // Run listener and executor in parallel
//!     tokio::spawn(async move {
//!         listener.run().await.unwrap();
//!     });
//!
//!     executor.run().await.unwrap();
//! }
//! ```
//!
//! ## Configuration
//!
//! Default configuration values:
//! - Execution timeout: 300,000ms (5 minutes)
//! - Max concurrent tasks: 1 (serial execution for MVP)
//! - Retry attempts: 0 (no retries for MVP)
//! - Poll interval: 100ms

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod executor;
pub mod listener;
pub mod runner;
pub mod submitter;

// Re-export main types
pub use error::{ExecutionError, Lane1Error, Lane1Result, ListenerError, SubmissionError};
pub use executor::{ExecutorConfig, ExecutorState, TaskExecutorService};
pub use listener::{ChainListener, ListenerConfig, TaskEvent};
pub use runner::{ExecutionOutput, ExecutionRunner, RunnerConfig, TaskProgress, TaskSpec};
pub use submitter::{ResultSubmitter, SubmitterConfig};
