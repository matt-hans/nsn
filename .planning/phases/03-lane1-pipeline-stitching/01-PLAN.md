# Phase 3, Plan 1: Lane 1 Pipeline Stitching

## Objective

Wire the Lane 1 task marketplace flow from submission to result delivery by implementing the `lane1` crate orchestration layer. This connects existing components: chain task events → scheduler queue management → sidecar execution → result submission → verification.

## Execution Context

**Files to read:**
- `node-core/crates/lane1/src/lib.rs` (currently empty placeholder)
- `node-core/crates/scheduler/src/state_machine.rs` (scheduler state and queue management)
- `node-core/crates/scheduler/src/task_queue.rs` (priority queue implementation)
- `node-core/crates/sidecar/src/client.rs` (gRPC client for execute_task)
- `node-core/crates/sidecar/proto/sidecar.proto` (gRPC interface definitions)
- `node-core/crates/chain-client/src/lib.rs` (chain submission and event handling)
- `nsn-chain/pallets/nsn-task-market/src/lib.rs` (task lifecycle extrinsics)
- `nsn-chain/pallets/nsn-task-market/src/types.rs` (TaskIntent, TaskStatus types)
- `node-core/crates/types/src/lib.rs` (shared types)
- `node-core/crates/lane0/src/lib.rs` (reference for crate structure pattern)

**Build commands:**
```bash
cd node-core && cargo build --release -p lane1
cd node-core && cargo test -p lane1
cd node-core && cargo clippy -p lane1 -- -D warnings
```

## Context

**Current State:**
- `lane1` crate is an empty placeholder (6 lines, just a comment)
- Scheduler has complete task queue with priority ordering (Lane0/Lane1 separation)
- Sidecar has complete gRPC service for ExecuteTask with model loading
- Task-market pallet has full lifecycle: create → assign → start → submit → verify → finalize
- Chain-client has executor registry and attestation submission
- Lane0 crate provides reference architecture (director, recipe, vortex_client, bft, publisher, error)

**Architecture Summary:**

```
Chain (nsn-task-market)              Lane1 Crate (TO IMPLEMENT)
┌─────────────────────────┐          ┌──────────────────────────────┐
│ Events:                 │          │ TaskExecutorService          │
│ - TaskCreated          ─┼──Event───▶│   ├── ChainListener          │
│ - TaskAssigned          │          │   ├── QueueManager            │
│ - TaskVerified          │          │   ├── ExecutionRunner         │
│ - TaskFailed            │          │   └── ResultSubmitter         │
└─────────────────────────┘          └──────────────────────────────┘
                                               │
              ┌────────────────────────────────┼────────────────────────┐
              ▼                                ▼                        ▼
     Scheduler                          Sidecar (gRPC)            Chain Client
     ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
     │ enqueue_lane1() │          │ execute_task()  │          │ submit_result() │
     │ next_task()     │          │ get_task_status │          │ start_task()    │
     │ complete_task() │          │ cancel_task()   │          │ attestations    │
     └─────────────────┘          └─────────────────┘          └─────────────────┘
```

**What Already Exists (no changes needed):**
- `scheduler/state_machine.rs`: enqueue_lane1(), next_task(), complete_task(), fail_task()
- `scheduler/task_queue.rs`: TaskQueue with priority ordering
- `sidecar/client.rs`: SidecarClient::execute_task(), get_task_status(), cancel_task()
- `chain-client/lib.rs`: ChainExecutorRegistry, ChainAttestationSubmitter
- `nsn-task-market`: Full pallet with task lifecycle
- `nsn-types`: Task, TaskStatus, Lane enums

**Integration Flow:**
1. Chain emits `TaskCreated` → Lane1 listens → Scheduler.enqueue_lane1()
2. Chain emits `TaskAssigned` (our node) → Lane1 calls Chain.start_task()
3. Scheduler.next_task() returns highest-priority Lane1 task
4. Lane1 calls Sidecar.execute_task() → waits for result
5. Sidecar returns output_cid → Lane1 calls Chain.submit_result()
6. Validators attest → Chain finalizes → payment distributed

## Tasks

### Task 1: Create Lane 1 Crate Module Structure

Set up the `lane1` crate with proper module organization and error types.

**Files to create/modify:**
- `node-core/crates/lane1/src/lib.rs` - Module exports and crate documentation
- `node-core/crates/lane1/src/executor.rs` - TaskExecutorService struct and state
- `node-core/crates/lane1/src/listener.rs` - Chain event listener
- `node-core/crates/lane1/src/runner.rs` - Sidecar execution wrapper
- `node-core/crates/lane1/src/submitter.rs` - Result submission to chain
- `node-core/crates/lane1/src/error.rs` - Error types with thiserror
- `node-core/crates/lane1/Cargo.toml` - Add required dependencies

**Implementation notes:**
- Follow lane0 crate structure as reference pattern
- Use thiserror for all error types
- Define separate error enums: ListenerError, ExecutionError, SubmissionError, Lane1Error
- All async functions use tokio runtime
- Use tracing for structured logging

**Acceptance criteria:**
- [ ] Module compiles with `cargo build -p lane1`
- [ ] All modules have proper documentation comments
- [ ] Error types defined for each failure mode
- [ ] TaskExecutorService struct defined with state management

**Checkpoint:** `cargo build -p lane1` succeeds

### Task 2: Implement Chain Event Listener

Implement listener that subscribes to task-market chain events and routes to scheduler.

**Implementation:**
```rust
pub struct ChainListener {
    chain_client: Arc<ChainClient>,
    scheduler: Arc<RwLock<SchedulerState>>,
    my_account: AccountId32,
    event_tx: mpsc::Sender<TaskEvent>,
}

pub enum TaskEvent {
    Created { task_id: u64, model_id: String, input_cid: String, priority: Priority },
    AssignedToMe { task_id: u64 },
    AssignedToOther { task_id: u64 },
    Verified { task_id: u64 },
    Rejected { task_id: u64 },
    Failed { task_id: u64, reason: String },
}
```

**Key methods:**
- `new()` - Initialize with chain client and scheduler reference
- `run()` - Main event loop (subscribe to chain events)
- `on_task_created()` - Enqueue to scheduler if model capability matches
- `on_task_assigned()` - Mark task for execution if assigned to us
- `on_task_verified()` - Clean up completed task
- `on_task_failed()` - Handle failure notification

**Event subscription pattern:**
- Use subxt event subscription (similar to chain-client patterns)
- Filter for nsn-task-market pallet events
- Map TaskCreated/TaskAssigned/TaskVerified/TaskFailed to internal events

**Files to modify:**
- `node-core/crates/lane1/src/listener.rs`

**Acceptance criteria:**
- [ ] Subscribes to chain events via subxt
- [ ] Routes TaskCreated to scheduler.enqueue_lane1()
- [ ] Detects when task is assigned to our node
- [ ] Handles all task lifecycle events
- [ ] Unit tests with mock chain events

**Checkpoint:** `cargo test -p lane1 -- listener` passes

### Task 3: Implement Execution Runner

Wrapper around sidecar gRPC to execute Lane 1 tasks with progress tracking.

**Implementation:**
```rust
pub struct ExecutionRunner {
    sidecar: SidecarClient,
    timeout_ms: u64,
    poll_interval_ms: u64,
}

impl ExecutionRunner {
    pub async fn execute(&self, task: &Task) -> Result<ExecutionResult, ExecutionError> {
        // 1. Call sidecar.execute_task()
        let response = self.sidecar.execute_task(ExecuteTaskRequest {
            task_id: task.id.to_string(),
            model_id: task.model_id.clone(),
            input_cid: task.input_cid.clone(),
            parameters: vec![],
            timeout_ms: self.timeout_ms,
            lane: 1,
            ..Default::default()
        }).await?;

        // 2. Check immediate success/failure
        if response.success {
            return Ok(ExecutionResult {
                task_id: task.id,
                output_cid: response.output_cid,
                execution_time_ms: response.execution_time_ms,
            });
        }

        Err(ExecutionError::SidecarFailed(response.error_message))
    }

    pub async fn poll_status(&self, task_id: &str) -> Result<TaskProgress, ExecutionError> {
        let status = self.sidecar.get_task_status(GetTaskStatusRequest {
            task_id: task_id.to_string(),
        }).await?;

        Ok(TaskProgress {
            progress: status.progress,
            stage: status.current_stage,
            status: status.status.into(),
        })
    }
}

pub struct ExecutionResult {
    pub task_id: TaskId,
    pub output_cid: String,
    pub execution_time_ms: u64,
}

pub struct TaskProgress {
    pub progress: f32,
    pub stage: String,
    pub status: TaskStatus,
}
```

**Files to modify:**
- `node-core/crates/lane1/src/runner.rs`

**Acceptance criteria:**
- [ ] Calls sidecar ExecuteTask with correct parameters
- [ ] Handles timeout and error cases
- [ ] Provides poll_status for progress monitoring
- [ ] Maps sidecar TaskStatus to internal status

**Checkpoint:** Integration test with mock sidecar passes

### Task 4: Implement Result Submitter

Submit execution results to chain and handle verification flow.

**Implementation:**
```rust
pub struct ResultSubmitter {
    chain_client: Arc<ChainClient>,
    keypair: sr25519::Pair,
}

impl ResultSubmitter {
    pub async fn start_task(&self, task_id: u64) -> Result<(), SubmissionError> {
        // Call pallet-nsn-task-market::start_task(task_id)
        self.chain_client.submit_extrinsic(
            "NsnTaskMarket",
            "start_task",
            (task_id,),
        ).await?;
        Ok(())
    }

    pub async fn submit_result(
        &self,
        task_id: u64,
        output_cid: &str,
        attestation_cid: Option<&str>,
    ) -> Result<(), SubmissionError> {
        // Call pallet-nsn-task-market::submit_result(task_id, output_cid, attestation_cid)
        self.chain_client.submit_extrinsic(
            "NsnTaskMarket",
            "submit_result",
            (task_id, output_cid.as_bytes().to_vec(), attestation_cid.map(|s| s.as_bytes().to_vec())),
        ).await?;
        Ok(())
    }

    pub async fn fail_task(&self, task_id: u64, reason: &str) -> Result<(), SubmissionError> {
        // Call pallet-nsn-task-market::fail_task(task_id, reason)
        self.chain_client.submit_extrinsic(
            "NsnTaskMarket",
            "fail_task",
            (task_id, reason.as_bytes().to_vec()),
        ).await?;
        Ok(())
    }
}
```

**Files to modify:**
- `node-core/crates/lane1/src/submitter.rs`

**Acceptance criteria:**
- [ ] Calls start_task extrinsic when beginning execution
- [ ] Submits result with output_cid after sidecar completion
- [ ] Handles fail_task for execution failures
- [ ] Proper error handling for chain submission failures

**Checkpoint:** `cargo test -p lane1 -- submitter` passes

### Task 5: Implement Task Executor Service Core

Main service that orchestrates the task execution lifecycle.

**Implementation:**
```rust
pub struct TaskExecutorService {
    state: ExecutorState,
    config: ExecutorConfig,
    listener: ChainListener,
    runner: ExecutionRunner,
    submitter: ResultSubmitter,
    scheduler: Arc<RwLock<SchedulerState>>,
    event_rx: mpsc::Receiver<TaskEvent>,
}

pub enum ExecutorState {
    Idle,                           // No active task
    Executing { task_id: u64 },    // Currently running task
    Submitting { task_id: u64 },   // Awaiting chain confirmation
}

pub struct ExecutorConfig {
    pub execution_timeout_ms: u64,  // Default 300_000 (5 minutes)
    pub max_concurrent: u32,        // Default 1 (serial execution)
    pub retry_attempts: u32,        // Default 0 (no retries for MVP)
    pub poll_interval_ms: u64,      // Default 1000
}
```

**Key methods:**
- `new()` - Initialize with all dependencies
- `run()` - Main event loop (event channel + scheduler polling)
- `process_next_task()` - Get next task from scheduler, execute, submit
- `on_task_event()` - Handle chain events (created, assigned, verified)
- `handle_execution()` - Orchestrate single task execution

**Main loop logic:**
```rust
async fn run(&mut self) -> Result<(), Lane1Error> {
    loop {
        tokio::select! {
            Some(event) = self.event_rx.recv() => {
                self.on_task_event(event).await?;
            }
            _ = tokio::time::sleep(Duration::from_millis(100)) => {
                if matches!(self.state, ExecutorState::Idle) {
                    self.process_next_task().await?;
                }
            }
        }
    }
}

async fn process_next_task(&mut self) -> Result<(), Lane1Error> {
    let scheduler = self.scheduler.read().await;
    if let Some(task) = scheduler.next_task() {
        drop(scheduler); // Release lock before async work

        self.state = ExecutorState::Executing { task_id: task.id };

        // 1. Notify chain we're starting
        self.submitter.start_task(task.id).await?;

        // 2. Execute via sidecar
        match self.runner.execute(&task).await {
            Ok(result) => {
                // 3. Submit result to chain
                self.state = ExecutorState::Submitting { task_id: task.id };
                self.submitter.submit_result(task.id, &result.output_cid, None).await?;

                // 4. Mark complete in scheduler
                let mut scheduler = self.scheduler.write().await;
                scheduler.complete_task(task.id, TaskResult::Success(result.output_cid))?;
            }
            Err(e) => {
                // Fail task on chain
                self.submitter.fail_task(task.id, &e.to_string()).await?;

                // Mark failed in scheduler
                let mut scheduler = self.scheduler.write().await;
                scheduler.fail_task(task.id, e.to_string())?;
            }
        }

        self.state = ExecutorState::Idle;
    }
    Ok(())
}
```

**Files to modify:**
- `node-core/crates/lane1/src/executor.rs`

**Acceptance criteria:**
- [ ] TaskExecutorService compiles and can be instantiated
- [ ] Main event loop handles both chain events and scheduler polling
- [ ] Execution flow: start_task → execute → submit_result → complete_task
- [ ] Failure handling: fail_task on sidecar errors
- [ ] State machine transitions correctly (Idle → Executing → Submitting → Idle)

**Checkpoint:** `cargo test -p lane1 -- executor` passes

### Task 6: Wire Module Exports and Crate Public API

Update lib.rs with complete module exports and public API.

**Implementation:**
```rust
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
//! ## Example
//!
//! ```rust,no_run
//! use nsn_lane1::{TaskExecutorService, ExecutorConfig};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = ExecutorConfig::default();
//!     let service = TaskExecutorService::new(config, /* ... */).await;
//!     service.run().await.unwrap();
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod executor;
pub mod listener;
pub mod runner;
pub mod submitter;

// Re-export main types
pub use error::{Lane1Error, ListenerError, ExecutionError, SubmissionError};
pub use executor::{TaskExecutorService, ExecutorConfig, ExecutorState};
pub use listener::{ChainListener, TaskEvent};
pub use runner::{ExecutionRunner, ExecutionResult, TaskProgress};
pub use submitter::ResultSubmitter;
```

**Files to modify:**
- `node-core/crates/lane1/src/lib.rs`

**Acceptance criteria:**
- [ ] All modules properly exported
- [ ] Re-exports for common types
- [ ] Module-level documentation
- [ ] Clippy and missing_docs warnings enabled

**Checkpoint:** `cargo doc -p lane1` generates documentation

### Task 7: Add Dependencies to Cargo.toml

Configure Cargo.toml with all required dependencies.

**Implementation:**
```toml
[package]
name = "nsn-lane1"
version = "0.1.0"
edition = "2021"
description = "Lane 1 task marketplace orchestration"
license = "Apache-2.0"

[dependencies]
# Workspace crates
nsn-types = { path = "../types" }
nsn-sidecar = { path = "../../sidecar" }
nsn-chain-client = { path = "../chain-client" }
nsn-scheduler = { path = "../scheduler" }

# Async runtime
tokio = { version = "1.0", features = ["full"] }

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Logging
tracing = "0.1"

# Chain interaction
subxt = { version = "0.35", features = ["native"] }
sp-core = { version = "28.0.0", default-features = false }

# Utilities
async-trait = "0.1"
futures = "0.3"

[dev-dependencies]
tokio-test = "0.4"
mockall = "0.12"
```

**Files to modify:**
- `node-core/crates/lane1/Cargo.toml`

**Acceptance criteria:**
- [ ] All dependencies resolve correctly
- [ ] Workspace crate references work
- [ ] Build succeeds with `cargo build -p lane1`

**Checkpoint:** `cargo check -p lane1` succeeds

### Task 8: End-to-End Integration Test

Create integration test that exercises full task execution lifecycle with mocks.

**Test scenario:**
1. Setup: Create TaskExecutorService with mock dependencies
2. Task creation: Mock chain emits TaskCreated event
3. Assignment: Mock chain emits TaskAssigned event (to our node)
4. Execution start: Verify start_task() called on chain
5. Sidecar execution: Mock sidecar returns ExecutionResult
6. Result submission: Verify submit_result() called on chain
7. Completion: Mock chain emits TaskVerified event
8. Cleanup: Verify task removed from scheduler

**Test structure:**
```rust
#[tokio::test]
async fn test_full_task_lifecycle() {
    // Setup mocks
    let mock_chain = MockChainClient::new();
    let mock_sidecar = MockSidecarClient::new();
    let scheduler = Arc::new(RwLock::new(SchedulerState::new(/* ... */)));

    // Configure expectations
    mock_chain.expect_submit_extrinsic()
        .with(eq("NsnTaskMarket"), eq("start_task"), any())
        .returning(|_, _, _| Ok(()));

    mock_sidecar.expect_execute_task()
        .returning(|_| Ok(ExecuteTaskResponse {
            success: true,
            output_cid: "QmResult123".to_string(),
            execution_time_ms: 5000,
            ..Default::default()
        }));

    mock_chain.expect_submit_extrinsic()
        .with(eq("NsnTaskMarket"), eq("submit_result"), any())
        .returning(|_, _, _| Ok(()));

    // Create service
    let config = ExecutorConfig::default();
    let service = TaskExecutorService::new_with_mocks(
        config, mock_chain, mock_sidecar, scheduler.clone()
    );

    // Simulate task creation event
    service.inject_event(TaskEvent::Created {
        task_id: 1,
        model_id: "flux-schnell".to_string(),
        input_cid: "QmInput123".to_string(),
        priority: Priority::Normal,
    }).await;

    // Simulate assignment event
    service.inject_event(TaskEvent::AssignedToMe { task_id: 1 }).await;

    // Run one iteration
    service.process_next_task().await.unwrap();

    // Verify task completed in scheduler
    let scheduler = scheduler.read().await;
    let result = scheduler.get_task_result(1);
    assert!(matches!(result, Some(TaskResult::Success(_))));
}

#[tokio::test]
async fn test_execution_failure_handling() {
    // Similar setup but mock sidecar failure
    // Verify fail_task() called on chain
    // Verify scheduler marks task as failed
}

#[tokio::test]
async fn test_preemption_during_execution() {
    // Test Lane 0 preemption cancels Lane 1 task
    // Verify cancel_task() called on sidecar
    // Verify task returned to queue or failed
}
```

**Files to create:**
- `node-core/crates/lane1/tests/integration.rs`

**Acceptance criteria:**
- [ ] Full lifecycle test passes
- [ ] All mock interactions verified
- [ ] State transitions logged correctly
- [ ] Error cases tested (sidecar failure, chain failure)
- [ ] Preemption test validates Lane 0 priority

**Checkpoint:** `cargo test -p lane1 --test integration` passes

## Verification

**Build and test:**
```bash
cd node-core && cargo build --release -p lane1
cd node-core && cargo test -p lane1
cd node-core && cargo clippy -p lane1 -- -D warnings
```

**Integration verification:**
```bash
# Run with mock dependencies
cd node-core && cargo test -p lane1 --test integration -- --nocapture
```

**Expected output:**
- All unit tests pass for each module
- Integration test completes full lifecycle
- No clippy warnings
- Compiles in release mode

## Success Criteria

- [ ] Lane 1 crate compiles with all modules
- [ ] ChainListener subscribes to task-market events
- [ ] ExecutionRunner calls sidecar successfully
- [ ] ResultSubmitter submits to chain
- [ ] TaskExecutorService orchestrates full lifecycle
- [ ] Scheduler integration (enqueue, next_task, complete)
- [ ] Error handling for all failure modes
- [ ] Integration test passes with mock dependencies
- [ ] No clippy warnings, cargo fmt clean

## Output

**Artifacts:**
- Fully implemented `node-core/crates/lane1/` crate
- TaskExecutorService with event-driven state machine
- Integration tests for task lifecycle
- Module documentation

**Dependencies for next phases:**
- Phase 4 (Viewer Web Extraction) independent - can proceed in parallel
- Phase 5 (Multi-Node E2E Simulation) will test multiple executors competing for tasks

**Open questions resolved from Phase 2:**
1. Execution timeout: 300_000ms (5 minutes) default, configurable
2. Retry attempts: 0 for MVP (single attempt, fail on error)
3. Max concurrent tasks: 1 for MVP (serial execution)
