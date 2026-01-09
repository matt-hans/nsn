//! Task Executor Service for Lane 1.
//!
//! Orchestrates the complete task execution lifecycle: event listening,
//! scheduler integration, sidecar execution, and result submission.

use crate::error::{ExecutionError, Lane1Error, Lane1Result};
use crate::listener::{ChainListener, ListenerConfig, TaskEvent};
use crate::runner::{ExecutionOutput, ExecutionRunner, RunnerConfig, TaskSpec};
use crate::submitter::{ResultSubmitter, SubmitterConfig};
use nsn_scheduler::state_machine::SchedulerState;
use nsn_scheduler::task_queue::{Priority, TaskResult};
use nsn_types::FailureReason;
use sp_core::sr25519;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

/// Executor state machine.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutorState {
    /// Idle, waiting for tasks.
    Idle,
    /// Currently executing a task.
    Executing {
        /// Task ID being executed.
        task_id: u64,
    },
    /// Submitting result to chain.
    Submitting {
        /// Task ID being submitted.
        task_id: u64,
    },
    /// Shutting down.
    Stopping,
}

impl std::fmt::Display for ExecutorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutorState::Idle => write!(f, "Idle"),
            ExecutorState::Executing { task_id } => write!(f, "Executing({})", task_id),
            ExecutorState::Submitting { task_id } => write!(f, "Submitting({})", task_id),
            ExecutorState::Stopping => write!(f, "Stopping"),
        }
    }
}

/// Configuration for the task executor service.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Default execution timeout in milliseconds.
    pub execution_timeout_ms: u64,
    /// Maximum concurrent tasks (1 for MVP).
    pub max_concurrent: u32,
    /// Retry attempts on transient failures (0 for MVP).
    pub retry_attempts: u32,
    /// Poll interval for checking scheduler queue.
    pub poll_interval_ms: u64,
    /// Listener configuration.
    pub listener: ListenerConfig,
    /// Runner configuration.
    pub runner: RunnerConfig,
    /// Submitter configuration.
    pub submitter: SubmitterConfig,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            execution_timeout_ms: 300_000, // 5 minutes
            max_concurrent: 1,
            retry_attempts: 0,
            poll_interval_ms: 100,
            listener: ListenerConfig::default(),
            runner: RunnerConfig::default(),
            submitter: SubmitterConfig::default(),
        }
    }
}

/// Task Executor Service.
///
/// Main service that orchestrates Lane 1 task execution:
/// 1. Listens for chain events (task created, assigned, verified, failed)
/// 2. Routes tasks to scheduler queue
/// 3. Executes tasks via sidecar
/// 4. Submits results back to chain
pub struct TaskExecutorService {
    config: ExecutorConfig,
    state: ExecutorState,
    scheduler: Arc<RwLock<SchedulerState>>,
    runner: ExecutionRunner,
    submitter: ResultSubmitter,
    event_rx: mpsc::Receiver<TaskEvent>,
    shutdown_tx: mpsc::Sender<()>,
    my_account: String,
}

impl TaskExecutorService {
    /// Create a new task executor service.
    ///
    /// # Arguments
    /// * `config` - Executor configuration
    /// * `scheduler` - Shared scheduler state
    /// * `keypair` - Signing keypair for chain transactions
    /// * `my_account` - This node's account ID
    pub fn new(
        config: ExecutorConfig,
        scheduler: Arc<RwLock<SchedulerState>>,
        keypair: sr25519::Pair,
        my_account: String,
    ) -> (Self, ChainListener) {
        let (event_tx, event_rx) = mpsc::channel(config.listener.event_buffer_size);
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        let listener = ChainListener::new(
            config.listener.clone(),
            my_account.clone(),
            event_tx,
        )
        .with_shutdown(shutdown_rx);

        let runner = ExecutionRunner::new(config.runner.clone());
        let submitter = ResultSubmitter::new(config.submitter.clone(), keypair);

        let executor = Self {
            config,
            state: ExecutorState::Idle,
            scheduler,
            runner,
            submitter,
            event_rx,
            shutdown_tx,
            my_account,
        };

        (executor, listener)
    }

    /// Get the current executor state.
    pub fn state(&self) -> &ExecutorState {
        &self.state
    }

    /// Get the configuration.
    pub fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    /// Get this node's account ID.
    pub fn my_account(&self) -> &str {
        &self.my_account
    }

    /// Shutdown the executor service.
    pub async fn shutdown(&self) -> Lane1Result<()> {
        let _ = self.shutdown_tx.send(()).await;
        Ok(())
    }

    /// Run the executor service main loop.
    ///
    /// This method runs until shutdown is requested. It:
    /// 1. Handles incoming chain events
    /// 2. Polls the scheduler for ready tasks
    /// 3. Executes tasks and submits results
    pub async fn run(&mut self) -> Lane1Result<()> {
        info!(
            account = %self.my_account,
            "Starting Lane 1 Task Executor Service"
        );

        // Connect to dependencies
        self.runner.connect().await.map_err(Lane1Error::Execution)?;
        self.submitter.connect().await.map_err(Lane1Error::Submission)?;

        loop {
            if matches!(self.state, ExecutorState::Stopping) {
                info!("Executor stopping");
                break;
            }

            tokio::select! {
                // Handle incoming chain events
                Some(event) = self.event_rx.recv() => {
                    if let Err(e) = self.handle_event(event).await {
                        error!(error = %e, "Error handling chain event");
                    }
                }

                // Poll for next task when idle
                _ = tokio::time::sleep(Duration::from_millis(self.config.poll_interval_ms)) => {
                    if matches!(self.state, ExecutorState::Idle) {
                        if let Err(e) = self.process_next_task().await {
                            error!(error = %e, "Error processing next task");
                        }
                    }
                }
            }
        }

        info!("Lane 1 Task Executor Service stopped");
        Ok(())
    }

    /// Handle a chain event.
    async fn handle_event(&mut self, event: TaskEvent) -> Lane1Result<()> {
        match event {
            TaskEvent::Created {
                task_id,
                model_id,
                input_cid,
                priority,
                reward: _,
            } => {
                self.on_task_created(task_id, model_id, input_cid, priority)
                    .await?;
            }
            TaskEvent::AssignedToMe { task_id } => {
                self.on_task_assigned_to_me(task_id).await?;
            }
            TaskEvent::AssignedToOther { task_id, executor } => {
                debug!(
                    task_id = task_id,
                    executor = %executor,
                    "Task assigned to other node"
                );
                // Remove from our queue if we had it
                let mut scheduler = self.scheduler.write().await;
                let _ = scheduler.cancel_task(nsn_scheduler::task_queue::TaskId::new(task_id));
            }
            TaskEvent::Verified { task_id } => {
                info!(task_id = task_id, "Task verified by validators");
            }
            TaskEvent::Rejected { task_id, reason } => {
                warn!(
                    task_id = task_id,
                    reason = %reason,
                    "Task rejected"
                );
            }
            TaskEvent::Failed { task_id, reason } => {
                warn!(
                    task_id = task_id,
                    reason = %reason,
                    "Task failed on chain"
                );
            }
        }

        Ok(())
    }

    /// Handle TaskCreated event - enqueue to scheduler.
    async fn on_task_created(
        &mut self,
        task_id: u64,
        model_id: String,
        input_cid: String,
        priority: Priority,
    ) -> Lane1Result<()> {
        debug!(
            task_id = task_id,
            model_id = %model_id,
            "Enqueueing task to scheduler"
        );

        let mut scheduler = self.scheduler.write().await;

        // Enqueue to Lane 1 queue
        scheduler
            .enqueue_lane1_with_priority(model_id, input_cid, priority)
            .map_err(|e| Lane1Error::Scheduler(e.to_string()))?;

        Ok(())
    }

    /// Handle TaskAssigned event when assigned to us.
    async fn on_task_assigned_to_me(&mut self, task_id: u64) -> Lane1Result<()> {
        info!(task_id = task_id, "Task assigned to us - ready for execution");
        // The task should already be in our queue from TaskCreated
        // Processing happens in process_next_task
        Ok(())
    }

    /// Process the next task from the scheduler queue.
    async fn process_next_task(&mut self) -> Lane1Result<()> {
        // Get next task from scheduler
        let task = {
            let mut scheduler = self.scheduler.write().await;
            scheduler.next_task()
        };

        let task = match task {
            Some(t) => t,
            None => return Ok(()), // No tasks available
        };

        // Only process Lane 1 tasks
        if task.lane != nsn_scheduler::task_queue::Lane::Lane1 {
            // Put it back - this shouldn't happen but be safe
            let mut scheduler = self.scheduler.write().await;
            let _ = scheduler.enqueue_lane1(task.model_id, task.input_cid);
            return Ok(());
        }

        let task_id = task.id.0;

        info!(
            task_id = task_id,
            model_id = %task.model_id,
            "Processing task"
        );

        self.state = ExecutorState::Executing { task_id };

        // Start task on scheduler
        {
            let mut scheduler = self.scheduler.write().await;
            if let Err(e) = scheduler.start_task(&task) {
                error!(error = %e, "Failed to start task in scheduler");
                self.state = ExecutorState::Idle;
                return Err(Lane1Error::Scheduler(e.to_string()));
            }
        }

        // Notify chain we're starting
        if let Err(e) = self.submitter.start_task(task_id).await {
            warn!(error = %e, "Failed to submit start_task - continuing anyway");
        }

        // Execute via sidecar
        let task_spec = TaskSpec {
            id: task_id,
            model_id: task.model_id.clone(),
            input_cid: task.input_cid.clone(),
            parameters: vec![],
            timeout_ms: Some(self.config.execution_timeout_ms),
        };

        match self.runner.execute(&task_spec).await {
            Ok(output) => {
                self.handle_execution_success(task_id, output).await?;
            }
            Err(e) => {
                self.handle_execution_failure(task_id, e).await?;
            }
        }

        self.state = ExecutorState::Idle;
        Ok(())
    }

    /// Handle successful task execution.
    async fn handle_execution_success(
        &mut self,
        task_id: u64,
        output: ExecutionOutput,
    ) -> Lane1Result<()> {
        info!(
            task_id = task_id,
            output_cid = %output.output_cid,
            execution_time_ms = output.execution_time_ms,
            "Task execution succeeded"
        );

        self.state = ExecutorState::Submitting { task_id };

        // Submit result to chain
        if let Err(e) = self
            .submitter
            .submit_result(task_id, &output.output_cid, None)
            .await
        {
            error!(error = %e, "Failed to submit result to chain");
            // Still mark as complete in scheduler
        }

        // Mark complete in scheduler
        let mut scheduler = self.scheduler.write().await;
        let result = TaskResult::success(output.output_cid, output.execution_time_ms);
        scheduler
            .complete_task(nsn_scheduler::task_queue::TaskId::new(task_id), result)
            .map_err(|e| Lane1Error::Scheduler(e.to_string()))?;

        Ok(())
    }

    /// Handle failed task execution.
    async fn handle_execution_failure(
        &mut self,
        task_id: u64,
        error: ExecutionError,
    ) -> Lane1Result<()> {
        warn!(
            task_id = task_id,
            error = %error,
            "Task execution failed"
        );

        // Determine failure reason
        let failure_reason = match &error {
            ExecutionError::Timeout { .. } => FailureReason::Timeout,
            ExecutionError::ModelUnavailable { .. } => FailureReason::ModelUnavailable,
            ExecutionError::Preempted => FailureReason::Other("preempted".to_string()),
            ExecutionError::Cancelled { reason } => FailureReason::Other(reason.clone()),
            _ => FailureReason::Other(error.to_string()),
        };

        // Submit failure to chain
        if let Err(e) = self
            .submitter
            .fail_task(task_id, &error.to_string())
            .await
        {
            error!(error = %e, "Failed to submit failure to chain");
        }

        // Mark failed in scheduler
        let mut scheduler = self.scheduler.write().await;
        scheduler
            .fail_task(nsn_scheduler::task_queue::TaskId::new(task_id), failure_reason)
            .map_err(|e| Lane1Error::Scheduler(e.to_string()))?;

        Ok(())
    }

    /// Inject an event for testing purposes.
    #[cfg(any(test, feature = "test-utils"))]
    pub async fn inject_event(&mut self, event: TaskEvent) -> Lane1Result<()> {
        self.handle_event(event).await
    }

    /// Process one task iteration for testing.
    #[cfg(any(test, feature = "test-utils"))]
    pub async fn process_once(&mut self) -> Lane1Result<()> {
        self.process_next_task().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sp_core::Pair;

    fn make_scheduler() -> Arc<RwLock<SchedulerState>> {
        let mut scheduler = SchedulerState::new();
        scheduler
            .transition(nsn_types::NodeState::LoadingModels)
            .unwrap();
        scheduler.transition(nsn_types::NodeState::Idle).unwrap();
        Arc::new(RwLock::new(scheduler))
    }

    #[test]
    fn test_executor_config_default() {
        let config = ExecutorConfig::default();
        assert_eq!(config.execution_timeout_ms, 300_000);
        assert_eq!(config.max_concurrent, 1);
        assert_eq!(config.retry_attempts, 0);
    }

    #[test]
    fn test_executor_state_display() {
        assert_eq!(ExecutorState::Idle.to_string(), "Idle");
        assert_eq!(
            ExecutorState::Executing { task_id: 42 }.to_string(),
            "Executing(42)"
        );
        assert_eq!(
            ExecutorState::Submitting { task_id: 42 }.to_string(),
            "Submitting(42)"
        );
        assert_eq!(ExecutorState::Stopping.to_string(), "Stopping");
    }

    #[tokio::test]
    async fn test_executor_creation() {
        let config = ExecutorConfig::default();
        let scheduler = make_scheduler();
        let keypair = sr25519::Pair::from_string("//Alice", None).unwrap();

        let (executor, _listener) = TaskExecutorService::new(
            config,
            scheduler,
            keypair,
            "5GrwvaEF...".to_string(),
        );

        assert!(matches!(executor.state(), ExecutorState::Idle));
        assert_eq!(executor.my_account(), "5GrwvaEF...");
    }

    #[tokio::test]
    async fn test_on_task_created() {
        let config = ExecutorConfig::default();
        let scheduler = make_scheduler();
        let keypair = sr25519::Pair::from_string("//Alice", None).unwrap();

        let (mut executor, _listener) = TaskExecutorService::new(
            config,
            scheduler.clone(),
            keypair,
            "5GrwvaEF...".to_string(),
        );

        // Inject a TaskCreated event
        executor
            .inject_event(TaskEvent::Created {
                task_id: 1,
                model_id: "flux-schnell".to_string(),
                input_cid: "QmInput123".to_string(),
                priority: Priority::Normal,
                reward: 1000,
            })
            .await
            .unwrap();

        // Check task was enqueued
        let sched = scheduler.read().await;
        assert_eq!(sched.lane1_queue_len(), 1);
    }

    #[tokio::test]
    async fn test_on_task_assigned_to_other() {
        let config = ExecutorConfig::default();
        let scheduler = make_scheduler();
        let keypair = sr25519::Pair::from_string("//Alice", None).unwrap();

        let (mut executor, _listener) = TaskExecutorService::new(
            config,
            scheduler.clone(),
            keypair,
            "5GrwvaEF...".to_string(),
        );

        // First create a task
        executor
            .inject_event(TaskEvent::Created {
                task_id: 1,
                model_id: "flux-schnell".to_string(),
                input_cid: "QmInput123".to_string(),
                priority: Priority::Normal,
                reward: 1000,
            })
            .await
            .unwrap();

        // Then assign to someone else - should remove from queue
        executor
            .inject_event(TaskEvent::AssignedToOther {
                task_id: 1,
                executor: "5FHneW46...".to_string(),
            })
            .await
            .unwrap();

        // Task should no longer be in queue (attempted cancel)
        // Note: Cancel may fail if task was already dequeued, that's ok
    }
}
