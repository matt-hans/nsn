//! Execution runner for Lane 1 tasks.
//!
//! Wraps the sidecar gRPC client to execute tasks with progress tracking.

use crate::error::{ExecutionError, ExecutionResult};
use async_trait::async_trait;
use nsn_sidecar::{SidecarClient, SidecarClientConfig};
use nsn_types::TaskStatus;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Trait for task execution via sidecar.
///
/// This trait enables mock implementations for testing without requiring
/// a real sidecar connection.
#[async_trait]
pub trait ExecutionRunnerTrait: Send + Sync {
    /// Connect to the execution backend.
    async fn connect(&mut self) -> ExecutionResult<()>;

    /// Disconnect from the execution backend.
    fn disconnect(&mut self);

    /// Check if connected to the backend.
    fn is_connected(&self) -> bool;

    /// Execute a task.
    ///
    /// # Arguments
    /// * `task` - Task specification to execute
    ///
    /// # Returns
    /// Execution result with output CID and timing information.
    async fn execute(&mut self, task: &TaskSpec) -> ExecutionResult<ExecutionOutput>;

    /// Poll task status.
    ///
    /// # Arguments
    /// * `task_id` - Task to check status for
    ///
    /// # Returns
    /// Current task progress and status.
    async fn poll_status(&mut self, task_id: u64) -> ExecutionResult<TaskProgress>;

    /// Cancel a running task.
    ///
    /// # Arguments
    /// * `task_id` - Task to cancel
    /// * `reason` - Cancellation reason
    async fn cancel(&mut self, task_id: u64, reason: &str) -> ExecutionResult<()>;
}

/// Configuration for the execution runner.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Sidecar gRPC endpoint.
    pub sidecar_endpoint: String,
    /// Default execution timeout in milliseconds.
    pub timeout_ms: u64,
    /// Status poll interval in milliseconds.
    pub poll_interval_ms: u64,
    /// Connection timeout.
    pub connect_timeout: Duration,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            sidecar_endpoint: "http://127.0.0.1:50050".to_string(),
            timeout_ms: 300_000, // 5 minutes
            poll_interval_ms: 1000,
            connect_timeout: Duration::from_secs(5),
        }
    }
}

/// Result of task execution.
#[derive(Debug, Clone)]
pub struct ExecutionOutput {
    /// Task ID.
    pub task_id: u64,
    /// Output data CID.
    pub output_cid: String,
    /// Execution time in milliseconds.
    pub execution_time_ms: u64,
}

/// Task execution progress.
#[derive(Debug, Clone)]
pub struct TaskProgress {
    /// Progress percentage (0.0 - 1.0).
    pub progress: f32,
    /// Current execution stage.
    pub stage: String,
    /// Task status.
    pub status: TaskStatus,
}

/// Task specification for execution.
#[derive(Debug, Clone)]
pub struct TaskSpec {
    /// Task ID.
    pub id: u64,
    /// Model to use.
    pub model_id: String,
    /// Input data CID.
    pub input_cid: String,
    /// Task parameters (JSON).
    pub parameters: Vec<u8>,
    /// Override timeout (optional).
    pub timeout_ms: Option<u64>,
}

/// Execution runner wrapping sidecar gRPC.
///
/// Provides a simplified interface for executing Lane 1 tasks via the
/// sidecar service, with automatic connection management and progress tracking.
pub struct ExecutionRunner {
    config: RunnerConfig,
    client: Option<SidecarClient>,
}

impl ExecutionRunner {
    /// Create a new execution runner.
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            config,
            client: None,
        }
    }

    /// Get the runner configuration.
    pub fn config(&self) -> &RunnerConfig {
        &self.config
    }

    /// Connect to the sidecar service.
    pub async fn connect(&mut self) -> ExecutionResult<()> {
        if self.client.is_some() {
            return Ok(());
        }

        info!(
            endpoint = %self.config.sidecar_endpoint,
            "Connecting to sidecar"
        );

        let sidecar_config = SidecarClientConfig::new(&self.config.sidecar_endpoint)
            .with_connect_timeout(self.config.connect_timeout);

        let client = SidecarClient::connect_with_config(sidecar_config)
            .await
            .map_err(|e| ExecutionError::Connection(e.to_string()))?;

        self.client = Some(client);
        Ok(())
    }

    /// Disconnect from the sidecar.
    pub fn disconnect(&mut self) {
        self.client = None;
    }

    /// Check if connected to sidecar.
    pub fn is_connected(&self) -> bool {
        self.client.is_some()
    }

    /// Execute a task via the sidecar.
    ///
    /// # Arguments
    /// * `task` - Task specification
    ///
    /// # Returns
    /// Execution result with output CID and timing information.
    pub async fn execute(&mut self, task: &TaskSpec) -> ExecutionResult<ExecutionOutput> {
        // Ensure connected
        self.connect().await?;

        let client = self.client.as_mut().ok_or_else(|| {
            ExecutionError::Connection("client not connected".to_string())
        })?;

        let timeout_ms = task.timeout_ms.unwrap_or(self.config.timeout_ms);

        debug!(
            task_id = task.id,
            model_id = %task.model_id,
            timeout_ms = timeout_ms,
            "Executing task via sidecar"
        );

        let start = std::time::Instant::now();

        // Call sidecar execute_task
        let response = client
            .execute_task_with_lane(
                task.id.to_string(),
                &task.model_id,
                &task.input_cid,
                task.parameters.clone(),
                1, // Lane 1
            )
            .await
            .map_err(|e| ExecutionError::SidecarFailed(e.to_string()))?;

        let execution_time_ms = start.elapsed().as_millis() as u64;

        if response.success {
            info!(
                task_id = task.id,
                output_cid = %response.output_cid,
                execution_time_ms = execution_time_ms,
                "Task execution succeeded"
            );

            Ok(ExecutionOutput {
                task_id: task.id,
                output_cid: response.output_cid,
                execution_time_ms,
            })
        } else {
            warn!(
                task_id = task.id,
                error = %response.error_message,
                "Task execution failed"
            );

            Err(ExecutionError::SidecarFailed(response.error_message))
        }
    }

    /// Poll task status from sidecar.
    ///
    /// # Arguments
    /// * `task_id` - Task to check status for
    ///
    /// # Returns
    /// Current task progress and status.
    pub async fn poll_status(&mut self, task_id: u64) -> ExecutionResult<TaskProgress> {
        // Ensure connected
        self.connect().await?;

        let client = self.client.as_mut().ok_or_else(|| {
            ExecutionError::Connection("client not connected".to_string())
        })?;

        let response = client
            .get_task_status(task_id.to_string())
            .await
            .map_err(|e| ExecutionError::SidecarFailed(e.to_string()))?;

        let status = match response.status.as_str() {
            "pending" => TaskStatus::Pending,
            "running" => TaskStatus::Running,
            "completed" => TaskStatus::Completed,
            "failed" => TaskStatus::Failed(nsn_types::FailureReason::Other(response.error_message.clone())),
            "cancelled" => TaskStatus::Cancelled,
            _ => TaskStatus::Failed(nsn_types::FailureReason::Other(format!("unknown status: {}", response.status))),
        };

        Ok(TaskProgress {
            progress: response.progress,
            stage: response.current_stage,
            status,
        })
    }

    /// Cancel a running task.
    ///
    /// # Arguments
    /// * `task_id` - Task to cancel
    /// * `reason` - Cancellation reason
    pub async fn cancel(&mut self, task_id: u64, reason: &str) -> ExecutionResult<()> {
        // Ensure connected
        self.connect().await?;

        let client = self.client.as_mut().ok_or_else(|| {
            ExecutionError::Connection("client not connected".to_string())
        })?;

        debug!(
            task_id = task_id,
            reason = %reason,
            "Cancelling task"
        );

        let response = client
            .cancel_task(task_id.to_string(), reason)
            .await
            .map_err(|e| ExecutionError::SidecarFailed(e.to_string()))?;

        if response.success {
            Ok(())
        } else {
            Err(ExecutionError::SidecarFailed(response.error_message))
        }
    }
}

#[async_trait]
impl ExecutionRunnerTrait for ExecutionRunner {
    async fn connect(&mut self) -> ExecutionResult<()> {
        self.connect().await
    }

    fn disconnect(&mut self) {
        self.disconnect()
    }

    fn is_connected(&self) -> bool {
        self.is_connected()
    }

    async fn execute(&mut self, task: &TaskSpec) -> ExecutionResult<ExecutionOutput> {
        self.execute(task).await
    }

    async fn poll_status(&mut self, task_id: u64) -> ExecutionResult<TaskProgress> {
        self.poll_status(task_id).await
    }

    async fn cancel(&mut self, task_id: u64, reason: &str) -> ExecutionResult<()> {
        self.cancel(task_id, reason).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_config_default() {
        let config = RunnerConfig::default();
        assert_eq!(config.timeout_ms, 300_000);
        assert_eq!(config.poll_interval_ms, 1000);
    }

    #[test]
    fn test_runner_creation() {
        let config = RunnerConfig::default();
        let runner = ExecutionRunner::new(config.clone());

        assert!(!runner.is_connected());
        assert_eq!(runner.config().sidecar_endpoint, config.sidecar_endpoint);
    }

    #[test]
    fn test_task_spec() {
        let task = TaskSpec {
            id: 42,
            model_id: "flux-schnell".to_string(),
            input_cid: "QmInput123".to_string(),
            parameters: b"{}".to_vec(),
            timeout_ms: Some(60_000),
        };

        assert_eq!(task.id, 42);
        assert_eq!(task.model_id, "flux-schnell");
        assert_eq!(task.timeout_ms, Some(60_000));
    }

    #[test]
    fn test_execution_output() {
        let output = ExecutionOutput {
            task_id: 1,
            output_cid: "QmOutput456".to_string(),
            execution_time_ms: 5000,
        };

        assert_eq!(output.task_id, 1);
        assert_eq!(output.output_cid, "QmOutput456");
        assert_eq!(output.execution_time_ms, 5000);
    }

    #[test]
    fn test_disconnect() {
        let config = RunnerConfig::default();
        let mut runner = ExecutionRunner::new(config);

        assert!(!runner.is_connected());
        runner.disconnect();
        assert!(!runner.is_connected());
    }
}
