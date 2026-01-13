//! Mock ExecutionRunner for testing Lane 1 task execution without actual sidecar.

use std::collections::{HashSet, VecDeque};
use std::time::Duration;

use async_trait::async_trait;
use nsn_lane1::{
    ExecutionError, ExecutionOutput, ExecutionResult, ExecutionRunnerTrait, TaskProgress, TaskSpec,
};
use nsn_types::TaskStatus;

/// A recorded execution event for verification.
#[derive(Debug, Clone)]
pub struct ExecutionEvent {
    /// Task ID that was executed.
    pub task_id: u64,
    /// Model ID used for execution.
    pub model_id: String,
    /// Input CID provided.
    pub input_cid: String,
    /// Whether execution succeeded.
    pub success: bool,
    /// Execution time in milliseconds.
    pub execution_time_ms: u64,
}

/// Mock ExecutionRunner for simulation testing.
///
/// Provides configurable success/failure behavior for task execution.
///
/// # Example
///
/// ```rust,ignore
/// use nsn_simulation::mocks::MockExecutionRunner;
/// use std::time::Duration;
///
/// let mut runner = MockExecutionRunner::new()
///     .with_success(&[1, 2, 3])
///     .with_latency(Duration::from_millis(100));
///
/// let output = runner.execute(&task_spec).await?;
/// assert_eq!(runner.execution_count(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct MockExecutionRunner {
    /// Task IDs that execute successfully.
    success_tasks: HashSet<u64>,
    /// Task IDs that timeout.
    timeout_tasks: HashSet<u64>,
    /// Simulated execution latency.
    latency: Option<Duration>,
    /// Execution events (for verification).
    pub executions: VecDeque<ExecutionEvent>,
    /// Whether connected to backend.
    connected: bool,
    /// Failure injection: probability of random failure (0.0 - 1.0).
    failure_rate: f64,
}

impl Default for MockExecutionRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl MockExecutionRunner {
    /// Create a new mock runner.
    pub fn new() -> Self {
        Self {
            success_tasks: HashSet::new(),
            timeout_tasks: HashSet::new(),
            latency: None,
            executions: VecDeque::new(),
            connected: false,
            failure_rate: 0.0,
        }
    }

    /// Configure task IDs that will succeed.
    pub fn with_success(mut self, task_ids: &[u64]) -> Self {
        self.success_tasks.extend(task_ids);
        self
    }

    /// Configure task IDs that will timeout.
    pub fn with_timeout(mut self, task_ids: &[u64]) -> Self {
        self.timeout_tasks.extend(task_ids);
        self
    }

    /// Configure simulated latency for execution.
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = Some(latency);
        self
    }

    /// Configure failure rate for random failures.
    pub fn with_failure_rate(mut self, rate: f64) -> Self {
        self.failure_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Add a success task ID dynamically.
    pub fn add_success_task(&mut self, task_id: u64) {
        self.success_tasks.insert(task_id);
    }

    /// Remove a success task (to simulate failures).
    pub fn remove_success_task(&mut self, task_id: u64) {
        self.success_tasks.remove(&task_id);
    }

    /// Check if a task is configured for success.
    pub fn is_success_task(&self, task_id: u64) -> bool {
        self.success_tasks.contains(&task_id)
    }

    /// Get the number of execution events.
    pub fn execution_count(&self) -> usize {
        self.executions.len()
    }

    /// Clear execution events.
    pub fn clear_executions(&mut self) {
        self.executions.clear();
    }

    /// Get executions for a specific task.
    pub fn executions_for_task(&self, task_id: u64) -> Vec<&ExecutionEvent> {
        self.executions
            .iter()
            .filter(|e| e.task_id == task_id)
            .collect()
    }
}

#[async_trait]
impl ExecutionRunnerTrait for MockExecutionRunner {
    async fn connect(&mut self) -> ExecutionResult<()> {
        self.connected = true;
        Ok(())
    }

    fn disconnect(&mut self) {
        self.connected = false;
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    async fn execute(&mut self, task: &TaskSpec) -> ExecutionResult<ExecutionOutput> {
        let task_id = task.id;

        // Apply latency if configured
        if let Some(latency) = self.latency {
            tokio::time::sleep(latency).await;
        }

        // Check for timeout
        if self.timeout_tasks.contains(&task_id) {
            self.executions.push_back(ExecutionEvent {
                task_id,
                model_id: task.model_id.clone(),
                input_cid: task.input_cid.clone(),
                success: false,
                execution_time_ms: task.timeout_ms.unwrap_or(300_000),
            });
            return Err(ExecutionError::Timeout {
                timeout_ms: task.timeout_ms.unwrap_or(300_000),
            });
        }

        // Check for success
        let success = self.success_tasks.contains(&task_id);
        let execution_time_ms = self.latency.map(|l| l.as_millis() as u64).unwrap_or(100);

        // Record execution event
        self.executions.push_back(ExecutionEvent {
            task_id,
            model_id: task.model_id.clone(),
            input_cid: task.input_cid.clone(),
            success,
            execution_time_ms,
        });

        if !success {
            return Err(ExecutionError::SidecarFailed(format!(
                "task {} not in success list",
                task_id
            )));
        }

        Ok(ExecutionOutput {
            task_id,
            output_cid: format!("QmOutput{}", task_id),
            execution_time_ms,
        })
    }

    async fn poll_status(&mut self, task_id: u64) -> ExecutionResult<TaskProgress> {
        // Check if task was executed successfully
        let completed = self
            .executions
            .iter()
            .any(|e| e.task_id == task_id && e.success);

        Ok(TaskProgress {
            progress: if completed { 1.0 } else { 0.0 },
            stage: if completed {
                "completed".to_string()
            } else {
                "pending".to_string()
            },
            status: if completed {
                TaskStatus::Completed
            } else {
                TaskStatus::Pending
            },
        })
    }

    async fn cancel(&mut self, _task_id: u64, _reason: &str) -> ExecutionResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_task_spec(task_id: u64) -> TaskSpec {
        TaskSpec {
            id: task_id,
            model_id: "test-model".to_string(),
            input_cid: format!("QmInput{}", task_id),
            parameters: vec![],
            timeout_ms: Some(300_000),
        }
    }

    #[tokio::test]
    async fn test_success_execution() {
        let mut runner = MockExecutionRunner::new().with_success(&[1, 2, 3]);

        let task = make_task_spec(1);
        let result = runner.execute(&task).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.task_id, 1);
        assert_eq!(output.output_cid, "QmOutput1");
    }

    #[tokio::test]
    async fn test_failure_execution() {
        let mut runner = MockExecutionRunner::new().with_success(&[1]);

        let task = make_task_spec(99);
        let result = runner.execute(&task).await;

        assert!(result.is_err());
        assert!(matches!(result, Err(ExecutionError::SidecarFailed(_))));
    }

    #[tokio::test]
    async fn test_timeout_execution() {
        let mut runner = MockExecutionRunner::new().with_timeout(&[5]);

        let task = make_task_spec(5);
        let result = runner.execute(&task).await;

        assert!(matches!(result, Err(ExecutionError::Timeout { .. })));
    }

    #[tokio::test]
    async fn test_execution_tracking() {
        let mut runner = MockExecutionRunner::new().with_success(&[1, 2, 3]);

        for task_id in 1..=3 {
            let task = make_task_spec(task_id);
            runner.execute(&task).await.unwrap();
        }

        assert_eq!(runner.execution_count(), 3);
        assert!(runner.executions.iter().all(|e| e.success));
    }

    #[tokio::test]
    async fn test_connect_disconnect() {
        let mut runner = MockExecutionRunner::new();

        assert!(!runner.is_connected());
        runner.connect().await.unwrap();
        assert!(runner.is_connected());
        runner.disconnect();
        assert!(!runner.is_connected());
    }

    #[tokio::test]
    async fn test_poll_status() {
        let mut runner = MockExecutionRunner::new().with_success(&[1]);

        // Before execution
        let status = runner.poll_status(1).await.unwrap();
        assert_eq!(status.progress, 0.0);

        // After execution
        let task = make_task_spec(1);
        runner.execute(&task).await.unwrap();
        let status = runner.poll_status(1).await.unwrap();
        assert_eq!(status.progress, 1.0);
    }

    #[tokio::test]
    async fn test_dynamic_configuration() {
        let mut runner = MockExecutionRunner::new();

        // Initially fails
        let task = make_task_spec(1);
        assert!(runner.execute(&task).await.is_err());

        // Add to success list
        runner.add_success_task(1);
        runner.clear_executions();
        assert!(runner.execute(&task).await.is_ok());

        // Remove from success list
        runner.remove_success_task(1);
        runner.clear_executions();
        assert!(runner.execute(&task).await.is_err());
    }
}
