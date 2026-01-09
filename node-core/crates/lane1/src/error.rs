//! Error types for Lane 1 task marketplace orchestration.
//!
//! Provides comprehensive error handling for chain event listening,
//! task execution, and result submission operations.

use thiserror::Error;

/// Top-level error type for Lane 1 operations.
#[derive(Debug, Error)]
pub enum Lane1Error {
    /// Chain event listener errors.
    #[error("listener error: {0}")]
    Listener(#[from] ListenerError),

    /// Task execution errors.
    #[error("execution error: {0}")]
    Execution(#[from] ExecutionError),

    /// Result submission errors.
    #[error("submission error: {0}")]
    Submission(#[from] SubmissionError),

    /// Scheduler integration errors.
    #[error("scheduler error: {0}")]
    Scheduler(String),

    /// Configuration errors.
    #[error("configuration error: {0}")]
    Config(String),

    /// Channel communication errors.
    #[error("channel error: {0}")]
    Channel(String),
}

/// Errors during chain event listening.
#[derive(Debug, Error)]
pub enum ListenerError {
    /// Chain connection failed.
    #[error("chain connection failed: {0}")]
    Connection(String),

    /// Event subscription failed.
    #[error("event subscription failed: {0}")]
    Subscription(String),

    /// Event decoding failed.
    #[error("event decoding failed: {0}")]
    Decode(String),

    /// Invalid task event.
    #[error("invalid task event: {0}")]
    InvalidEvent(String),

    /// Scheduler queue full.
    #[error("scheduler queue full")]
    QueueFull,
}

/// Errors during task execution via sidecar.
#[derive(Debug, Error)]
pub enum ExecutionError {
    /// Sidecar gRPC connection failed.
    #[error("sidecar connection failed: {0}")]
    Connection(String),

    /// Task execution failed on sidecar.
    #[error("sidecar execution failed: {0}")]
    SidecarFailed(String),

    /// Task timed out.
    #[error("task timed out after {timeout_ms}ms")]
    Timeout {
        /// Timeout duration in milliseconds.
        timeout_ms: u64,
    },

    /// Model not available.
    #[error("model not available: {model}")]
    ModelUnavailable {
        /// Model identifier.
        model: String,
    },

    /// Invalid task parameters.
    #[error("invalid parameters: {0}")]
    InvalidParameters(String),

    /// Task was cancelled.
    #[error("task cancelled: {reason}")]
    Cancelled {
        /// Cancellation reason.
        reason: String,
    },

    /// Preempted by Lane 0 task.
    #[error("preempted by Lane 0 task")]
    Preempted,
}

/// Errors during result submission to chain.
#[derive(Debug, Error)]
pub enum SubmissionError {
    /// Chain connection failed.
    #[error("chain connection failed: {0}")]
    Connection(String),

    /// Extrinsic submission failed.
    #[error("extrinsic submission failed: {0}")]
    ExtrinsicFailed(String),

    /// Transaction rejected.
    #[error("transaction rejected: {0}")]
    Rejected(String),

    /// Task not found on chain.
    #[error("task not found on chain: {task_id}")]
    TaskNotFound {
        /// Task ID that was not found.
        task_id: u64,
    },

    /// Invalid task state for operation.
    #[error("invalid task state for {operation}: current state is {current_state}")]
    InvalidState {
        /// Operation that was attempted.
        operation: String,
        /// Current state of the task.
        current_state: String,
    },

    /// Signer error.
    #[error("signer error: {0}")]
    SignerError(String),
}

/// Result type alias for Lane 1 operations.
pub type Lane1Result<T> = Result<T, Lane1Error>;

/// Result type alias for listener operations.
pub type ListenerResult<T> = Result<T, ListenerError>;

/// Result type alias for execution operations.
pub type ExecutionResult<T> = Result<T, ExecutionError>;

/// Result type alias for submission operations.
pub type SubmissionResult<T> = Result<T, SubmissionError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ExecutionError::Timeout { timeout_ms: 5000 };
        assert!(err.to_string().contains("5000"));

        let err = SubmissionError::TaskNotFound { task_id: 42 };
        assert!(err.to_string().contains("42"));
    }

    #[test]
    fn test_error_conversion() {
        let listener_err = ListenerError::Connection("connection refused".to_string());
        let lane1_err: Lane1Error = listener_err.into();
        assert!(matches!(lane1_err, Lane1Error::Listener(_)));

        let exec_err = ExecutionError::Preempted;
        let lane1_err: Lane1Error = exec_err.into();
        assert!(matches!(lane1_err, Lane1Error::Execution(_)));
    }
}
