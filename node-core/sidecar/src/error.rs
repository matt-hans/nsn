//! Error types for the sidecar service.
//!
//! This module defines all error types that can occur during sidecar operations,
//! including container management, model loading, task execution, and gRPC errors.

use std::fmt;

use tonic::Status;

/// Result type alias for sidecar operations.
pub type SidecarResult<T> = Result<T, SidecarError>;

/// Errors that can occur during sidecar operations.
#[derive(Debug, Clone)]
pub enum SidecarError {
    /// Container with the given ID was not found
    ContainerNotFound(String),

    /// Attempted to start a container that already exists
    ContainerAlreadyExists(String),

    /// Maximum number of containers reached
    ContainerLimitReached(usize),

    /// Container is not in a healthy state
    ContainerUnhealthy(String),

    /// Model with the given ID is not loaded
    ModelNotLoaded(String),

    /// Model is already loaded in the container
    ModelAlreadyLoaded(String),

    /// Not enough VRAM to load the model
    InsufficientVram {
        /// VRAM required in GB
        required: f32,
        /// VRAM available in GB
        available: f32,
    },

    /// Task with the given ID was not found
    TaskNotFound(String),

    /// Task is already running
    TaskAlreadyRunning(String),

    /// Task execution failed
    TaskExecutionFailed(String),

    /// Task was cancelled
    TaskCancelled(String),

    /// Task timed out
    TaskTimeout {
        /// Task that timed out
        task_id: String,
        /// Timeout in milliseconds
        timeout_ms: u64,
    },

    /// gRPC communication error
    GrpcError(String),

    /// Invalid request parameters
    InvalidRequest(String),

    /// Internal error
    Internal(String),
}

impl fmt::Display for SidecarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContainerNotFound(id) => write!(f, "Container not found: {}", id),
            Self::ContainerAlreadyExists(id) => write!(f, "Container already exists: {}", id),
            Self::ContainerLimitReached(limit) => {
                write!(f, "Container limit reached: maximum {} containers", limit)
            }
            Self::ContainerUnhealthy(id) => write!(f, "Container is unhealthy: {}", id),
            Self::ModelNotLoaded(id) => write!(f, "Model not loaded: {}", id),
            Self::ModelAlreadyLoaded(id) => write!(f, "Model already loaded: {}", id),
            Self::InsufficientVram { required, available } => {
                write!(
                    f,
                    "Insufficient VRAM: required {} GB, available {} GB",
                    required, available
                )
            }
            Self::TaskNotFound(id) => write!(f, "Task not found: {}", id),
            Self::TaskAlreadyRunning(id) => write!(f, "Task already running: {}", id),
            Self::TaskExecutionFailed(msg) => write!(f, "Task execution failed: {}", msg),
            Self::TaskCancelled(id) => write!(f, "Task was cancelled: {}", id),
            Self::TaskTimeout { task_id, timeout_ms } => {
                write!(f, "Task {} timed out after {} ms", task_id, timeout_ms)
            }
            Self::GrpcError(msg) => write!(f, "gRPC error: {}", msg),
            Self::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for SidecarError {}

impl From<SidecarError> for Status {
    fn from(err: SidecarError) -> Self {
        match &err {
            SidecarError::ContainerNotFound(_)
            | SidecarError::ModelNotLoaded(_)
            | SidecarError::TaskNotFound(_) => Status::not_found(err.to_string()),

            SidecarError::ContainerAlreadyExists(_)
            | SidecarError::ModelAlreadyLoaded(_)
            | SidecarError::TaskAlreadyRunning(_) => Status::already_exists(err.to_string()),

            SidecarError::ContainerLimitReached(_) | SidecarError::InsufficientVram { .. } => {
                Status::resource_exhausted(err.to_string())
            }

            SidecarError::ContainerUnhealthy(_) | SidecarError::TaskExecutionFailed(_) => {
                Status::failed_precondition(err.to_string())
            }

            SidecarError::TaskCancelled(_) => Status::cancelled(err.to_string()),

            SidecarError::TaskTimeout { .. } => Status::deadline_exceeded(err.to_string()),

            SidecarError::InvalidRequest(_) => Status::invalid_argument(err.to_string()),

            SidecarError::GrpcError(_) | SidecarError::Internal(_) => {
                Status::internal(err.to_string())
            }
        }
    }
}

impl From<tonic::transport::Error> for SidecarError {
    fn from(err: tonic::transport::Error) -> Self {
        SidecarError::GrpcError(err.to_string())
    }
}

impl From<anyhow::Error> for SidecarError {
    fn from(err: anyhow::Error) -> Self {
        SidecarError::Internal(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SidecarError::ContainerNotFound("test-container".to_string());
        assert_eq!(err.to_string(), "Container not found: test-container");

        let err = SidecarError::InsufficientVram {
            required: 6.0,
            available: 4.0,
        };
        assert!(err.to_string().contains("6"));
        assert!(err.to_string().contains("4"));
    }

    #[test]
    fn test_error_to_status() {
        let err = SidecarError::ContainerNotFound("test".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::NotFound);

        let err = SidecarError::ContainerAlreadyExists("test".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::AlreadyExists);

        let err = SidecarError::InsufficientVram {
            required: 6.0,
            available: 4.0,
        };
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);
    }
}
