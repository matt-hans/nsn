//! gRPC client for connecting to the sidecar service.
//!
//! This module provides a client wrapper for the Sidecar gRPC service,
//! making it easy for the NSN node scheduler to communicate with the
//! sidecar and Python AI containers.

use std::time::Duration;

use tonic::transport::{Channel, Endpoint};
use tracing::{debug, info};

use crate::error::{SidecarError, SidecarResult};
use crate::proto::sidecar_client::SidecarClient as TonicSidecarClient;
use crate::proto::{
    CancelTaskRequest, CancelTaskResponse, ExecuteTaskRequest, ExecuteTaskResponse,
    GetLoadedModelsRequest, GetTaskStatusRequest, GetTaskStatusResponse, GetVramStatusRequest,
    GetVramStatusResponse, HealthCheckRequest, HealthCheckResponse, LoadModelRequest,
    LoadModelResponse, LoadedModel, StartContainerRequest, StartContainerResponse,
    StopContainerRequest, StopContainerResponse, UnloadModelRequest, UnloadModelResponse,
};

/// Default connection timeout
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Default request timeout
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Client configuration
#[derive(Debug, Clone)]
pub struct SidecarClientConfig {
    /// Sidecar gRPC endpoint (e.g., "http://127.0.0.1:50050")
    pub endpoint: String,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Request timeout
    pub request_timeout: Duration,
}

impl Default for SidecarClientConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:50050".to_string(),
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }
}

impl SidecarClientConfig {
    /// Create a new config with the given endpoint.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Default::default()
        }
    }

    /// Set the connect timeout.
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Set the request timeout.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }
}

/// Client for communicating with the Sidecar gRPC service.
///
/// This client wraps the generated Tonic client and provides a more
/// ergonomic interface for the NSN node scheduler.
#[derive(Debug, Clone)]
pub struct SidecarClient {
    /// Underlying Tonic gRPC client
    inner: TonicSidecarClient<Channel>,
    /// Client configuration
    config: SidecarClientConfig,
}

impl SidecarClient {
    /// Connect to the sidecar service with default configuration.
    pub async fn connect(endpoint: impl Into<String>) -> SidecarResult<Self> {
        let config = SidecarClientConfig::new(endpoint);
        Self::connect_with_config(config).await
    }

    /// Connect to the sidecar service with custom configuration.
    pub async fn connect_with_config(config: SidecarClientConfig) -> SidecarResult<Self> {
        info!(endpoint = %config.endpoint, "Connecting to sidecar");

        let channel = Endpoint::from_shared(config.endpoint.clone())
            .map_err(|e| SidecarError::GrpcError(e.to_string()))?
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .connect()
            .await
            .map_err(|e| SidecarError::GrpcError(e.to_string()))?;

        let inner = TonicSidecarClient::new(channel);

        info!("Connected to sidecar successfully");

        Ok(Self { inner, config })
    }

    /// Create a client from an existing channel (for testing).
    pub fn from_channel(channel: Channel) -> Self {
        Self {
            inner: TonicSidecarClient::new(channel),
            config: SidecarClientConfig::default(),
        }
    }

    /// Get the client configuration.
    pub fn config(&self) -> &SidecarClientConfig {
        &self.config
    }

    // =========================================================================
    // Container Lifecycle
    // =========================================================================

    /// Start a new container with GPU support.
    ///
    /// # Arguments
    /// * `container_id` - Unique identifier for the container
    /// * `image_cid` - Container image reference (IPFS CID or registry path)
    /// * `gpu_ids` - GPU device IDs to assign to the container
    ///
    /// # Returns
    /// The gRPC endpoint for communicating with the container
    pub async fn start_container(
        &mut self,
        container_id: impl Into<String>,
        image_cid: impl Into<String>,
        gpu_ids: Vec<String>,
    ) -> SidecarResult<StartContainerResponse> {
        let request = StartContainerRequest {
            container_id: container_id.into(),
            image_cid: image_cid.into(),
            gpu_ids,
            env_vars: Default::default(),
            resource_limits: None,
        };

        debug!(
            container_id = %request.container_id,
            image = %request.image_cid,
            "Starting container"
        );

        let response = self
            .inner
            .start_container(request)
            .await
            .map_err(|e| SidecarError::GrpcError(e.to_string()))?;

        Ok(response.into_inner())
    }

    /// Stop a running container.
    ///
    /// # Arguments
    /// * `container_id` - Container to stop
    /// * `force` - If true, force kill without graceful shutdown
    pub async fn stop_container(
        &mut self,
        container_id: impl Into<String>,
        force: bool,
    ) -> SidecarResult<StopContainerResponse> {
        let request = StopContainerRequest {
            container_id: container_id.into(),
            force,
            timeout_seconds: 30,
        };

        let response = self
            .inner
            .stop_container(request)
            .await
            .map_err(|e| SidecarError::GrpcError(e.to_string()))?;

        Ok(response.into_inner())
    }

    /// Check container health status.
    ///
    /// # Arguments
    /// * `container_id` - Container to check
    /// * `include_metrics` - If true, include detailed metrics
    pub async fn health_check(
        &mut self,
        container_id: impl Into<String>,
        include_metrics: bool,
    ) -> SidecarResult<HealthCheckResponse> {
        let request = HealthCheckRequest {
            container_id: container_id.into(),
            include_metrics,
        };

        let response = self
            .inner
            .health_check(request)
            .await
            .map_err(|e| SidecarError::GrpcError(e.to_string()))?;

        Ok(response.into_inner())
    }

    // =========================================================================
    // Model Management
    // =========================================================================

    /// Load a model into a container.
    ///
    /// # Arguments
    /// * `model_id` - Model identifier (e.g., "flux-schnell")
    /// * `container_id` - Container to load the model into
    /// * `priority` - Loading priority (0 = critical)
    pub async fn load_model(
        &mut self,
        model_id: impl Into<String>,
        container_id: impl Into<String>,
        priority: u32,
    ) -> SidecarResult<LoadModelResponse> {
        let request = LoadModelRequest {
            model_id: model_id.into(),
            container_id: container_id.into(),
            priority,
            config: vec![],
        };

        debug!(
            model_id = %request.model_id,
            container_id = %request.container_id,
            "Loading model"
        );

        let response = self
            .inner
            .load_model(request)
            .await
            .map_err(|e| SidecarError::GrpcError(e.to_string()))?;

        Ok(response.into_inner())
    }

    /// Unload a model from a container.
    ///
    /// # Arguments
    /// * `model_id` - Model to unload
    /// * `container_id` - Container to unload from
    pub async fn unload_model(
        &mut self,
        model_id: impl Into<String>,
        container_id: impl Into<String>,
    ) -> SidecarResult<UnloadModelResponse> {
        let request = UnloadModelRequest {
            model_id: model_id.into(),
            container_id: container_id.into(),
            wait: true,
        };

        let response = self
            .inner
            .unload_model(request)
            .await
            .map_err(|e| SidecarError::GrpcError(e.to_string()))?;

        Ok(response.into_inner())
    }

    /// Get list of loaded models.
    ///
    /// # Arguments
    /// * `container_id` - Optional container to filter by (None = all containers)
    pub async fn get_loaded_models(
        &mut self,
        container_id: Option<String>,
    ) -> SidecarResult<Vec<LoadedModel>> {
        let request = GetLoadedModelsRequest {
            container_id: container_id.unwrap_or_default(),
        };

        let response = self
            .inner
            .get_loaded_models(request)
            .await
            .map_err(|e| SidecarError::GrpcError(e.to_string()))?;

        Ok(response.into_inner().models)
    }

    // =========================================================================
    // Task Execution
    // =========================================================================

    /// Execute a task using a loaded model.
    ///
    /// # Arguments
    /// * `task_id` - Unique task identifier
    /// * `model_id` - Model to use for execution
    /// * `input_cid` - Input data CID
    /// * `parameters` - JSON-encoded task parameters
    pub async fn execute_task(
        &mut self,
        task_id: impl Into<String>,
        model_id: impl Into<String>,
        input_cid: impl Into<String>,
        parameters: Vec<u8>,
    ) -> SidecarResult<ExecuteTaskResponse> {
        let request = ExecuteTaskRequest {
            task_id: task_id.into(),
            model_id: model_id.into(),
            input_cid: input_cid.into(),
            parameters,
            timeout_ms: 0,
            callback_endpoint: String::new(),
        };

        debug!(
            task_id = %request.task_id,
            model_id = %request.model_id,
            "Executing task"
        );

        let response = self
            .inner
            .execute_task(request)
            .await
            .map_err(|e| SidecarError::GrpcError(e.to_string()))?;

        Ok(response.into_inner())
    }

    /// Execute a task with a timeout.
    ///
    /// # Arguments
    /// * `task_id` - Unique task identifier
    /// * `model_id` - Model to use for execution
    /// * `input_cid` - Input data CID
    /// * `parameters` - JSON-encoded task parameters
    /// * `timeout` - Maximum execution time
    pub async fn execute_task_with_timeout(
        &mut self,
        task_id: impl Into<String>,
        model_id: impl Into<String>,
        input_cid: impl Into<String>,
        parameters: Vec<u8>,
        timeout: Duration,
    ) -> SidecarResult<ExecuteTaskResponse> {
        let request = ExecuteTaskRequest {
            task_id: task_id.into(),
            model_id: model_id.into(),
            input_cid: input_cid.into(),
            parameters,
            timeout_ms: timeout.as_millis() as u64,
            callback_endpoint: String::new(),
        };

        let response = self
            .inner
            .execute_task(request)
            .await
            .map_err(|e| SidecarError::GrpcError(e.to_string()))?;

        Ok(response.into_inner())
    }

    /// Get the status of a task.
    pub async fn get_task_status(
        &mut self,
        task_id: impl Into<String>,
    ) -> SidecarResult<GetTaskStatusResponse> {
        let request = GetTaskStatusRequest {
            task_id: task_id.into(),
        };

        let response = self
            .inner
            .get_task_status(request)
            .await
            .map_err(|e| SidecarError::GrpcError(e.to_string()))?;

        Ok(response.into_inner())
    }

    /// Cancel a running task.
    ///
    /// # Arguments
    /// * `task_id` - Task to cancel
    /// * `reason` - Reason for cancellation
    pub async fn cancel_task(
        &mut self,
        task_id: impl Into<String>,
        reason: impl Into<String>,
    ) -> SidecarResult<CancelTaskResponse> {
        let request = CancelTaskRequest {
            task_id: task_id.into(),
            reason: reason.into(),
        };

        let response = self
            .inner
            .cancel_task(request)
            .await
            .map_err(|e| SidecarError::GrpcError(e.to_string()))?;

        Ok(response.into_inner())
    }

    // =========================================================================
    // Resource Monitoring
    // =========================================================================

    /// Get VRAM usage status.
    ///
    /// # Arguments
    /// * `gpu_id` - Optional GPU ID to filter by
    pub async fn get_vram_status(
        &mut self,
        gpu_id: Option<String>,
    ) -> SidecarResult<GetVramStatusResponse> {
        let request = GetVramStatusRequest {
            gpu_id: gpu_id.unwrap_or_default(),
        };

        let response = self
            .inner
            .get_vram_status(request)
            .await
            .map_err(|e| SidecarError::GrpcError(e.to_string()))?;

        Ok(response.into_inner())
    }

    /// Check if there's enough VRAM available for a model.
    ///
    /// # Arguments
    /// * `required_gb` - Required VRAM in GB
    pub async fn has_available_vram(&mut self, required_gb: f32) -> SidecarResult<bool> {
        let status = self.get_vram_status(None).await?;
        Ok(status.available_vram_gb >= required_gb)
    }
}

/// Builder for SidecarClient with fluent API.
#[derive(Debug, Clone, Default)]
pub struct SidecarClientBuilder {
    config: SidecarClientConfig,
}

impl SidecarClientBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the sidecar endpoint.
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.config.endpoint = endpoint.into();
        self
    }

    /// Set the connection timeout.
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.config.connect_timeout = timeout;
        self
    }

    /// Set the request timeout.
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.config.request_timeout = timeout;
        self
    }

    /// Build and connect the client.
    pub async fn build(self) -> SidecarResult<SidecarClient> {
        SidecarClient::connect_with_config(self.config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = SidecarClientConfig::default();
        assert_eq!(config.endpoint, "http://127.0.0.1:50050");
        assert_eq!(config.connect_timeout, DEFAULT_CONNECT_TIMEOUT);
        assert_eq!(config.request_timeout, DEFAULT_REQUEST_TIMEOUT);
    }

    #[test]
    fn test_config_builder() {
        let config = SidecarClientConfig::new("http://localhost:9999")
            .with_connect_timeout(Duration::from_secs(10))
            .with_request_timeout(Duration::from_secs(120));

        assert_eq!(config.endpoint, "http://localhost:9999");
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert_eq!(config.request_timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_builder_pattern() {
        let builder = SidecarClientBuilder::new()
            .endpoint("http://sidecar:50050")
            .connect_timeout(Duration::from_secs(3))
            .request_timeout(Duration::from_secs(30));

        assert_eq!(builder.config.endpoint, "http://sidecar:50050");
        assert_eq!(builder.config.connect_timeout, Duration::from_secs(3));
        assert_eq!(builder.config.request_timeout, Duration::from_secs(30));
    }
}
