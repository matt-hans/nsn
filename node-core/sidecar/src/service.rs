//! gRPC service implementation for the sidecar.
//!
//! This module implements the Sidecar gRPC service that bridges the Rust
//! scheduler with Python AI model containers. It handles:
//!
//! - Container lifecycle (start, stop, health check)
//! - Model management (load, unload, query)
//! - Task execution
//! - VRAM status reporting

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

use crate::container::{ContainerManager, ModelLoadState};
use crate::error::SidecarError;
use crate::proto;
use crate::proto::sidecar_server::Sidecar;
use crate::vram::VramTracker;

/// Task execution state tracked by the sidecar.
#[derive(Debug, Clone)]
pub struct TaskState {
    /// Unique task identifier
    pub task_id: String,
    /// Model used for this task
    pub model_id: String,
    /// Container executing the task
    pub container_id: String,
    /// Current task status
    pub status: TaskStatus,
    /// Progress (0.0 to 1.0)
    pub progress: f32,
    /// Current execution stage description
    pub current_stage: String,
    /// When the task started
    pub started_at: Option<Instant>,
    /// When the task completed (success or failure)
    pub completed_at: Option<Instant>,
    /// Error message if task failed
    pub error_message: Option<String>,
    /// Output CID if task completed successfully
    pub output_cid: Option<String>,
}

/// Task status enumeration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task created but not yet queued
    Pending,
    /// Task queued for execution
    Queued,
    /// Task currently executing
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed with error
    Failed,
    /// Task was cancelled
    Cancelled,
}

impl TaskStatus {
    /// Convert the task status to a string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskStatus::Pending => "pending",
            TaskStatus::Queued => "queued",
            TaskStatus::Running => "running",
            TaskStatus::Completed => "completed",
            TaskStatus::Failed => "failed",
            TaskStatus::Cancelled => "cancelled",
        }
    }
}

/// Configuration for the sidecar service
#[derive(Debug, Clone)]
pub struct SidecarServiceConfig {
    /// Address to bind the gRPC server
    pub bind_addr: SocketAddr,
    /// Maximum task execution time
    pub max_task_duration: Duration,
    /// Task history retention
    pub task_retention: Duration,
}

impl Default for SidecarServiceConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:50050".parse().unwrap(),
            max_task_duration: Duration::from_secs(300), // 5 minutes
            task_retention: Duration::from_secs(3600),   // 1 hour
        }
    }
}

/// Sidecar gRPC service implementation.
///
/// Bridges the Rust scheduler with Python AI model containers,
/// handling container lifecycle, model management, and task execution.
pub struct SidecarService {
    /// Container lifecycle manager
    containers: Arc<ContainerManager>,
    /// VRAM usage tracker
    vram_tracker: Arc<RwLock<VramTracker>>,
    /// Active and recent tasks
    tasks: Arc<RwLock<HashMap<String, TaskState>>>,
    /// Service configuration
    config: SidecarServiceConfig,
}

impl SidecarService {
    /// Create a new sidecar service with default configuration.
    pub fn new() -> Self {
        Self::with_config(SidecarServiceConfig::default())
    }

    /// Create a new sidecar service with custom configuration.
    pub fn with_config(config: SidecarServiceConfig) -> Self {
        Self {
            containers: Arc::new(ContainerManager::new()),
            vram_tracker: Arc::new(RwLock::new(VramTracker::new())),
            tasks: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Create a new sidecar service with injected dependencies.
    pub fn with_dependencies(
        containers: Arc<ContainerManager>,
        vram_tracker: Arc<RwLock<VramTracker>>,
        config: SidecarServiceConfig,
    ) -> Self {
        Self {
            containers,
            vram_tracker,
            tasks: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Get the service configuration.
    pub fn config(&self) -> &SidecarServiceConfig {
        &self.config
    }

    /// Get a reference to the container manager.
    pub fn containers(&self) -> &Arc<ContainerManager> {
        &self.containers
    }

    /// Get a reference to the VRAM tracker.
    pub fn vram_tracker(&self) -> &Arc<RwLock<VramTracker>> {
        &self.vram_tracker
    }

    /// Find a container with the requested model loaded.
    async fn find_container_with_model(&self, model_id: &str) -> Option<String> {
        let models = self.containers.get_loaded_models(None).await;
        models
            .iter()
            .find(|(_, m)| m.model_id == model_id && m.state == ModelLoadState::Hot)
            .map(|(cid, _)| cid.clone())
    }
}

impl Default for SidecarService {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl Sidecar for SidecarService {
    /// Start a new container with GPU support.
    async fn start_container(
        &self,
        request: Request<proto::StartContainerRequest>,
    ) -> Result<Response<proto::StartContainerResponse>, Status> {
        let req = request.into_inner();

        info!(
            container_id = %req.container_id,
            image = %req.image_cid,
            "StartContainer request"
        );

        let resource_limits = req.resource_limits.map(|l| l.into());

        match self
            .containers
            .start_container(
                req.container_id.clone(),
                req.image_cid,
                req.gpu_ids,
                resource_limits,
            )
            .await
        {
            Ok(endpoint) => {
                info!(
                    container_id = %req.container_id,
                    endpoint = %endpoint,
                    "Container started successfully"
                );
                Ok(Response::new(proto::StartContainerResponse {
                    success: true,
                    error_message: String::new(),
                    container_endpoint: endpoint,
                }))
            }
            Err(e) => {
                error!(
                    container_id = %req.container_id,
                    error = %e,
                    "Failed to start container"
                );
                Ok(Response::new(proto::StartContainerResponse {
                    success: false,
                    error_message: e.to_string(),
                    container_endpoint: String::new(),
                }))
            }
        }
    }

    /// Stop a running container.
    async fn stop_container(
        &self,
        request: Request<proto::StopContainerRequest>,
    ) -> Result<Response<proto::StopContainerResponse>, Status> {
        let req = request.into_inner();

        info!(
            container_id = %req.container_id,
            force = req.force,
            "StopContainer request"
        );

        let timeout = if req.timeout_seconds > 0 {
            Some(Duration::from_secs(req.timeout_seconds as u64))
        } else {
            None
        };

        match self
            .containers
            .stop_container(&req.container_id, req.force, timeout)
            .await
        {
            Ok(()) => {
                info!(container_id = %req.container_id, "Container stopped");
                Ok(Response::new(proto::StopContainerResponse {
                    success: true,
                    error_message: String::new(),
                }))
            }
            Err(e) => {
                error!(
                    container_id = %req.container_id,
                    error = %e,
                    "Failed to stop container"
                );
                Ok(Response::new(proto::StopContainerResponse {
                    success: false,
                    error_message: e.to_string(),
                }))
            }
        }
    }

    /// Check container health status.
    async fn health_check(
        &self,
        request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        let req = request.into_inner();

        debug!(container_id = %req.container_id, "HealthCheck request");

        match self
            .containers
            .health_check(&req.container_id, req.include_metrics)
            .await
        {
            Ok((healthy, status, uptime, metrics)) => {
                let container_metrics = metrics.map(|m| proto::ContainerMetrics {
                    cpu_usage_percent: m.cpu_usage_percent,
                    memory_used_bytes: m.memory_used_bytes,
                    memory_limit_bytes: m.memory_limit_bytes,
                    gpu_utilization_percent: m.gpu_utilization_percent,
                });

                Ok(Response::new(proto::HealthCheckResponse {
                    healthy,
                    status,
                    uptime_seconds: uptime,
                    metrics: container_metrics,
                }))
            }
            Err(e) => {
                warn!(
                    container_id = %req.container_id,
                    error = %e,
                    "Health check failed"
                );
                Err(e.into())
            }
        }
    }

    /// Load a model into a container.
    async fn load_model(
        &self,
        request: Request<proto::LoadModelRequest>,
    ) -> Result<Response<proto::LoadModelResponse>, Status> {
        let req = request.into_inner();

        info!(
            model_id = %req.model_id,
            container_id = %req.container_id,
            priority = req.priority,
            "LoadModel request"
        );

        // Verify container exists
        if let Err(e) = self.containers.get_container(&req.container_id).await {
            return Err(e.into());
        }

        // Estimate VRAM for the model (MVP: use hardcoded estimates)
        let vram_gb = estimate_model_vram(&req.model_id);

        // Check VRAM availability
        {
            let vram = self.vram_tracker.read().await;
            if !vram.can_allocate(vram_gb) {
                return Ok(Response::new(proto::LoadModelResponse {
                    success: false,
                    error_message: format!(
                        "Insufficient VRAM: need {} GB, available {} GB",
                        vram_gb,
                        vram.available()
                    ),
                    vram_used_gb: 0.0,
                    load_time_ms: 0,
                }));
            }
        }

        // MVP: Simulate model loading time
        let load_start = Instant::now();

        // In production, this would call the Python container's model loading endpoint
        // For MVP, we just register the model
        if let Err(e) = self
            .containers
            .register_model(
                &req.container_id,
                req.model_id.clone(),
                vram_gb,
                req.priority,
            )
            .await
        {
            return Err(e.into());
        }

        // Track VRAM allocation
        {
            let mut vram = self.vram_tracker.write().await;
            vram.allocate(&req.model_id, vram_gb);
        }

        let load_time_ms = load_start.elapsed().as_millis() as u64;

        info!(
            model_id = %req.model_id,
            container_id = %req.container_id,
            vram_gb = vram_gb,
            load_time_ms = load_time_ms,
            "Model loaded successfully"
        );

        Ok(Response::new(proto::LoadModelResponse {
            success: true,
            error_message: String::new(),
            vram_used_gb: vram_gb,
            load_time_ms,
        }))
    }

    /// Unload a model from a container.
    async fn unload_model(
        &self,
        request: Request<proto::UnloadModelRequest>,
    ) -> Result<Response<proto::UnloadModelResponse>, Status> {
        let req = request.into_inner();

        info!(
            model_id = %req.model_id,
            container_id = %req.container_id,
            "UnloadModel request"
        );

        match self
            .containers
            .unregister_model(&req.container_id, &req.model_id)
            .await
        {
            Ok(vram_freed) => {
                // Free VRAM allocation
                {
                    let mut vram = self.vram_tracker.write().await;
                    vram.deallocate(&req.model_id);
                }

                info!(
                    model_id = %req.model_id,
                    vram_freed_gb = vram_freed,
                    "Model unloaded"
                );

                Ok(Response::new(proto::UnloadModelResponse {
                    success: true,
                    error_message: String::new(),
                    vram_freed_gb: vram_freed,
                }))
            }
            Err(e) => {
                error!(
                    model_id = %req.model_id,
                    error = %e,
                    "Failed to unload model"
                );
                Err(e.into())
            }
        }
    }

    /// Get list of loaded models.
    async fn get_loaded_models(
        &self,
        request: Request<proto::GetLoadedModelsRequest>,
    ) -> Result<Response<proto::GetLoadedModelsResponse>, Status> {
        let req = request.into_inner();

        let container_id = if req.container_id.is_empty() {
            None
        } else {
            Some(req.container_id.as_str())
        };

        let models = self.containers.get_loaded_models(container_id).await;

        let loaded_models: Vec<proto::LoadedModel> = models
            .into_iter()
            .map(|(container_id, model)| proto::LoadedModel {
                model_id: model.model_id,
                container_id,
                vram_gb: model.vram_gb,
                state: model.state.as_str().to_string(),
                priority: model.priority,
                last_used_timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
                    .saturating_sub(model.last_used.elapsed().as_secs()),
            })
            .collect();

        Ok(Response::new(proto::GetLoadedModelsResponse {
            models: loaded_models,
        }))
    }

    /// Execute a task using a loaded model.
    async fn execute_task(
        &self,
        request: Request<proto::ExecuteTaskRequest>,
    ) -> Result<Response<proto::ExecuteTaskResponse>, Status> {
        let req = request.into_inner();

        info!(
            task_id = %req.task_id,
            model_id = %req.model_id,
            input_cid = %req.input_cid,
            "ExecuteTask request"
        );

        // Find container with the model
        let container_id = self
            .find_container_with_model(&req.model_id)
            .await
            .ok_or_else(|| SidecarError::ModelNotLoaded(req.model_id.clone()))?;

        // Check if task already exists
        {
            let tasks = self.tasks.read().await;
            if tasks.contains_key(&req.task_id) {
                return Err(SidecarError::TaskAlreadyRunning(req.task_id).into());
            }
        }

        // Create task state
        let task_state = TaskState {
            task_id: req.task_id.clone(),
            model_id: req.model_id.clone(),
            container_id: container_id.clone(),
            status: TaskStatus::Running,
            progress: 0.0,
            current_stage: "initializing".to_string(),
            started_at: Some(Instant::now()),
            completed_at: None,
            error_message: None,
            output_cid: None,
        };

        // Register task
        {
            let mut tasks = self.tasks.write().await;
            tasks.insert(req.task_id.clone(), task_state);
        }

        // MVP: Simulate task execution
        // In production, this would:
        // 1. Call the Python container's execute endpoint
        // 2. Stream progress updates
        // 3. Return the output CID

        let execution_start = Instant::now();

        // Simulate some work
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Update model last used time
        let _ = self
            .containers
            .touch_model(&container_id, &req.model_id)
            .await;

        // Generate mock output CID
        let output_cid = format!(
            "ipfs://Qm{}Output",
            &req.task_id[..8.min(req.task_id.len())]
        );
        let execution_time_ms = execution_start.elapsed().as_millis() as u64;

        // Update task state to completed
        {
            let mut tasks = self.tasks.write().await;
            if let Some(task) = tasks.get_mut(&req.task_id) {
                task.status = TaskStatus::Completed;
                task.progress = 1.0;
                task.current_stage = "completed".to_string();
                task.completed_at = Some(Instant::now());
                task.output_cid = Some(output_cid.clone());
            }
        }

        info!(
            task_id = %req.task_id,
            output_cid = %output_cid,
            execution_time_ms = execution_time_ms,
            "Task completed"
        );

        Ok(Response::new(proto::ExecuteTaskResponse {
            success: true,
            error_message: String::new(),
            output_cid,
            execution_time_ms,
            result_metadata: vec![],
        }))
    }

    /// Get the status of a task.
    async fn get_task_status(
        &self,
        request: Request<proto::GetTaskStatusRequest>,
    ) -> Result<Response<proto::GetTaskStatusResponse>, Status> {
        let req = request.into_inner();

        let tasks = self.tasks.read().await;

        let task = tasks
            .get(&req.task_id)
            .ok_or_else(|| SidecarError::TaskNotFound(req.task_id.clone()))?;

        let eta_ms = if task.status == TaskStatus::Running {
            // MVP: Estimate remaining time based on progress
            if task.progress > 0.0 {
                let elapsed = task
                    .started_at
                    .map(|s| s.elapsed().as_millis())
                    .unwrap_or(0);
                let estimated_total = (elapsed as f32 / task.progress) as u64;
                estimated_total.saturating_sub(elapsed as u64)
            } else {
                0
            }
        } else {
            0
        };

        Ok(Response::new(proto::GetTaskStatusResponse {
            status: task.status.as_str().to_string(),
            progress: task.progress,
            error_message: task.error_message.clone().unwrap_or_default(),
            current_stage: task.current_stage.clone(),
            eta_ms,
        }))
    }

    /// Cancel a running task.
    async fn cancel_task(
        &self,
        request: Request<proto::CancelTaskRequest>,
    ) -> Result<Response<proto::CancelTaskResponse>, Status> {
        let req = request.into_inner();

        info!(
            task_id = %req.task_id,
            reason = %req.reason,
            "CancelTask request"
        );

        let mut tasks = self.tasks.write().await;

        let task = tasks
            .get_mut(&req.task_id)
            .ok_or_else(|| SidecarError::TaskNotFound(req.task_id.clone()))?;

        let was_running = task.status == TaskStatus::Running;

        // MVP: Just update the state
        // In production, this would also send a cancellation signal to the container
        task.status = TaskStatus::Cancelled;
        task.completed_at = Some(Instant::now());
        task.error_message = Some(format!("Cancelled: {}", req.reason));

        info!(
            task_id = %req.task_id,
            was_running = was_running,
            "Task cancelled"
        );

        Ok(Response::new(proto::CancelTaskResponse {
            success: true,
            error_message: String::new(),
            was_running,
        }))
    }

    /// Get VRAM usage status.
    async fn get_vram_status(
        &self,
        request: Request<proto::GetVramStatusRequest>,
    ) -> Result<Response<proto::GetVramStatusResponse>, Status> {
        let req = request.into_inner();

        debug!(gpu_id = %req.gpu_id, "GetVramStatus request");

        let vram = self.vram_tracker.read().await;

        let model_allocations: Vec<proto::ModelVram> = vram
            .allocations()
            .iter()
            .map(|(model_id, &vram_gb)| proto::ModelVram {
                model_id: model_id.clone(),
                vram_gb,
                container_id: String::new(), // MVP: Not tracking per-container
            })
            .collect();

        // MVP: Return single GPU status
        let gpu_statuses = vec![proto::GpuStatus {
            gpu_id: if req.gpu_id.is_empty() {
                "0".to_string()
            } else {
                req.gpu_id
            },
            gpu_name: "NVIDIA RTX 3060".to_string(), // MVP: Hardcoded
            total_vram_gb: vram.total(),
            used_vram_gb: vram.used(),
            temperature_celsius: 45.0, // MVP: Stubbed
            utilization_percent: 30.0, // MVP: Stubbed
        }];

        Ok(Response::new(proto::GetVramStatusResponse {
            total_vram_gb: vram.total(),
            used_vram_gb: vram.used(),
            available_vram_gb: vram.available(),
            model_allocations,
            gpu_statuses,
        }))
    }
}

/// Estimate VRAM usage for a model.
///
/// MVP: Uses hardcoded estimates based on model ID.
/// In production, this would query the model registry.
fn estimate_model_vram(model_id: &str) -> f32 {
    match model_id {
        "flux-schnell" => 6.0,
        "liveportrait" => 3.5,
        "kokoro-82m" => 0.4,
        "clip-vit-b-32" => 0.3,
        "clip-vit-l-14" => 0.6,
        _ => 1.0, // Default estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_task() {
        let service = SidecarService::new();

        // Start a container first
        let start_req = Request::new(proto::StartContainerRequest {
            container_id: "test-container".to_string(),
            image_cid: "test-image".to_string(),
            gpu_ids: vec!["0".to_string()],
            env_vars: HashMap::new(),
            resource_limits: None,
        });
        let resp = service.start_container(start_req).await.unwrap();
        assert!(resp.into_inner().success);

        // Load a model
        let load_req = Request::new(proto::LoadModelRequest {
            model_id: "flux-schnell".to_string(),
            container_id: "test-container".to_string(),
            priority: 0,
            config: vec![],
        });
        let resp = service.load_model(load_req).await.unwrap();
        assert!(resp.into_inner().success);

        // Execute a task
        let exec_req = Request::new(proto::ExecuteTaskRequest {
            task_id: "task-001".to_string(),
            model_id: "flux-schnell".to_string(),
            input_cid: "ipfs://QmInput".to_string(),
            parameters: b"{}".to_vec(),
            timeout_ms: 0,
            callback_endpoint: String::new(),
        });
        let resp = service.execute_task(exec_req).await.unwrap();
        let inner = resp.into_inner();
        assert!(inner.success);
        assert!(!inner.output_cid.is_empty());
    }

    #[tokio::test]
    async fn test_vram_tracking() {
        let service = SidecarService::new();

        // Start container
        let start_req = Request::new(proto::StartContainerRequest {
            container_id: "vram-test".to_string(),
            image_cid: "test-image".to_string(),
            gpu_ids: vec![],
            env_vars: HashMap::new(),
            resource_limits: None,
        });
        service.start_container(start_req).await.unwrap();

        // Check initial VRAM
        let vram_req = Request::new(proto::GetVramStatusRequest::default());
        let resp = service.get_vram_status(vram_req).await.unwrap();
        let initial_used = resp.into_inner().used_vram_gb;

        // Load model
        let load_req = Request::new(proto::LoadModelRequest {
            model_id: "flux-schnell".to_string(),
            container_id: "vram-test".to_string(),
            priority: 0,
            config: vec![],
        });
        service.load_model(load_req).await.unwrap();

        // Check VRAM after loading
        let vram_req = Request::new(proto::GetVramStatusRequest::default());
        let resp = service.get_vram_status(vram_req).await.unwrap();
        let after_load = resp.into_inner().used_vram_gb;

        assert!(after_load > initial_used);
        assert!((after_load - initial_used - 6.0).abs() < 0.1); // flux-schnell uses 6.0 GB

        // Unload model
        let unload_req = Request::new(proto::UnloadModelRequest {
            model_id: "flux-schnell".to_string(),
            container_id: "vram-test".to_string(),
            wait: false,
        });
        let resp = service.unload_model(unload_req).await.unwrap();
        assert!((resp.into_inner().vram_freed_gb - 6.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_cancel_task() {
        let service = SidecarService::new();

        // Setup
        let start_req = Request::new(proto::StartContainerRequest {
            container_id: "cancel-test".to_string(),
            image_cid: "test-image".to_string(),
            gpu_ids: vec![],
            env_vars: HashMap::new(),
            resource_limits: None,
        });
        service.start_container(start_req).await.unwrap();

        let load_req = Request::new(proto::LoadModelRequest {
            model_id: "flux-schnell".to_string(),
            container_id: "cancel-test".to_string(),
            priority: 0,
            config: vec![],
        });
        service.load_model(load_req).await.unwrap();

        // Execute task
        let exec_req = Request::new(proto::ExecuteTaskRequest {
            task_id: "cancel-task-001".to_string(),
            model_id: "flux-schnell".to_string(),
            input_cid: "ipfs://QmInput".to_string(),
            parameters: vec![],
            timeout_ms: 0,
            callback_endpoint: String::new(),
        });
        service.execute_task(exec_req).await.unwrap();

        // Cancel should work even on completed tasks
        let cancel_req = Request::new(proto::CancelTaskRequest {
            task_id: "cancel-task-001".to_string(),
            reason: "Test cancellation".to_string(),
        });
        let resp = service.cancel_task(cancel_req).await.unwrap();
        assert!(resp.into_inner().success);
    }
}
