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
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use tokio::sync::{mpsc, RwLock};
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

use crate::container::ContainerManager;
use crate::error::SidecarError;
use crate::plugins::{PluginPolicy, PluginRegistry};
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
    /// Plugin used for this task (if any)
    pub plugin_name: Option<String>,
    /// Container executing the task
    pub container_id: String,
    /// Lane for execution (if provided)
    pub lane: Option<u32>,
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

/// Completion event emitted when a task finishes successfully.
#[derive(Debug, Clone)]
pub struct TaskCompletionEvent {
    /// Completed task identifier.
    pub task_id: String,
    /// Output CID produced by the task.
    pub output_cid: String,
    /// Lane used for execution.
    pub lane: u32,
    /// Plugin name if provided.
    pub plugin_name: Option<String>,
    /// Model identifier used for execution.
    pub model_id: String,
    /// Execution time in milliseconds.
    pub execution_time_ms: u64,
}

/// Sender for task completion events.
pub type TaskCompletionSender = mpsc::UnboundedSender<TaskCompletionEvent>;

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

#[derive(Debug, Deserialize)]
struct VortexPluginIndex {
    plugins: Vec<VortexPluginEntry>,
}

#[derive(Debug, Deserialize)]
struct VortexPluginEntry {
    name: String,
    version: String,
    schema_version: String,
    supported_lanes: Vec<String>,
    deterministic: bool,
    max_latency_ms: u32,
    vram_gb: f32,
    max_concurrency: u32,
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
    /// Enable plugin registry
    pub plugins_enabled: bool,
    /// Plugin directory (manifest.yaml files)
    pub plugins_dir: Option<PathBuf>,
    /// Plugin policy constraints
    pub plugin_policy: PluginPolicy,
    /// Optional Vortex plugin index file for reconciliation
    pub vortex_plugin_index: Option<PathBuf>,
    /// Require Vortex plugin index to match sidecar registry
    pub require_vortex_plugin_match: bool,
    /// Command to execute plugin tasks (e.g., python runner)
    pub plugin_exec_command: Option<Vec<String>>,
}

impl Default for SidecarServiceConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:50050".parse().unwrap(),
            max_task_duration: Duration::from_secs(300), // 5 minutes
            task_retention: Duration::from_secs(3600),   // 1 hour
            plugins_enabled: true,
            plugins_dir: Some(PathBuf::from("plugins")),
            plugin_policy: PluginPolicy::default(),
            vortex_plugin_index: Some(PathBuf::from("plugins/index.json")),
            require_vortex_plugin_match: false,
            plugin_exec_command: Some(vec![
                "python3".to_string(),
                "-m".to_string(),
                "vortex.plugins.runner".to_string(),
            ]),
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
    /// Plugin registry
    plugins: Arc<PluginRegistry>,
    /// Whether plugins are enabled
    plugins_enabled: bool,
    /// Service configuration
    config: SidecarServiceConfig,
    /// Optional task completion notifier.
    task_completion_tx: Option<TaskCompletionSender>,
}

impl SidecarService {
    /// Create a new sidecar service with default configuration.
    pub fn new() -> Self {
        Self::with_config(SidecarServiceConfig::default())
    }

    /// Create a new sidecar service with custom configuration.
    pub fn with_config(config: SidecarServiceConfig) -> Self {
        let (plugins, plugins_enabled) = Self::build_plugins(&config);
        Self {
            containers: Arc::new(ContainerManager::new()),
            vram_tracker: Arc::new(RwLock::new(VramTracker::new())),
            tasks: Arc::new(RwLock::new(HashMap::new())),
            plugins: Arc::new(plugins),
            plugins_enabled,
            config,
            task_completion_tx: None,
        }
    }

    /// Create a new sidecar service with injected dependencies.
    pub fn with_dependencies(
        containers: Arc<ContainerManager>,
        vram_tracker: Arc<RwLock<VramTracker>>,
        config: SidecarServiceConfig,
    ) -> Self {
        let (plugins, plugins_enabled) = Self::build_plugins(&config);
        Self {
            containers,
            vram_tracker,
            tasks: Arc::new(RwLock::new(HashMap::new())),
            plugins: Arc::new(plugins),
            plugins_enabled,
            config,
            task_completion_tx: None,
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

    /// Attach a task completion sender to emit task completion events.
    pub fn set_task_completion_sender(&mut self, sender: TaskCompletionSender) {
        self.task_completion_tx = Some(sender);
    }

    fn emit_task_completion(&self, event: TaskCompletionEvent) {
        if let Some(sender) = &self.task_completion_tx {
            let _ = sender.send(event);
        }
    }

    fn build_plugins(config: &SidecarServiceConfig) -> (PluginRegistry, bool) {
        if !config.plugins_enabled {
            return (PluginRegistry::empty(config.plugin_policy.clone()), false);
        }

        let Some(dir) = &config.plugins_dir else {
            warn!("Plugin registry enabled but plugins_dir not set; disabling plugins");
            return (PluginRegistry::empty(config.plugin_policy.clone()), false);
        };

        match PluginRegistry::load_from_dir(dir, config.plugin_policy.clone()) {
            Ok(registry) => {
                if config.require_vortex_plugin_match {
                    if let Err(err) = Self::reconcile_registry(&registry, config) {
                        warn!(error = %err, "Plugin reconciliation failed; disabling plugins");
                        return (PluginRegistry::empty(config.plugin_policy.clone()), false);
                    }
                }
                (registry, true)
            }
            Err(err) => {
                warn!(error = %err, "Failed to load plugins; disabling registry");
                (PluginRegistry::empty(config.plugin_policy.clone()), false)
            }
        }
    }

    fn reconcile_registry(
        registry: &PluginRegistry,
        config: &SidecarServiceConfig,
    ) -> Result<(), SidecarError> {
        let Some(index_path) = &config.vortex_plugin_index else {
            return Err(SidecarError::PluginMismatch(
                "vortex plugin index not configured".to_string(),
            ));
        };

        let data = std::fs::read_to_string(index_path)
            .map_err(|e| SidecarError::PluginRegistryError(e.to_string()))?;

        let index: VortexPluginIndex = serde_json::from_str(&data)
            .map_err(|e| SidecarError::PluginRegistryError(e.to_string()))?;

        for manifest in registry.list() {
            let entry = index
                .plugins
                .iter()
                .find(|p| p.name == manifest.name)
                .ok_or_else(|| SidecarError::PluginMismatch(manifest.name.clone()))?;

            if entry.version != manifest.version {
                return Err(SidecarError::PluginMismatch(format!(
                    "{} version mismatch (sidecar={}, vortex={})",
                    manifest.name, manifest.version, entry.version
                )));
            }

            if entry.schema_version != manifest.schema_version {
                return Err(SidecarError::PluginMismatch(format!(
                    "{} schema_version mismatch (sidecar={}, vortex={})",
                    manifest.name, manifest.schema_version, entry.schema_version
                )));
            }

            let mut sidecar_lanes: Vec<String> = manifest
                .supported_lanes
                .iter()
                .map(|lane| lane.to_lowercase())
                .collect();
            let mut vortex_lanes: Vec<String> = entry
                .supported_lanes
                .iter()
                .map(|lane| lane.to_lowercase())
                .collect();
            sidecar_lanes.sort();
            vortex_lanes.sort();

            if sidecar_lanes != vortex_lanes {
                return Err(SidecarError::PluginMismatch(format!(
                    "{} supported_lanes mismatch (sidecar={:?}, vortex={:?})",
                    manifest.name, sidecar_lanes, vortex_lanes
                )));
            }

            if entry.deterministic != manifest.deterministic {
                return Err(SidecarError::PluginMismatch(format!(
                    "{} deterministic mismatch (sidecar={}, vortex={})",
                    manifest.name, manifest.deterministic, entry.deterministic
                )));
            }

            if entry.max_latency_ms != manifest.resources.max_latency_ms {
                return Err(SidecarError::PluginMismatch(format!(
                    "{} max_latency_ms mismatch (sidecar={}, vortex={})",
                    manifest.name, manifest.resources.max_latency_ms, entry.max_latency_ms
                )));
            }

            if entry.max_concurrency != manifest.resources.max_concurrency {
                return Err(SidecarError::PluginMismatch(format!(
                    "{} max_concurrency mismatch (sidecar={}, vortex={})",
                    manifest.name, manifest.resources.max_concurrency, entry.max_concurrency
                )));
            }

            if (entry.vram_gb - manifest.resources.vram_gb).abs() > 0.0001 {
                return Err(SidecarError::PluginMismatch(format!(
                    "{} vram_gb mismatch (sidecar={}, vortex={})",
                    manifest.name, manifest.resources.vram_gb, entry.vram_gb
                )));
            }
        }

        Ok(())
    }

    async fn execute_plugin_command(
        &self,
        req: &proto::ExecuteTaskRequest,
    ) -> Result<(String, Vec<u8>, u64), SidecarError> {
        if self.plugins.policy().allow_untrusted
            && std::env::var("NSN_ALLOW_UNSANDBOXED_PLUGINS").is_err()
        {
            return Err(SidecarError::PluginPolicyViolation(
                "untrusted plugins require sandboxed execution".to_string(),
            ));
        }

        let command = self.config.plugin_exec_command.as_ref().ok_or_else(|| {
            SidecarError::TaskExecutionFailed("plugin runner not configured".to_string())
        })?;

        if command.is_empty() {
            return Err(SidecarError::TaskExecutionFailed(
                "plugin runner command is empty".to_string(),
            ));
        }

        let mut payload = if req.parameters.is_empty() {
            serde_json::Map::new()
        } else {
            let value: serde_json::Value = serde_json::from_slice(&req.parameters)
                .map_err(|e| SidecarError::InvalidRequest(e.to_string()))?;
            value.as_object().cloned().ok_or_else(|| {
                SidecarError::InvalidRequest("parameters must be a JSON object".to_string())
            })?
        };

        payload
            .entry("input_cid".to_string())
            .or_insert(serde_json::Value::String(req.input_cid.clone()));
        payload
            .entry("task_id".to_string())
            .or_insert(serde_json::Value::String(req.task_id.clone()));
        payload
            .entry("lane".to_string())
            .or_insert(serde_json::Value::Number(serde_json::Number::from(
                req.lane,
            )));
        payload
            .entry("plugin".to_string())
            .or_insert(serde_json::Value::String(req.plugin_name.clone()));
        if !req.model_id.is_empty() {
            payload
                .entry("model_id".to_string())
                .or_insert(serde_json::Value::String(req.model_id.clone()));
        }

        let payload_str = serde_json::Value::Object(payload).to_string();

        let mut cmd = tokio::process::Command::new(&command[0]);
        if command.len() > 1 {
            cmd.args(&command[1..]);
        }

        cmd.arg("--plugin")
            .arg(req.plugin_name.clone())
            .arg("--payload")
            .arg(payload_str);

        if req.timeout_ms > 0 {
            cmd.arg("--timeout-ms").arg(req.timeout_ms.to_string());
        }

        let output = cmd
            .output()
            .await
            .map_err(|e| SidecarError::TaskExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(SidecarError::TaskExecutionFailed(stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let response: serde_json::Value = serde_json::from_str(&stdout)
            .map_err(|e| SidecarError::TaskExecutionFailed(e.to_string()))?;

        let output_obj = response
            .get("output")
            .and_then(|val| val.as_object())
            .ok_or_else(|| {
                SidecarError::TaskExecutionFailed(
                    "plugin output missing 'output' object".to_string(),
                )
            })?;

        let output_cid = output_obj
            .get("output_cid")
            .and_then(|val| val.as_str())
            .ok_or_else(|| {
                SidecarError::TaskExecutionFailed("plugin output missing 'output_cid'".to_string())
            })?;

        let duration_ms = response
            .get("duration_ms")
            .and_then(|val| val.as_f64())
            .unwrap_or(0.0) as u64;

        Ok((
            output_cid.to_string(),
            response.to_string().into_bytes(),
            duration_ms,
        ))
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
        let mut req = request.into_inner();

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
        let mut req = request.into_inner();

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
            // Allocation should always succeed here since we checked before loading
            // If it fails, log error but don't fail the operation (model is already loaded)
            if let Err(e) = vram.allocate(&req.model_id, vram_gb) {
                warn!(
                    model_id = %req.model_id,
                    vram_gb = vram_gb,
                    error = %e,
                    "Failed to track VRAM allocation (model already loaded)"
                );
            }
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

    /// List available plugins from the registry.
    async fn list_plugins(
        &self,
        request: Request<proto::ListPluginsRequest>,
    ) -> Result<Response<proto::ListPluginsResponse>, Status> {
        if !self.plugins_enabled {
            return Err(SidecarError::PluginDisabled.into());
        }

        let req = request.into_inner();
        let lane_filter = req.lane.trim().to_lowercase();
        let plugins = self
            .plugins
            .list()
            .into_iter()
            .filter(|manifest| {
                if lane_filter.is_empty() {
                    true
                } else {
                    manifest
                        .supported_lanes
                        .iter()
                        .any(|lane| lane.eq_ignore_ascii_case(&lane_filter))
                }
            })
            .map(|manifest| proto::PluginInfo {
                name: manifest.name,
                version: manifest.version,
                supported_lanes: manifest.supported_lanes,
                deterministic: manifest.deterministic,
                max_latency_ms: manifest.resources.max_latency_ms,
                vram_required_mb: (manifest.resources.vram_gb * 1024.0) as u32,
            })
            .collect();

        Ok(Response::new(proto::ListPluginsResponse { plugins }))
    }

    /// Execute a task using a loaded model.
    async fn execute_task(
        &self,
        request: Request<proto::ExecuteTaskRequest>,
    ) -> Result<Response<proto::ExecuteTaskResponse>, Status> {
        let mut req = request.into_inner();

        info!(
            task_id = %req.task_id,
            model_id = %req.model_id,
            plugin_name = %req.plugin_name,
            input_cid = %req.input_cid,
            "ExecuteTask request"
        );

        if req.plugin_name.trim().is_empty() {
            let model_name = req.model_id.trim();
            if !model_name.is_empty()
                && self.plugins_enabled
                && self.plugins.get(model_name).is_some()
            {
                req.plugin_name = model_name.to_string();
            }
        }

        req.plugin_name = req.plugin_name.trim().to_string();
        req.model_id = req.model_id.trim().to_string();

        let using_plugin = !req.plugin_name.trim().is_empty();
        let lane = if using_plugin { Some(req.lane) } else { None };

        if !using_plugin {
            return Err(SidecarError::TaskExecutionFailed(
                "non-plugin execution is not supported".to_string(),
            )
            .into());
        }

        // Resolve container and validate plugin policy if needed
        if !self.plugins_enabled {
            return Err(SidecarError::PluginDisabled.into());
        }

        let plugin_name = req.plugin_name.trim();
        let manifest = self
            .plugins
            .get(plugin_name)
            .ok_or_else(|| SidecarError::PluginNotFound(plugin_name.to_string()))?;

        if req.lane > 1 {
            return Err(SidecarError::InvalidRequest("invalid lane value".to_string()).into());
        }

        if !manifest.supports_lane(req.lane) {
            return Err(SidecarError::PluginPolicyViolation(format!(
                "plugin '{}' does not support lane {}",
                manifest.name, req.lane
            ))
            .into());
        }

        self.plugins.policy().check(manifest, req.lane)?;

        let container_id = format!("plugin:{}", req.plugin_name);

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
            model_id: if using_plugin && req.model_id.is_empty() {
                req.plugin_name.clone()
            } else {
                req.model_id.clone()
            },
            plugin_name: if using_plugin {
                Some(req.plugin_name.clone())
            } else {
                None
            },
            container_id: container_id.clone(),
            lane,
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

        let execution_start = Instant::now();
        let mut output_cid = String::new();
        let mut result_metadata: Vec<u8> = vec![];

        match self.execute_plugin_command(&req).await {
            Ok((cid, metadata, _duration_ms)) => {
                output_cid = cid;
                result_metadata = metadata;
            }
            Err(err) => {
                let mut tasks = self.tasks.write().await;
                if let Some(task) = tasks.get_mut(&req.task_id) {
                    task.status = TaskStatus::Failed;
                    task.error_message = Some(err.to_string());
                    task.completed_at = Some(Instant::now());
                }

                return Ok(Response::new(proto::ExecuteTaskResponse {
                    success: false,
                    error_message: err.to_string(),
                    output_cid: String::new(),
                    execution_time_ms: execution_start.elapsed().as_millis() as u64,
                    result_metadata: vec![],
                }));
            }
        }

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

        if req.lane == 0 && !output_cid.is_empty() {
            self.emit_task_completion(TaskCompletionEvent {
                task_id: req.task_id.clone(),
                output_cid: output_cid.clone(),
                lane: req.lane,
                plugin_name: if req.plugin_name.is_empty() {
                    None
                } else {
                    Some(req.plugin_name.clone())
                },
                model_id: req.model_id.clone(),
                execution_time_ms,
            });
        }

        Ok(Response::new(proto::ExecuteTaskResponse {
            success: true,
            error_message: String::new(),
            output_cid,
            execution_time_ms,
            result_metadata,
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
    use bollard::Docker;
    use std::env;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    async fn docker_available() -> bool {
        if env::var("NSN_DOCKER_TESTS").ok().as_deref() != Some("1") {
            return false;
        }

        match Docker::connect_with_local_defaults() {
            Ok(client) => client.ping().await.is_ok(),
            Err(_) => false,
        }
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let mut path = env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        path.push(format!("nsn-{}-{}", prefix, nanos));
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    fn write_plugin_manifest(plugins_dir: &PathBuf) {
        let plugin_dir = plugins_dir.join("flux-schnell");
        fs::create_dir_all(&plugin_dir).expect("create plugin dir");
        let manifest = r#"
schema_version: "1"
name: "flux-schnell"
version: "0.1.0"
entrypoint: "runner"
description: "test plugin"
supported_lanes:
  - lane0
  - lane1
deterministic: true
resources:
  vram_gb: 6.0
  max_latency_ms: 1000
  max_concurrency: 1
io:
  input_schema: {}
  output_schema: {}
"#;
        fs::write(plugin_dir.join("manifest.yaml"), manifest).expect("write manifest");
    }

    fn write_runner_script(base_dir: &PathBuf) -> PathBuf {
        let script_path = base_dir.join("runner.sh");
        let script = r#"#!/bin/sh
echo '{"output":{"output_cid":"ipfs://QmTest"},"duration_ms":1}'
"#;
        fs::write(&script_path, script).expect("write runner script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script_path)
                .expect("script metadata")
                .permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script_path, perms).expect("set script permissions");
        }
        script_path
    }

    fn test_service_config() -> SidecarServiceConfig {
        let base_dir = unique_temp_dir("sidecar-tests");
        let plugins_dir = base_dir.join("plugins");
        fs::create_dir_all(&plugins_dir).expect("create plugins dir");
        write_plugin_manifest(&plugins_dir);
        let runner_path = write_runner_script(&base_dir);

        let mut policy = PluginPolicy::default();
        policy.allowlist.insert("flux-schnell".to_string());

        SidecarServiceConfig {
            plugins_enabled: true,
            plugins_dir: Some(plugins_dir),
            plugin_policy: policy,
            plugin_exec_command: Some(vec![runner_path.to_string_lossy().to_string()]),
            ..SidecarServiceConfig::default()
        }
    }

    #[tokio::test]
    async fn test_execute_task() {
        if !docker_available().await {
            return;
        }
        let service = SidecarService::with_config(test_service_config());

        // Start a container first
        let start_req = Request::new(proto::StartContainerRequest {
            container_id: "test-container".to_string(),
            image_cid: "alpine:latest".to_string(),
            gpu_ids: vec![],
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
            plugin_name: String::new(),
            lane: 0,
        });
        let resp = service.execute_task(exec_req).await.unwrap();
        let inner = resp.into_inner();
        assert!(inner.success);
        assert!(!inner.output_cid.is_empty());
    }

    #[tokio::test]
    async fn test_vram_tracking() {
        if !docker_available().await {
            return;
        }
        let service = SidecarService::with_config(test_service_config());

        // Start container
        let start_req = Request::new(proto::StartContainerRequest {
            container_id: "vram-test".to_string(),
            image_cid: "alpine:latest".to_string(),
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
        if !docker_available().await {
            return;
        }
        let service = SidecarService::with_config(test_service_config());

        // Setup
        let start_req = Request::new(proto::StartContainerRequest {
            container_id: "cancel-test".to_string(),
            image_cid: "alpine:latest".to_string(),
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
            plugin_name: String::new(),
            lane: 0,
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
