//! Container lifecycle management for the sidecar.
//!
//! This module handles:
//! - Starting and stopping Docker containers with GPU support
//! - Health checking containers
//! - Managing container state and metadata
//!
//! Note: For MVP, Docker/containerd integration is stubbed. The focus is on
//! the data structures and interfaces that will be used when real container
//! integration is implemented.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::error::{SidecarError, SidecarResult};
use crate::proto;

/// Container operational status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainerStatus {
    /// Container is starting up
    Starting,
    /// Container is running and healthy
    Running,
    /// Container is stopping gracefully
    Stopping,
    /// Container has stopped
    Stopped,
    /// Container is in an error state
    Error(String),
}

impl ContainerStatus {
    /// Returns true if the container is in a healthy running state
    pub fn is_healthy(&self) -> bool {
        matches!(self, ContainerStatus::Running)
    }

    /// Convert to string representation for gRPC responses
    pub fn as_str(&self) -> &'static str {
        match self {
            ContainerStatus::Starting => "starting",
            ContainerStatus::Running => "running",
            ContainerStatus::Stopping => "stopping",
            ContainerStatus::Stopped => "stopped",
            ContainerStatus::Error(_) => "error",
        }
    }
}

/// Information about a loaded model within a container
#[derive(Debug, Clone)]
pub struct LoadedModelInfo {
    /// Model identifier (e.g., "flux-schnell", "clip-vit-l-14")
    pub model_id: String,
    /// VRAM used by this model in GB
    pub vram_gb: f32,
    /// Current model state
    pub state: ModelLoadState,
    /// Loading priority (0 = critical, higher = lower priority)
    pub priority: u32,
    /// When the model was last used
    pub last_used: Instant,
}

/// Model loading state within a container
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelLoadState {
    /// Model is actively being loaded
    Loading,
    /// Model is hot (ready for immediate use)
    Hot,
    /// Model is warm (cached but may need warmup)
    Warm,
    /// Model is being unloaded
    Unloading,
}

impl ModelLoadState {
    /// Convert the model load state to a string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelLoadState::Loading => "loading",
            ModelLoadState::Hot => "hot",
            ModelLoadState::Warm => "warm",
            ModelLoadState::Unloading => "unloading",
        }
    }
}

/// Container metadata and state
#[derive(Debug, Clone)]
pub struct ContainerInfo {
    /// Unique container identifier
    pub id: String,
    /// Container image CID (IPFS or registry reference)
    pub image_cid: String,
    /// gRPC endpoint for communicating with the container
    pub endpoint: String,
    /// Current container status
    pub status: ContainerStatus,
    /// Assigned GPU device IDs
    pub gpu_ids: Vec<String>,
    /// Models currently loaded in this container
    pub loaded_models: Vec<LoadedModelInfo>,
    /// When the container was started
    pub started_at: Instant,
    /// Resource limits applied to this container
    pub resource_limits: ResourceLimits,
}

/// Resource limits for a container
#[derive(Debug, Clone, Default)]
pub struct ResourceLimits {
    /// Memory limit in bytes (0 = unlimited)
    pub memory_bytes: u64,
    /// CPU shares (relative weight)
    pub cpu_shares: u32,
    /// VRAM limit in GB (informational)
    pub vram_limit_gb: f32,
}

impl From<proto::ResourceLimits> for ResourceLimits {
    fn from(limits: proto::ResourceLimits) -> Self {
        Self {
            memory_bytes: limits.memory_bytes,
            cpu_shares: limits.cpu_shares,
            vram_limit_gb: limits.vram_limit_gb,
        }
    }
}

/// Container runtime metrics
#[derive(Debug, Clone, Default)]
pub struct ContainerMetrics {
    /// CPU usage percentage (0-100)
    pub cpu_usage_percent: f32,
    /// Memory used in bytes
    pub memory_used_bytes: u64,
    /// Memory limit in bytes
    pub memory_limit_bytes: u64,
    /// GPU utilization percentage (0-100)
    pub gpu_utilization_percent: f32,
}

/// Manages container lifecycle operations.
///
/// This struct handles starting, stopping, and monitoring Docker containers
/// that run the Python AI models. For MVP, Docker integration is stubbed.
#[derive(Debug)]
pub struct ContainerManager {
    /// Active containers indexed by ID
    containers: Arc<RwLock<HashMap<String, ContainerInfo>>>,
    /// Default container configuration
    config: ContainerManagerConfig,
}

/// Configuration for the container manager
#[derive(Debug, Clone)]
pub struct ContainerManagerConfig {
    /// Default gRPC port for containers
    pub default_grpc_port: u16,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Graceful shutdown timeout
    pub shutdown_timeout: Duration,
    /// Maximum containers allowed
    pub max_containers: usize,
}

impl Default for ContainerManagerConfig {
    fn default() -> Self {
        Self {
            default_grpc_port: 50051,
            health_check_interval: Duration::from_secs(10),
            shutdown_timeout: Duration::from_secs(30),
            max_containers: 10,
        }
    }
}

impl ContainerManager {
    /// Create a new container manager with default configuration
    pub fn new() -> Self {
        Self::with_config(ContainerManagerConfig::default())
    }

    /// Create a new container manager with custom configuration
    pub fn with_config(config: ContainerManagerConfig) -> Self {
        Self {
            containers: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Start a new container with the given configuration.
    ///
    /// # Arguments
    /// * `container_id` - Unique identifier for the container
    /// * `image_cid` - Container image reference (IPFS CID or registry path)
    /// * `gpu_ids` - GPU device IDs to assign
    /// * `resource_limits` - Optional resource limits
    ///
    /// # Returns
    /// The gRPC endpoint for the started container
    pub async fn start_container(
        &self,
        container_id: String,
        image_cid: String,
        gpu_ids: Vec<String>,
        resource_limits: Option<ResourceLimits>,
    ) -> SidecarResult<String> {
        let mut containers = self.containers.write().await;

        // Check if container already exists
        if containers.contains_key(&container_id) {
            return Err(SidecarError::ContainerAlreadyExists(container_id));
        }

        // Check container limit
        if containers.len() >= self.config.max_containers {
            return Err(SidecarError::ContainerLimitReached(self.config.max_containers));
        }

        info!(
            container_id = %container_id,
            image = %image_cid,
            gpu_count = gpu_ids.len(),
            "Starting container"
        );

        // MVP: Stub Docker integration
        // In production, this would:
        // 1. Pull image if not present
        // 2. Create container with GPU runtime
        // 3. Start container
        // 4. Wait for gRPC endpoint to be ready

        // Generate endpoint (in production, this comes from Docker)
        let port = self.config.default_grpc_port + containers.len() as u16;
        let endpoint = format!("127.0.0.1:{}", port);

        let container_info = ContainerInfo {
            id: container_id.clone(),
            image_cid,
            endpoint: endpoint.clone(),
            status: ContainerStatus::Running,
            gpu_ids,
            loaded_models: Vec::new(),
            started_at: Instant::now(),
            resource_limits: resource_limits.unwrap_or_default(),
        };

        containers.insert(container_id.clone(), container_info);

        debug!(container_id = %container_id, endpoint = %endpoint, "Container started");

        Ok(endpoint)
    }

    /// Stop a container gracefully or forcefully.
    ///
    /// # Arguments
    /// * `container_id` - Container to stop
    /// * `force` - If true, force kill without graceful shutdown
    /// * `timeout` - Graceful shutdown timeout (uses default if None)
    pub async fn stop_container(
        &self,
        container_id: &str,
        force: bool,
        timeout: Option<Duration>,
    ) -> SidecarResult<()> {
        let mut containers = self.containers.write().await;

        let container = containers
            .get_mut(container_id)
            .ok_or_else(|| SidecarError::ContainerNotFound(container_id.to_string()))?;

        info!(
            container_id = %container_id,
            force = force,
            "Stopping container"
        );

        // Update status to stopping
        container.status = ContainerStatus::Stopping;

        // MVP: Stub Docker integration
        // In production, this would:
        // 1. Send SIGTERM to container
        // 2. Wait for graceful shutdown or timeout
        // 3. Send SIGKILL if force or timeout exceeded
        // 4. Remove container

        let _timeout = timeout.unwrap_or(self.config.shutdown_timeout);

        // Remove from active containers
        containers.remove(container_id);

        debug!(container_id = %container_id, "Container stopped");

        Ok(())
    }

    /// Check container health status.
    ///
    /// # Arguments
    /// * `container_id` - Container to check
    /// * `include_metrics` - If true, include detailed metrics
    ///
    /// # Returns
    /// Tuple of (healthy, status, uptime_seconds, optional_metrics)
    pub async fn health_check(
        &self,
        container_id: &str,
        include_metrics: bool,
    ) -> SidecarResult<(bool, String, u64, Option<ContainerMetrics>)> {
        let containers = self.containers.read().await;

        let container = containers
            .get(container_id)
            .ok_or_else(|| SidecarError::ContainerNotFound(container_id.to_string()))?;

        let healthy = container.status.is_healthy();
        let status = container.status.as_str().to_string();
        let uptime_seconds = container.started_at.elapsed().as_secs();

        // MVP: Return stubbed metrics
        let metrics = if include_metrics {
            Some(ContainerMetrics {
                cpu_usage_percent: 15.0,
                memory_used_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
                memory_limit_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
                gpu_utilization_percent: 45.0,
            })
        } else {
            None
        };

        Ok((healthy, status, uptime_seconds, metrics))
    }

    /// Get information about a container.
    pub async fn get_container(&self, container_id: &str) -> SidecarResult<ContainerInfo> {
        let containers = self.containers.read().await;

        containers
            .get(container_id)
            .cloned()
            .ok_or_else(|| SidecarError::ContainerNotFound(container_id.to_string()))
    }

    /// List all active containers.
    pub async fn list_containers(&self) -> Vec<ContainerInfo> {
        let containers = self.containers.read().await;
        containers.values().cloned().collect()
    }

    /// Add a loaded model to a container's registry.
    pub async fn register_model(
        &self,
        container_id: &str,
        model_id: String,
        vram_gb: f32,
        priority: u32,
    ) -> SidecarResult<()> {
        let mut containers = self.containers.write().await;

        let container = containers
            .get_mut(container_id)
            .ok_or_else(|| SidecarError::ContainerNotFound(container_id.to_string()))?;

        // Check if model already loaded
        if container.loaded_models.iter().any(|m| m.model_id == model_id) {
            warn!(
                container_id = %container_id,
                model_id = %model_id,
                "Model already registered"
            );
            return Ok(());
        }

        container.loaded_models.push(LoadedModelInfo {
            model_id: model_id.clone(),
            vram_gb,
            state: ModelLoadState::Hot,
            priority,
            last_used: Instant::now(),
        });

        debug!(
            container_id = %container_id,
            model_id = %model_id,
            vram_gb = vram_gb,
            "Model registered"
        );

        Ok(())
    }

    /// Remove a model from a container's registry.
    pub async fn unregister_model(
        &self,
        container_id: &str,
        model_id: &str,
    ) -> SidecarResult<f32> {
        let mut containers = self.containers.write().await;

        let container = containers
            .get_mut(container_id)
            .ok_or_else(|| SidecarError::ContainerNotFound(container_id.to_string()))?;

        let idx = container
            .loaded_models
            .iter()
            .position(|m| m.model_id == model_id)
            .ok_or_else(|| SidecarError::ModelNotLoaded(model_id.to_string()))?;

        let model = container.loaded_models.remove(idx);

        debug!(
            container_id = %container_id,
            model_id = %model_id,
            vram_freed = model.vram_gb,
            "Model unregistered"
        );

        Ok(model.vram_gb)
    }

    /// Get all loaded models across all containers or for a specific container.
    pub async fn get_loaded_models(
        &self,
        container_id: Option<&str>,
    ) -> Vec<(String, LoadedModelInfo)> {
        let containers = self.containers.read().await;

        let mut models = Vec::new();

        for (cid, container) in containers.iter() {
            if let Some(filter_id) = container_id {
                if cid != filter_id {
                    continue;
                }
            }

            for model in &container.loaded_models {
                models.push((cid.clone(), model.clone()));
            }
        }

        models
    }

    /// Update the last used time for a model (used after task execution).
    pub async fn touch_model(&self, container_id: &str, model_id: &str) -> SidecarResult<()> {
        let mut containers = self.containers.write().await;

        let container = containers
            .get_mut(container_id)
            .ok_or_else(|| SidecarError::ContainerNotFound(container_id.to_string()))?;

        if let Some(model) = container
            .loaded_models
            .iter_mut()
            .find(|m| m.model_id == model_id)
        {
            model.last_used = Instant::now();
            Ok(())
        } else {
            Err(SidecarError::ModelNotLoaded(model_id.to_string()))
        }
    }
}

impl Default for ContainerManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_start_stop_container() {
        let manager = ContainerManager::new();

        // Start container
        let endpoint = manager
            .start_container(
                "test-container-1".to_string(),
                "ipfs://QmTest".to_string(),
                vec!["0".to_string()],
                None,
            )
            .await
            .expect("Failed to start container");

        assert!(endpoint.contains("127.0.0.1"));

        // Verify container exists
        let container = manager
            .get_container("test-container-1")
            .await
            .expect("Container not found");
        assert_eq!(container.id, "test-container-1");
        assert!(container.status.is_healthy());

        // Stop container
        manager
            .stop_container("test-container-1", false, None)
            .await
            .expect("Failed to stop container");

        // Verify container removed
        assert!(manager.get_container("test-container-1").await.is_err());
    }

    #[tokio::test]
    async fn test_health_check() {
        let manager = ContainerManager::new();

        manager
            .start_container(
                "health-test".to_string(),
                "ipfs://QmTest".to_string(),
                vec![],
                None,
            )
            .await
            .expect("Failed to start container");

        let (healthy, status, uptime, metrics) = manager
            .health_check("health-test", true)
            .await
            .expect("Health check failed");

        assert!(healthy);
        assert_eq!(status, "running");
        assert!(uptime < 5); // Just started
        assert!(metrics.is_some());
    }

    #[tokio::test]
    async fn test_load_unload_model() {
        let manager = ContainerManager::new();

        manager
            .start_container(
                "model-test".to_string(),
                "ipfs://QmTest".to_string(),
                vec!["0".to_string()],
                None,
            )
            .await
            .expect("Failed to start container");

        // Register model
        manager
            .register_model("model-test", "flux-schnell".to_string(), 6.0, 0)
            .await
            .expect("Failed to register model");

        // Check model is registered
        let models = manager.get_loaded_models(Some("model-test")).await;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].1.model_id, "flux-schnell");
        assert_eq!(models[0].1.vram_gb, 6.0);

        // Unregister model
        let freed = manager
            .unregister_model("model-test", "flux-schnell")
            .await
            .expect("Failed to unregister model");
        assert_eq!(freed, 6.0);

        // Verify model removed
        let models = manager.get_loaded_models(Some("model-test")).await;
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_container_limit() {
        let config = ContainerManagerConfig {
            max_containers: 2,
            ..Default::default()
        };
        let manager = ContainerManager::with_config(config);

        // Start first container
        manager
            .start_container("c1".to_string(), "img".to_string(), vec![], None)
            .await
            .expect("Failed to start c1");

        // Start second container
        manager
            .start_container("c2".to_string(), "img".to_string(), vec![], None)
            .await
            .expect("Failed to start c2");

        // Third should fail
        let result = manager
            .start_container("c3".to_string(), "img".to_string(), vec![], None)
            .await;
        assert!(matches!(result, Err(SidecarError::ContainerLimitReached(2))));
    }

    #[tokio::test]
    async fn test_duplicate_container() {
        let manager = ContainerManager::new();

        manager
            .start_container("dup".to_string(), "img".to_string(), vec![], None)
            .await
            .expect("Failed to start container");

        let result = manager
            .start_container("dup".to_string(), "img".to_string(), vec![], None)
            .await;

        assert!(matches!(
            result,
            Err(SidecarError::ContainerAlreadyExists(_))
        ));
    }
}
