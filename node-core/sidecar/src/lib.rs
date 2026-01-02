//! NSN Sidecar - gRPC bridge to Python AI containers
//!
//! This crate provides the gRPC sidecar service that bridges the Rust
//! NSN node scheduler with Python AI model containers. The sidecar handles:
//!
//! - **Container Lifecycle**: Starting, stopping, and health checking containers
//! - **Model Management**: Loading, unloading, and tracking models in VRAM
//! - **Task Execution**: Executing AI generation tasks on loaded models
//! - **VRAM Monitoring**: Tracking and reporting GPU memory usage
//!
//! # Architecture
//!
//! ```text
//! NSN Node (Rust) <--gRPC--> Sidecar (Rust) <--gRPC--> Python Container (Vortex)
//! ```
//!
//! The sidecar runs alongside the NSN node and manages Python AI containers
//! that run models like Flux, LivePortrait, Kokoro, and CLIP.
//!
//! # Example
//!
//! ```no_run
//! use nsn_sidecar::{SidecarClient, SidecarClientBuilder};
//! use std::time::Duration;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Connect to the sidecar
//! let mut client = SidecarClientBuilder::new()
//!     .endpoint("http://127.0.0.1:50050")
//!     .connect_timeout(Duration::from_secs(5))
//!     .build()
//!     .await?;
//!
//! // Start a container
//! let resp = client.start_container(
//!     "vortex-container",
//!     "ghcr.io/nsn/vortex:latest",
//!     vec!["0".to_string()],
//! ).await?;
//!
//! if resp.success {
//!     println!("Container started at: {}", resp.container_endpoint);
//! }
//!
//! // Load a model
//! let resp = client.load_model("flux-schnell", "vortex-container", 0).await?;
//! if resp.success {
//!     println!("Model loaded, VRAM used: {} GB", resp.vram_used_gb);
//! }
//!
//! // Execute a plugin task (Lane 1 by default)
//! let resp = client.execute_plugin_task(
//!     "task-001",
//!     "flux-schnell",
//!     "ipfs://QmInput",
//!     b"{}".to_vec(),
//!     1,
//!     None,
//! ).await?;
//!
//! if resp.success {
//!     println!("Task completed, output: {}", resp.output_cid);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Server Example
//!
//! ```no_run
//! use nsn_sidecar::{SidecarService, SidecarServiceConfig};
//! use nsn_sidecar::proto::sidecar_server::SidecarServer;
//! use tonic::transport::Server;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = SidecarServiceConfig::default();
//! let service = SidecarService::with_config(config.clone());
//!
//! Server::builder()
//!     .add_service(SidecarServer::new(service))
//!     .serve(config.bind_addr)
//!     .await?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod client;
pub mod container;
pub mod error;
pub mod plugins;
pub mod service;
pub mod vram;

// Re-export main types for convenience
pub use client::{SidecarClient, SidecarClientBuilder, SidecarClientConfig};
pub use container::{
    ContainerInfo, ContainerManager, ContainerStatus, LoadedModelInfo, PreemptionManager,
    PreemptionReason, PreemptionResult, PreemptionStrategy,
};
pub use error::{SidecarError, SidecarResult};
pub use plugins::{PluginManifest, PluginPolicy, PluginRegistry, PluginResources};
pub use service::{SidecarService, SidecarServiceConfig, TaskState, TaskStatus};
pub use vram::{
    AllocationPolicy, NvidiaError, NvidiaGpu, VramBudget, VramError, VramManager, VramStatus,
    VramTracker,
};

/// Generated protobuf types and gRPC service definitions.
///
/// This module contains types generated from `proto/sidecar.proto` including:
/// - Request/Response types for all gRPC methods
/// - Service client (`sidecar_client::SidecarClient`)
/// - Service server trait (`sidecar_server::Sidecar`)
#[allow(missing_docs)]
pub mod proto {
    tonic::include_proto!("nsn.sidecar");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that all main types are accessible from the crate root.
    #[test]
    fn test_exports() {
        // Container types
        let _ = ContainerStatus::Running;
        let _ = container::ModelLoadState::Hot;

        // Service types
        let _ = TaskStatus::Pending;

        // Error types
        let err = SidecarError::ContainerNotFound("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    /// Verify that the proto module is accessible and contains expected types.
    #[test]
    fn test_proto_types() {
        // Request types
        let _ = proto::StartContainerRequest::default();
        let _ = proto::LoadModelRequest::default();
        let _ = proto::ExecuteTaskRequest::default();
        let _ = proto::GetVramStatusRequest::default();

        // Response types
        let _ = proto::StartContainerResponse::default();
        let _ = proto::LoadModelResponse::default();
        let _ = proto::ExecuteTaskResponse::default();
        let _ = proto::GetVramStatusResponse::default();
    }
}
