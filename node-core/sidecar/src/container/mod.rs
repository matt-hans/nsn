//! Container lifecycle management for the sidecar.
//!
//! This module handles:
//! - Starting and stopping Docker containers with GPU support
//! - Health checking containers
//! - Managing container state and metadata
//! - Preempting containers when necessary
//!
//! Note: For MVP, Docker/containerd integration is stubbed. The focus is on
//! the data structures and interfaces that will be used when real container
//! integration is implemented.

mod manager;
mod preempt;

pub use manager::{
    ContainerInfo, ContainerManager, ContainerManagerConfig, ContainerMetrics, ContainerStatus,
    LoadedModelInfo, ModelLoadState, ResourceLimits,
};
pub use preempt::{PreemptionManager, PreemptionReason, PreemptionResult, PreemptionStrategy};
