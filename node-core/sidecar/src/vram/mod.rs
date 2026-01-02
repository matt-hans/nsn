//! VRAM management module for the NSN sidecar.
//!
//! This module provides comprehensive VRAM tracking, budget management,
//! and optional NVIDIA GPU integration via NVML.
//!
//! # Components
//!
//! - [`VramTracker`]: Tracks VRAM allocations per model
//! - [`VramBudget`]: Defines budget policies and limits
//! - [`NvidiaGpu`]: NVML integration for real GPU queries (feature-gated)
//! - [`VramManager`]: High-level manager combining all components
//!
//! # Feature Gates
//!
//! The `nvidia` feature enables NVML integration for querying actual GPU
//! statistics. When disabled, the manager operates in tracking-only mode.
//!
//! ```toml
//! nsn-sidecar = { version = "*", features = ["nvidia"] }
//! ```
//!
//! # Example
//!
//! ```no_run
//! use nsn_sidecar::vram::{VramManager, AllocationPolicy};
//!
//! // Create manager (will try to detect GPU if nvidia feature enabled)
//! let mut manager = VramManager::new();
//!
//! // Allocate VRAM for models
//! manager.allocate("flux-schnell", 6.0).expect("allocation failed");
//! manager.allocate("liveportrait", 3.5).expect("allocation failed");
//!
//! // Check status
//! let status = manager.status();
//! println!("Used: {} GB, Available: {} GB", status.used_gb, status.available_gb);
//!
//! // Deallocate when done
//! manager.deallocate("flux-schnell");
//! ```

pub mod budget;
pub mod nvidia;
pub mod tracker;

pub use budget::{presets, AllocationPolicy, VramBudget};
pub use nvidia::{is_nvml_available, try_create_gpu, GpuInfo, NvidiaError, NvidiaGpu};
pub use tracker::{VramTracker, DEFAULT_TOTAL_VRAM_GB};

use tracing::debug;

#[cfg(feature = "nvidia")]
use tracing::{info, warn};

/// Error type for VRAM management operations.
#[derive(Debug, Clone)]
pub enum VramError {
    /// Allocation would exceed the VRAM budget.
    BudgetExceeded {
        /// The requested allocation size.
        requested_gb: f32,
        /// The currently used VRAM.
        used_gb: f32,
        /// The total allocatable VRAM.
        allocatable_gb: f32,
    },
    /// Model is already allocated.
    AlreadyAllocated(String),
    /// Model was not found for deallocation.
    NotFound(String),
    /// GPU query failed.
    GpuError(NvidiaError),
}

impl std::fmt::Display for VramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VramError::BudgetExceeded {
                requested_gb,
                used_gb,
                allocatable_gb,
            } => write!(
                f,
                "Allocation of {} GB would exceed budget (used: {}, allocatable: {})",
                requested_gb, used_gb, allocatable_gb
            ),
            VramError::AlreadyAllocated(model) => {
                write!(f, "Model '{}' is already allocated", model)
            }
            VramError::NotFound(model) => write!(f, "Model '{}' not found", model),
            VramError::GpuError(e) => write!(f, "GPU error: {}", e),
        }
    }
}

impl std::error::Error for VramError {}

impl From<NvidiaError> for VramError {
    fn from(e: NvidiaError) -> Self {
        VramError::GpuError(e)
    }
}

/// Current VRAM status snapshot.
#[derive(Debug, Clone)]
pub struct VramStatus {
    /// Total VRAM budget in GB.
    pub total_gb: f32,
    /// Currently used VRAM (tracked) in GB.
    pub used_gb: f32,
    /// Available VRAM (tracked) in GB.
    pub available_gb: f32,
    /// VRAM tracked by allocations in GB.
    pub tracked_gb: f32,
    /// Actual GPU VRAM usage if available (from GPU query).
    pub actual_gb: Option<f32>,
    /// GPU name if available.
    pub gpu_name: Option<String>,
    /// GPU temperature if available.
    pub gpu_temp_c: Option<u32>,
    /// GPU utilization if available.
    pub gpu_utilization: Option<u32>,
    /// Number of allocated models.
    pub model_count: usize,
    /// Whether GPU queries are available.
    pub gpu_available: bool,
}

/// High-level VRAM manager combining tracker with GPU queries.
///
/// The VramManager provides a unified interface for:
/// - Tracking VRAM allocations per model
/// - Enforcing budget policies (strict, soft, dynamic)
/// - Querying actual GPU statistics (when nvidia feature enabled)
/// - Syncing tracked state with real GPU usage
#[derive(Debug)]
pub struct VramManager {
    /// Model allocation tracker.
    tracker: VramTracker,
    /// Budget configuration.
    budget: VramBudget,
    /// GPU interface (None if NVML unavailable).
    #[cfg(feature = "nvidia")]
    gpu: Option<NvidiaGpu>,
    #[cfg(not(feature = "nvidia"))]
    gpu: Option<()>,
}

impl VramManager {
    /// Create a new VramManager with default settings.
    ///
    /// If the nvidia feature is enabled, attempts to detect a GPU.
    /// Falls back to tracking-only mode if no GPU is found.
    pub fn new() -> Self {
        #[cfg(feature = "nvidia")]
        {
            match NvidiaGpu::new(0) {
                Ok(gpu) => {
                    // Detected GPU - use its total memory for budget
                    match gpu.total_memory_gb() {
                        Ok(total) => {
                            info!(
                                gpu_name = %gpu.name().unwrap_or_default(),
                                total_gb = total,
                                "Detected NVIDIA GPU"
                            );
                            let budget = presets::from_total(total);
                            let tracker = VramTracker::with_budget(budget.allocatable());
                            Self {
                                tracker,
                                budget,
                                gpu: Some(gpu),
                            }
                        }
                        Err(e) => {
                            warn!(error = %e, "Failed to query GPU memory, using defaults");
                            Self::without_gpu()
                        }
                    }
                }
                Err(e) => {
                    debug!(error = %e, "No NVIDIA GPU detected, using tracking-only mode");
                    Self::without_gpu()
                }
            }
        }

        #[cfg(not(feature = "nvidia"))]
        {
            debug!("NVML feature disabled, using tracking-only mode");
            Self::without_gpu()
        }
    }

    /// Create a VramManager without GPU integration.
    fn without_gpu() -> Self {
        let budget = VramBudget::default();
        let tracker = VramTracker::with_budget(budget.allocatable());
        Self {
            tracker,
            budget,
            gpu: None,
        }
    }

    /// Create a VramManager for a specific GPU device.
    ///
    /// # Arguments
    /// * `device_index` - The GPU device index (0 for first GPU)
    ///
    /// # Errors
    /// Returns an error if the GPU cannot be accessed.
    #[cfg(feature = "nvidia")]
    pub fn with_gpu(device_index: u32) -> Result<Self, NvidiaError> {
        let gpu = NvidiaGpu::new(device_index)?;
        let total = gpu.total_memory_gb()?;

        info!(
            device_index = device_index,
            gpu_name = %gpu.name().unwrap_or_default(),
            total_gb = total,
            "Initialized VramManager with GPU"
        );

        let budget = presets::from_total(total);
        let tracker = VramTracker::with_budget(budget.allocatable());

        Ok(Self {
            tracker,
            budget,
            gpu: Some(gpu),
        })
    }

    /// Create a VramManager for a specific GPU device (stub when nvidia feature disabled).
    ///
    /// Always returns `NvidiaError::FeatureDisabled` when nvidia feature is not enabled.
    #[cfg(not(feature = "nvidia"))]
    pub fn with_gpu(_device_index: u32) -> Result<Self, NvidiaError> {
        Err(NvidiaError::FeatureDisabled)
    }

    /// Create a VramManager with custom budget.
    pub fn with_budget(budget: VramBudget) -> Self {
        let tracker = VramTracker::with_budget(budget.allocatable());

        #[cfg(feature = "nvidia")]
        let gpu = NvidiaGpu::new(0).ok();

        #[cfg(not(feature = "nvidia"))]
        let gpu = None;

        Self {
            tracker,
            budget,
            gpu,
        }
    }

    /// Sync the tracker with actual GPU usage.
    ///
    /// When using dynamic policy, this updates the tracked usage
    /// to match what the GPU reports.
    #[cfg(feature = "nvidia")]
    pub fn sync_with_gpu(&mut self) -> Result<(), NvidiaError> {
        if let Some(ref gpu) = self.gpu {
            let used = gpu.used_memory_gb()?;
            let total = gpu.total_memory_gb()?;

            // Update budget with actual total
            self.budget.set_total(total);
            self.tracker.set_total(total - self.budget.reserved());

            debug!(
                actual_used_gb = used,
                tracked_used_gb = self.tracker.used(),
                "Synced with GPU"
            );

            Ok(())
        } else {
            Err(NvidiaError::DeviceNotFound)
        }
    }

    /// Sync the tracker with actual GPU usage (stub when nvidia feature disabled).
    ///
    /// Always returns `NvidiaError::FeatureDisabled` when nvidia feature is not enabled.
    #[cfg(not(feature = "nvidia"))]
    pub fn sync_with_gpu(&mut self) -> Result<(), NvidiaError> {
        Err(NvidiaError::FeatureDisabled)
    }

    /// Allocate VRAM for a model.
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier for the model
    /// * `size_gb` - VRAM size in GB
    ///
    /// # Errors
    /// Returns an error if:
    /// - The model is already allocated
    /// - The allocation would exceed budget (strict policy)
    pub fn allocate(&mut self, model_id: &str, size_gb: f32) -> Result<(), VramError> {
        // Check if already allocated
        if self.tracker.get_allocation(model_id).is_some() {
            return Err(VramError::AlreadyAllocated(model_id.to_string()));
        }

        // Check budget policy
        let current_used = self.tracker.used();
        match self.budget.check_allocation(size_gb, current_used) {
            Ok(_fits) => {
                // Allocation allowed (either fits or soft policy)
                // Use unchecked allocation since we've already done the policy check
                self.tracker.allocate_unchecked(model_id, size_gb)?;
                debug!(
                    model_id = %model_id,
                    size_gb = size_gb,
                    used = self.tracker.used(),
                    "Model allocated"
                );
                Ok(())
            }
            Err(_msg) => {
                // Strict policy denied
                Err(VramError::BudgetExceeded {
                    requested_gb: size_gb,
                    used_gb: current_used,
                    allocatable_gb: self.budget.allocatable(),
                })
            }
        }
    }

    /// Deallocate VRAM for a model.
    ///
    /// # Returns
    /// The amount of VRAM freed in GB.
    pub fn deallocate(&mut self, model_id: &str) -> f32 {
        let freed = self.tracker.deallocate(model_id);
        if freed > 0.0 {
            debug!(model_id = %model_id, freed_gb = freed, "Model deallocated");
        }
        freed
    }

    /// Get the allocation size for a specific model.
    pub fn get_allocation(&self, model_id: &str) -> Option<f32> {
        self.tracker.get_allocation(model_id)
    }

    /// Get the current VRAM status.
    pub fn status(&self) -> VramStatus {
        let tracked = self.tracker.used();
        let available = self.budget.remaining(tracked);

        #[cfg(feature = "nvidia")]
        let (actual_gb, gpu_name, gpu_temp, gpu_util, gpu_available) =
            if let Some(ref gpu) = self.gpu {
                let actual = gpu.used_memory_gb().ok();
                let name = gpu.name().ok();
                let temp = gpu.temperature().ok();
                let util = gpu.utilization_percent().ok();
                (actual, name, temp, util, true)
            } else {
                (None, None, None, None, false)
            };

        #[cfg(not(feature = "nvidia"))]
        let (actual_gb, gpu_name, gpu_temp, gpu_util, gpu_available) =
            (None, None, None, None, false);

        VramStatus {
            total_gb: self.budget.total(),
            used_gb: tracked,
            available_gb: available,
            tracked_gb: tracked,
            actual_gb,
            gpu_name,
            gpu_temp_c: gpu_temp,
            gpu_utilization: gpu_util,
            model_count: self.tracker.model_count(),
            gpu_available,
        }
    }

    /// Check if the manager has GPU access.
    pub fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Get the allocation policy.
    pub fn policy(&self) -> AllocationPolicy {
        self.budget.policy()
    }

    /// Set the allocation policy.
    pub fn set_policy(&mut self, policy: AllocationPolicy) {
        self.budget.set_policy(policy);
    }

    /// Get a reference to the budget.
    pub fn budget(&self) -> &VramBudget {
        &self.budget
    }

    /// Get a reference to the tracker.
    pub fn tracker(&self) -> &VramTracker {
        &self.tracker
    }

    /// Reset all allocations.
    pub fn reset(&mut self) {
        self.tracker.reset();
        debug!("VramManager reset");
    }

    /// Check if an allocation of the given size can fit.
    pub fn can_allocate(&self, size_gb: f32) -> bool {
        self.budget.can_fit(size_gb, self.tracker.used())
    }

    /// Get the utilization percentage (0-100).
    pub fn utilization_percent(&self) -> f32 {
        self.budget.utilization_percent(self.tracker.used())
    }
}

impl Default for VramManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_manager_creation() {
        let manager = VramManager::new();
        assert!(manager.tracker().used() == 0.0);
    }

    #[test]
    fn test_vram_manager_with_budget() {
        let budget = VramBudget::with_policy(16.0, 1.0, AllocationPolicy::Strict);
        let manager = VramManager::with_budget(budget);

        assert_eq!(manager.budget().total(), 16.0);
        assert_eq!(manager.budget().allocatable(), 15.0);
        assert_eq!(manager.policy(), AllocationPolicy::Strict);
    }

    #[test]
    fn test_allocation_success() {
        let mut manager = VramManager::with_budget(VramBudget::new(12.0));

        assert!(manager.allocate("flux", 6.0).is_ok());
        assert_eq!(manager.get_allocation("flux"), Some(6.0));
        assert_eq!(manager.tracker().used(), 6.0);
    }

    #[test]
    fn test_allocation_already_allocated() {
        let mut manager = VramManager::new();

        manager.allocate("flux", 6.0).unwrap();
        let result = manager.allocate("flux", 3.0);

        assert!(matches!(result, Err(VramError::AlreadyAllocated(_))));
    }

    #[test]
    fn test_budget_strict_policy() {
        let budget = VramBudget::with_policy(10.0, 1.0, AllocationPolicy::Strict);
        let mut manager = VramManager::with_budget(budget);

        // Should fit (9.0 allocatable)
        assert!(manager.allocate("small", 5.0).is_ok());

        // Should fail (would need 5.0 more, only 4.0 available)
        let result = manager.allocate("large", 5.0);
        assert!(matches!(result, Err(VramError::BudgetExceeded { .. })));
    }

    #[test]
    fn test_budget_soft_policy() {
        let budget = VramBudget::with_policy(10.0, 1.0, AllocationPolicy::Soft);
        let mut manager = VramManager::with_budget(budget);

        // Should fit
        assert!(manager.allocate("small", 5.0).is_ok());

        // Should allow (soft policy)
        assert!(manager.allocate("large", 5.0).is_ok());
        assert!(manager.tracker().used() > manager.budget().allocatable());
    }

    #[test]
    fn test_deallocation() {
        let mut manager = VramManager::new();

        manager.allocate("model1", 4.0).unwrap();
        manager.allocate("model2", 3.0).unwrap();

        assert_eq!(manager.tracker().used(), 7.0);

        let freed = manager.deallocate("model1");
        assert_eq!(freed, 4.0);
        assert_eq!(manager.tracker().used(), 3.0);
    }

    #[test]
    fn test_status() {
        let mut manager = VramManager::with_budget(VramBudget::new(12.0));
        manager.allocate("flux", 6.0).unwrap();

        let status = manager.status();
        assert_eq!(status.total_gb, 12.0);
        assert_eq!(status.used_gb, 6.0);
        assert_eq!(status.tracked_gb, 6.0);
        assert_eq!(status.available_gb, 5.0); // 11.0 allocatable - 6.0 used
        assert_eq!(status.model_count, 1);
    }

    #[test]
    fn test_vram_status_fields() {
        let status = VramStatus {
            total_gb: 12.0,
            used_gb: 6.0,
            available_gb: 5.0,
            tracked_gb: 6.0,
            actual_gb: Some(6.5),
            gpu_name: Some("RTX 3060".to_string()),
            gpu_temp_c: Some(65),
            gpu_utilization: Some(50),
            model_count: 2,
            gpu_available: true,
        };

        assert_eq!(status.total_gb, 12.0);
        assert_eq!(status.actual_gb, Some(6.5));
        assert!(status.gpu_available);
    }

    #[test]
    fn test_can_allocate() {
        let mut manager = VramManager::with_budget(VramBudget::new(12.0));

        assert!(manager.can_allocate(10.0));
        manager.allocate("m1", 8.0).unwrap();
        assert!(manager.can_allocate(3.0));
        assert!(!manager.can_allocate(4.0));
    }

    #[test]
    fn test_utilization_percent() {
        let mut manager = VramManager::with_budget(VramBudget::new(10.0)); // 9.0 allocatable

        assert_eq!(manager.utilization_percent(), 0.0);

        manager.allocate("m1", 4.5).unwrap();
        assert!((manager.utilization_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_reset() {
        let mut manager = VramManager::new();

        manager.allocate("m1", 1.0).unwrap();
        manager.allocate("m2", 2.0).unwrap();

        assert_eq!(manager.tracker().model_count(), 2);

        manager.reset();

        assert_eq!(manager.tracker().model_count(), 0);
        assert_eq!(manager.tracker().used(), 0.0);
    }

    #[test]
    fn test_error_display() {
        let err = VramError::BudgetExceeded {
            requested_gb: 5.0,
            used_gb: 8.0,
            allocatable_gb: 11.0,
        };
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("8"));

        let err = VramError::AlreadyAllocated("flux".to_string());
        assert!(err.to_string().contains("flux"));

        let err = VramError::NotFound("model".to_string());
        assert!(err.to_string().contains("model"));
    }

    // NVML-specific tests (only run with nvidia feature)
    #[test]
    #[cfg(not(feature = "nvidia"))]
    fn test_with_gpu_disabled() {
        let result = VramManager::with_gpu(0);
        assert!(matches!(result, Err(NvidiaError::FeatureDisabled)));
    }

    #[test]
    #[cfg(not(feature = "nvidia"))]
    fn test_sync_with_gpu_disabled() {
        let mut manager = VramManager::new();
        let result = manager.sync_with_gpu();
        assert!(matches!(result, Err(NvidiaError::FeatureDisabled)));
    }

    #[test]
    fn test_has_gpu() {
        let manager = VramManager::without_gpu();
        assert!(!manager.has_gpu());
    }

    #[test]
    fn test_icn_model_stack() {
        // Test that ICN's typical model set fits in RTX 3060 12GB
        // Using preset which gives 12GB total, 1GB reserved = 11GB allocatable
        let mut manager = VramManager::with_budget(presets::rtx_3060());

        // ICN model stack from PRD
        assert!(manager.allocate("flux-schnell", 6.0).is_ok());
        assert!(manager.allocate("liveportrait", 3.5).is_ok());
        assert!(manager.allocate("kokoro-82m", 0.4).is_ok());
        assert!(manager.allocate("clip-vit-b-32", 0.3).is_ok());
        assert!(manager.allocate("clip-vit-l-14", 0.6).is_ok());

        // Total: 10.8 GB, should have ~0.2GB headroom (11.0 - 10.8)
        let status = manager.status();
        assert_eq!(status.model_count, 5);
        assert!((status.used_gb - 10.8).abs() < 0.01);
        // Use tolerance for floating-point comparison
        assert!(
            (status.available_gb - 0.2).abs() < 0.01,
            "available_gb was {}",
            status.available_gb
        );
    }
}
