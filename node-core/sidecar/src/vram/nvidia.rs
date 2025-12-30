//! NVIDIA GPU integration via NVML.
//!
//! This module provides access to real GPU statistics through NVIDIA's
//! Management Library (NVML). It is feature-gated and gracefully degrades
//! when NVML is unavailable.
//!
//! # Feature Gate
//!
//! This module requires the `nvidia` feature to be enabled:
//!
//! ```toml
//! nsn-sidecar = { version = "*", features = ["nvidia"] }
//! ```

use std::fmt;

/// Errors that can occur when interacting with NVIDIA GPUs.
#[derive(Debug, Clone)]
pub enum NvidiaError {
    /// NVML library initialization failed.
    InitFailed(String),
    /// The requested GPU device was not found.
    DeviceNotFound,
    /// A query to the GPU failed.
    QueryFailed(String),
    /// NVML feature is not enabled.
    FeatureDisabled,
}

impl fmt::Display for NvidiaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NvidiaError::InitFailed(msg) => write!(f, "NVML initialization failed: {}", msg),
            NvidiaError::DeviceNotFound => write!(f, "GPU device not found"),
            NvidiaError::QueryFailed(msg) => write!(f, "GPU query failed: {}", msg),
            NvidiaError::FeatureDisabled => write!(f, "NVML feature not enabled"),
        }
    }
}

impl std::error::Error for NvidiaError {}

/// GPU information snapshot.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU name (e.g., "NVIDIA GeForce RTX 3060")
    pub name: String,
    /// Total VRAM in GB
    pub total_memory_gb: f32,
    /// Used VRAM in GB
    pub used_memory_gb: f32,
    /// Free VRAM in GB
    pub free_memory_gb: f32,
    /// GPU utilization percentage (0-100)
    pub utilization_percent: u32,
    /// GPU temperature in Celsius
    pub temperature_c: u32,
}

// Implementation when nvidia feature is enabled
#[cfg(feature = "nvidia")]
mod nvml_impl {
    use super::*;
    use nvml_wrapper::{Device, Nvml};
    use tracing::{debug, info, warn};

    /// NVIDIA GPU wrapper providing access to GPU statistics via NVML.
    ///
    /// This struct wraps the NVML library and provides a safe, ergonomic
    /// interface for querying GPU information.
    pub struct NvidiaGpu {
        nvml: Nvml,
        device_index: u32,
    }

    impl NvidiaGpu {
        /// Create a new NvidiaGpu for the specified device index.
        ///
        /// # Arguments
        /// * `device_index` - The GPU device index (0 for the first GPU)
        ///
        /// # Errors
        /// Returns an error if NVML initialization fails or the device is not found.
        pub fn new(device_index: u32) -> Result<Self, NvidiaError> {
            let nvml = Nvml::init().map_err(|e| {
                NvidiaError::InitFailed(format!("Failed to initialize NVML: {}", e))
            })?;

            // Verify device exists
            let device_count = nvml.device_count().map_err(|e| {
                NvidiaError::QueryFailed(format!("Failed to get device count: {}", e))
            })?;

            if device_index >= device_count {
                return Err(NvidiaError::DeviceNotFound);
            }

            info!(
                device_index = device_index,
                device_count = device_count,
                "NVML initialized successfully"
            );

            Ok(Self { nvml, device_index })
        }

        /// Get the device handle for the current GPU.
        fn device(&self) -> Result<Device<'_>, NvidiaError> {
            self.nvml
                .device_by_index(self.device_index)
                .map_err(|e| NvidiaError::QueryFailed(format!("Failed to get device: {}", e)))
        }

        /// Get the total VRAM in GB.
        pub fn total_memory_gb(&self) -> Result<f32, NvidiaError> {
            let device = self.device()?;
            let memory_info = device.memory_info().map_err(|e| {
                NvidiaError::QueryFailed(format!("Failed to get memory info: {}", e))
            })?;

            let gb = memory_info.total as f32 / (1024.0 * 1024.0 * 1024.0);
            debug!(total_gb = gb, "Queried total memory");
            Ok(gb)
        }

        /// Get the currently used VRAM in GB.
        pub fn used_memory_gb(&self) -> Result<f32, NvidiaError> {
            let device = self.device()?;
            let memory_info = device.memory_info().map_err(|e| {
                NvidiaError::QueryFailed(format!("Failed to get memory info: {}", e))
            })?;

            let gb = memory_info.used as f32 / (1024.0 * 1024.0 * 1024.0);
            debug!(used_gb = gb, "Queried used memory");
            Ok(gb)
        }

        /// Get the free VRAM in GB.
        pub fn free_memory_gb(&self) -> Result<f32, NvidiaError> {
            let device = self.device()?;
            let memory_info = device.memory_info().map_err(|e| {
                NvidiaError::QueryFailed(format!("Failed to get memory info: {}", e))
            })?;

            let gb = memory_info.free as f32 / (1024.0 * 1024.0 * 1024.0);
            debug!(free_gb = gb, "Queried free memory");
            Ok(gb)
        }

        /// Get the GPU utilization percentage (0-100).
        pub fn utilization_percent(&self) -> Result<u32, NvidiaError> {
            let device = self.device()?;
            let utilization = device.utilization_rates().map_err(|e| {
                NvidiaError::QueryFailed(format!("Failed to get utilization: {}", e))
            })?;

            let percent = utilization.gpu;
            debug!(utilization_percent = percent, "Queried utilization");
            Ok(percent)
        }

        /// Get the GPU name.
        pub fn name(&self) -> Result<String, NvidiaError> {
            let device = self.device()?;
            device
                .name()
                .map_err(|e| NvidiaError::QueryFailed(format!("Failed to get name: {}", e)))
        }

        /// Get the GPU temperature in Celsius.
        pub fn temperature(&self) -> Result<u32, NvidiaError> {
            use nvml_wrapper::enum_wrappers::device::TemperatureSensor;

            let device = self.device()?;
            device
                .temperature(TemperatureSensor::Gpu)
                .map_err(|e| NvidiaError::QueryFailed(format!("Failed to get temperature: {}", e)))
        }

        /// Get a complete snapshot of GPU information.
        pub fn info(&self) -> Result<GpuInfo, NvidiaError> {
            let device = self.device()?;

            let name = device
                .name()
                .map_err(|e| NvidiaError::QueryFailed(format!("Failed to get name: {}", e)))?;

            let memory_info = device.memory_info().map_err(|e| {
                NvidiaError::QueryFailed(format!("Failed to get memory info: {}", e))
            })?;

            let utilization = device.utilization_rates().map_err(|e| {
                NvidiaError::QueryFailed(format!("Failed to get utilization: {}", e))
            })?;

            use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
            let temperature = device
                .temperature(TemperatureSensor::Gpu)
                .map_err(|e| NvidiaError::QueryFailed(format!("Failed to get temperature: {}", e)))?;

            let bytes_to_gb = |b: u64| b as f32 / (1024.0 * 1024.0 * 1024.0);

            Ok(GpuInfo {
                name,
                total_memory_gb: bytes_to_gb(memory_info.total),
                used_memory_gb: bytes_to_gb(memory_info.used),
                free_memory_gb: bytes_to_gb(memory_info.free),
                utilization_percent: utilization.gpu,
                temperature_c: temperature,
            })
        }

        /// Get the device index.
        pub fn device_index(&self) -> u32 {
            self.device_index
        }

        /// Get the number of available GPU devices.
        pub fn device_count(&self) -> Result<u32, NvidiaError> {
            self.nvml
                .device_count()
                .map_err(|e| NvidiaError::QueryFailed(format!("Failed to get device count: {}", e)))
        }
    }

    impl fmt::Debug for NvidiaGpu {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("NvidiaGpu")
                .field("device_index", &self.device_index)
                .field("name", &self.name().ok())
                .finish()
        }
    }
}

// Stub implementation when nvidia feature is disabled
#[cfg(not(feature = "nvidia"))]
mod nvml_impl {
    use super::*;

    /// Stub GPU wrapper when NVML feature is disabled.
    ///
    /// All methods return `NvidiaError::FeatureDisabled`.
    #[derive(Debug)]
    pub struct NvidiaGpu {
        device_index: u32,
    }

    impl NvidiaGpu {
        /// Create a new NvidiaGpu (always returns FeatureDisabled).
        pub fn new(_device_index: u32) -> Result<Self, NvidiaError> {
            Err(NvidiaError::FeatureDisabled)
        }

        /// Get the total VRAM in GB (not available).
        pub fn total_memory_gb(&self) -> Result<f32, NvidiaError> {
            Err(NvidiaError::FeatureDisabled)
        }

        /// Get the currently used VRAM in GB (not available).
        pub fn used_memory_gb(&self) -> Result<f32, NvidiaError> {
            Err(NvidiaError::FeatureDisabled)
        }

        /// Get the free VRAM in GB (not available).
        pub fn free_memory_gb(&self) -> Result<f32, NvidiaError> {
            Err(NvidiaError::FeatureDisabled)
        }

        /// Get the GPU utilization percentage (not available).
        pub fn utilization_percent(&self) -> Result<u32, NvidiaError> {
            Err(NvidiaError::FeatureDisabled)
        }

        /// Get the GPU name (not available).
        pub fn name(&self) -> Result<String, NvidiaError> {
            Err(NvidiaError::FeatureDisabled)
        }

        /// Get the GPU temperature (not available).
        pub fn temperature(&self) -> Result<u32, NvidiaError> {
            Err(NvidiaError::FeatureDisabled)
        }

        /// Get GPU info snapshot (not available).
        pub fn info(&self) -> Result<GpuInfo, NvidiaError> {
            Err(NvidiaError::FeatureDisabled)
        }

        /// Get the device index.
        pub fn device_index(&self) -> u32 {
            self.device_index
        }

        /// Get the number of available GPU devices (not available).
        pub fn device_count(&self) -> Result<u32, NvidiaError> {
            Err(NvidiaError::FeatureDisabled)
        }
    }
}

// Re-export the appropriate implementation
pub use nvml_impl::NvidiaGpu;

/// Check if NVML support is available at compile time.
pub const fn is_nvml_available() -> bool {
    cfg!(feature = "nvidia")
}

/// Try to create an NvidiaGpu, returning None if unavailable.
///
/// This is a convenience function that doesn't require error handling
/// when the nvidia feature is disabled.
pub fn try_create_gpu(device_index: u32) -> Option<NvidiaGpu> {
    NvidiaGpu::new(device_index).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvidia_error_display() {
        let err = NvidiaError::InitFailed("test error".to_string());
        assert!(err.to_string().contains("test error"));

        let err = NvidiaError::DeviceNotFound;
        assert!(err.to_string().contains("not found"));

        let err = NvidiaError::QueryFailed("query error".to_string());
        assert!(err.to_string().contains("query error"));

        let err = NvidiaError::FeatureDisabled;
        assert!(err.to_string().contains("not enabled"));
    }

    #[test]
    fn test_gpu_info_fields() {
        let info = GpuInfo {
            name: "Test GPU".to_string(),
            total_memory_gb: 12.0,
            used_memory_gb: 6.0,
            free_memory_gb: 6.0,
            utilization_percent: 50,
            temperature_c: 65,
        };

        assert_eq!(info.name, "Test GPU");
        assert_eq!(info.total_memory_gb, 12.0);
        assert_eq!(info.used_memory_gb, 6.0);
        assert_eq!(info.free_memory_gb, 6.0);
        assert_eq!(info.utilization_percent, 50);
        assert_eq!(info.temperature_c, 65);
    }

    #[test]
    fn test_is_nvml_available() {
        // This should return true if compiled with nvidia feature, false otherwise
        let available = is_nvml_available();
        #[cfg(feature = "nvidia")]
        assert!(available);
        #[cfg(not(feature = "nvidia"))]
        assert!(!available);
    }

    #[test]
    #[cfg(not(feature = "nvidia"))]
    fn test_stub_returns_feature_disabled() {
        // When nvidia feature is disabled, all methods should fail
        let result = NvidiaGpu::new(0);
        assert!(matches!(result, Err(NvidiaError::FeatureDisabled)));

        // try_create_gpu should return None
        assert!(try_create_gpu(0).is_none());
    }

    // Tests that require actual NVIDIA hardware
    #[test]
    #[cfg(feature = "nvidia")]
    #[ignore = "Requires NVIDIA GPU hardware"]
    fn test_nvidia_gpu_init() {
        let gpu = NvidiaGpu::new(0);
        match gpu {
            Ok(gpu) => {
                // If we have a GPU, verify we can query it
                assert!(gpu.total_memory_gb().is_ok());
                assert!(gpu.name().is_ok());
                assert!(gpu.info().is_ok());
            }
            Err(NvidiaError::DeviceNotFound) => {
                // No GPU available, test still passes
            }
            Err(e) => {
                panic!("Unexpected error: {}", e);
            }
        }
    }
}
