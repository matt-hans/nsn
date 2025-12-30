//! VRAM allocation tracking for the sidecar.
//!
//! This module tracks VRAM usage across all loaded models to prevent
//! out-of-memory conditions. The tracker maintains a budget and validates
//! allocations before allowing models to be loaded.

use std::collections::HashMap;

use tracing::{debug, warn};

/// Default total VRAM budget (RTX 3060 12GB target, with 0.2GB safety margin)
const DEFAULT_TOTAL_VRAM_GB: f32 = 11.8;

/// Tracks VRAM allocations across loaded models.
///
/// Maintains a budget to prevent OOM conditions when loading models.
/// Used by the sidecar service to validate model load requests.
#[derive(Debug, Clone)]
pub struct VramTracker {
    /// Total VRAM budget in GB
    total_gb: f32,
    /// Current allocations by model ID
    allocations: HashMap<String, f32>,
    /// Total allocated VRAM
    used_gb: f32,
}

impl VramTracker {
    /// Create a new VRAM tracker with default budget.
    pub fn new() -> Self {
        Self::with_budget(DEFAULT_TOTAL_VRAM_GB)
    }

    /// Create a new VRAM tracker with custom budget.
    pub fn with_budget(total_gb: f32) -> Self {
        Self {
            total_gb,
            allocations: HashMap::new(),
            used_gb: 0.0,
        }
    }

    /// Get total VRAM budget in GB.
    pub fn total(&self) -> f32 {
        self.total_gb
    }

    /// Get currently used VRAM in GB.
    pub fn used(&self) -> f32 {
        self.used_gb
    }

    /// Get available VRAM in GB.
    pub fn available(&self) -> f32 {
        (self.total_gb - self.used_gb).max(0.0)
    }

    /// Get all current allocations.
    pub fn allocations(&self) -> &HashMap<String, f32> {
        &self.allocations
    }

    /// Check if an allocation of the given size can be made.
    pub fn can_allocate(&self, size_gb: f32) -> bool {
        self.available() >= size_gb
    }

    /// Allocate VRAM for a model.
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier for the model
    /// * `size_gb` - VRAM size in GB
    ///
    /// # Returns
    /// `true` if allocation succeeded, `false` if model already allocated
    ///
    /// # Panics
    /// Does not panic, but logs a warning if allocation would exceed budget.
    pub fn allocate(&mut self, model_id: &str, size_gb: f32) -> bool {
        if self.allocations.contains_key(model_id) {
            warn!(model_id = %model_id, "Model already allocated");
            return false;
        }

        if !self.can_allocate(size_gb) {
            warn!(
                model_id = %model_id,
                size_gb = size_gb,
                available = self.available(),
                "Allocation would exceed budget"
            );
        }

        self.allocations.insert(model_id.to_string(), size_gb);
        self.used_gb += size_gb;

        debug!(
            model_id = %model_id,
            size_gb = size_gb,
            used = self.used_gb,
            available = self.available(),
            "VRAM allocated"
        );

        true
    }

    /// Deallocate VRAM for a model.
    ///
    /// # Arguments
    /// * `model_id` - Model to deallocate
    ///
    /// # Returns
    /// The amount of VRAM freed, or 0.0 if model was not allocated
    pub fn deallocate(&mut self, model_id: &str) -> f32 {
        if let Some(size_gb) = self.allocations.remove(model_id) {
            self.used_gb = (self.used_gb - size_gb).max(0.0);

            debug!(
                model_id = %model_id,
                freed_gb = size_gb,
                used = self.used_gb,
                available = self.available(),
                "VRAM deallocated"
            );

            size_gb
        } else {
            warn!(model_id = %model_id, "Model not found for deallocation");
            0.0
        }
    }

    /// Get the allocation size for a specific model.
    pub fn get_allocation(&self, model_id: &str) -> Option<f32> {
        self.allocations.get(model_id).copied()
    }

    /// Reset all allocations.
    pub fn reset(&mut self) {
        self.allocations.clear();
        self.used_gb = 0.0;
        debug!("VRAM tracker reset");
    }

    /// Get the number of allocated models.
    pub fn model_count(&self) -> usize {
        self.allocations.len()
    }

    /// Calculate utilization percentage (0-100).
    pub fn utilization_percent(&self) -> f32 {
        if self.total_gb > 0.0 {
            (self.used_gb / self.total_gb) * 100.0
        } else {
            0.0
        }
    }
}

impl Default for VramTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let mut tracker = VramTracker::with_budget(12.0);

        assert_eq!(tracker.total(), 12.0);
        assert_eq!(tracker.used(), 0.0);
        assert_eq!(tracker.available(), 12.0);

        // Allocate
        assert!(tracker.allocate("flux", 6.0));
        assert_eq!(tracker.used(), 6.0);
        assert_eq!(tracker.available(), 6.0);

        // Allocate another
        assert!(tracker.allocate("clip", 0.5));
        assert_eq!(tracker.used(), 6.5);
        assert_eq!(tracker.available(), 5.5);

        // Deallocate
        let freed = tracker.deallocate("flux");
        assert_eq!(freed, 6.0);
        assert_eq!(tracker.used(), 0.5);
        assert_eq!(tracker.available(), 11.5);
    }

    #[test]
    fn test_can_allocate() {
        let mut tracker = VramTracker::with_budget(10.0);

        assert!(tracker.can_allocate(5.0));
        assert!(tracker.can_allocate(10.0));
        assert!(!tracker.can_allocate(10.1));

        tracker.allocate("model1", 8.0);

        assert!(tracker.can_allocate(2.0));
        assert!(!tracker.can_allocate(2.1));
    }

    #[test]
    fn test_duplicate_allocation() {
        let mut tracker = VramTracker::new();

        assert!(tracker.allocate("model1", 1.0));
        assert!(!tracker.allocate("model1", 2.0)); // Duplicate

        // Original allocation unchanged
        assert_eq!(tracker.get_allocation("model1"), Some(1.0));
    }

    #[test]
    fn test_deallocate_nonexistent() {
        let mut tracker = VramTracker::new();

        let freed = tracker.deallocate("nonexistent");
        assert_eq!(freed, 0.0);
    }

    #[test]
    fn test_reset() {
        let mut tracker = VramTracker::new();

        tracker.allocate("m1", 1.0);
        tracker.allocate("m2", 2.0);
        tracker.allocate("m3", 3.0);

        assert_eq!(tracker.model_count(), 3);
        assert_eq!(tracker.used(), 6.0);

        tracker.reset();

        assert_eq!(tracker.model_count(), 0);
        assert_eq!(tracker.used(), 0.0);
    }

    #[test]
    fn test_utilization_percent() {
        let mut tracker = VramTracker::with_budget(10.0);

        assert_eq!(tracker.utilization_percent(), 0.0);

        tracker.allocate("m1", 5.0);
        assert_eq!(tracker.utilization_percent(), 50.0);

        tracker.allocate("m2", 2.5);
        assert_eq!(tracker.utilization_percent(), 75.0);
    }

    #[test]
    fn test_icn_vram_budget() {
        // Test that ICN's typical model set fits in budget
        let mut tracker = VramTracker::new(); // Uses 11.8 GB budget

        // ICN model stack from PRD
        assert!(tracker.allocate("flux-schnell", 6.0));
        assert!(tracker.allocate("liveportrait", 3.5));
        assert!(tracker.allocate("kokoro-82m", 0.4));
        assert!(tracker.allocate("clip-vit-b-32", 0.3));
        assert!(tracker.allocate("clip-vit-l-14", 0.6));

        // Total: 10.8 GB
        assert_eq!(tracker.used(), 10.8);
        assert!(tracker.available() >= 1.0); // At least 1GB headroom

        // System overhead should fit
        assert!(tracker.can_allocate(1.0));
    }
}
