//! VRAM budget policies for allocation management.
//!
//! This module defines budget policies that control how VRAM allocations
//! are handled when approaching or exceeding the budget limit.

use std::fmt;

use tracing::{debug, warn};

use super::tracker::DEFAULT_TOTAL_VRAM_GB;

/// Default reserved VRAM for system/PyTorch overhead in GB.
pub const DEFAULT_RESERVED_GB: f32 = 1.0;

/// Allocation policy for handling VRAM budget constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocationPolicy {
    /// Strict: fail immediately if allocation would exceed budget.
    ///
    /// Use this policy when VRAM budget must never be exceeded.
    /// Allocations that would exceed the budget return an error.
    Strict,

    /// Soft: warn but allow allocation to proceed.
    ///
    /// Use this policy for development/testing or when the budget
    /// is a soft limit. Logs a warning but allows over-allocation.
    #[default]
    Soft,

    /// Dynamic: adjust behavior based on actual GPU usage.
    ///
    /// Use this policy with NVML integration. When GPU is available,
    /// checks actual VRAM usage rather than tracked usage.
    Dynamic,
}

impl fmt::Display for AllocationPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocationPolicy::Strict => write!(f, "strict"),
            AllocationPolicy::Soft => write!(f, "soft"),
            AllocationPolicy::Dynamic => write!(f, "dynamic"),
        }
    }
}

/// VRAM budget configuration.
///
/// Controls the total VRAM budget, reserved space, and allocation policy.
#[derive(Debug, Clone)]
pub struct VramBudget {
    /// Total budget (may be less than physical for safety margin).
    total_gb: f32,
    /// Reserved for system/PyTorch overhead.
    reserved_gb: f32,
    /// Allocation policy.
    policy: AllocationPolicy,
}

impl VramBudget {
    /// Create a new VRAM budget with default reserved space.
    ///
    /// # Arguments
    /// * `total_gb` - Total VRAM budget in GB
    pub fn new(total_gb: f32) -> Self {
        Self::with_reserved(total_gb, DEFAULT_RESERVED_GB)
    }

    /// Create a new VRAM budget with custom reserved space.
    ///
    /// # Arguments
    /// * `total_gb` - Total VRAM budget in GB
    /// * `reserved_gb` - Reserved VRAM for system overhead in GB
    pub fn with_reserved(total_gb: f32, reserved_gb: f32) -> Self {
        Self {
            total_gb,
            reserved_gb,
            policy: AllocationPolicy::default(),
        }
    }

    /// Create a new VRAM budget with all settings.
    ///
    /// # Arguments
    /// * `total_gb` - Total VRAM budget in GB
    /// * `reserved_gb` - Reserved VRAM for system overhead in GB
    /// * `policy` - Allocation policy
    pub fn with_policy(total_gb: f32, reserved_gb: f32, policy: AllocationPolicy) -> Self {
        Self {
            total_gb,
            reserved_gb,
            policy,
        }
    }

    /// Get the total VRAM budget.
    pub fn total(&self) -> f32 {
        self.total_gb
    }

    /// Get the reserved VRAM.
    pub fn reserved(&self) -> f32 {
        self.reserved_gb
    }

    /// Get the allocatable VRAM (total minus reserved).
    pub fn allocatable(&self) -> f32 {
        (self.total_gb - self.reserved_gb).max(0.0)
    }

    /// Get the allocation policy.
    pub fn policy(&self) -> AllocationPolicy {
        self.policy
    }

    /// Set the allocation policy.
    pub fn set_policy(&mut self, policy: AllocationPolicy) {
        debug!(old = %self.policy, new = %policy, "Allocation policy changed");
        self.policy = policy;
    }

    /// Set the total budget.
    pub fn set_total(&mut self, total_gb: f32) {
        debug!(old = self.total_gb, new = total_gb, "Total budget changed");
        self.total_gb = total_gb;
    }

    /// Set the reserved space.
    pub fn set_reserved(&mut self, reserved_gb: f32) {
        debug!(
            old = self.reserved_gb,
            new = reserved_gb,
            "Reserved space changed"
        );
        self.reserved_gb = reserved_gb;
    }

    /// Check if a given allocation can fit within the budget.
    ///
    /// # Arguments
    /// * `size_gb` - Size of the allocation in GB
    /// * `current_used` - Current used VRAM in GB
    ///
    /// # Returns
    /// `true` if the allocation fits, `false` otherwise.
    pub fn can_fit(&self, size_gb: f32, current_used: f32) -> bool {
        let available = self.allocatable() - current_used;
        available >= size_gb
    }

    /// Check allocation and return result based on policy.
    ///
    /// # Arguments
    /// * `size_gb` - Size of the allocation in GB
    /// * `current_used` - Current used VRAM in GB
    ///
    /// # Returns
    /// * `Ok(true)` - Allocation fits and is allowed
    /// * `Ok(false)` - Allocation doesn't fit but is allowed (Soft policy)
    /// * `Err(msg)` - Allocation is denied (Strict policy)
    pub fn check_allocation(&self, size_gb: f32, current_used: f32) -> Result<bool, String> {
        let fits = self.can_fit(size_gb, current_used);

        if fits {
            return Ok(true);
        }

        match self.policy {
            AllocationPolicy::Strict => Err(format!(
                "Allocation of {} GB would exceed budget (used: {}, allocatable: {})",
                size_gb,
                current_used,
                self.allocatable()
            )),
            AllocationPolicy::Soft => {
                warn!(
                    size_gb = size_gb,
                    current_used = current_used,
                    allocatable = self.allocatable(),
                    "Allocation exceeds budget (soft policy - allowing)"
                );
                Ok(false)
            }
            AllocationPolicy::Dynamic => {
                // For dynamic policy, defer to the caller who should
                // check actual GPU usage
                warn!(
                    size_gb = size_gb,
                    current_used = current_used,
                    allocatable = self.allocatable(),
                    "Allocation exceeds tracked budget (dynamic policy - deferring to GPU check)"
                );
                Ok(false)
            }
        }
    }

    /// Get the remaining available VRAM given current usage.
    pub fn remaining(&self, current_used: f32) -> f32 {
        (self.allocatable() - current_used).max(0.0)
    }

    /// Check if the budget is over-committed.
    pub fn is_over_budget(&self, current_used: f32) -> bool {
        current_used > self.allocatable()
    }

    /// Calculate utilization percentage (0-100).
    pub fn utilization_percent(&self, current_used: f32) -> f32 {
        let allocatable = self.allocatable();
        if allocatable > 0.0 {
            (current_used / allocatable) * 100.0
        } else {
            0.0
        }
    }
}

impl Default for VramBudget {
    fn default() -> Self {
        Self::new(DEFAULT_TOTAL_VRAM_GB)
    }
}

/// Predefined budget configurations for common GPU types.
pub mod presets {
    use super::*;

    /// Budget for RTX 3060 12GB (ICN reference hardware).
    pub fn rtx_3060() -> VramBudget {
        VramBudget::with_reserved(12.0, 1.0)
    }

    /// Budget for RTX 3070 8GB.
    pub fn rtx_3070() -> VramBudget {
        VramBudget::with_reserved(8.0, 0.8)
    }

    /// Budget for RTX 3080 10GB.
    pub fn rtx_3080() -> VramBudget {
        VramBudget::with_reserved(10.0, 1.0)
    }

    /// Budget for RTX 3090 24GB.
    pub fn rtx_3090() -> VramBudget {
        VramBudget::with_reserved(24.0, 1.5)
    }

    /// Budget for RTX 4080 16GB.
    pub fn rtx_4080() -> VramBudget {
        VramBudget::with_reserved(16.0, 1.0)
    }

    /// Budget for RTX 4090 24GB.
    pub fn rtx_4090() -> VramBudget {
        VramBudget::with_reserved(24.0, 1.5)
    }

    /// Budget from detected GPU total memory.
    ///
    /// Reserves 10% of total or 1.0 GB minimum for system overhead.
    pub fn from_total(total_gb: f32) -> VramBudget {
        let reserved = (total_gb * 0.1).max(1.0);
        VramBudget::with_reserved(total_gb, reserved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_creation() {
        let budget = VramBudget::new(12.0);
        assert_eq!(budget.total(), 12.0);
        assert_eq!(budget.reserved(), DEFAULT_RESERVED_GB);
        assert_eq!(budget.allocatable(), 11.0);
        assert_eq!(budget.policy(), AllocationPolicy::Soft);
    }

    #[test]
    fn test_budget_with_reserved() {
        let budget = VramBudget::with_reserved(16.0, 2.0);
        assert_eq!(budget.total(), 16.0);
        assert_eq!(budget.reserved(), 2.0);
        assert_eq!(budget.allocatable(), 14.0);
    }

    #[test]
    fn test_budget_with_policy() {
        let budget = VramBudget::with_policy(12.0, 1.0, AllocationPolicy::Strict);
        assert_eq!(budget.policy(), AllocationPolicy::Strict);
    }

    #[test]
    fn test_can_fit() {
        let budget = VramBudget::new(12.0); // 11.0 allocatable

        assert!(budget.can_fit(5.0, 0.0));
        assert!(budget.can_fit(11.0, 0.0));
        assert!(!budget.can_fit(11.1, 0.0));

        assert!(budget.can_fit(5.0, 6.0));
        assert!(!budget.can_fit(5.1, 6.0));
    }

    #[test]
    fn test_check_allocation_strict() {
        let budget = VramBudget::with_policy(12.0, 1.0, AllocationPolicy::Strict);

        // Fits - should return Ok(true)
        let result = budget.check_allocation(5.0, 0.0);
        assert!(result.is_ok());
        assert!(result.unwrap());

        // Doesn't fit - should return Err
        let result = budget.check_allocation(12.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_allocation_soft() {
        let budget = VramBudget::with_policy(12.0, 1.0, AllocationPolicy::Soft);

        // Fits - should return Ok(true)
        let result = budget.check_allocation(5.0, 0.0);
        assert!(result.is_ok());
        assert!(result.unwrap());

        // Doesn't fit - should return Ok(false) (soft policy allows)
        let result = budget.check_allocation(12.0, 0.0);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_remaining() {
        let budget = VramBudget::new(12.0); // 11.0 allocatable

        assert_eq!(budget.remaining(0.0), 11.0);
        assert_eq!(budget.remaining(5.0), 6.0);
        assert_eq!(budget.remaining(11.0), 0.0);
        assert_eq!(budget.remaining(12.0), 0.0); // Clamped to 0
    }

    #[test]
    fn test_is_over_budget() {
        let budget = VramBudget::new(12.0); // 11.0 allocatable

        assert!(!budget.is_over_budget(10.0));
        assert!(!budget.is_over_budget(11.0));
        assert!(budget.is_over_budget(11.1));
    }

    #[test]
    fn test_utilization_percent() {
        let budget = VramBudget::new(10.0); // 9.0 allocatable

        assert_eq!(budget.utilization_percent(0.0), 0.0);
        assert!((budget.utilization_percent(4.5) - 50.0).abs() < 0.01);
        assert!((budget.utilization_percent(9.0) - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_presets() {
        let rtx_3060 = presets::rtx_3060();
        assert_eq!(rtx_3060.total(), 12.0);
        assert_eq!(rtx_3060.allocatable(), 11.0);

        let rtx_4090 = presets::rtx_4090();
        assert_eq!(rtx_4090.total(), 24.0);
        assert_eq!(rtx_4090.allocatable(), 22.5);
    }

    #[test]
    fn test_preset_from_total() {
        let budget = presets::from_total(8.0);
        assert_eq!(budget.total(), 8.0);
        // 10% of 8 = 0.8, but minimum is 1.0
        assert_eq!(budget.reserved(), 1.0);
        assert_eq!(budget.allocatable(), 7.0);

        let budget = presets::from_total(24.0);
        assert_eq!(budget.total(), 24.0);
        // 10% of 24 = 2.4
        assert_eq!(budget.reserved(), 2.4);
        assert_eq!(budget.allocatable(), 21.6);
    }

    #[test]
    fn test_policy_display() {
        assert_eq!(format!("{}", AllocationPolicy::Strict), "strict");
        assert_eq!(format!("{}", AllocationPolicy::Soft), "soft");
        assert_eq!(format!("{}", AllocationPolicy::Dynamic), "dynamic");
    }

    #[test]
    fn test_setters() {
        let mut budget = VramBudget::new(12.0);

        budget.set_total(16.0);
        assert_eq!(budget.total(), 16.0);

        budget.set_reserved(2.0);
        assert_eq!(budget.reserved(), 2.0);

        budget.set_policy(AllocationPolicy::Strict);
        assert_eq!(budget.policy(), AllocationPolicy::Strict);
    }
}
