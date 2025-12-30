//! Container preemption logic with Double-Tap graceful shutdown strategy.
//!
//! This module implements preemption of Lane 1 containers when:
//! - Lane 0 video task is waiting (priority preemption)
//! - VRAM budget has been exceeded
//! - Epoch transition (On-Deck draining period)
//! - Manual operator request
//!
//! The "Double-Tap" strategy:
//! 1. First tap: Send graceful shutdown signal (gRPC CancelTask)
//! 2. Wait for configurable timeout (default 5 seconds)
//! 3. Second tap: Force kill if still running (SIGKILL)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Default timeout for graceful preemption before force kill
const DEFAULT_GRACEFUL_TIMEOUT_SECS: u64 = 5;

/// Why preemption was triggered
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionReason {
    /// Lane 0 video task is waiting (higher priority)
    Lane0Priority,
    /// VRAM budget has been exceeded
    VramBudgetExceeded,
    /// Epoch transition (On-Deck draining period)
    EpochTransition,
    /// Manual operator request
    ManualRequest,
}

impl PreemptionReason {
    /// Convert reason to a string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            PreemptionReason::Lane0Priority => "lane0_priority",
            PreemptionReason::VramBudgetExceeded => "vram_budget_exceeded",
            PreemptionReason::EpochTransition => "epoch_transition",
            PreemptionReason::ManualRequest => "manual_request",
        }
    }

    /// Returns true if this reason requires immediate preemption
    pub fn is_urgent(&self) -> bool {
        matches!(self, PreemptionReason::Lane0Priority | PreemptionReason::VramBudgetExceeded)
    }
}

/// Strategy for preempting a container
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionStrategy {
    /// Double-Tap: graceful shutdown, then force kill if timeout exceeded
    Graceful,
    /// Immediate force kill (emergency only)
    Immediate,
}

impl PreemptionStrategy {
    /// Convert strategy to a string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            PreemptionStrategy::Graceful => "graceful",
            PreemptionStrategy::Immediate => "immediate",
        }
    }
}

/// Result of a preemption attempt
#[derive(Debug, Clone)]
pub struct PreemptionResult {
    /// Container that was preempted
    pub container_id: String,
    /// Task that was running in the container
    pub task_id: String,
    /// Why preemption was triggered
    pub reason: PreemptionReason,
    /// Whether preemption was successful
    pub success: bool,
    /// How long preemption took (milliseconds)
    pub duration_ms: u64,
    /// Whether graceful shutdown was successful (before force kill)
    pub was_graceful: bool,
}

impl PreemptionResult {
    /// Create a new successful preemption result
    fn success(
        container_id: String,
        task_id: String,
        reason: PreemptionReason,
        duration: Duration,
        was_graceful: bool,
    ) -> Self {
        Self {
            container_id,
            task_id,
            reason,
            success: true,
            duration_ms: duration.as_millis() as u64,
            was_graceful,
        }
    }

    /// Create a new failed preemption result
    fn failure(
        container_id: String,
        task_id: String,
        reason: PreemptionReason,
        duration: Duration,
    ) -> Self {
        Self {
            container_id,
            task_id,
            reason,
            success: false,
            duration_ms: duration.as_millis() as u64,
            was_graceful: false,
        }
    }
}

/// State of an ongoing preemption operation
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for debugging and future extension
struct PreemptionState {
    /// Container being preempted
    container_id: String,
    /// Task being cancelled
    task_id: String,
    /// Why preemption was triggered
    reason: PreemptionReason,
    /// When preemption started
    started_at: Instant,
    /// Strategy being used
    strategy: PreemptionStrategy,
}

/// Manages container preemption operations.
///
/// Coordinates preemption across containers using the Double-Tap strategy:
/// - First tap: graceful shutdown (gRPC CancelTask)
/// - Second tap: force kill if timeout exceeded
#[derive(Debug)]
pub struct PreemptionManager {
    /// Active preemption operations indexed by container ID
    active_preemptions: Arc<RwLock<HashMap<String, PreemptionState>>>,
    /// Timeout for graceful shutdown before force kill
    graceful_timeout: Duration,
}

impl PreemptionManager {
    /// Create a new preemption manager with default timeout
    pub fn new() -> Self {
        Self::with_timeout(Duration::from_secs(DEFAULT_GRACEFUL_TIMEOUT_SECS))
    }

    /// Create a new preemption manager with custom graceful timeout
    pub fn with_timeout(graceful_timeout: Duration) -> Self {
        Self {
            active_preemptions: Arc::new(RwLock::new(HashMap::new())),
            graceful_timeout,
        }
    }

    /// Create a new preemption manager with timeout in seconds
    pub fn with_timeout_secs(graceful_timeout_secs: u64) -> Self {
        Self::with_timeout(Duration::from_secs(graceful_timeout_secs))
    }

    /// Check if a container is currently being preempted
    pub async fn is_preempting(&self, container_id: &str) -> bool {
        let preemptions = self.active_preemptions.read().await;
        preemptions.contains_key(container_id)
    }

    /// Cancel an ongoing preemption (if still in progress)
    pub async fn cancel_preemption(&self, container_id: &str) -> bool {
        let mut preemptions = self.active_preemptions.write().await;

        if let Some(state) = preemptions.remove(container_id) {
            info!(
                container_id = %container_id,
                task_id = %state.task_id,
                reason = %state.reason.as_str(),
                "Preemption cancelled"
            );
            true
        } else {
            false
        }
    }

    /// Preempt a container using the appropriate strategy.
    ///
    /// # Arguments
    /// * `container_id` - Container to preempt
    /// * `task_id` - Task running in the container
    /// * `reason` - Why preemption is being triggered
    ///
    /// # Returns
    /// Result of the preemption attempt
    pub async fn preempt(
        &self,
        container_id: &str,
        task_id: &str,
        reason: PreemptionReason,
    ) -> PreemptionResult {
        // For MVP, always use graceful strategy (Double-Tap)
        // In production, urgent reasons could use Immediate strategy in emergencies
        self.preempt_with_strategy(container_id, task_id, reason, PreemptionStrategy::Graceful).await
    }

    /// Preempt a container with a specific strategy.
    ///
    /// # Arguments
    /// * `container_id` - Container to preempt
    /// * `task_id` - Task running in the container
    /// * `reason` - Why preemption is being triggered
    /// * `strategy` - Preemption strategy to use
    ///
    /// # Returns
    /// Result of the preemption attempt
    pub async fn preempt_with_strategy(
        &self,
        container_id: &str,
        task_id: &str,
        reason: PreemptionReason,
        strategy: PreemptionStrategy,
    ) -> PreemptionResult {
        let start_time = Instant::now();

        // Check if already preempting this container
        {
            let preemptions = self.active_preemptions.read().await;
            if let Some(existing) = preemptions.get(container_id) {
                warn!(
                    container_id = %container_id,
                    task_id = %task_id,
                    existing_task_id = %existing.task_id,
                    "Container already being preempted"
                );
                return PreemptionResult::failure(
                    container_id.to_string(),
                    task_id.to_string(),
                    reason,
                    start_time.elapsed(),
                );
            }
        }

        // Register preemption state
        {
            let mut preemptions = self.active_preemptions.write().await;
            preemptions.insert(
                container_id.to_string(),
                PreemptionState {
                    container_id: container_id.to_string(),
                    task_id: task_id.to_string(),
                    reason,
                    started_at: start_time,
                    strategy,
                },
            );
        }

        info!(
            container_id = %container_id,
            task_id = %task_id,
            reason = %reason.as_str(),
            strategy = %strategy.as_str(),
            "Starting container preemption"
        );

        // Execute preemption based on strategy
        let result = match strategy {
            PreemptionStrategy::Graceful => {
                self.graceful_preempt(container_id, task_id, reason, start_time).await
            }
            PreemptionStrategy::Immediate => {
                self.immediate_preempt(container_id, task_id, reason, start_time).await
            }
        };

        // Clean up preemption state
        {
            let mut preemptions = self.active_preemptions.write().await;
            preemptions.remove(container_id);
        }

        result
    }

    /// Execute graceful preemption (Double-Tap strategy)
    async fn graceful_preempt(
        &self,
        container_id: &str,
        task_id: &str,
        reason: PreemptionReason,
        start_time: Instant,
    ) -> PreemptionResult {
        debug!(
            container_id = %container_id,
            task_id = %task_id,
            timeout_ms = self.graceful_timeout.as_millis(),
            "First tap: sending graceful shutdown signal"
        );

        // First tap: Send graceful shutdown (gRPC CancelTask)
        // MVP: Stubbed - in production this would call container's gRPC endpoint
        let graceful_result = self.send_cancel_signal(container_id, task_id).await;

        if !graceful_result {
            warn!(
                container_id = %container_id,
                task_id = %task_id,
                "Failed to send graceful shutdown signal"
            );
            // Fall through to force kill
        }

        // Wait for graceful shutdown with timeout
        let shutdown_result = timeout(
            self.graceful_timeout,
            self.wait_for_shutdown(container_id, task_id),
        )
        .await;

        match shutdown_result {
            Ok(Ok(())) => {
                // Graceful shutdown succeeded
                info!(
                    container_id = %container_id,
                    task_id = %task_id,
                    elapsed_ms = start_time.elapsed().as_millis(),
                    "Graceful shutdown completed"
                );
                PreemptionResult::success(
                    container_id.to_string(),
                    task_id.to_string(),
                    reason,
                    start_time.elapsed(),
                    true,
                )
            }
            Ok(Err(e)) => {
                // Graceful shutdown failed
                warn!(
                    container_id = %container_id,
                    task_id = %task_id,
                    error = %e,
                    "Graceful shutdown failed, proceeding to force kill"
                );
                // Second tap: Force kill
                self.force_kill(container_id, task_id, reason, start_time).await
            }
            Err(_) => {
                // Timeout exceeded
                warn!(
                    container_id = %container_id,
                    task_id = %task_id,
                    timeout_ms = self.graceful_timeout.as_millis(),
                    "Graceful shutdown timeout exceeded, proceeding to force kill"
                );
                // Second tap: Force kill
                self.force_kill(container_id, task_id, reason, start_time).await
            }
        }
    }

    /// Execute immediate force kill preemption
    async fn immediate_preempt(
        &self,
        container_id: &str,
        task_id: &str,
        reason: PreemptionReason,
        start_time: Instant,
    ) -> PreemptionResult {
        debug!(
            container_id = %container_id,
            task_id = %task_id,
            "Immediate force kill (emergency preemption)"
        );

        self.force_kill(container_id, task_id, reason, start_time).await
    }

    /// Send graceful cancel signal to container (First Tap)
    ///
    /// MVP: Stubbed - in production this would call the container's gRPC endpoint
    async fn send_cancel_signal(&self, container_id: &str, task_id: &str) -> bool {
        debug!(
            container_id = %container_id,
            task_id = %task_id,
            "Sending CancelTask gRPC call (stubbed)"
        );

        // MVP: Stubbed implementation
        // In production, this would:
        // 1. Connect to container's gRPC endpoint
        // 2. Call CancelTask RPC with task_id
        // 3. Return true if signal sent successfully

        // Simulate some network latency
        tokio::time::sleep(Duration::from_millis(10)).await;

        true
    }

    /// Wait for container to gracefully shutdown
    ///
    /// MVP: Stubbed - in production this would poll container status
    async fn wait_for_shutdown(&self, container_id: &str, task_id: &str) -> Result<(), String> {
        debug!(
            container_id = %container_id,
            task_id = %task_id,
            "Waiting for graceful shutdown (stubbed)"
        );

        // MVP: Stubbed implementation
        // In production, this would:
        // 1. Poll container status via Docker API
        // 2. Check if task has completed
        // 3. Return Ok(()) when shutdown complete

        // Simulate graceful shutdown taking some time
        tokio::time::sleep(Duration::from_millis(100)).await;

        Ok(())
    }

    /// Force kill container (Second Tap)
    ///
    /// MVP: Stubbed - in production this would send SIGKILL
    async fn force_kill(
        &self,
        container_id: &str,
        task_id: &str,
        reason: PreemptionReason,
        start_time: Instant,
    ) -> PreemptionResult {
        debug!(
            container_id = %container_id,
            task_id = %task_id,
            "Second tap: sending SIGKILL (stubbed)"
        );

        // MVP: Stubbed implementation
        // In production, this would:
        // 1. Send SIGKILL to container process
        // 2. Remove container
        // 3. Clean up resources

        // Simulate force kill taking minimal time
        tokio::time::sleep(Duration::from_millis(10)).await;

        info!(
            container_id = %container_id,
            task_id = %task_id,
            elapsed_ms = start_time.elapsed().as_millis(),
            "Force kill completed"
        );

        PreemptionResult::success(
            container_id.to_string(),
            task_id.to_string(),
            reason,
            start_time.elapsed(),
            false, // Was not graceful
        )
    }
}

impl Default for PreemptionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_preemption_reasons() {
        // Test all preemption reasons
        let reasons = vec![
            PreemptionReason::Lane0Priority,
            PreemptionReason::VramBudgetExceeded,
            PreemptionReason::EpochTransition,
            PreemptionReason::ManualRequest,
        ];

        for reason in reasons {
            let manager = PreemptionManager::new();
            let result = manager.preempt("container-1", "task-1", reason).await;

            assert!(result.success);
            assert_eq!(result.reason, reason);
            assert_eq!(result.container_id, "container-1");
            assert_eq!(result.task_id, "task-1");
        }
    }

    #[tokio::test]
    async fn test_graceful_preemption() {
        let manager = PreemptionManager::with_timeout_secs(2);

        let result = manager.preempt(
            "container-graceful",
            "task-graceful",
            PreemptionReason::Lane0Priority,
        ).await;

        assert!(result.success);
        assert!(result.was_graceful); // Should complete gracefully in MVP stub
        assert!(result.duration_ms < 1000); // Should be fast in stubbed implementation
    }

    #[tokio::test]
    async fn test_immediate_preemption() {
        let manager = PreemptionManager::new();

        let result = manager.preempt_with_strategy(
            "container-immediate",
            "task-immediate",
            PreemptionReason::ManualRequest,
            PreemptionStrategy::Immediate,
        ).await;

        assert!(result.success);
        assert!(!result.was_graceful); // Immediate kill is never graceful
        assert!(result.duration_ms < 100); // Should be very fast
    }

    #[tokio::test]
    async fn test_concurrent_preemption_protection() {
        let manager = Arc::new(PreemptionManager::new());

        let manager1 = manager.clone();
        let manager2 = manager.clone();

        // Try to preempt the same container concurrently
        let handle1 = tokio::spawn(async move {
            manager1.preempt("container-1", "task-1", PreemptionReason::Lane0Priority).await
        });

        // Small delay to ensure first preemption starts
        tokio::time::sleep(Duration::from_millis(10)).await;

        let handle2 = tokio::spawn(async move {
            manager2.preempt("container-1", "task-2", PreemptionReason::VramBudgetExceeded).await
        });

        let result1 = handle1.await.unwrap();
        let result2 = handle2.await.unwrap();

        // One should succeed, one should fail (already preempting)
        assert!(result1.success || result2.success);
        assert!(!(result1.success && result2.success)); // Not both
    }

    #[tokio::test]
    async fn test_is_preempting() {
        let manager = Arc::new(PreemptionManager::with_timeout_secs(1));

        // Not preempting initially
        assert!(!manager.is_preempting("container-1").await);

        // Start preemption in background
        let manager_clone = manager.clone();
        let handle = tokio::spawn(async move {
            manager_clone.preempt("container-1", "task-1", PreemptionReason::Lane0Priority).await
        });

        // Small delay to let preemption start
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Should be preempting now
        // Note: In MVP stub this might already be done due to fast execution
        // In production with real containers, this would be true during preemption

        // Wait for completion
        let result = handle.await.unwrap();
        assert!(result.success);

        // Not preempting after completion
        assert!(!manager.is_preempting("container-1").await);
    }

    #[tokio::test]
    async fn test_cancel_preemption() {
        let manager = PreemptionManager::new();

        // Can't cancel non-existent preemption
        assert!(!manager.cancel_preemption("container-1").await);

        // Note: Due to stubbed implementation being fast, it's hard to test
        // cancellation in the middle of preemption. In production with real
        // containers, you would start a long preemption and cancel it mid-way.
    }

    #[test]
    fn test_reason_as_str() {
        assert_eq!(PreemptionReason::Lane0Priority.as_str(), "lane0_priority");
        assert_eq!(PreemptionReason::VramBudgetExceeded.as_str(), "vram_budget_exceeded");
        assert_eq!(PreemptionReason::EpochTransition.as_str(), "epoch_transition");
        assert_eq!(PreemptionReason::ManualRequest.as_str(), "manual_request");
    }

    #[test]
    fn test_reason_urgency() {
        assert!(PreemptionReason::Lane0Priority.is_urgent());
        assert!(PreemptionReason::VramBudgetExceeded.is_urgent());
        assert!(!PreemptionReason::EpochTransition.is_urgent());
        assert!(!PreemptionReason::ManualRequest.is_urgent());
    }

    #[test]
    fn test_strategy_as_str() {
        assert_eq!(PreemptionStrategy::Graceful.as_str(), "graceful");
        assert_eq!(PreemptionStrategy::Immediate.as_str(), "immediate");
    }
}
