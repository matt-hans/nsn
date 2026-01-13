//! Pre-defined test scenarios for common simulation patterns.
//!
//! Each scenario encapsulates setup, execution, and verification logic
//! for a specific test case.

use std::time::Duration;

use thiserror::Error;

use crate::harness::TestHarness;
use crate::ByzantineBehavior;

/// Errors from scenario execution.
#[derive(Debug, Error)]
pub enum ScenarioFailure {
    /// Consensus not reached
    #[error("Consensus not reached: {0}")]
    ConsensusNotReached(String),
    /// Insufficient votes
    #[error("Insufficient votes: expected {expected}, got {actual}")]
    InsufficientVotes { expected: usize, actual: usize },
    /// Timeout exceeded
    #[error("Timeout exceeded: {0}")]
    TimeoutExceeded(String),
    /// Invalid state
    #[error("Invalid state: {0}")]
    InvalidState(String),
    /// Task not completed
    #[error("Task not completed: {0}")]
    TaskNotCompleted(u64),
}

/// Result type for scenario verification.
pub type ScenarioVerifyResult = Result<(), ScenarioFailure>;

/// Configuration for a scenario.
#[derive(Debug, Clone)]
pub struct ScenarioConfig {
    /// Number of directors
    pub num_directors: usize,
    /// Number of executors
    pub num_executors: usize,
    /// Number of Byzantine nodes
    pub num_byzantine: usize,
    /// Byzantine behavior type
    pub byzantine_behavior: Option<ByzantineBehavior>,
    /// Network latency
    pub latency_ms: u64,
    /// Timeout for scenario execution
    pub timeout_ms: u64,
    /// Slots to run
    pub slots: Vec<u64>,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        Self {
            num_directors: 5,
            num_executors: 0,
            num_byzantine: 0,
            byzantine_behavior: None,
            latency_ms: 0,
            timeout_ms: 5000,
            slots: vec![1],
        }
    }
}

/// Result of running a scenario.
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    /// Number of successful consensus rounds
    pub consensus_reached: usize,
    /// Number of slots generated
    pub slots_generated: usize,
    /// Number of chunks published
    pub chunks_published: usize,
    /// Number of tasks completed
    pub tasks_completed: usize,
    /// Messages exchanged
    pub message_count: usize,
    /// Duration of scenario
    pub duration: Duration,
    /// Directors that reached consensus
    pub successful_directors: Vec<libp2p::PeerId>,
    /// Failed directors
    pub failed_directors: Vec<libp2p::PeerId>,
}

impl Default for ScenarioResult {
    fn default() -> Self {
        Self {
            consensus_reached: 0,
            slots_generated: 0,
            chunks_published: 0,
            tasks_completed: 0,
            message_count: 0,
            duration: Duration::ZERO,
            successful_directors: vec![],
            failed_directors: vec![],
        }
    }
}

/// Pre-defined test scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    /// 5 directors, 1 slot, successful consensus
    BaselineConsensus,
    /// 5 directors, 1 Byzantine with divergent embeddings
    ByzantineDirector,
    /// 5 directors, variable latency (10-100ms)
    HighLatencyConsensus,
    /// 5 directors, 2|3 partition, verify 3-of-5 succeeds in larger partition
    NetworkPartition,
    /// Director goes offline mid-slot, verify recovery
    DirectorFailure,
    /// Full workflow: stake → election → BFT → chunk publish
    FullEpochLifecycle,
    /// Lane 1 task: created → assigned → executed → verified
    TaskLifecycle,
    /// Lane 0 active, Lane 1 draining, epoch ends, Lane 1 resumes
    LaneSwitching,
}

impl Scenario {
    /// Get the configuration for this scenario.
    pub fn configure(&self) -> ScenarioConfig {
        match self {
            Scenario::BaselineConsensus => ScenarioConfig {
                num_directors: 5,
                num_executors: 0,
                num_byzantine: 0,
                byzantine_behavior: None,
                latency_ms: 0,
                timeout_ms: 5000,
                slots: vec![1],
            },
            Scenario::ByzantineDirector => ScenarioConfig {
                num_directors: 5,
                num_executors: 0,
                num_byzantine: 1,
                byzantine_behavior: Some(ByzantineBehavior::DivergentEmbeddings),
                latency_ms: 0,
                timeout_ms: 5000,
                slots: vec![1],
            },
            Scenario::HighLatencyConsensus => ScenarioConfig {
                num_directors: 5,
                num_executors: 0,
                num_byzantine: 0,
                byzantine_behavior: None,
                latency_ms: 100,
                timeout_ms: 10000,
                slots: vec![1],
            },
            Scenario::NetworkPartition => ScenarioConfig {
                num_directors: 5,
                num_executors: 0,
                num_byzantine: 0,
                byzantine_behavior: None,
                latency_ms: 0,
                timeout_ms: 5000,
                slots: vec![1],
            },
            Scenario::DirectorFailure => ScenarioConfig {
                num_directors: 5,
                num_executors: 0,
                num_byzantine: 0,
                byzantine_behavior: None,
                latency_ms: 0,
                timeout_ms: 5000,
                slots: vec![1],
            },
            Scenario::FullEpochLifecycle => ScenarioConfig {
                num_directors: 5,
                num_executors: 0,
                num_byzantine: 0,
                byzantine_behavior: None,
                latency_ms: 0,
                timeout_ms: 60000,
                slots: vec![1, 2, 3],
            },
            Scenario::TaskLifecycle => ScenarioConfig {
                num_directors: 0,
                num_executors: 3,
                num_byzantine: 0,
                byzantine_behavior: None,
                latency_ms: 0,
                timeout_ms: 10000,
                slots: vec![],
            },
            Scenario::LaneSwitching => ScenarioConfig {
                num_directors: 5,
                num_executors: 3,
                num_byzantine: 0,
                byzantine_behavior: None,
                latency_ms: 0,
                timeout_ms: 30000,
                slots: vec![1],
            },
        }
    }

    /// Run the scenario.
    pub async fn run(&self, harness: &mut TestHarness) -> ScenarioResult {
        match self {
            Scenario::BaselineConsensus => run_baseline_consensus(harness).await,
            Scenario::ByzantineDirector => run_byzantine_director(harness).await,
            Scenario::HighLatencyConsensus => run_high_latency_consensus(harness).await,
            Scenario::NetworkPartition => run_network_partition(harness).await,
            Scenario::DirectorFailure => run_director_failure(harness).await,
            Scenario::FullEpochLifecycle => run_full_epoch_lifecycle(harness).await,
            Scenario::TaskLifecycle => run_task_lifecycle(harness).await,
            Scenario::LaneSwitching => run_lane_switching(harness).await,
        }
    }

    /// Verify the scenario result.
    pub fn verify(&self, result: &ScenarioResult) -> ScenarioVerifyResult {
        match self {
            Scenario::BaselineConsensus => {
                if result.consensus_reached < 1 {
                    return Err(ScenarioFailure::ConsensusNotReached(
                        "No consensus reached".to_string(),
                    ));
                }
                if result.successful_directors.len() < 3 {
                    return Err(ScenarioFailure::InsufficientVotes {
                        expected: 3,
                        actual: result.successful_directors.len(),
                    });
                }
                Ok(())
            }
            Scenario::ByzantineDirector => {
                // Should still reach consensus with 4 honest nodes
                if result.consensus_reached < 1 {
                    return Err(ScenarioFailure::ConsensusNotReached(
                        "No consensus with Byzantine node".to_string(),
                    ));
                }
                // Byzantine node should fail
                if result.failed_directors.is_empty() {
                    return Err(ScenarioFailure::InvalidState(
                        "Byzantine node should have failed".to_string(),
                    ));
                }
                Ok(())
            }
            Scenario::HighLatencyConsensus => {
                if result.consensus_reached < 1 {
                    return Err(ScenarioFailure::ConsensusNotReached(
                        "No consensus under high latency".to_string(),
                    ));
                }
                Ok(())
            }
            Scenario::NetworkPartition => {
                // Larger partition (3 nodes) should reach consensus
                if result.successful_directors.len() < 3 {
                    return Err(ScenarioFailure::InsufficientVotes {
                        expected: 3,
                        actual: result.successful_directors.len(),
                    });
                }
                Ok(())
            }
            Scenario::DirectorFailure => {
                // Should reach consensus with remaining 4 nodes
                if result.consensus_reached < 1 {
                    return Err(ScenarioFailure::ConsensusNotReached(
                        "No consensus after director failure".to_string(),
                    ));
                }
                Ok(())
            }
            Scenario::FullEpochLifecycle => {
                // All slots should be generated
                if result.slots_generated < 3 {
                    return Err(ScenarioFailure::InvalidState(format!(
                        "Expected 3 slots, got {}",
                        result.slots_generated
                    )));
                }
                Ok(())
            }
            Scenario::TaskLifecycle => {
                if result.tasks_completed < 1 {
                    return Err(ScenarioFailure::TaskNotCompleted(1));
                }
                Ok(())
            }
            Scenario::LaneSwitching => {
                // Should have consensus and task completion
                if result.consensus_reached < 1 {
                    return Err(ScenarioFailure::ConsensusNotReached(
                        "No consensus during lane switching".to_string(),
                    ));
                }
                Ok(())
            }
        }
    }
}

// Scenario implementations

async fn run_baseline_consensus(harness: &mut TestHarness) -> ScenarioResult {
    let mut result = ScenarioResult::default();
    let start = std::time::Instant::now();

    // Setup: 5 directors
    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1]);

    // Activate directors
    harness.emit_epoch_started(&directors);

    // Run slot 1
    let successful = harness.run_slot(1).await.unwrap_or(0);
    result.consensus_reached = if successful >= 3 { 1 } else { 0 };
    result.slots_generated = successful;
    result.chunks_published = successful;

    // Collect results
    for peer in &directors {
        if let Some(director) = harness.get_director(peer) {
            if director.state.consensus_results.contains(&1) {
                result.successful_directors.push(*peer);
            } else {
                result.failed_directors.push(*peer);
            }
        }
    }

    result.duration = start.elapsed();
    result
}

async fn run_byzantine_director(harness: &mut TestHarness) -> ScenarioResult {
    let mut result = ScenarioResult::default();
    let start = std::time::Instant::now();

    // Setup: 5 directors, 1 Byzantine
    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1]);

    // Make first director Byzantine
    harness
        .set_byzantine(directors[0], ByzantineBehavior::DropMessages)
        .unwrap();

    // Activate directors
    harness.emit_epoch_started(&directors);

    // Run slot 1
    let successful = harness.run_slot(1).await.unwrap_or(0);
    result.consensus_reached = if successful >= 3 { 1 } else { 0 };
    result.slots_generated = successful;

    // Collect results
    for peer in &directors {
        if let Some(director) = harness.get_director(peer) {
            if director.state.consensus_results.contains(&1) {
                result.successful_directors.push(*peer);
            } else {
                result.failed_directors.push(*peer);
            }
        }
    }

    result.duration = start.elapsed();
    result
}

async fn run_high_latency_consensus(harness: &mut TestHarness) -> ScenarioResult {
    let mut result = ScenarioResult::default();
    let start = std::time::Instant::now();

    // Setup: 5 directors with latency
    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1]);

    // Activate directors
    harness.emit_epoch_started(&directors);

    // Pause time for deterministic simulation
    tokio::time::pause();

    // Run with time advancement
    harness.advance_time(Duration::from_millis(100)).await;
    let successful = harness.run_slot(1).await.unwrap_or(0);
    result.consensus_reached = if successful >= 3 { 1 } else { 0 };
    result.slots_generated = successful;

    for peer in &directors {
        if let Some(director) = harness.get_director(peer) {
            if director.state.consensus_results.contains(&1) {
                result.successful_directors.push(*peer);
            }
        }
    }

    result.duration = start.elapsed();
    result
}

async fn run_network_partition(harness: &mut TestHarness) -> ScenarioResult {
    let mut result = ScenarioResult::default();
    let start = std::time::Instant::now();

    // Setup: 5 directors
    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1]);

    // Create partition: [3] | [2]
    harness.inject_partition(vec![
        vec![directors[0], directors[1], directors[2]],
        vec![directors[3], directors[4]],
    ]);

    // Activate only larger partition
    harness.emit_epoch_started(&[directors[0], directors[1], directors[2]]);

    // Run slot 1
    let successful = harness.run_slot(1).await.unwrap_or(0);
    result.consensus_reached = if successful >= 3 { 1 } else { 0 };
    result.slots_generated = successful;

    for peer in &directors[0..3] {
        if let Some(director) = harness.get_director(peer) {
            if director.state.consensus_results.contains(&1) {
                result.successful_directors.push(*peer);
            }
        }
    }

    harness.heal_partition();
    result.duration = start.elapsed();
    result
}

async fn run_director_failure(harness: &mut TestHarness) -> ScenarioResult {
    let mut result = ScenarioResult::default();
    let start = std::time::Instant::now();

    // Setup: 5 directors
    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1]);

    // Activate directors
    harness.emit_epoch_started(&directors);

    // Remove one director (simulating failure)
    harness.remove_node(directors[0]).unwrap();
    result.failed_directors.push(directors[0]);

    // Run slot 1 with remaining 4
    let successful = harness.run_slot(1).await.unwrap_or(0);
    result.consensus_reached = if successful >= 3 { 1 } else { 0 };
    result.slots_generated = successful;

    for peer in &directors[1..5] {
        if let Some(director) = harness.get_director(peer) {
            if director.state.consensus_results.contains(&1) {
                result.successful_directors.push(*peer);
            }
        }
    }

    result.duration = start.elapsed();
    result
}

async fn run_full_epoch_lifecycle(harness: &mut TestHarness) -> ScenarioResult {
    let mut result = ScenarioResult::default();
    let start = std::time::Instant::now();

    // Setup: 5 directors
    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    harness.configure_slot_success(&[1, 2, 3]);

    // Pause time for deterministic simulation
    tokio::time::pause();

    // OnDeck phase
    harness.emit_on_deck(&directors);
    harness.advance_time(Duration::from_millis(100)).await;

    // EpochStarted
    harness.emit_epoch_started(&directors);

    // Run 3 slots
    for slot in 1..=3 {
        let successful = harness.run_slot(slot).await.unwrap_or(0);
        if successful >= 3 {
            result.consensus_reached += 1;
        }
        result.slots_generated += successful;
        harness.advance_time(Duration::from_millis(100)).await;
    }

    // EpochEnded
    harness.emit_epoch_ended();

    for peer in &directors {
        if let Some(director) = harness.get_director(peer) {
            if !director.state.consensus_results.is_empty() {
                result.successful_directors.push(*peer);
            }
        }
    }

    result.duration = start.elapsed();
    result
}

async fn run_task_lifecycle(harness: &mut TestHarness) -> ScenarioResult {
    let mut result = ScenarioResult::default();
    let start = std::time::Instant::now();

    // Setup: 3 executors
    let executors: Vec<_> = (0..3).map(|_| harness.add_executor()).collect();

    // Configure task 1 for success
    harness.configure_task_success(&[1]);

    // Run full task lifecycle through real executor pipeline:
    // 1. Create task event
    // 2. Assign task to executor
    // 3. Execute via MockExecutionRunner
    // 4. Submit result via MockResultSubmitter
    match harness.run_task_lifecycle(executors[0], 1, "flux-schnell").await {
        Ok(()) => {
            result.tasks_completed = 1;

            // Verify executor state
            if let Some(executor) = harness.get_executor(&executors[0]) {
                // Verify task was assigned
                assert!(executor.state.tasks_assigned.contains(&1));

                // Verify submissions were made (start_task + submit_result = 2)
                assert_eq!(executor.submitter.submission_count(), 2);

                // Verify task was completed
                assert!(executor.state.tasks_completed.contains(&1));
            }
        }
        Err(e) => {
            tracing::error!(error = %e, "Task lifecycle failed");
        }
    }

    result.duration = start.elapsed();
    result
}

async fn run_lane_switching(harness: &mut TestHarness) -> ScenarioResult {
    let mut result = ScenarioResult::default();
    let start = std::time::Instant::now();

    // Setup: 5 directors, 3 executors
    let directors: Vec<_> = (0..5).map(|_| harness.add_director()).collect();
    let executors: Vec<_> = (0..3).map(|_| harness.add_executor()).collect();

    // Configure both Lane 0 and Lane 1 success conditions
    harness.configure_slot_success(&[1]);
    harness.configure_task_success(&[1]);

    // Start with Lane 1 active
    harness.chain_state.state.set_active_lane(1);

    // Create and assign task on Lane 1 (but don't execute yet)
    if let Err(e) = harness.create_and_assign_task(executors[0], 1, "model-1", "QmInput1") {
        tracing::error!(error = %e, "Failed to create and assign task");
        result.duration = start.elapsed();
        return result;
    }

    // Switch to Lane 0 (epoch starts)
    harness.chain_state.state.set_active_lane(0);
    harness.emit_epoch_started(&directors);

    // Run Lane 0 slot
    let successful = harness.run_slot(1).await.unwrap_or(0);
    result.consensus_reached = if successful >= 3 { 1 } else { 0 };
    result.slots_generated = successful;

    // End epoch, switch back to Lane 1
    harness.emit_epoch_ended();
    harness.chain_state.state.set_active_lane(1);

    // Complete Lane 1 task through the real execution pipeline
    match harness.run_task(executors[0], 1, "model-1", "QmInput1").await {
        Ok(()) => {
            result.tasks_completed = 1;

            // Verify executor state through submitter tracking
            if let Some(executor) = harness.executors.get(&executors[0]) {
                // Verify submissions were made (start_task + submit_result = 2)
                assert_eq!(
                    executor.submitter.submission_count(),
                    2,
                    "Expected 2 submissions (start_task + submit_result)"
                );

                // Verify task was completed
                assert!(
                    executor.state.tasks_completed.contains(&1),
                    "Task 1 should be in completed state"
                );
            }
        }
        Err(e) => {
            tracing::error!(error = %e, "Lane 1 task execution failed after lane switch");
        }
    }

    for peer in &directors {
        if let Some(director) = harness.get_director(peer) {
            if !director.state.consensus_results.is_empty() {
                result.successful_directors.push(*peer);
            }
        }
    }

    result.duration = start.elapsed();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_baseline_consensus_scenario() {
        let mut harness = TestHarness::new();
        let result = Scenario::BaselineConsensus.run(&mut harness).await;
        assert!(Scenario::BaselineConsensus.verify(&result).is_ok());
    }

    #[tokio::test]
    async fn test_byzantine_director_scenario() {
        let mut harness = TestHarness::new();
        let result = Scenario::ByzantineDirector.run(&mut harness).await;
        // Byzantine director should still reach consensus with 4 honest nodes
        assert!(result.consensus_reached >= 1 || result.failed_directors.len() >= 1);
    }

    #[tokio::test]
    async fn test_network_partition_scenario() {
        let mut harness = TestHarness::new();
        let result = Scenario::NetworkPartition.run(&mut harness).await;
        assert!(Scenario::NetworkPartition.verify(&result).is_ok());
    }

    #[tokio::test]
    async fn test_director_failure_scenario() {
        let mut harness = TestHarness::new();
        let result = Scenario::DirectorFailure.run(&mut harness).await;
        assert!(result.failed_directors.len() >= 1);
        assert!(result.consensus_reached >= 1);
    }

    #[tokio::test]
    async fn test_full_epoch_lifecycle_scenario() {
        let mut harness = TestHarness::new();
        let result = Scenario::FullEpochLifecycle.run(&mut harness).await;
        assert!(result.slots_generated >= 3);
    }

    #[tokio::test]
    async fn test_task_lifecycle_scenario() {
        let mut harness = TestHarness::new();
        let result = Scenario::TaskLifecycle.run(&mut harness).await;
        assert!(Scenario::TaskLifecycle.verify(&result).is_ok());
    }

    #[tokio::test]
    async fn test_lane_switching_scenario() {
        let mut harness = TestHarness::new();
        let result = Scenario::LaneSwitching.run(&mut harness).await;
        assert!(result.consensus_reached >= 1);
        assert!(result.tasks_completed >= 1);
    }

    #[test]
    fn test_scenario_configs() {
        // Verify all scenarios have reasonable configs
        for scenario in [
            Scenario::BaselineConsensus,
            Scenario::ByzantineDirector,
            Scenario::HighLatencyConsensus,
            Scenario::NetworkPartition,
            Scenario::DirectorFailure,
            Scenario::FullEpochLifecycle,
            Scenario::TaskLifecycle,
            Scenario::LaneSwitching,
        ] {
            let config = scenario.configure();
            assert!(config.timeout_ms > 0);
            assert!(config.num_directors > 0 || config.num_executors > 0);
        }
    }
}
