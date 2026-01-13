//! Test harness for multi-node scenario orchestration.
//!
//! Provides high-level API for setting up and running multi-node simulations.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use libp2p::PeerId;
use thiserror::Error;

use crate::mocks::{
    MockBftParticipant, MockChainClient, MockChunkPublisher, MockExecutionRunner,
    MockResultSubmitter, MockVortexClient,
};
use crate::network::{LatencyProfile, SimulatedNetwork};
use crate::{ByzantineBehavior, NodeRole};
use nsn_lane1::{ExecutionRunnerTrait, ResultSubmitterTrait, TaskSpec};

/// Errors from harness operations.
#[derive(Debug, Error)]
pub enum HarnessError {
    /// Node not found
    #[error("Node {0} not found")]
    NodeNotFound(PeerId),
    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    /// Scenario execution failed
    #[error("Scenario failed: {0}")]
    ScenarioFailed(String),
}

/// Result type for harness operations.
pub type HarnessResult<T> = Result<T, HarnessError>;

/// A simulated director node.
pub struct SimulatedDirector {
    /// Peer ID
    pub peer_id: PeerId,
    /// Mock Vortex client
    pub vortex: MockVortexClient,
    /// Mock BFT participant
    pub bft: MockBftParticipant,
    /// Mock chunk publisher
    pub publisher: MockChunkPublisher,
    /// Current state
    pub state: DirectorSimState,
    /// Byzantine behavior (if any)
    pub byzantine: Option<ByzantineBehavior>,
}

/// State tracking for simulated director.
#[derive(Debug, Clone, Default)]
pub struct DirectorSimState {
    /// Current epoch
    pub epoch: Option<u64>,
    /// Is on deck for next epoch
    pub on_deck: bool,
    /// Is active this epoch
    pub active: bool,
    /// Slots generated
    pub slots_generated: Vec<u64>,
    /// Consensus results
    pub consensus_results: Vec<u64>,
    /// Chunks published
    pub chunks_published: Vec<u64>,
}

/// A simulated executor node with full Lane 1 mock stack.
pub struct SimulatedExecutor {
    /// Peer ID
    pub peer_id: PeerId,
    /// Mock chain client (for event injection)
    pub chain: MockChainClient,
    /// Mock execution runner (sidecar simulation)
    pub runner: MockExecutionRunner,
    /// Mock result submitter (chain submission tracking)
    pub submitter: MockResultSubmitter,
    /// Current state
    pub state: ExecutorSimState,
    /// Account ID for this executor
    account_id: String,
}

impl SimulatedExecutor {
    /// Create a new simulated executor with the given peer ID.
    pub fn new(peer_id: PeerId) -> Self {
        let account_id = format!("5Executor{:x}", peer_id.to_bytes()[0..4].iter().fold(0u32, |acc, &x| acc * 256 + x as u32));
        Self {
            peer_id,
            chain: MockChainClient::new().with_account(account_id.clone()),
            runner: MockExecutionRunner::new(),
            submitter: MockResultSubmitter::new(),
            state: ExecutorSimState::default(),
            account_id,
        }
    }

    /// Configure which tasks succeed execution.
    pub fn with_success_tasks(mut self, task_ids: &[u64]) -> Self {
        self.runner = self.runner.with_success(task_ids);
        self
    }

    /// Configure execution latency.
    pub fn with_execution_latency(mut self, latency: Duration) -> Self {
        self.runner = self.runner.with_latency(latency);
        self
    }

    /// Get the account ID for this executor.
    pub fn account_id(&self) -> &str {
        &self.account_id
    }

    /// Process a single task through the full execution pipeline.
    ///
    /// This simulates the complete TaskExecutorService flow:
    /// 1. Submit start_task extrinsic
    /// 2. Execute task via sidecar (MockExecutionRunner)
    /// 3. On success: submit result to chain
    /// 4. On failure: submit failure to chain
    pub async fn process_task(&mut self, task_id: u64, model_id: &str, input_cid: &str) -> Result<(), String> {
        // Track that we started this task
        self.state.tasks_started.push(task_id);

        // 1. Submit start_task extrinsic
        self.submitter
            .start_task(task_id)
            .await
            .map_err(|e| e.to_string())?;

        // 2. Build task spec and execute
        let task_spec = TaskSpec {
            id: task_id,
            model_id: model_id.to_string(),
            input_cid: input_cid.to_string(),
            parameters: vec![],
            timeout_ms: Some(300_000),
        };

        match self.runner.execute(&task_spec).await {
            Ok(output) => {
                // 3a. Submit result on success
                self.submitter
                    .submit_result(task_id, &output.output_cid, None)
                    .await
                    .map_err(|e| e.to_string())?;

                self.state.tasks_completed.push(task_id);

                // Update chain state
                self.chain.state.completed_tasks.push(task_id);

                Ok(())
            }
            Err(e) => {
                // 3b. Submit failure
                self.submitter
                    .fail_task(task_id, &e.to_string())
                    .await
                    .map_err(|e| e.to_string())?;

                self.state.tasks_failed.push(task_id);

                // Update chain state
                self.chain.state.failed_tasks.push(task_id);

                Err(e.to_string())
            }
        }
    }

    /// Inject a task created event into this executor's chain.
    pub fn inject_task_created(&mut self, task_id: u64, model_id: &str, input_cid: &str) {
        self.chain.create_task(task_id, model_id, input_cid, 1000);
    }

    /// Inject a task assigned event (assigned to this executor).
    pub fn inject_task_assigned(&mut self, task_id: u64) {
        self.chain.assign_task_to_me(task_id);
        self.state.tasks_assigned.push(task_id);
    }
}

/// State tracking for simulated executor.
#[derive(Debug, Clone, Default)]
pub struct ExecutorSimState {
    /// Tasks assigned to this executor
    pub tasks_assigned: Vec<u64>,
    /// Tasks where start_task was called
    pub tasks_started: Vec<u64>,
    /// Tasks completed successfully
    pub tasks_completed: Vec<u64>,
    /// Tasks that failed
    pub tasks_failed: Vec<u64>,
}

/// Test harness for multi-node simulations.
///
/// Orchestrates multiple simulated nodes and provides high-level
/// operations for testing distributed scenarios.
///
/// # Example
///
/// ```rust,ignore
/// use nsn_simulation::TestHarness;
///
/// let mut harness = TestHarness::new();
///
/// // Add 5 directors and 3 executors
/// for _ in 0..5 {
///     harness.add_director();
/// }
/// for _ in 0..3 {
///     harness.add_executor();
/// }
///
/// // Emit epoch event
/// harness.emit_epoch_event(EpochEvent::OnDeck { ... }).await;
///
/// // Run simulation
/// harness.advance_time(Duration::from_secs(10)).await;
///
/// // Verify results
/// harness.assert_consensus_reached(1);
/// ```
pub struct TestHarness {
    /// Simulated network
    pub network: SimulatedNetwork,
    /// Director nodes
    pub directors: HashMap<PeerId, SimulatedDirector>,
    /// Executor nodes
    pub executors: HashMap<PeerId, SimulatedExecutor>,
    /// Mock chain state
    pub chain_state: MockChainClient,
    /// Current simulation time
    time: Instant,
    /// Metrics collection
    metrics: HarnessMetrics,
}

/// Metrics collected during simulation.
#[derive(Debug, Clone, Default)]
pub struct HarnessMetrics {
    /// Total messages sent
    pub messages_sent: usize,
    /// Total consensus rounds
    pub consensus_rounds: usize,
    /// Total slots generated
    pub slots_generated: usize,
    /// Total tasks completed
    pub tasks_completed: usize,
    /// Simulation duration
    pub duration: Duration,
}

impl Default for TestHarness {
    fn default() -> Self {
        Self::new()
    }
}

impl TestHarness {
    /// Create a new test harness.
    pub fn new() -> Self {
        Self {
            network: SimulatedNetwork::new(),
            directors: HashMap::new(),
            executors: HashMap::new(),
            chain_state: MockChainClient::new(),
            time: Instant::now(),
            metrics: HarnessMetrics::default(),
        }
    }

    /// Configure network latency.
    pub fn with_latency(mut self, profile: LatencyProfile) -> Self {
        self.network = self.network.with_latency(profile);
        self
    }

    /// Add a director node.
    pub fn add_director(&mut self) -> PeerId {
        let peer_id = self.network.add_node(NodeRole::Director);

        let director = SimulatedDirector {
            peer_id,
            vortex: MockVortexClient::new(),
            bft: MockBftParticipant::new(),
            publisher: MockChunkPublisher::new(),
            state: DirectorSimState::default(),
            byzantine: None,
        };

        self.directors.insert(peer_id, director);
        peer_id
    }

    /// Add an executor node.
    pub fn add_executor(&mut self) -> PeerId {
        let peer_id = self.network.add_node(NodeRole::Executor);
        let executor = SimulatedExecutor::new(peer_id);
        self.executors.insert(peer_id, executor);
        peer_id
    }

    /// Get a mutable reference to an executor.
    pub fn get_executor_mut(&mut self, peer: &PeerId) -> Option<&mut SimulatedExecutor> {
        self.executors.get_mut(peer)
    }

    /// Remove a node from the simulation.
    pub fn remove_node(&mut self, peer: PeerId) -> HarnessResult<()> {
        self.network
            .remove_node(peer)
            .map_err(|_| HarnessError::NodeNotFound(peer))?;
        self.directors.remove(&peer);
        self.executors.remove(&peer);
        Ok(())
    }

    /// Set Byzantine behavior for a director.
    pub fn set_byzantine(
        &mut self,
        peer: PeerId,
        behavior: ByzantineBehavior,
    ) -> HarnessResult<()> {
        if let Some(director) = self.directors.get_mut(&peer) {
            director.byzantine = Some(behavior);
            Ok(())
        } else {
            Err(HarnessError::NodeNotFound(peer))
        }
    }

    /// Get a reference to a director.
    pub fn get_director(&self, peer: &PeerId) -> Option<&SimulatedDirector> {
        self.directors.get(peer)
    }

    /// Get a mutable reference to a director.
    pub fn get_director_mut(&mut self, peer: &PeerId) -> Option<&mut SimulatedDirector> {
        self.directors.get_mut(peer)
    }

    /// Get a reference to an executor.
    pub fn get_executor(&self, peer: &PeerId) -> Option<&SimulatedExecutor> {
        self.executors.get(peer)
    }

    /// Get all director peer IDs.
    pub fn director_peers(&self) -> Vec<PeerId> {
        self.directors.keys().copied().collect()
    }

    /// Get all executor peer IDs.
    pub fn executor_peers(&self) -> Vec<PeerId> {
        self.executors.keys().copied().collect()
    }

    /// Get count of directors.
    pub fn director_count(&self) -> usize {
        self.directors.len()
    }

    /// Get count of executors.
    pub fn executor_count(&self) -> usize {
        self.executors.len()
    }

    /// Configure success slots for all directors.
    pub fn configure_slot_success(&mut self, slots: &[u64]) {
        for director in self.directors.values_mut() {
            for slot in slots {
                director.vortex.add_success_slot(*slot);
                director.bft.add_consensus_slot(*slot);
            }
        }
    }

    /// Advance simulation time.
    pub async fn advance_time(&mut self, duration: Duration) {
        // Note: caller should call tokio::time::pause() before using this
        // We use tokio::time::advance only - caller manages pause state
        tokio::time::advance(duration).await;
        self.network.advance_time(duration);
        self.metrics.duration += duration;
    }

    /// Run until a condition is met.
    pub async fn run_until<F>(&mut self, max_duration: Duration, mut condition: F) -> bool
    where
        F: FnMut(&Self) -> bool,
    {
        tokio::time::pause();
        let step = Duration::from_millis(10);
        let mut elapsed = Duration::ZERO;

        while elapsed < max_duration {
            if condition(self) {
                return true;
            }
            tokio::time::advance(step).await;
            self.network.advance_time(step);
            elapsed += step;
        }

        false
    }

    /// Emit OnDeck event to selected directors.
    pub fn emit_on_deck(&mut self, directors: &[PeerId]) {
        let epoch = self.chain_state.state.current_epoch;
        for peer in directors {
            if let Some(director) = self.directors.get_mut(peer) {
                director.state.on_deck = true;
                director.state.epoch = Some(epoch);
            }
        }
    }

    /// Emit EpochStarted event to activate directors.
    pub fn emit_epoch_started(&mut self, directors: &[PeerId]) {
        let epoch = self.chain_state.state.current_epoch;
        for peer in directors {
            if let Some(director) = self.directors.get_mut(peer) {
                director.state.active = true;
                director.state.on_deck = false;
                director.state.epoch = Some(epoch);
            }
        }
    }

    /// Emit EpochEnded event.
    pub fn emit_epoch_ended(&mut self) {
        for director in self.directors.values_mut() {
            director.state.active = false;
            director.state.on_deck = false;
        }
        self.chain_state.state.advance_epoch();
    }

    /// Simulate slot generation for active directors.
    pub async fn run_slot(&mut self, slot: u64) -> HarnessResult<usize> {
        let active_directors: Vec<PeerId> = self
            .directors
            .iter()
            .filter(|(_, d)| d.state.active)
            .map(|(p, _)| *p)
            .collect();

        let mut successful = 0;

        for peer in active_directors {
            if let Some(director) = self.directors.get_mut(&peer) {
                // Skip if Byzantine crash mode
                if matches!(director.byzantine, Some(ByzantineBehavior::DropMessages)) {
                    continue;
                }

                // Simulate generation
                let recipe = create_mock_recipe(slot);
                if let Ok(output) = director.vortex.generate_slot(&recipe).await {
                    director.state.slots_generated.push(slot);

                    // Simulate BFT consensus
                    if let Ok(_result) = director
                        .bft
                        .run_consensus(slot, output.clip_embedding.clone(), 5000)
                        .await
                    {
                        director.state.consensus_results.push(slot);

                        // Simulate publishing
                        if let Ok(_headers) = director
                            .publisher
                            .publish_video(slot, &output.content_id, &output.video_data)
                            .await
                        {
                            director.state.chunks_published.push(slot);
                            successful += 1;
                        }
                    }
                }
            }
        }

        self.metrics.slots_generated += successful;
        self.metrics.consensus_rounds += 1;
        Ok(successful)
    }

    /// Create a network partition.
    pub fn inject_partition(&mut self, groups: Vec<Vec<PeerId>>) {
        let hash_groups: Vec<std::collections::HashSet<PeerId>> =
            groups.into_iter().map(|g| g.into_iter().collect()).collect();
        self.network.inject_partition(hash_groups);
    }

    /// Heal network partition.
    pub fn heal_partition(&mut self) {
        self.network.heal_partition();
    }

    /// Assert that consensus was reached for a slot.
    pub fn assert_consensus_reached(&self, slot: u64) {
        let consensus_count = self
            .directors
            .values()
            .filter(|d| d.state.consensus_results.contains(&slot))
            .count();

        assert!(
            consensus_count >= 3,
            "Expected at least 3 directors to reach consensus for slot {}, got {}",
            slot,
            consensus_count
        );
    }

    /// Assert that a chunk was published.
    pub fn assert_chunk_published(&self, slot: u64) {
        let published = self
            .directors
            .values()
            .any(|d| d.state.chunks_published.contains(&slot));

        assert!(
            published,
            "Expected chunk to be published for slot {}",
            slot
        );
    }

    /// Assert that a task was completed.
    pub fn assert_task_completed(&self, task_id: u64) {
        let completed = self
            .executors
            .values()
            .any(|e| e.state.tasks_completed.contains(&task_id));

        assert!(completed, "Expected task {} to be completed", task_id);
    }

    /// Get collected metrics.
    pub fn metrics(&self) -> &HarnessMetrics {
        &self.metrics
    }

    /// Reset metrics.
    pub fn reset_metrics(&mut self) {
        self.metrics = HarnessMetrics::default();
    }

    // ========== Lane 1 Executor Coordination Methods ==========

    /// Configure which tasks succeed for all executors.
    pub fn configure_task_success(&mut self, task_ids: &[u64]) {
        for executor in self.executors.values_mut() {
            for task_id in task_ids {
                executor.runner.add_success_task(*task_id);
            }
        }
    }

    /// Run a task through an executor's full pipeline.
    ///
    /// # Arguments
    /// * `executor_peer` - The executor to run the task on
    /// * `task_id` - Task ID to execute
    /// * `model_id` - Model to use for execution
    /// * `input_cid` - Input data CID
    pub async fn run_task(
        &mut self,
        executor_peer: PeerId,
        task_id: u64,
        model_id: &str,
        input_cid: &str,
    ) -> Result<(), String> {
        if let Some(executor) = self.executors.get_mut(&executor_peer) {
            let result = executor.process_task(task_id, model_id, input_cid).await;
            if result.is_ok() {
                self.metrics.tasks_completed += 1;
            }
            result
        } else {
            Err(format!("Executor {:?} not found", executor_peer))
        }
    }

    /// Create and assign a task to an executor.
    ///
    /// Injects TaskCreated and TaskAssigned events into the executor's chain.
    pub fn create_and_assign_task(
        &mut self,
        executor_peer: PeerId,
        task_id: u64,
        model_id: &str,
        input_cid: &str,
    ) -> Result<(), String> {
        if let Some(executor) = self.executors.get_mut(&executor_peer) {
            executor.inject_task_created(task_id, model_id, input_cid);
            executor.inject_task_assigned(task_id);
            Ok(())
        } else {
            Err(format!("Executor {:?} not found", executor_peer))
        }
    }

    /// Run full task lifecycle: create -> assign -> execute -> submit.
    ///
    /// This is a convenience method that combines task creation, assignment,
    /// and execution into a single call.
    pub async fn run_task_lifecycle(
        &mut self,
        executor_peer: PeerId,
        task_id: u64,
        model_id: &str,
    ) -> Result<(), String> {
        let input_cid = format!("QmInput{}", task_id);

        // Ensure task is configured for success
        if let Some(executor) = self.executors.get_mut(&executor_peer) {
            executor.runner.add_success_task(task_id);
        }

        // Create and assign
        self.create_and_assign_task(executor_peer, task_id, model_id, &input_cid)?;

        // Execute
        self.run_task(executor_peer, task_id, model_id, &input_cid).await
    }

    /// Assert that an executor has a specific number of submissions.
    pub fn assert_submissions(&self, executor_peer: &PeerId, expected_count: usize) {
        if let Some(executor) = self.executors.get(executor_peer) {
            let actual = executor.submitter.submission_count();
            assert_eq!(
                actual, expected_count,
                "Expected {} submissions for executor {:?}, got {}",
                expected_count, executor_peer, actual
            );
        } else {
            panic!("Executor {:?} not found", executor_peer);
        }
    }

    /// Assert that an executor completed a task.
    pub fn assert_executor_completed(&self, executor_peer: &PeerId, task_id: u64) {
        if let Some(executor) = self.executors.get(executor_peer) {
            assert!(
                executor.state.tasks_completed.contains(&task_id),
                "Task {} not in completed list for executor {:?}. Completed: {:?}",
                task_id,
                executor_peer,
                executor.state.tasks_completed
            );
        } else {
            panic!("Executor {:?} not found", executor_peer);
        }
    }

    /// Assert that an executor failed a task.
    pub fn assert_executor_failed(&self, executor_peer: &PeerId, task_id: u64) {
        if let Some(executor) = self.executors.get(executor_peer) {
            assert!(
                executor.state.tasks_failed.contains(&task_id),
                "Task {} not in failed list for executor {:?}. Failed: {:?}",
                task_id,
                executor_peer,
                executor.state.tasks_failed
            );
        } else {
            panic!("Executor {:?} not found", executor_peer);
        }
    }

    /// Assert that an executor started a task (called start_task extrinsic).
    pub fn assert_executor_started(&self, executor_peer: &PeerId, task_id: u64) {
        if let Some(executor) = self.executors.get(executor_peer) {
            assert!(
                executor.state.tasks_started.contains(&task_id),
                "Task {} not in started list for executor {:?}. Started: {:?}",
                task_id,
                executor_peer,
                executor.state.tasks_started
            );
            assert!(
                executor.submitter.has_start_task(task_id),
                "start_task extrinsic not submitted for task {}",
                task_id
            );
        } else {
            panic!("Executor {:?} not found", executor_peer);
        }
    }
}

/// Create a mock recipe for testing.
fn create_mock_recipe(slot: u64) -> nsn_types::Recipe {
    nsn_types::Recipe {
        recipe_id: format!("recipe-{}", slot),
        version: "1.0".to_string(),
        slot_params: nsn_types::SlotParams {
            slot_number: slot,
            duration_sec: 30,
            resolution: "1920x1080".to_string(),
            fps: 30,
        },
        audio_track: nsn_types::AudioTrack {
            script: "Test script".to_string(),
            voice_id: "voice-1".to_string(),
            speed: 1.0,
            emotion: "neutral".to_string(),
        },
        visual_track: nsn_types::VisualTrack {
            prompt: "Test scene".to_string(),
            negative_prompt: "".to_string(),
            motion_preset: "default".to_string(),
            expression_sequence: vec![],
            camera_motion: "static".to_string(),
        },
        semantic_constraints: nsn_types::SemanticConstraints {
            min_clip_score: 0.85,
            banned_concepts: vec![],
            required_concepts: vec![],
        },
        security: nsn_types::SecurityMetadata {
            director_id: "test-director".to_string(),
            ed25519_signature: vec![],
            timestamp: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_nodes() {
        let mut harness = TestHarness::new();

        let d1 = harness.add_director();
        let d2 = harness.add_director();
        let e1 = harness.add_executor();

        assert_eq!(harness.director_count(), 2);
        assert_eq!(harness.executor_count(), 1);
        assert!(harness.get_director(&d1).is_some());
        assert!(harness.get_director(&d2).is_some());
        assert!(harness.get_executor(&e1).is_some());
    }

    #[tokio::test]
    async fn test_remove_node() {
        let mut harness = TestHarness::new();

        let d1 = harness.add_director();
        assert_eq!(harness.director_count(), 1);

        harness.remove_node(d1).unwrap();
        assert_eq!(harness.director_count(), 0);
    }

    #[tokio::test]
    async fn test_configure_slot_success() {
        let mut harness = TestHarness::new();

        let d1 = harness.add_director();
        harness.configure_slot_success(&[1, 2, 3]);

        let director = harness.get_director(&d1).unwrap();
        assert!(director.vortex.is_success_slot(1));
        assert!(director.vortex.is_success_slot(2));
        assert!(director.vortex.is_success_slot(3));
    }

    #[tokio::test]
    async fn test_emit_epoch_events() {
        let mut harness = TestHarness::new();

        let d1 = harness.add_director();
        let d2 = harness.add_director();

        // Emit OnDeck
        harness.emit_on_deck(&[d1, d2]);
        assert!(harness.get_director(&d1).unwrap().state.on_deck);
        assert!(harness.get_director(&d2).unwrap().state.on_deck);

        // Emit EpochStarted
        harness.emit_epoch_started(&[d1, d2]);
        assert!(harness.get_director(&d1).unwrap().state.active);
        assert!(!harness.get_director(&d1).unwrap().state.on_deck);

        // Emit EpochEnded
        harness.emit_epoch_ended();
        assert!(!harness.get_director(&d1).unwrap().state.active);
    }

    #[tokio::test]
    async fn test_run_slot() {
        let mut harness = TestHarness::new();

        let d1 = harness.add_director();
        let d2 = harness.add_director();
        let d3 = harness.add_director();

        harness.configure_slot_success(&[1]);
        harness.emit_epoch_started(&[d1, d2, d3]);

        let successful = harness.run_slot(1).await.unwrap();
        assert_eq!(successful, 3);

        harness.assert_consensus_reached(1);
        harness.assert_chunk_published(1);
    }

    #[tokio::test]
    async fn test_byzantine_drop_messages() {
        let mut harness = TestHarness::new();

        let d1 = harness.add_director();
        let d2 = harness.add_director();
        let d3 = harness.add_director();

        harness.configure_slot_success(&[1]);
        harness.emit_epoch_started(&[d1, d2, d3]);

        // Make d1 Byzantine
        harness
            .set_byzantine(d1, ByzantineBehavior::DropMessages)
            .unwrap();

        let successful = harness.run_slot(1).await.unwrap();
        assert_eq!(successful, 2); // Only d2 and d3 succeed
    }

    #[tokio::test]
    async fn test_partition() {
        let mut harness = TestHarness::new();

        let d1 = harness.add_director();
        let d2 = harness.add_director();
        let d3 = harness.add_director();
        let d4 = harness.add_director();
        let d5 = harness.add_director();

        // Partition into [d1, d2, d3] | [d4, d5]
        harness.inject_partition(vec![vec![d1, d2, d3], vec![d4, d5]]);

        // Verify partition is set
        let can_d1_d2 = harness.network.get_node(&d1).is_some();
        assert!(can_d1_d2);

        // Heal partition
        harness.heal_partition();
    }

    #[tokio::test]
    async fn test_run_until() {
        let mut harness = TestHarness::new();

        harness.add_director();
        harness.add_director();
        harness.add_director();

        let mut count = 0;
        let found = harness
            .run_until(Duration::from_millis(100), |_| {
                count += 1;
                count >= 5
            })
            .await;

        assert!(found);
        assert!(count >= 5);
    }
}
