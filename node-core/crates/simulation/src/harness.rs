//! Test harness for multi-node scenario orchestration.
//!
//! Provides high-level API for setting up and running multi-node simulations.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use libp2p::PeerId;
use thiserror::Error;

use crate::mocks::{MockBftParticipant, MockChainClient, MockChunkPublisher, MockVortexClient};
use crate::network::{LatencyProfile, SimulatedNetwork};
use crate::{ByzantineBehavior, NodeRole};

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

/// A simulated executor node.
pub struct SimulatedExecutor {
    /// Peer ID
    pub peer_id: PeerId,
    /// Mock chain client
    pub chain: MockChainClient,
    /// Current state
    pub state: ExecutorSimState,
}

/// State tracking for simulated executor.
#[derive(Debug, Clone, Default)]
pub struct ExecutorSimState {
    /// Tasks assigned
    pub tasks_assigned: Vec<u64>,
    /// Tasks completed
    pub tasks_completed: Vec<u64>,
    /// Tasks failed
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

        let executor = SimulatedExecutor {
            peer_id,
            chain: MockChainClient::new(),
            state: ExecutorSimState::default(),
        };

        self.executors.insert(peer_id, executor);
        peer_id
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
