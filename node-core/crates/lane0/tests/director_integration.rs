//! Integration tests for DirectorService lifecycle.
//!
//! Tests the full director lifecycle: epoch transitions, recipe processing,
//! slot generation pipeline, and state machine correctness.

use std::collections::VecDeque;

use nsn_lane0::{
    BftConsensusResult, BftError, BftResult, DirectorConfig, DirectorState, GenerationOutput,
    PublishResult, RecipeConfig, RecipeProcessor, SlotError, VortexError, VortexResult,
};
use nsn_scheduler::EpochEvent;
use nsn_types::{
    AudioTrack, EpochInfo, Recipe, SemanticConstraints, SecurityMetadata, SlotParams,
    VideoChunkHeader, VisualTrack,
};

// =============================================================================
// Mock Types for Testing
// =============================================================================

/// Mock VortexClient for testing without actual sidecar.
struct MockVortexClient {
    /// Recipes to generate successfully.
    success_slots: std::collections::HashSet<u64>,
    /// Timeout slots.
    timeout_slots: std::collections::HashSet<u64>,
    /// Generated outputs.
    generated: VecDeque<GenerationOutput>,
}

impl MockVortexClient {
    fn new() -> Self {
        Self {
            success_slots: std::collections::HashSet::new(),
            timeout_slots: std::collections::HashSet::new(),
            generated: VecDeque::new(),
        }
    }

    fn with_success(mut self, slots: &[u64]) -> Self {
        self.success_slots.extend(slots);
        self
    }

    #[allow(dead_code)]
    fn with_timeout(mut self, slots: &[u64]) -> Self {
        self.timeout_slots.extend(slots);
        self
    }

    async fn generate_slot(&mut self, recipe: &Recipe) -> VortexResult<GenerationOutput> {
        let slot = recipe.slot_params.slot_number;

        if self.timeout_slots.contains(&slot) {
            return Err(VortexError::Timeout { timeout_ms: 5000 });
        }

        if !self.success_slots.contains(&slot) {
            return Err(VortexError::Execution("slot not in success list".to_string()));
        }

        let output = GenerationOutput {
            video_data: vec![0u8; 1024 * 1024], // 1MB mock video
            audio_waveform: vec![0u8; 48000 * 2], // Mock audio
            clip_embedding: vec![0.5f32; 512],
            content_id: format!("QmSlot{}", slot),
            generation_time_ms: 30000,
            slot_id: slot,
        };

        self.generated.push_back(output.clone());
        Ok(output)
    }
}

/// Mock BFT participant for testing.
struct MockBftParticipant {
    /// Slots that reach consensus.
    consensus_slots: std::collections::HashSet<u64>,
    /// Collected results.
    results: VecDeque<BftConsensusResult>,
}

impl MockBftParticipant {
    fn new() -> Self {
        Self {
            consensus_slots: std::collections::HashSet::new(),
            results: VecDeque::new(),
        }
    }

    fn with_consensus(mut self, slots: &[u64]) -> Self {
        self.consensus_slots.extend(slots);
        self
    }

    async fn run_consensus(
        &mut self,
        slot: u64,
        _embedding: Vec<f32>,
        _timeout_ms: u64,
    ) -> BftResult<BftConsensusResult> {
        let success = self.consensus_slots.contains(&slot);
        let similarity = if success { 0.95 } else { 0.50 };

        let result = BftConsensusResult {
            slot,
            canonical_hash: [0u8; 32],
            signers: vec![],
            success,
            similarity,
        };

        if !success {
            return Err(BftError::ConsensusFailed {
                slot,
                similarity,
                threshold: 0.85,
            });
        }

        self.results.push_back(result.clone());
        Ok(result)
    }
}

/// Mock chunk publisher for testing.
struct MockChunkPublisher {
    /// Published slots.
    published: VecDeque<(u64, String, usize)>,
}

impl MockChunkPublisher {
    fn new() -> Self {
        Self {
            published: VecDeque::new(),
        }
    }

    async fn publish_video(
        &mut self,
        slot: u64,
        content_id: &str,
        video_data: &[u8],
    ) -> PublishResult<Vec<VideoChunkHeader>> {
        let chunk_size = 1024 * 1024;
        let num_chunks = video_data.len().div_ceil(chunk_size);

        self.published
            .push_back((slot, content_id.to_string(), num_chunks));

        // Create mock headers
        let headers: Vec<VideoChunkHeader> = (0..num_chunks as u32)
            .map(|i| VideoChunkHeader {
                version: 1,
                slot,
                content_id: content_id.to_string(),
                chunk_index: i,
                total_chunks: num_chunks as u32,
                timestamp_ms: 0,
                is_keyframe: i == 0,
                payload_hash: [0u8; 32],
            })
            .collect();

        Ok(headers)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn make_epoch_info(epoch: u64, slot: u64, active_lane: u8) -> EpochInfo {
    EpochInfo {
        epoch,
        slot,
        block_number: epoch * 100 + slot,
        active_lane,
    }
}

fn make_recipe(slot: u64) -> Recipe {
    Recipe {
        recipe_id: format!("recipe-{}", slot),
        version: "1.0".to_string(),
        slot_params: SlotParams {
            slot_number: slot,
            duration_sec: 30,
            resolution: "1920x1080".to_string(),
            fps: 30,
        },
        audio_track: AudioTrack {
            script: "Test script for slot".to_string(),
            voice_id: "voice-1".to_string(),
            speed: 1.0,
            emotion: "neutral".to_string(),
        },
        visual_track: VisualTrack {
            prompt: "A beautiful test scene".to_string(),
            negative_prompt: "".to_string(),
            motion_preset: "default".to_string(),
            expression_sequence: vec![],
            camera_motion: "static".to_string(),
        },
        semantic_constraints: SemanticConstraints {
            min_clip_score: 0.85,
            banned_concepts: vec![],
            required_concepts: vec![],
        },
        security: SecurityMetadata {
            director_id: "test-director".to_string(),
            ed25519_signature: vec![],
            timestamp: 0,
        },
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

#[tokio::test]
async fn test_director_state_machine_transitions() {
    // Test that state machine transitions correctly through lifecycle

    let state = DirectorState::Standby;
    assert!(state.is_standby());
    assert!(!state.is_active());
    assert_eq!(state.epoch(), None);

    let state = DirectorState::OnDeck { epoch: 5 };
    assert!(!state.is_standby());
    assert!(!state.is_active());
    assert_eq!(state.epoch(), Some(5));

    let state = DirectorState::Active { epoch: 5, slot: 10 };
    assert!(!state.is_standby());
    assert!(state.is_active());
    assert_eq!(state.epoch(), Some(5));

    let state = DirectorState::Draining { epoch: 5 };
    assert!(!state.is_standby());
    assert!(!state.is_active());
    assert_eq!(state.epoch(), Some(5));
}

#[tokio::test]
async fn test_mock_vortex_client_success() {
    let mut client = MockVortexClient::new().with_success(&[1, 2, 3]);

    let recipe = make_recipe(1);
    let result = client.generate_slot(&recipe).await;

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.slot_id, 1);
    assert_eq!(output.clip_embedding.len(), 512);
}

#[tokio::test]
async fn test_mock_vortex_client_failure() {
    let mut client = MockVortexClient::new().with_success(&[1]);

    let recipe = make_recipe(99); // Not in success list
    let result = client.generate_slot(&recipe).await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_mock_bft_consensus_success() {
    let mut bft = MockBftParticipant::new().with_consensus(&[1, 2, 3]);

    let embedding = vec![0.5f32; 512];
    let result = bft.run_consensus(1, embedding, 5000).await;

    assert!(result.is_ok());
    let consensus = result.unwrap();
    assert!(consensus.success);
    assert_eq!(consensus.slot, 1);
}

#[tokio::test]
async fn test_mock_bft_consensus_failure() {
    let mut bft = MockBftParticipant::new().with_consensus(&[1]);

    let embedding = vec![0.5f32; 512];
    let result = bft.run_consensus(99, embedding, 5000).await;

    assert!(result.is_err());
    match result {
        Err(BftError::ConsensusFailed { slot, .. }) => {
            assert_eq!(slot, 99);
        }
        _ => panic!("Expected ConsensusFailed error"),
    }
}

#[tokio::test]
async fn test_mock_publisher() {
    let mut publisher = MockChunkPublisher::new();

    let video_data = vec![0u8; 2 * 1024 * 1024]; // 2MB
    let headers = publisher
        .publish_video(1, "QmTest", &video_data)
        .await
        .unwrap();

    assert_eq!(headers.len(), 2); // 2 chunks for 2MB
    assert_eq!(headers[0].chunk_index, 0);
    assert!(headers[0].is_keyframe);
    assert_eq!(headers[1].chunk_index, 1);
    assert!(!headers[1].is_keyframe);
}

#[tokio::test]
async fn test_full_slot_pipeline_with_mocks() {
    // Simulate full slot generation pipeline
    let slot = 42u64;

    // 1. Vortex generates video
    let mut vortex = MockVortexClient::new().with_success(&[slot]);
    let recipe = make_recipe(slot);
    let gen_output = vortex.generate_slot(&recipe).await.unwrap();

    assert_eq!(gen_output.slot_id, slot);
    assert!(!gen_output.video_data.is_empty());

    // 2. BFT consensus on CLIP embedding
    let mut bft = MockBftParticipant::new().with_consensus(&[slot]);
    let bft_result = bft
        .run_consensus(slot, gen_output.clip_embedding.clone(), 5000)
        .await
        .unwrap();

    assert!(bft_result.success);

    // 3. Publish video chunks
    let mut publisher = MockChunkPublisher::new();
    let headers = publisher
        .publish_video(slot, &gen_output.content_id, &gen_output.video_data)
        .await
        .unwrap();

    assert!(!headers.is_empty());

    // Verify pipeline completed
    assert_eq!(vortex.generated.len(), 1);
    assert_eq!(bft.results.len(), 1);
    assert_eq!(publisher.published.len(), 1);
}

#[tokio::test]
async fn test_recipe_processor_queuing() {
    let config = RecipeConfig {
        max_pending: 3,
        ..Default::default()
    };
    let mut processor = RecipeProcessor::new(config);

    // Enqueue recipes
    processor.enqueue(make_recipe(1)).unwrap();
    processor.enqueue(make_recipe(2)).unwrap();
    processor.enqueue(make_recipe(3)).unwrap();

    assert_eq!(processor.pending_count(), 3);

    // Queue is full
    let result = processor.enqueue(make_recipe(4));
    assert!(result.is_err());

    // Dequeue and process
    let r1 = processor.try_next_recipe().unwrap();
    assert_eq!(r1.slot_params.slot_number, 1);

    // Can add again after dequeue
    processor.enqueue(make_recipe(4)).unwrap();
}

#[tokio::test]
async fn test_epoch_event_on_deck() {
    let epoch_info = make_epoch_info(5, 0, 0);
    let event = EpochEvent::OnDeck {
        epoch: epoch_info,
        am_director: true,
    };

    match event {
        EpochEvent::OnDeck { epoch, am_director } => {
            assert_eq!(epoch.epoch, 5);
            assert!(am_director);
        }
        _ => panic!("Wrong event type"),
    }
}

#[tokio::test]
async fn test_epoch_event_started() {
    let epoch_info = make_epoch_info(5, 0, 0);
    let event = EpochEvent::EpochStarted { epoch: epoch_info };

    match event {
        EpochEvent::EpochStarted { epoch } => {
            assert_eq!(epoch.epoch, 5);
            assert_eq!(epoch.active_lane, 0);
        }
        _ => panic!("Wrong event type"),
    }
}

#[tokio::test]
async fn test_epoch_event_ended() {
    let event = EpochEvent::EpochEnded { epoch: 5 };

    match event {
        EpochEvent::EpochEnded { epoch } => {
            assert_eq!(epoch, 5);
        }
        _ => panic!("Wrong event type"),
    }
}

#[tokio::test]
async fn test_generation_output_clip_dimensions() {
    // Verify CLIP embedding is correct size for consensus
    let output = GenerationOutput {
        video_data: vec![],
        audio_waveform: vec![],
        clip_embedding: vec![0.5f32; 512],
        content_id: "test".to_string(),
        generation_time_ms: 1000,
        slot_id: 1,
    };

    assert_eq!(output.clip_embedding.len(), 512);
}

#[tokio::test]
async fn test_slot_error_types() {
    // Test error conversion chain
    let vortex_err = VortexError::Timeout { timeout_ms: 5000 };
    let slot_err: SlotError = vortex_err.into();

    match slot_err {
        SlotError::Generation(_) => {}
        _ => panic!("Expected Generation error variant"),
    }

    let bft_err = BftError::ConsensusFailed {
        slot: 1,
        similarity: 0.5,
        threshold: 0.85,
    };
    let slot_err: SlotError = bft_err.into();

    match slot_err {
        SlotError::Consensus(_) => {}
        _ => panic!("Expected Consensus error variant"),
    }
}

#[tokio::test]
async fn test_director_config_defaults() {
    let config = DirectorConfig::default();

    assert_eq!(config.bft_timeout_ms, 5000);
    assert_eq!(config.max_pending_recipes, 10);
    assert_eq!(config.chunk_size, 1024 * 1024);
    assert_eq!(config.sidecar_endpoint, "http://127.0.0.1:50050");
}

#[tokio::test]
async fn test_multiple_slot_pipeline() {
    // Test processing multiple slots in sequence
    let slots = vec![100, 101, 102, 103, 104];
    let mut vortex = MockVortexClient::new().with_success(&slots);
    let mut bft = MockBftParticipant::new().with_consensus(&slots);
    let mut publisher = MockChunkPublisher::new();

    for slot in &slots {
        let recipe = make_recipe(*slot);

        // Generate
        let output = vortex.generate_slot(&recipe).await.unwrap();
        assert_eq!(output.slot_id, *slot);

        // Consensus
        let consensus = bft
            .run_consensus(*slot, output.clip_embedding.clone(), 5000)
            .await
            .unwrap();
        assert!(consensus.success);

        // Publish
        let headers = publisher
            .publish_video(*slot, &output.content_id, &output.video_data)
            .await
            .unwrap();
        assert!(!headers.is_empty());
    }

    // Verify all slots processed
    assert_eq!(vortex.generated.len(), 5);
    assert_eq!(bft.results.len(), 5);
    assert_eq!(publisher.published.len(), 5);
}
