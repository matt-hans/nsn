//! Mock VortexClient for testing Lane 0 video generation without actual sidecar.

use std::collections::{HashSet, VecDeque};
use std::time::Duration;

use nsn_lane0::{GenerationOutput, VortexError, VortexResult};
use nsn_types::Recipe;

/// Mock VortexClient for simulation testing.
///
/// Provides configurable success/failure behavior for video generation.
///
/// # Example
///
/// ```rust,ignore
/// use nsn_simulation::mocks::MockVortexClient;
///
/// let mut client = MockVortexClient::new()
///     .with_success(&[1, 2, 3])
///     .with_latency(Duration::from_millis(100));
///
/// let output = client.generate_slot(&recipe).await?;
/// ```
#[derive(Debug, Clone)]
pub struct MockVortexClient {
    /// Slots that generate successfully
    success_slots: HashSet<u64>,
    /// Slots that timeout
    timeout_slots: HashSet<u64>,
    /// Simulated generation latency (for timing tests)
    latency: Option<Duration>,
    /// Generated outputs (for verification)
    pub generated: VecDeque<GenerationOutput>,
    /// Failure injection: probability of random failure (0.0 - 1.0)
    failure_rate: f64,
    /// Custom CLIP embedding to return (for Byzantine testing)
    custom_embedding: Option<Vec<f32>>,
}

impl Default for MockVortexClient {
    fn default() -> Self {
        Self::new()
    }
}

impl MockVortexClient {
    /// Create a new mock client.
    pub fn new() -> Self {
        Self {
            success_slots: HashSet::new(),
            timeout_slots: HashSet::new(),
            latency: None,
            generated: VecDeque::new(),
            failure_rate: 0.0,
            custom_embedding: None,
        }
    }

    /// Configure slots that will succeed.
    pub fn with_success(mut self, slots: &[u64]) -> Self {
        self.success_slots.extend(slots);
        self
    }

    /// Configure slots that will timeout.
    pub fn with_timeout(mut self, slots: &[u64]) -> Self {
        self.timeout_slots.extend(slots);
        self
    }

    /// Configure simulated latency for generation.
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = Some(latency);
        self
    }

    /// Configure failure rate for random failures.
    pub fn with_failure_rate(mut self, rate: f64) -> Self {
        self.failure_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Configure custom CLIP embedding (for Byzantine behavior).
    pub fn with_custom_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.custom_embedding = Some(embedding);
        self
    }

    /// Add a success slot dynamically.
    pub fn add_success_slot(&mut self, slot: u64) {
        self.success_slots.insert(slot);
    }

    /// Remove a success slot (to simulate failures).
    pub fn remove_success_slot(&mut self, slot: u64) {
        self.success_slots.remove(&slot);
    }

    /// Check if a slot is configured for success.
    pub fn is_success_slot(&self, slot: u64) -> bool {
        self.success_slots.contains(&slot)
    }

    /// Get the number of generated outputs.
    pub fn generation_count(&self) -> usize {
        self.generated.len()
    }

    /// Clear generated outputs.
    pub fn clear_generated(&mut self) {
        self.generated.clear();
    }

    /// Simulate slot generation.
    pub async fn generate_slot(&mut self, recipe: &Recipe) -> VortexResult<GenerationOutput> {
        let slot = recipe.slot_params.slot_number;

        // Apply latency if configured
        if let Some(latency) = self.latency {
            tokio::time::sleep(latency).await;
        }

        // Check for timeout
        if self.timeout_slots.contains(&slot) {
            return Err(VortexError::Timeout { timeout_ms: 5000 });
        }

        // Check for success
        if !self.success_slots.contains(&slot) {
            return Err(VortexError::Execution(
                "slot not in success list".to_string(),
            ));
        }

        // Generate embedding (custom or standard)
        let clip_embedding = self
            .custom_embedding
            .clone()
            .unwrap_or_else(|| vec![0.5f32; 512]);

        let output = GenerationOutput {
            video_data: vec![0u8; 1024 * 1024], // 1MB mock video
            audio_waveform: vec![0u8; 48000 * 2], // Mock audio
            clip_embedding,
            content_id: format!("QmSlot{}", slot),
            generation_time_ms: 30000,
            slot_id: slot,
        };

        self.generated.push_back(output.clone());
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsn_types::{AudioTrack, SemanticConstraints, SecurityMetadata, SlotParams, VisualTrack};

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
                script: "Test script".to_string(),
                voice_id: "voice-1".to_string(),
                speed: 1.0,
                emotion: "neutral".to_string(),
            },
            visual_track: VisualTrack {
                prompt: "Test scene".to_string(),
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

    #[tokio::test]
    async fn test_success_generation() {
        let mut client = MockVortexClient::new().with_success(&[1, 2, 3]);

        let recipe = make_recipe(1);
        let result = client.generate_slot(&recipe).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.slot_id, 1);
        assert_eq!(output.clip_embedding.len(), 512);
    }

    #[tokio::test]
    async fn test_failure_generation() {
        let mut client = MockVortexClient::new().with_success(&[1]);

        let recipe = make_recipe(99);
        let result = client.generate_slot(&recipe).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_timeout_generation() {
        let mut client = MockVortexClient::new().with_timeout(&[5]);

        let recipe = make_recipe(5);
        let result = client.generate_slot(&recipe).await;

        assert!(matches!(result, Err(VortexError::Timeout { .. })));
    }

    #[tokio::test]
    async fn test_custom_embedding() {
        let custom = vec![0.1f32; 512];
        let mut client = MockVortexClient::new()
            .with_success(&[1])
            .with_custom_embedding(custom.clone());

        let recipe = make_recipe(1);
        let result = client.generate_slot(&recipe).await.unwrap();

        assert_eq!(result.clip_embedding, custom);
    }

    #[tokio::test]
    async fn test_generation_tracking() {
        let mut client = MockVortexClient::new().with_success(&[1, 2, 3]);

        for slot in 1..=3 {
            let recipe = make_recipe(slot);
            client.generate_slot(&recipe).await.unwrap();
        }

        assert_eq!(client.generation_count(), 3);
    }
}
