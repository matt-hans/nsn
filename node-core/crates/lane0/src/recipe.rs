//! Recipe processing for Lane 0 video generation.
//!
//! Handles incoming recipes from P2P GossipSub, validates them,
//! and queues them for slot generation.

use std::collections::VecDeque;

use parity_scale_codec::Decode;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use nsn_types::Recipe;

use crate::error::{RecipeError, RecipeResult};

/// Configuration for recipe processing.
#[derive(Debug, Clone)]
pub struct RecipeConfig {
    /// Maximum number of pending recipes.
    pub max_pending: usize,
    /// Minimum slot duration in seconds.
    pub min_duration_sec: u32,
    /// Maximum slot duration in seconds.
    pub max_duration_sec: u32,
    /// Minimum FPS.
    pub min_fps: u32,
    /// Maximum FPS.
    pub max_fps: u32,
}

impl Default for RecipeConfig {
    fn default() -> Self {
        Self {
            max_pending: 10,
            min_duration_sec: 5,
            max_duration_sec: 60,
            min_fps: 24,
            max_fps: 60,
        }
    }
}

/// Processes and queues recipes for slot generation.
pub struct RecipeProcessor {
    /// Configuration.
    config: RecipeConfig,
    /// Queue of pending recipes.
    pending: VecDeque<Recipe>,
    /// Receiver for raw recipe bytes from P2P.
    p2p_rx: Option<mpsc::Receiver<Vec<u8>>>,
    /// Set of processed slot numbers to detect duplicates.
    processed_slots: std::collections::HashSet<u64>,
}

impl RecipeProcessor {
    /// Create a new recipe processor.
    pub fn new(config: RecipeConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
            p2p_rx: None,
            processed_slots: std::collections::HashSet::new(),
        }
    }

    /// Create a recipe processor with P2P subscription.
    pub fn with_p2p(config: RecipeConfig, p2p_rx: mpsc::Receiver<Vec<u8>>) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
            p2p_rx: Some(p2p_rx),
            processed_slots: std::collections::HashSet::new(),
        }
    }

    /// Get the number of pending recipes.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Check if the queue has capacity.
    pub fn has_capacity(&self) -> bool {
        self.pending.len() < self.config.max_pending
    }

    /// Add a recipe to the queue.
    ///
    /// Validates the recipe before queuing.
    pub fn enqueue(&mut self, recipe: Recipe) -> RecipeResult<()> {
        // Validate recipe
        self.validate(&recipe)?;

        // Check for duplicates
        let slot = recipe.slot_params.slot_number;
        if self.processed_slots.contains(&slot) {
            return Err(RecipeError::Duplicate { slot });
        }

        // Check capacity
        if !self.has_capacity() {
            return Err(RecipeError::QueueFull {
                capacity: self.config.max_pending,
            });
        }

        self.pending.push_back(recipe);
        debug!(slot, pending = self.pending.len(), "Recipe queued");
        Ok(())
    }

    /// Parse and enqueue a recipe from raw bytes (CBOR or JSON).
    pub fn enqueue_bytes(&mut self, data: &[u8]) -> RecipeResult<()> {
        let recipe = self.parse_recipe(data)?;
        self.enqueue(recipe)
    }

    /// Get the next recipe from the queue (async, waits for P2P).
    pub async fn next_recipe(&mut self) -> Option<Recipe> {
        // First check pending queue
        if let Some(recipe) = self.pending.pop_front() {
            self.processed_slots.insert(recipe.slot_params.slot_number);
            return Some(recipe);
        }

        // Then wait for P2P if available
        let rx = self.p2p_rx.as_mut()?;
        loop {
            let data = rx.recv().await?;
            match Self::parse_recipe_static(&self.config, &data) {
                Ok(recipe) => {
                    let slot = recipe.slot_params.slot_number;
                    if !self.processed_slots.contains(&slot) {
                        self.processed_slots.insert(slot);
                        return Some(recipe);
                    }
                    debug!(slot, "Duplicate recipe from P2P, skipping");
                }
                Err(e) => {
                    warn!(error = %e, "Failed to parse recipe from P2P");
                }
            }
        }
    }

    /// Try to get the next recipe without waiting.
    pub fn try_next_recipe(&mut self) -> Option<Recipe> {
        if let Some(recipe) = self.pending.pop_front() {
            self.processed_slots.insert(recipe.slot_params.slot_number);
            return Some(recipe);
        }

        // Check P2P non-blocking
        let rx = self.p2p_rx.as_mut()?;
        while let Ok(data) = rx.try_recv() {
            match Self::parse_recipe_static(&self.config, &data) {
                Ok(recipe) => {
                    let slot = recipe.slot_params.slot_number;
                    if !self.processed_slots.contains(&slot) {
                        self.processed_slots.insert(slot);
                        return Some(recipe);
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Failed to parse recipe from P2P");
                }
            }
        }

        None
    }

    /// Clear the processed slots tracking (call on epoch transition).
    pub fn clear_processed(&mut self) {
        self.processed_slots.clear();
        debug!("Cleared processed slots tracking");
    }

    /// Parse a recipe from bytes.
    fn parse_recipe(&self, data: &[u8]) -> RecipeResult<Recipe> {
        Self::parse_recipe_bytes(data)
    }

    /// Parse a recipe from bytes (static version).
    fn parse_recipe_bytes(data: &[u8]) -> RecipeResult<Recipe> {
        // Try SCALE decoding first (Substrate standard)
        if let Ok(recipe) = Recipe::decode(&mut &data[..]) {
            return Ok(recipe);
        }

        // Try JSON as fallback
        serde_json::from_slice(data)
            .map_err(|e| RecipeError::Deserialization(e.to_string()))
    }

    /// Parse and validate a recipe (static version for use without &mut self).
    fn parse_recipe_static(config: &RecipeConfig, data: &[u8]) -> RecipeResult<Recipe> {
        let recipe = Self::parse_recipe_bytes(data)?;
        Self::validate_recipe(config, &recipe)?;
        Ok(recipe)
    }

    /// Parse and validate a recipe.
    #[allow(dead_code)] // Kept for API consistency with static version
    fn parse_and_validate(&self, data: &[u8]) -> RecipeResult<Recipe> {
        let recipe = self.parse_recipe(data)?;
        self.validate(&recipe)?;
        Ok(recipe)
    }

    /// Validate a recipe's fields.
    fn validate(&self, recipe: &Recipe) -> RecipeResult<()> {
        Self::validate_recipe(&self.config, recipe)
    }

    /// Validate a recipe's fields (static version).
    fn validate_recipe(config: &RecipeConfig, recipe: &Recipe) -> RecipeResult<()> {
        // Check required fields
        if recipe.recipe_id.is_empty() {
            return Err(RecipeError::MissingField {
                field: "recipe_id".to_string(),
            });
        }

        // Validate slot parameters
        let params = &recipe.slot_params;

        if params.duration_sec < config.min_duration_sec {
            return Err(RecipeError::InvalidSlotParams(format!(
                "duration {}s below minimum {}s",
                params.duration_sec, config.min_duration_sec
            )));
        }

        if params.duration_sec > config.max_duration_sec {
            return Err(RecipeError::InvalidSlotParams(format!(
                "duration {}s exceeds maximum {}s",
                params.duration_sec, config.max_duration_sec
            )));
        }

        if params.fps < config.min_fps {
            return Err(RecipeError::InvalidSlotParams(format!(
                "fps {} below minimum {}",
                params.fps, config.min_fps
            )));
        }

        if params.fps > config.max_fps {
            return Err(RecipeError::InvalidSlotParams(format!(
                "fps {} exceeds maximum {}",
                params.fps, config.max_fps
            )));
        }

        // Validate audio track
        if recipe.audio_track.script.is_empty() {
            return Err(RecipeError::MissingField {
                field: "audio_track.script".to_string(),
            });
        }

        // Validate visual track
        if recipe.visual_track.prompt.is_empty() {
            return Err(RecipeError::MissingField {
                field: "visual_track.prompt".to_string(),
            });
        }

        // Validate semantic constraints
        if recipe.semantic_constraints.min_clip_score < 0.0
            || recipe.semantic_constraints.min_clip_score > 1.0
        {
            return Err(RecipeError::Validation(format!(
                "min_clip_score {} must be between 0.0 and 1.0",
                recipe.semantic_constraints.min_clip_score
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsn_types::{AudioTrack, SemanticConstraints, SecurityMetadata, SlotParams, VisualTrack};

    fn make_valid_recipe(slot: u64) -> Recipe {
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
                script: "Hello world".to_string(),
                voice_id: "voice-1".to_string(),
                speed: 1.0,
                emotion: "neutral".to_string(),
            },
            visual_track: VisualTrack {
                prompt: "A beautiful sunset".to_string(),
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
                director_id: "director-1".to_string(),
                ed25519_signature: vec![],
                timestamp: 0,
            },
        }
    }

    #[test]
    fn test_recipe_processor_new() {
        let processor = RecipeProcessor::new(RecipeConfig::default());
        assert_eq!(processor.pending_count(), 0);
        assert!(processor.has_capacity());
    }

    #[test]
    fn test_enqueue_valid_recipe() {
        let mut processor = RecipeProcessor::new(RecipeConfig::default());
        let recipe = make_valid_recipe(1);

        processor.enqueue(recipe).expect("should enqueue");
        assert_eq!(processor.pending_count(), 1);
    }

    #[test]
    fn test_enqueue_duplicate_rejected() {
        let mut processor = RecipeProcessor::new(RecipeConfig::default());
        let recipe1 = make_valid_recipe(1);
        let recipe2 = make_valid_recipe(1); // Same slot

        processor.enqueue(recipe1).expect("first should succeed");

        // Consume the first recipe
        let _ = processor.try_next_recipe();

        // Try to add same slot again
        let result = processor.enqueue(recipe2);
        assert!(matches!(result, Err(RecipeError::Duplicate { slot: 1 })));
    }

    #[test]
    fn test_queue_capacity() {
        let config = RecipeConfig {
            max_pending: 2,
            ..Default::default()
        };
        let mut processor = RecipeProcessor::new(config);

        processor.enqueue(make_valid_recipe(1)).unwrap();
        processor.enqueue(make_valid_recipe(2)).unwrap();

        let result = processor.enqueue(make_valid_recipe(3));
        assert!(matches!(result, Err(RecipeError::QueueFull { capacity: 2 })));
    }

    #[test]
    fn test_validation_missing_recipe_id() {
        let mut processor = RecipeProcessor::new(RecipeConfig::default());
        let mut recipe = make_valid_recipe(1);
        recipe.recipe_id = "".to_string();

        let result = processor.enqueue(recipe);
        assert!(matches!(
            result,
            Err(RecipeError::MissingField { field }) if field == "recipe_id"
        ));
    }

    #[test]
    fn test_validation_duration_too_short() {
        let mut processor = RecipeProcessor::new(RecipeConfig::default());
        let mut recipe = make_valid_recipe(1);
        recipe.slot_params.duration_sec = 1; // Below minimum of 5

        let result = processor.enqueue(recipe);
        assert!(matches!(result, Err(RecipeError::InvalidSlotParams(_))));
    }

    #[test]
    fn test_validation_fps_too_high() {
        let mut processor = RecipeProcessor::new(RecipeConfig::default());
        let mut recipe = make_valid_recipe(1);
        recipe.slot_params.fps = 120; // Above maximum of 60

        let result = processor.enqueue(recipe);
        assert!(matches!(result, Err(RecipeError::InvalidSlotParams(_))));
    }

    #[test]
    fn test_validation_missing_script() {
        let mut processor = RecipeProcessor::new(RecipeConfig::default());
        let mut recipe = make_valid_recipe(1);
        recipe.audio_track.script = "".to_string();

        let result = processor.enqueue(recipe);
        assert!(matches!(
            result,
            Err(RecipeError::MissingField { field }) if field == "audio_track.script"
        ));
    }

    #[test]
    fn test_validation_invalid_clip_score() {
        let mut processor = RecipeProcessor::new(RecipeConfig::default());
        let mut recipe = make_valid_recipe(1);
        recipe.semantic_constraints.min_clip_score = 1.5; // Above 1.0

        let result = processor.enqueue(recipe);
        assert!(matches!(result, Err(RecipeError::Validation(_))));
    }

    #[test]
    fn test_try_next_recipe() {
        let mut processor = RecipeProcessor::new(RecipeConfig::default());

        // Empty queue
        assert!(processor.try_next_recipe().is_none());

        // Add recipes
        processor.enqueue(make_valid_recipe(1)).unwrap();
        processor.enqueue(make_valid_recipe(2)).unwrap();

        // Get them in order
        let r1 = processor.try_next_recipe().unwrap();
        assert_eq!(r1.slot_params.slot_number, 1);

        let r2 = processor.try_next_recipe().unwrap();
        assert_eq!(r2.slot_params.slot_number, 2);

        // Empty again
        assert!(processor.try_next_recipe().is_none());
    }

    #[test]
    fn test_clear_processed() {
        let mut processor = RecipeProcessor::new(RecipeConfig::default());

        processor.enqueue(make_valid_recipe(1)).unwrap();
        let _ = processor.try_next_recipe();

        // Slot 1 is now processed
        let result = processor.enqueue(make_valid_recipe(1));
        assert!(matches!(result, Err(RecipeError::Duplicate { .. })));

        // Clear processed slots
        processor.clear_processed();

        // Now we can add slot 1 again
        processor.enqueue(make_valid_recipe(1)).unwrap();
        assert_eq!(processor.pending_count(), 1);
    }

    #[test]
    fn test_parse_json_recipe() {
        let processor = RecipeProcessor::new(RecipeConfig::default());
        let recipe = make_valid_recipe(42);
        let json = serde_json::to_vec(&recipe).unwrap();

        let parsed = processor.parse_recipe(&json).unwrap();
        assert_eq!(parsed.recipe_id, recipe.recipe_id);
        assert_eq!(parsed.slot_params.slot_number, 42);
    }
}
