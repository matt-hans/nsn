//! Vortex client wrapper for Lane 0 video generation.
//!
//! Provides a high-level interface to call the Vortex pipeline via the
//! sidecar gRPC service. Handles request construction, response parsing,
//! and error handling.

use std::time::Duration;

use nsn_sidecar::{SidecarClient, SidecarClientConfig};
use nsn_types::Recipe;
use tracing::{debug, info};
use uuid::Uuid;

use crate::error::{VortexError, VortexResult};

/// Configuration for the Vortex client.
#[derive(Debug, Clone)]
pub struct VortexClientConfig {
    /// Sidecar gRPC endpoint.
    pub endpoint: String,
    /// Default timeout for generation in milliseconds.
    pub timeout_ms: u64,
    /// Plugin name for Lane 0 generation.
    pub plugin_name: String,
    /// Connection timeout.
    pub connect_timeout: Duration,
}

impl Default for VortexClientConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:50050".to_string(),
            timeout_ms: 45_000, // 45s for glass-to-glass latency target
            plugin_name: "vortex-lane0".to_string(),
            connect_timeout: Duration::from_secs(5),
        }
    }
}

/// Output from Vortex video generation.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Raw video data bytes.
    pub video_data: Vec<u8>,
    /// Audio waveform bytes.
    pub audio_waveform: Vec<u8>,
    /// CLIP embedding from dual-CLIP ensemble (512 dimensions).
    pub clip_embedding: Vec<f32>,
    /// Content identifier (CID) for the generated video.
    pub content_id: String,
    /// Time spent on generation in milliseconds.
    pub generation_time_ms: u64,
    /// Slot ID that was generated.
    pub slot_id: u64,
}

/// Client for calling Vortex pipeline via sidecar gRPC.
pub struct VortexClient {
    /// Underlying sidecar client.
    sidecar: SidecarClient,
    /// Client configuration.
    config: VortexClientConfig,
}

impl VortexClient {
    /// Create a new Vortex client.
    pub async fn new(config: VortexClientConfig) -> VortexResult<Self> {
        let sidecar_config = SidecarClientConfig::new(&config.endpoint)
            .with_connect_timeout(config.connect_timeout)
            .with_request_timeout(Duration::from_millis(config.timeout_ms + 5000));

        let sidecar = SidecarClient::connect_with_config(sidecar_config)
            .await
            .map_err(|e| VortexError::Connection(e.to_string()))?;

        info!(endpoint = %config.endpoint, "Connected to sidecar");

        Ok(Self { sidecar, config })
    }

    /// Create a Vortex client from an existing sidecar client.
    pub fn from_sidecar(sidecar: SidecarClient, config: VortexClientConfig) -> Self {
        Self { sidecar, config }
    }

    /// Get the client configuration.
    pub fn config(&self) -> &VortexClientConfig {
        &self.config
    }

    /// Generate a slot from a recipe.
    ///
    /// Calls the Vortex pipeline via sidecar and returns the generation output
    /// including video data and CLIP embedding for BFT consensus.
    pub async fn generate_slot(&mut self, recipe: &Recipe) -> VortexResult<GenerationOutput> {
        let slot_id = recipe.slot_params.slot_number;
        let task_id = Uuid::new_v4().to_string();

        debug!(
            slot = slot_id,
            task_id = %task_id,
            plugin = %self.config.plugin_name,
            "Starting slot generation"
        );

        // Serialize recipe to JSON for the plugin
        let parameters = serde_json::to_vec(recipe)
            .map_err(|e| VortexError::Execution(format!("failed to serialize recipe: {}", e)))?;

        // Execute the plugin task
        let start = std::time::Instant::now();
        let response = self
            .sidecar
            .execute_plugin_task(
                &task_id,
                &self.config.plugin_name,
                "", // No input CID, recipe is in parameters
                parameters,
                0, // Lane 0
                Some(Duration::from_millis(self.config.timeout_ms)),
            )
            .await
            .map_err(|e| VortexError::Execution(e.to_string()))?;

        let elapsed_ms = start.elapsed().as_millis() as u64;

        if !response.success {
            let error_msg = if response.error_message.is_empty() {
                "unknown error".to_string()
            } else {
                response.error_message.clone()
            };
            return Err(VortexError::Execution(error_msg));
        }

        // Parse the response
        let output = self.parse_response(&response, slot_id, elapsed_ms)?;

        info!(
            slot = slot_id,
            generation_ms = elapsed_ms,
            clip_dims = output.clip_embedding.len(),
            "Slot generation complete"
        );

        Ok(output)
    }

    /// Parse the sidecar response into GenerationOutput.
    fn parse_response(
        &self,
        response: &nsn_sidecar::proto::ExecuteTaskResponse,
        slot_id: u64,
        generation_time_ms: u64,
    ) -> VortexResult<GenerationOutput> {
        // Response contains output_cid pointing to video data
        // and result_metadata containing JSON-encoded CLIP embedding and audio
        let content_id = response.output_cid.clone();

        // Parse result metadata as JSON
        if response.result_metadata.is_empty() {
            return Err(VortexError::ResponseParse("missing result_metadata".to_string()));
        }

        let result: VortexResultJson = serde_json::from_slice(&response.result_metadata)
            .map_err(|e| VortexError::ResponseParse(format!("failed to parse result: {}", e)))?;

        // Validate CLIP embedding dimensions
        if result.clip_embedding.len() != 512 {
            return Err(VortexError::InvalidOutput(format!(
                "expected 512-dim CLIP embedding, got {}",
                result.clip_embedding.len()
            )));
        }

        Ok(GenerationOutput {
            video_data: result.video_data,
            audio_waveform: result.audio_waveform,
            clip_embedding: result.clip_embedding,
            content_id,
            generation_time_ms,
            slot_id,
        })
    }
}

/// JSON structure returned by Vortex plugin in result_json.
#[derive(Debug, serde::Deserialize)]
struct VortexResultJson {
    /// Raw video data (base64 decoded by serde).
    #[serde(with = "base64_bytes")]
    video_data: Vec<u8>,
    /// Audio waveform data (base64 decoded).
    #[serde(with = "base64_bytes")]
    audio_waveform: Vec<u8>,
    /// CLIP embedding from dual-CLIP ensemble.
    clip_embedding: Vec<f32>,
}

mod base64_bytes {
    use base64::prelude::*;
    use serde::{Deserialize, Deserializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s: String = Deserialize::deserialize(deserializer)?;
        BASE64_STANDARD
            .decode(&s)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = VortexClientConfig::default();
        assert_eq!(config.endpoint, "http://127.0.0.1:50050");
        assert_eq!(config.timeout_ms, 45_000);
        assert_eq!(config.plugin_name, "vortex-lane0");
    }

    #[test]
    fn test_parse_vortex_result_json() {
        let json = r#"{
            "video_data": "dmlkZW9fZGF0YQ==",
            "audio_waveform": "YXVkaW9fZGF0YQ==",
            "clip_embedding": [0.1, 0.2, 0.3]
        }"#;

        let result: VortexResultJson = serde_json::from_str(json).unwrap();
        assert_eq!(result.video_data, b"video_data");
        assert_eq!(result.audio_waveform, b"audio_data");
        assert_eq!(result.clip_embedding.len(), 3);
    }

    #[test]
    fn test_generation_output_fields() {
        let output = GenerationOutput {
            video_data: vec![1, 2, 3],
            audio_waveform: vec![4, 5, 6],
            clip_embedding: vec![0.5; 512],
            content_id: "QmTest".to_string(),
            generation_time_ms: 1000,
            slot_id: 42,
        };

        assert_eq!(output.slot_id, 42);
        assert_eq!(output.clip_embedding.len(), 512);
        assert_eq!(output.content_id, "QmTest");
    }
}
