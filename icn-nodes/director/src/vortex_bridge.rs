use crate::types::{ClipEmbedding, Recipe, VideoOutput};
use pyo3::prelude::*;
use tracing::{debug, info};

/// Bridge to Python Vortex engine via PyO3 FFI
pub struct VortexBridge {
    _python: Python<'static>,
}

#[cfg_attr(feature = "stub", allow(dead_code))]
impl VortexBridge {
    /// Initialize Python interpreter and Vortex module
    pub fn initialize() -> crate::error::Result<Self> {
        info!("Initializing PyO3 Python bridge for Vortex");

        // Initialize Python interpreter (GIL acquired)
        pyo3::prepare_freethreaded_python();

        // Get Python handle
        // SAFETY: GIL is acquired via pyo3::prepare_freethreaded_python() on line 16.
        // This call initializes the Python interpreter and holds the GIL until program exit.
        let python = unsafe { Python::assume_gil_acquired() };

        debug!("Python {} initialized", python.version());

        Ok(Self { _python: python })
    }

    /// Generate video using Vortex pipeline (STUB - returns mock data)
    /// Real implementation in T014 will call actual Python functions
    pub fn generate_video(&self, recipe: &Recipe) -> crate::error::Result<VideoOutput> {
        debug!(
            "Vortex generate_video called for slot {} (STUB)",
            recipe.slot
        );

        // STUB: Return mock video output
        // Real implementation will call: vortex.pipeline.generate(recipe)
        Ok(VideoOutput {
            slot: recipe.slot,
            video_path: format!("/tmp/video_{}.mp4", recipe.slot),
            clip_embedding: self.mock_embedding(),
        })
    }

    /// Compute CLIP embedding for video (STUB - returns mock embedding)
    pub fn compute_clip_embedding(&self, video_path: &str) -> crate::error::Result<ClipEmbedding> {
        debug!("Vortex compute_clip_embedding for {} (STUB)", video_path);

        // STUB: Return mock embedding
        // Real implementation will call: vortex.clip.compute_embedding(video_path)
        Ok(self.mock_embedding())
    }

    /// Mock embedding for testing (512 dimensions, normalized)
    fn mock_embedding(&self) -> ClipEmbedding {
        // Deterministic mock embedding (ViT-B-32 has 512 dims)
        let mut emb = vec![0.0; 512];
        for (i, value) in emb.iter_mut().enumerate() {
            *value = ((i as f32) / 512.0).sin();
        }

        // Normalize
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        emb.iter_mut().for_each(|x| *x /= norm);

        emb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Case 6: PyO3 mock call
    /// Validates PyO3 integration without real Vortex
    /// Ignored by default due to Python dependency
    #[test]
    #[ignore = "Requires Python runtime"]
    fn test_vortex_bridge_init() {
        let bridge = VortexBridge::initialize().expect("Failed to init bridge");

        let recipe = Recipe {
            slot: 100,
            script: "Test script".to_string(),
            prompt: "Test prompt".to_string(),
        };

        let output = bridge.generate_video(&recipe).expect("Failed to generate");

        assert_eq!(output.slot, 100);
        assert_eq!(output.clip_embedding.len(), 512);

        // Embedding should be normalized
        let norm: f32 = output
            .clip_embedding
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
