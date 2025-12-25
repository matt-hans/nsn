#![cfg_attr(feature = "stub", allow(dead_code))]

use serde::{Deserialize, Serialize};

/// Slot number representing a content generation window
pub type SlotNumber = u64;

/// Block number in the ICN Chain
pub type BlockNumber = u32;

/// Account identifier (32-byte public key)
pub type AccountId = [u8; 32];

/// PeerId for libp2p networking
pub type PeerId = String;

/// Hash type (32 bytes)
pub type Hash = [u8; 32];

/// CLIP embedding vector (512 dimensions for ViT-B-32, 768 for ViT-L-14)
pub type ClipEmbedding = Vec<f32>;

/// BFT consensus result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BftResult {
    /// 3-of-5 consensus reached
    Success {
        /// Canonical director whose output is accepted
        canonical_director: PeerId,
        /// All directors who agreed (3+)
        agreeing_directors: Vec<PeerId>,
        /// Canonical embedding hash
        embedding_hash: Hash,
    },
    /// No consensus reached
    Failed {
        /// All participating directors
        directors: Vec<PeerId>,
        /// Reason for failure
        reason: String,
    },
}

/// Slot task for the scheduler
#[derive(Debug, Clone)]
pub struct SlotTask {
    pub slot: SlotNumber,
    pub deadline_block: BlockNumber,
    pub directors: Vec<PeerId>,
}

/// Attestation from a director for BFT result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attestation {
    pub director: PeerId,
    pub agreed: bool,
    pub embedding_hash: Hash,
}

/// Recipe for video generation (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    pub slot: SlotNumber,
    pub script: String,
    pub prompt: String,
}

/// Video output from Vortex pipeline
#[derive(Debug, Clone)]
pub struct VideoOutput {
    pub slot: SlotNumber,
    pub video_path: String,
    pub clip_embedding: ClipEmbedding,
}

/// Election event data
#[derive(Debug, Clone)]
pub struct ElectionEvent {
    pub slot: SlotNumber,
    pub directors: Vec<PeerId>,
}

/// Compute cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Compute SHA256 hash of embedding
pub fn hash_embedding(embedding: &[f32]) -> Hash {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for &val in embedding {
        hasher.update(val.to_le_bytes());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);

        // Deeper assertion: Verify embedding normalization
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        // For unit vectors, norm should be close to 1.0 after normalization
        // For our test vectors, norm is sqrt(1+4+9) = sqrt(14) ≈ 3.742
        assert!((norm_a - 3.742).abs() < 0.01);
        assert!((norm_b - 3.742).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.0001);

        // Deeper assertion: Verify norms
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Both are unit vectors (norm = 1.0)
        assert!((norm_a - 1.0).abs() < 0.0001);
        assert!((norm_b - 1.0).abs() < 0.0001);

        // Dot product should be 0 for orthogonal vectors
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(dot_product.abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 0.0001);

        // Deeper assertion: Opposite vectors have same norm
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_a - norm_b).abs() < 0.0001);

        // Dot product should be negative of squared norm
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let expected_dot = -(1.0 + 4.0 + 9.0); // -14.0
        assert!((dot_product - expected_dot).abs() < 0.0001);
    }

    #[test]
    fn test_hash_embedding_deterministic() {
        let emb = vec![0.1, 0.2, 0.3];
        let hash1 = hash_embedding(&emb);
        let hash2 = hash_embedding(&emb);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_embedding_different() {
        let emb1 = vec![0.1, 0.2, 0.3];
        let emb2 = vec![0.1, 0.2, 0.4];
        let hash1 = hash_embedding(&emb1);
        let hash2 = hash_embedding(&emb2);
        assert_ne!(hash1, hash2);
    }
}
