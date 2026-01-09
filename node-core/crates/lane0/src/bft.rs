//! BFT consensus participant for Lane 0 video verification.
//!
//! Handles Byzantine Fault Tolerant consensus using CLIP embeddings to verify
//! that all directors generated semantically equivalent video content. Uses
//! cosine similarity to compare embeddings and requires 3-of-5 threshold.

use std::collections::HashMap;

use libp2p::identity::Keypair;
use libp2p::PeerId;
use parity_scale_codec::{Decode, Encode};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::time::{timeout, Duration, Instant};
use tracing::{debug, info, warn};

use nsn_p2p::{ServiceCommand, TopicCategory};

use crate::error::{BftError, BftResult};

/// Default BFT timeout in milliseconds.
pub const DEFAULT_BFT_TIMEOUT_MS: u64 = 5000;

/// Default consensus threshold (3 of 5).
pub const DEFAULT_THRESHOLD: u8 = 3;

/// Minimum cosine similarity for embeddings to be considered equivalent.
pub const SIMILARITY_THRESHOLD: f32 = 0.85;

/// Configuration for BFT participant.
#[derive(Debug, Clone)]
pub struct BftConfig {
    /// Timeout for collecting embeddings in milliseconds.
    pub timeout_ms: u64,
    /// Threshold for consensus (e.g., 3 of 5).
    pub threshold: u8,
    /// Expected number of directors.
    pub expected_directors: u8,
    /// Minimum cosine similarity for consensus.
    pub similarity_threshold: f32,
}

impl Default for BftConfig {
    fn default() -> Self {
        Self {
            timeout_ms: DEFAULT_BFT_TIMEOUT_MS,
            threshold: DEFAULT_THRESHOLD,
            expected_directors: 5,
            similarity_threshold: SIMILARITY_THRESHOLD,
        }
    }
}

/// BFT signal message for embedding exchange.
#[derive(Debug, Clone, Encode, Decode, Serialize, Deserialize)]
pub enum BftSignal {
    /// Director publishes their CLIP embedding.
    Embedding {
        /// Slot number.
        slot: u64,
        /// CLIP embedding (512 dimensions).
        embedding: Vec<f32>,
        /// Peer ID of the sender.
        signer: Vec<u8>,
        /// Ed25519 signature over slot + embedding hash.
        signature: Vec<u8>,
    },
    /// Director votes for a canonical embedding.
    Vote {
        /// Slot number.
        slot: u64,
        /// Blake3 hash of the canonical embedding.
        embedding_hash: [u8; 32],
        /// Peer ID of the voter.
        voter: Vec<u8>,
        /// Signature over the vote.
        signature: Vec<u8>,
    },
}

/// Result of BFT consensus for a slot.
#[derive(Debug, Clone)]
pub struct BftConsensusResult {
    /// Slot number.
    pub slot: u64,
    /// Blake3 hash of the agreed canonical embedding.
    pub canonical_hash: [u8; 32],
    /// Peer IDs of directors who agreed.
    pub signers: Vec<PeerId>,
    /// Whether consensus was reached.
    pub success: bool,
    /// Average similarity between embeddings.
    pub similarity: f32,
}

/// Collected embedding from a director.
#[derive(Debug, Clone)]
struct CollectedEmbedding {
    peer_id: PeerId,
    embedding: Vec<f32>,
    #[allow(dead_code)] // Kept for potential future verification needs
    signature: Vec<u8>,
}

/// BFT consensus participant for Lane 0.
pub struct BftParticipant {
    /// Node keypair for signing.
    keypair: Keypair,
    /// Configuration.
    config: BftConfig,
    /// P2P command sender.
    p2p_tx: mpsc::UnboundedSender<ServiceCommand>,
    /// Receiver for BFT signals from P2P.
    signal_rx: Option<mpsc::Receiver<Vec<u8>>>,
}

impl BftParticipant {
    /// Create a new BFT participant.
    pub fn new(
        keypair: Keypair,
        config: BftConfig,
        p2p_tx: mpsc::UnboundedSender<ServiceCommand>,
    ) -> Self {
        Self {
            keypair,
            config,
            p2p_tx,
            signal_rx: None,
        }
    }

    /// Create a BFT participant with P2P subscription for signals.
    pub fn with_subscription(
        keypair: Keypair,
        config: BftConfig,
        p2p_tx: mpsc::UnboundedSender<ServiceCommand>,
        signal_rx: mpsc::Receiver<Vec<u8>>,
    ) -> Self {
        Self {
            keypair,
            config,
            p2p_tx,
            signal_rx: Some(signal_rx),
        }
    }

    /// Get the local peer ID.
    pub fn peer_id(&self) -> PeerId {
        self.keypair.public().to_peer_id()
    }

    /// Run BFT consensus for a slot.
    ///
    /// 1. Publish our CLIP embedding
    /// 2. Collect embeddings from other directors
    /// 3. Verify all embeddings are similar enough
    /// 4. Return consensus result
    pub async fn run_consensus(
        &mut self,
        slot: u64,
        my_embedding: Vec<f32>,
        timeout_ms: u64,
    ) -> BftResult<BftConsensusResult> {
        let timeout_ms = if timeout_ms == 0 {
            self.config.timeout_ms
        } else {
            timeout_ms
        };

        info!(slot, timeout_ms, "Starting BFT consensus");

        // 1. Publish our embedding
        self.publish_embedding(slot, &my_embedding).await?;

        // 2. Collect embeddings from other directors
        let embeddings = self.collect_embeddings(slot, timeout_ms).await?;

        // Add our own embedding
        let mut all_embeddings = vec![CollectedEmbedding {
            peer_id: self.peer_id(),
            embedding: my_embedding.clone(),
            signature: vec![], // Self doesn't need signature
        }];
        all_embeddings.extend(embeddings);

        // 3. Verify consensus
        let result = self.verify_consensus(slot, &all_embeddings)?;

        if result.success {
            info!(
                slot,
                signers = result.signers.len(),
                similarity = result.similarity,
                "BFT consensus reached"
            );
        } else {
            warn!(
                slot,
                collected = all_embeddings.len(),
                similarity = result.similarity,
                "BFT consensus failed"
            );
        }

        Ok(result)
    }

    /// Publish our CLIP embedding to the BFT signals topic.
    async fn publish_embedding(&self, slot: u64, embedding: &[f32]) -> BftResult<()> {
        // Create signing payload: slot || embedding_hash
        let embedding_hash = blake3::hash(bytemuck::cast_slice(embedding));
        let mut sign_payload = slot.to_le_bytes().to_vec();
        sign_payload.extend_from_slice(embedding_hash.as_bytes());

        // Sign the payload
        let signature = self
            .keypair
            .sign(&sign_payload)
            .map_err(|_| BftError::PublishFailed("signing failed".to_string()))?;

        let signal = BftSignal::Embedding {
            slot,
            embedding: embedding.to_vec(),
            signer: self.keypair.public().encode_protobuf(),
            signature,
        };

        let encoded = signal.encode();

        // Publish via P2P
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.p2p_tx
            .send(ServiceCommand::Publish(TopicCategory::BftSignals, encoded, tx))
            .map_err(|e| BftError::PublishFailed(e.to_string()))?;

        // Wait for ack
        match timeout(Duration::from_millis(1000), rx).await {
            Ok(Ok(Ok(_))) => {
                debug!(slot, "Embedding published successfully");
                Ok(())
            }
            Ok(Ok(Err(e))) => Err(BftError::PublishFailed(e.to_string())),
            Ok(Err(_)) => Err(BftError::PublishFailed("channel dropped".to_string())),
            Err(_) => Err(BftError::PublishFailed("publish timeout".to_string())),
        }
    }

    /// Collect embeddings from other directors.
    async fn collect_embeddings(
        &mut self,
        slot: u64,
        timeout_ms: u64,
    ) -> BftResult<Vec<CollectedEmbedding>> {
        let mut collected = HashMap::<PeerId, CollectedEmbedding>::new();
        let start = Instant::now();
        let deadline = Duration::from_millis(timeout_ms);

        // Need threshold - 1 other directors (we have our own)
        let needed = (self.config.threshold.saturating_sub(1)) as usize;
        let my_peer_id = self.peer_id();

        let rx = match self.signal_rx.as_mut() {
            Some(rx) => rx,
            None => {
                // No P2P subscription, can't collect embeddings
                return Err(BftError::Timeout {
                    timeout_ms,
                    collected: 0,
                    expected: needed,
                });
            }
        };

        while start.elapsed() < deadline && collected.len() < needed {
            let remaining = deadline.saturating_sub(start.elapsed());

            match timeout(remaining, rx.recv()).await {
                Ok(Some(data)) => {
                    match Self::parse_and_validate_signal_static(slot, &data) {
                        Ok(embedding) => {
                            if embedding.peer_id != my_peer_id {
                                collected.insert(embedding.peer_id, embedding);
                                debug!(
                                    slot,
                                    collected = collected.len(),
                                    needed,
                                    "Received embedding"
                                );
                            }
                        }
                        Err(e) => {
                            debug!(slot, error = %e, "Invalid BFT signal received");
                        }
                    }
                }
                Ok(None) => break, // Channel closed
                Err(_) => break,   // Timeout
            }
        }

        if collected.len() < needed {
            return Err(BftError::Timeout {
                timeout_ms,
                collected: collected.len(),
                expected: needed,
            });
        }

        Ok(collected.into_values().collect())
    }

    /// Parse and validate a BFT signal from P2P.
    #[allow(dead_code)] // Kept for API consistency with non-static version
    fn parse_and_validate_signal(
        &self,
        expected_slot: u64,
        data: &[u8],
    ) -> BftResult<CollectedEmbedding> {
        Self::parse_and_validate_signal_static(expected_slot, data)
    }

    /// Parse and validate a BFT signal from P2P (static version).
    fn parse_and_validate_signal_static(
        expected_slot: u64,
        data: &[u8],
    ) -> BftResult<CollectedEmbedding> {
        let signal = BftSignal::decode(&mut &data[..])
            .map_err(|e| BftError::InvalidEmbedding {
                peer: "unknown".to_string(),
                reason: format!("decode failed: {}", e),
            })?;

        match signal {
            BftSignal::Embedding {
                slot,
                embedding,
                signer,
                signature,
            } => {
                // Check slot
                if slot != expected_slot {
                    return Err(BftError::InvalidEmbedding {
                        peer: "unknown".to_string(),
                        reason: format!("wrong slot: expected {}, got {}", expected_slot, slot),
                    });
                }

                // Check embedding dimensions
                if embedding.len() != 512 {
                    return Err(BftError::InvalidEmbedding {
                        peer: "unknown".to_string(),
                        reason: format!("wrong embedding size: {}", embedding.len()),
                    });
                }

                // Decode signer public key
                let public_key = libp2p::identity::PublicKey::try_decode_protobuf(&signer)
                    .map_err(|_| BftError::InvalidSignature {
                        peer: "unknown".to_string(),
                    })?;

                let peer_id = public_key.to_peer_id();

                // Verify signature
                let embedding_hash = blake3::hash(bytemuck::cast_slice(&embedding));
                let mut sign_payload = slot.to_le_bytes().to_vec();
                sign_payload.extend_from_slice(embedding_hash.as_bytes());

                if !public_key.verify(&sign_payload, &signature) {
                    return Err(BftError::InvalidSignature {
                        peer: peer_id.to_string(),
                    });
                }

                Ok(CollectedEmbedding {
                    peer_id,
                    embedding,
                    signature,
                })
            }
            BftSignal::Vote { .. } => Err(BftError::InvalidEmbedding {
                peer: "unknown".to_string(),
                reason: "unexpected vote message".to_string(),
            }),
        }
    }

    /// Verify consensus among collected embeddings.
    fn verify_consensus(
        &self,
        slot: u64,
        embeddings: &[CollectedEmbedding],
    ) -> BftResult<BftConsensusResult> {
        if embeddings.len() < self.config.threshold as usize {
            return Err(BftError::InsufficientEmbeddings {
                got: embeddings.len(),
                need: self.config.threshold as usize,
            });
        }

        // Calculate pairwise cosine similarities
        let mut similarities = Vec::new();
        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                let sim = cosine_similarity(&embeddings[i].embedding, &embeddings[j].embedding);
                similarities.push(sim);
            }
        }

        // Average similarity
        let avg_similarity = if similarities.is_empty() {
            1.0
        } else {
            similarities.iter().sum::<f32>() / similarities.len() as f32
        };

        // Check if all are above threshold
        let all_similar = similarities
            .iter()
            .all(|&s| s >= self.config.similarity_threshold);

        // Compute canonical hash (hash of first embedding as reference)
        let canonical_hash = if !embeddings.is_empty() {
            *blake3::hash(bytemuck::cast_slice(&embeddings[0].embedding)).as_bytes()
        } else {
            [0u8; 32]
        };

        let signers: Vec<PeerId> = embeddings.iter().map(|e| e.peer_id).collect();

        Ok(BftConsensusResult {
            slot,
            canonical_hash,
            signers,
            success: all_similar && embeddings.len() >= self.config.threshold as usize,
            similarity: avg_similarity,
        })
    }
}

/// Calculate cosine similarity between two embeddings.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_bft_config_default() {
        let config = BftConfig::default();
        assert_eq!(config.timeout_ms, 5000);
        assert_eq!(config.threshold, 3);
        assert_eq!(config.expected_directors, 5);
        assert_eq!(config.similarity_threshold, 0.85);
    }

    #[test]
    fn test_bft_signal_encoding() {
        let signal = BftSignal::Embedding {
            slot: 42,
            embedding: vec![0.1, 0.2, 0.3],
            signer: vec![1, 2, 3],
            signature: vec![4, 5, 6],
        };

        let encoded = signal.encode();
        let decoded = BftSignal::decode(&mut &encoded[..]).unwrap();

        match decoded {
            BftSignal::Embedding {
                slot, embedding, ..
            } => {
                assert_eq!(slot, 42);
                assert_eq!(embedding.len(), 3);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_bft_consensus_result_fields() {
        let result = BftConsensusResult {
            slot: 100,
            canonical_hash: [0u8; 32],
            signers: vec![],
            success: true,
            similarity: 0.95,
        };

        assert_eq!(result.slot, 100);
        assert!(result.success);
        assert_eq!(result.similarity, 0.95);
    }
}
