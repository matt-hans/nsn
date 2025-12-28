//! Merkle proof verification for cached shard content
//!
//! This module implements Merkle proof verification to ensure the integrity
//! of shards fetched from Super-Nodes before caching them. This prevents
//! cache poisoning attacks where malicious actors could inject fake content.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use crate::error::Result;

/// Merkle proof for a single shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    /// Content identifier (CID) for the full video
    pub cid: String,
    /// Shard index (0-13 for 10+4 Reed-Solomon encoding)
    pub shard_index: usize,
    /// Hash of the shard data
    pub shard_hash: String,
    /// Merkle path from leaf to root (sibling hashes)
    pub proof_path: Vec<String>,
    /// Expected root hash for the full content
    pub expected_root: String,
}

/// Verified shard with Merkle proof
#[derive(Debug, Clone)]
pub struct VerifiedShard {
    /// Shard data bytes
    pub data: Vec<u8>,
    /// Merkle proof used for verification
    pub proof: MerkleProof,
}

/// Merkle proof verifier
pub struct MerkleVerifier;

impl MerkleVerifier {
    /// Verify a Merkle proof for shard data
    ///
    /// # Arguments
    /// * `shard_data` - The raw shard bytes to verify
    /// * `proof` - The Merkle proof for this shard
    ///
    /// # Returns
    /// Ok(()) if verification succeeds, Err otherwise
    pub fn verify_shard(shard_data: &[u8], proof: &MerkleProof) -> Result<()> {
        // Step 1: Compute hash of shard data
        let shard_hash = Self::hash_data(shard_data);
        let shard_hash_hex = hex::encode(shard_hash);

        // Step 2: Verify shard hash matches proof
        if shard_hash_hex != proof.shard_hash {
            return Err(crate::error::RelayError::MerkleProofVerificationFailed(
                format!(
                    "shard hash mismatch: computed={}, expected={}",
                    shard_hash_hex, proof.shard_hash
                ),
            ));
        }

        // Step 3: Reconstruct Merkle root using proof path
        let computed_root =
            Self::compute_merkle_root(&shard_hash, &proof.proof_path, proof.shard_index);

        // Step 4: Verify computed root matches expected root
        if computed_root != proof.expected_root {
            warn!(
                "Merkle root mismatch: computed={}, expected={}",
                computed_root, proof.expected_root
            );
            return Err(crate::error::RelayError::MerkleProofVerificationFailed(
                format!(
                    "root hash mismatch for shard {} of CID {}",
                    proof.shard_index, proof.cid
                ),
            ));
        }

        debug!(
            "Merkle proof verified: CID={}, shard={}",
            proof.cid, proof.shard_index
        );

        Ok(())
    }

    /// Verify a batch of Merkle proofs
    ///
    /// More efficient for verifying multiple shards at once
    pub fn verify_batch(shards: &[(Vec<u8>, MerkleProof)]) -> Result<()> {
        for (data, proof) in shards {
            Self::verify_shard(data, proof)?;
        }
        Ok(())
    }

    /// Hash data using SHA-256
    fn hash_data(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut array = [0u8; 32];
        array.copy_from_slice(&result);
        array
    }

    /// Compute Merkle root from leaf hash and proof path
    ///
    /// This reconstructs the Merkle root by hashing up the tree
    /// using the provided proof path (sibling hashes at each level)
    fn compute_merkle_root(leaf_hash: &[u8], proof_path: &[String], leaf_index: usize) -> String {
        // Convert leaf hash to fixed-size array
        let mut current_hash = [0u8; 32];
        current_hash.copy_from_slice(leaf_hash);
        let mut index = leaf_index;

        for sibling_hex in proof_path {
            // Decode sibling hash from hex
            let sibling_hash = match hex::decode(sibling_hex) {
                Ok(hash) if hash.len() == 32 => {
                    let mut array = [0u8; 32];
                    array.copy_from_slice(&hash);
                    array
                }
                _ => {
                    warn!("Invalid sibling hash in Merkle proof path");
                    // Continue with best-effort verification
                    [0u8; 32]
                }
            };

            // Hash parent node: H(left || right) or H(right || left)
            // Order depends on whether the node is a left or right child
            let mut hasher = Sha256::new();

            if index & 1 == 0 {
                // Current node is left child
                hasher.update(current_hash);
                hasher.update(sibling_hash);
            } else {
                // Current node is right child
                hasher.update(sibling_hash);
                hasher.update(current_hash);
            }

            let result = hasher.finalize();
            current_hash.copy_from_slice(&result);
            index /= 2;
        }

        hex::encode(current_hash)
    }
}

/// Helper function to create a Merkle proof for testing
#[cfg(test)]
#[cfg(feature = "dev-mode")]
pub fn create_test_proof(cid: &str, shard_index: usize, shard_data: &[u8]) -> MerkleProof {
    let shard_hash = hex::encode(Sha256::digest(shard_data));
    let expected_root = format!("root_{}", cid); // Simplified for testing

    MerkleProof {
        cid: cid.to_string(),
        shard_index,
        shard_hash,
        proof_path: vec![],
        expected_root,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_data() {
        let data = b"test shard data";
        let hash = MerkleVerifier::hash_data(data);

        // SHA-256 produces 32 bytes
        assert_eq!(hash.len(), 32);

        // Same input should produce same hash
        let hash2 = MerkleVerifier::hash_data(data);
        assert_eq!(hash, hash2);

        // Different input should produce different hash
        let hash3 = MerkleVerifier::hash_data(b"different data");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_compute_merkle_root_simple() {
        // Simple Merkle tree with 2 leaves
        //     root
        //    /    \
        // leaf0  leaf1

        let leaf0_hash = [0u8; 32];
        let leaf1_hash = [1u8; 32];

        // Proof path for leaf0 (left child) should contain leaf1's hash
        let proof_path = vec![hex::encode(leaf1_hash)];

        let computed_root = MerkleVerifier::compute_merkle_root(&leaf0_hash, &proof_path, 0);

        // Root should be hash of (leaf0 || leaf1)
        let mut hasher = Sha256::new();
        hasher.update(&leaf0_hash);
        hasher.update(&leaf1_hash);
        let expected_root = hasher.finalize();

        assert_eq!(computed_root, hex::encode(expected_root));
    }

    #[test]
    fn test_verify_shard_with_valid_proof() {
        let shard_data = b"test shard content";
        let shard_hash = hex::encode(Sha256::digest(shard_data));

        // Create a valid proof (simplified - empty proof path means shard IS the root)
        let proof = MerkleProof {
            cid: "test_cid".to_string(),
            shard_index: 0,
            shard_hash: shard_hash.clone(),
            proof_path: vec![],
            expected_root: shard_hash,
        };

        let result = MerkleVerifier::verify_shard(shard_data, &proof);
        assert!(result.is_ok(), "Should verify valid proof");
    }

    #[test]
    fn test_verify_shard_with_invalid_hash() {
        let shard_data = b"test shard content";
        let wrong_hash = hex::encode([1u8; 32]);

        let proof = MerkleProof {
            cid: "test_cid".to_string(),
            shard_index: 0,
            shard_hash: wrong_hash.clone(),
            proof_path: vec![],
            expected_root: wrong_hash,
        };

        let result = MerkleVerifier::verify_shard(shard_data, &proof);
        assert!(result.is_err(), "Should reject proof with invalid hash");
    }

    #[test]
    fn test_verify_shard_with_invalid_root() {
        let shard_data = b"test shard content";
        let shard_hash = hex::encode(Sha256::digest(shard_data));
        let wrong_root = hex::encode([2u8; 32]);

        let proof = MerkleProof {
            cid: "test_cid".to_string(),
            shard_index: 0,
            shard_hash,
            proof_path: vec![],
            expected_root: wrong_root,
        };

        let result = MerkleVerifier::verify_shard(shard_data, &proof);
        assert!(result.is_err(), "Should reject proof with invalid root");
    }
}
