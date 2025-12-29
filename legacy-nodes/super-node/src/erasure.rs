//! Reed-Solomon erasure coding (10+4)
//!
//! Encodes video chunks into 14 shards (10 data + 4 parity) using Reed-Solomon algorithm.
//! Any 10 of 14 shards can reconstruct the original content.

use reed_solomon_erasure::galois_8::ReedSolomon;

/// Erasure coder for Reed-Solomon (10+4) encoding/decoding
pub struct ErasureCoder {
    encoder: ReedSolomon,
    data_shards: usize,
    parity_shards: usize,
}

impl ErasureCoder {
    /// Create new erasure coder with 10 data shards + 4 parity shards
    pub fn new() -> crate::error::Result<Self> {
        let encoder = ReedSolomon::new(10, 4).map_err(|e| {
            crate::error::SuperNodeError::ErasureCoding(format!("Failed to create encoder: {}", e))
        })?;

        Ok(Self {
            encoder,
            data_shards: 10,
            parity_shards: 4,
        })
    }

    /// Encode data into 14 shards (10 data + 4 parity)
    ///
    /// # Arguments
    /// * `data` - Input video chunk bytes
    ///
    /// # Returns
    /// Vector of 14 shard byte vectors
    pub fn encode(&self, data: &[u8]) -> crate::error::Result<Vec<Vec<u8>>> {
        // Calculate shard size (round up to ensure all data fits)
        // Minimum shard size of 1 to satisfy reed-solomon-erasure library
        let shard_size = if data.is_empty() {
            1
        } else {
            data.len().div_ceil(self.data_shards)
        };

        // Split data into 10 equal-sized chunks
        let mut shards: Vec<Vec<u8>> = data
            .chunks(shard_size)
            .map(|chunk| {
                let mut shard = chunk.to_vec();
                shard.resize(shard_size, 0); // Pad with zeros if needed
                shard
            })
            .collect();

        // Ensure we have exactly 10 data shards
        while shards.len() < self.data_shards {
            shards.push(vec![0u8; shard_size]);
        }

        // Add 4 empty parity shards
        for _ in 0..self.parity_shards {
            shards.push(vec![0u8; shard_size]);
        }

        // Compute parity shards
        self.encoder.encode(&mut shards).map_err(|e| {
            crate::error::SuperNodeError::ErasureCoding(format!("Encoding failed: {}", e))
        })?;

        Ok(shards)
    }

    /// Decode shards back to original data
    ///
    /// # Arguments
    /// * `shards` - Vector of Option<Vec<u8>>, where None indicates missing shard
    /// * `original_size` - Original data size before encoding (for trimming padding)
    ///
    /// # Returns
    /// Reconstructed original data
    pub fn decode(
        &self,
        mut shards: Vec<Option<Vec<u8>>>,
        original_size: usize,
    ) -> crate::error::Result<Vec<u8>> {
        // Count available shards
        let available = shards.iter().filter(|s| s.is_some()).count();

        if available < self.data_shards {
            return Err(crate::error::SuperNodeError::ErasureCoding(format!(
                "Insufficient shards for reconstruction: have {}, need {}",
                available, self.data_shards
            )));
        }

        // Reconstruct missing shards
        self.encoder.reconstruct(&mut shards).map_err(|e| {
            crate::error::SuperNodeError::ErasureCoding(format!("Reconstruction failed: {}", e))
        })?;

        // Concatenate data shards (skip parity)
        let mut data: Vec<u8> = shards
            .into_iter()
            .take(self.data_shards)
            .flatten()
            .flatten()
            .collect();

        // Trim to original size (remove padding)
        data.truncate(original_size);

        Ok(data)
    }
}

impl Default for ErasureCoder {
    fn default() -> Self {
        Self::new().expect("Failed to create default ErasureCoder")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Case 1: Encode 50MB video chunk into 14 shards
    /// Purpose: Verify Reed-Solomon encoding produces correct shard count
    /// Contract: 50MB input → 14 shards × ~7MB each
    #[test]
    fn test_erasure_encoding_50mb() {
        let coder = ErasureCoder::new().unwrap();

        // 50MB test data
        let data = vec![42u8; 50 * 1024 * 1024];
        let original_size = data.len();

        let shards = coder.encode(&data).expect("Encoding failed");

        // Verify shard count
        assert_eq!(shards.len(), 14, "Should produce 14 shards");

        // Verify each shard is ~7MB (50MB / 10 data shards = 5MB per shard)
        let expected_shard_size = (original_size + 9) / 10;
        for (i, shard) in shards.iter().enumerate() {
            assert_eq!(
                shard.len(),
                expected_shard_size,
                "Shard {} size mismatch",
                i
            );
        }
    }

    /// Test Case 2: Decode with all 14 shards (no data loss)
    /// Purpose: Verify reconstruction with full shard set
    /// Contract: Reconstructed data matches original bit-for-bit
    #[test]
    fn test_erasure_decode_all_shards() {
        let coder = ErasureCoder::new().unwrap();

        let original_data = b"Hello, ICN Super-Node! This is test data for erasure coding.";
        let original_size = original_data.len();

        // Encode
        let shards = coder.encode(original_data).unwrap();

        // Decode with all shards
        let shards_opt: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();
        let decoded = coder.decode(shards_opt, original_size).unwrap();

        assert_eq!(decoded, original_data);
    }

    /// Test Case 3: Decode with exactly 10 shards (4 missing)
    /// Purpose: Verify reconstruction with minimum required shards
    /// Contract: Any 10 of 14 shards can reconstruct original
    #[test]
    fn test_erasure_decode_minimum_shards() {
        let coder = ErasureCoder::new().unwrap();

        let original_data = vec![123u8; 10_000];
        let original_size = original_data.len();

        // Encode
        let shards = coder.encode(&original_data).unwrap();

        // Simulate losing shards 2, 5, 11, 13 (4 shards missing)
        let mut shards_opt: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();
        shards_opt[2] = None;
        shards_opt[5] = None;
        shards_opt[11] = None;
        shards_opt[13] = None;

        // Should still reconstruct
        let decoded = coder.decode(shards_opt, original_size).unwrap();
        assert_eq!(decoded, original_data);
    }

    /// Test Case 4: Decode fails with <10 shards
    /// Purpose: Verify insufficient shards returns error
    /// Contract: Must have at least 10 shards for reconstruction
    #[test]
    fn test_erasure_decode_insufficient_shards() {
        let coder = ErasureCoder::new().unwrap();

        let original_data = vec![99u8; 1000];

        // Encode
        let shards = coder.encode(&original_data).unwrap();

        // Simulate losing 5 shards (only 9 remaining)
        let mut shards_opt: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();
        for i in 0..5 {
            shards_opt[i] = None;
        }

        // Should fail
        let result = coder.decode(shards_opt, original_data.len());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Insufficient shards"));
    }

    /// Test Case 5: Checksum verification (bit-for-bit reconstruction)
    /// Purpose: Verify decoded data matches original exactly
    /// Contract: sha256(reconstructed) == sha256(original)
    #[test]
    fn test_erasure_checksum_verification() {
        use sha2::{Digest, Sha256};

        let coder = ErasureCoder::new().unwrap();

        // Large test data with variety
        let original_data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let original_size = original_data.len();
        let original_hash = Sha256::digest(&original_data);

        // Encode
        let shards = coder.encode(&original_data).unwrap();

        // Lose 4 random shards
        let mut shards_opt: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();
        shards_opt[1] = None;
        shards_opt[4] = None;
        shards_opt[8] = None;
        shards_opt[12] = None;

        // Decode
        let decoded = coder.decode(shards_opt, original_size).unwrap();
        let decoded_hash = Sha256::digest(&decoded);

        assert_eq!(original_hash, decoded_hash, "Checksums must match");
        assert_eq!(decoded, original_data, "Data must match byte-for-byte");
    }

    /// Test Case 6: Encode empty data
    /// Purpose: Verify edge case handling
    /// Contract: Should handle gracefully
    #[test]
    fn test_erasure_encode_empty() {
        let coder = ErasureCoder::new().unwrap();
        let data = vec![];

        let shards = coder.encode(&data).unwrap();

        assert_eq!(shards.len(), 14);
        // All shards should be empty or minimal padding
        for shard in &shards {
            assert!(shard.is_empty() || shard.iter().all(|&b| b == 0));
        }
    }

    /// Test Case 7: Encode 1-byte data
    /// Purpose: Verify minimal data handling
    #[test]
    fn test_erasure_encode_single_byte() {
        let coder = ErasureCoder::new().unwrap();
        let data = vec![255u8];

        let shards = coder.encode(&data).unwrap();
        assert_eq!(shards.len(), 14);

        // Decode
        let shards_opt: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();
        let decoded = coder.decode(shards_opt, 1).unwrap();

        assert_eq!(decoded, vec![255u8]);
    }
}
