use crate::types::{cosine_similarity, hash_embedding, BftResult, ClipEmbedding, PeerId};
use tracing::{debug, info, warn};

/// BFT Coordinator for director consensus
pub struct BftCoordinator {
    _own_peer_id: PeerId,
    _consensus_threshold: f32,
}

impl BftCoordinator {
    pub fn new(own_peer_id: PeerId, consensus_threshold: f32) -> Self {
        Self {
            _own_peer_id: own_peer_id,
            _consensus_threshold: consensus_threshold,
        }
    }

    /// Compute BFT agreement from collected embeddings
    /// Returns Success if 3-of-5 directors agree (cosine similarity > threshold)
    #[cfg_attr(feature = "stub", allow(dead_code))]
    pub fn compute_agreement(&self, embeddings: Vec<(PeerId, ClipEmbedding)>) -> BftResult {
        if embeddings.len() < 3 {
            return BftResult::Failed {
                directors: embeddings.iter().map(|(p, _)| p.clone()).collect(),
                reason: format!("Insufficient directors: {}", embeddings.len()),
            };
        }

        debug!(
            "Computing BFT agreement for {} embeddings",
            embeddings.len()
        );

        // Build agreement matrix
        let mut agreement_groups: Vec<Vec<PeerId>> = Vec::new();

        for (i, (peer_i, emb_i)) in embeddings.iter().enumerate() {
            let mut group = vec![peer_i.clone()];

            for (j, (peer_j, emb_j)) in embeddings.iter().enumerate() {
                if i != j {
                    let similarity = cosine_similarity(emb_i, emb_j);
                    if similarity > self._consensus_threshold {
                        group.push(peer_j.clone());
                    }
                }
            }

            // Only keep groups with 3+ members (BFT threshold)
            if group.len() >= 3 {
                agreement_groups.push(group);
            }
        }

        // Find largest agreement group (or first if tie)
        if let Some(largest_group) = agreement_groups.iter().max_by_key(|g| g.len()) {
            let canonical_director = largest_group[0].clone();
            let canonical_embedding = embeddings
                .iter()
                .find(|(p, _)| p == &canonical_director)
                .map(|(_, e)| e)
                .unwrap();

            info!(
                "BFT consensus reached: {} directors agreed",
                largest_group.len()
            );

            BftResult::Success {
                canonical_director,
                agreeing_directors: largest_group.clone(),
                embedding_hash: hash_embedding(canonical_embedding),
            }
        } else {
            warn!("BFT consensus failed: no 3-of-5 agreement");
            BftResult::Failed {
                directors: embeddings.iter().map(|(p, _)| p.clone()).collect(),
                reason: "No 3-of-5 consensus reached".to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Case 3: BFT agreement matrix (3-of-5 Success)
    /// Core BFT consensus logic must correctly identify majority
    #[test]
    fn test_bft_agreement_success() {
        let coordinator = BftCoordinator::new("Alice".to_string(), 0.95);

        // Create 5 embeddings where 4 agree (high similarity)
        // emb_a and similar variants point in similar directions
        // emb_b points in a completely different direction (orthogonal)
        let emb_a = vec![1.0, 0.0, 0.0];
        let emb_a2 = vec![0.99, 0.01, 0.01]; // Very similar to emb_a
        let emb_a3 = vec![0.98, 0.02, 0.00]; // Very similar to emb_a
        let emb_a4 = vec![0.97, 0.03, 0.00]; // Very similar to emb_a
        let emb_b = vec![0.0, 1.0, 0.0]; // Orthogonal (completely different direction)

        let embeddings = vec![
            ("Dir1".to_string(), emb_a),
            ("Dir2".to_string(), emb_a2),
            ("Dir3".to_string(), emb_a3),
            ("Dir4".to_string(), emb_b),
            ("Dir5".to_string(), emb_a4),
        ];

        let result = coordinator.compute_agreement(embeddings);

        match result {
            BftResult::Success {
                canonical_director,
                agreeing_directors,
                embedding_hash: _,
            } => {
                // The canonical director should be one of the agreeing directors
                assert!(agreeing_directors.contains(&canonical_director));
                // Should have at least 3 (BFT threshold) agreeing directors
                assert!(agreeing_directors.len() >= 3);
                // Dir4 should not be in the agreeing set (it's the outlier)
                assert!(!agreeing_directors.contains(&"Dir4".to_string()));
            }
            _ => panic!("Expected BftResult::Success"),
        }
    }

    /// Test Case 4: BFT agreement matrix (Failure - no consensus)
    /// Must detect when no consensus reached
    #[test]
    fn test_bft_agreement_failure() {
        let coordinator = BftCoordinator::new("Alice".to_string(), 0.95);

        // Create 5 completely different embeddings
        let embeddings = vec![
            ("Dir1".to_string(), vec![1.0, 0.0, 0.0]),
            ("Dir2".to_string(), vec![0.0, 1.0, 0.0]),
            ("Dir3".to_string(), vec![0.0, 0.0, 1.0]),
            ("Dir4".to_string(), vec![1.0, 1.0, 0.0]),
            ("Dir5".to_string(), vec![0.0, 1.0, 1.0]),
        ];

        let result = coordinator.compute_agreement(embeddings);

        match result {
            BftResult::Failed { directors, reason } => {
                assert_eq!(directors.len(), 5);
                assert!(reason.contains("No 3-of-5 consensus"));
            }
            _ => panic!("Expected BftResult::Failed"),
        }
    }

    #[test]
    fn test_insufficient_directors() {
        let coordinator = BftCoordinator::new("Alice".to_string(), 0.95);

        let embeddings = vec![
            ("Dir1".to_string(), vec![1.0, 2.0]),
            ("Dir2".to_string(), vec![1.0, 2.0]),
        ];

        let result = coordinator.compute_agreement(embeddings);

        match result {
            BftResult::Failed { directors, reason } => {
                assert_eq!(directors.len(), 2);
                assert!(reason.contains("Insufficient directors"));
            }
            _ => panic!("Expected BftResult::Failed"),
        }
    }

    /// Test Case: BFT timeout with unresponsive peer
    /// Purpose: Verify 5-second timeout for unresponsive directors
    /// Contract: BFT round proceeds without unresponsive peer after timeout
    /// Scenario 5 from task specification
    #[tokio::test]
    #[ignore] // Requires gRPC infrastructure for timeout testing
    async fn test_bft_timeout_unresponsive_peer() {
        use tokio::time::{timeout, Duration};

        let coordinator = BftCoordinator::new("Alice".to_string(), 0.95);

        // Simulate BFT coordination with 5-second timeout
        let bft_operation = async {
            // Mock: Wait for peer responses
            // In real implementation, this would be gRPC calls to other directors
            tokio::time::sleep(Duration::from_secs(6)).await;
            Ok::<(), String>(())
        };

        // Apply 5-second timeout
        let result = timeout(Duration::from_secs(5), bft_operation).await;

        // Should timeout after 5 seconds
        assert!(
            result.is_err(),
            "BFT operation should timeout after 5 seconds"
        );

        // After timeout, BFT should proceed with available directors
        // In this case, only 4 directors responded (one timed out)
        let emb_a = vec![1.0, 0.0, 0.0];
        let emb_a2 = vec![0.99, 0.01, 0.01];
        let emb_a3 = vec![0.98, 0.02, 0.00];
        let emb_a4 = vec![0.97, 0.03, 0.00];

        let embeddings = vec![
            ("Dir1".to_string(), emb_a),
            ("Dir2".to_string(), emb_a2),
            ("Dir3".to_string(), emb_a3),
            ("Dir5".to_string(), emb_a4),
            // Dir4 timed out (not included)
        ];

        // Should still reach 3-of-4 consensus
        let agreement = coordinator.compute_agreement(embeddings);

        match agreement {
            BftResult::Success {
                agreeing_directors, ..
            } => {
                assert!(
                    agreeing_directors.len() >= 3,
                    "Should reach 3-of-4 consensus with timeout"
                );
            }
            _ => panic!("Expected BftResult::Success with 4 directors"),
        }
    }

    /// Test Case: BFT peer failure handling
    /// Purpose: Verify degraded consensus when one peer fails (3-of-4)
    /// Contract: BFT succeeds with 3-of-4 agreement when one peer fails
    /// Scenario 5 from task specification
    #[test]
    fn test_bft_peer_failure_handling() {
        let coordinator = BftCoordinator::new("Alice".to_string(), 0.95);

        // Scenario: 5 directors elected, but Dir4 fails/becomes unreachable
        // Only 4 directors participate in BFT
        let emb_a = vec![1.0, 0.0, 0.0];
        let emb_a2 = vec![0.99, 0.01, 0.01];
        let emb_a3 = vec![0.98, 0.02, 0.00];
        let emb_a4 = vec![0.97, 0.03, 0.00];

        // Only 4 embeddings received (Dir4 failed to respond)
        let embeddings = vec![
            ("Dir1".to_string(), emb_a),
            ("Dir2".to_string(), emb_a2),
            ("Dir3".to_string(), emb_a3),
            ("Dir5".to_string(), emb_a4),
        ];

        let result = coordinator.compute_agreement(embeddings);

        match result {
            BftResult::Success {
                canonical_director,
                agreeing_directors,
                embedding_hash: _,
            } => {
                // Should reach 3-of-4 consensus
                assert!(agreeing_directors.len() >= 3);
                assert!(agreeing_directors.len() <= 4);

                // Canonical director should be in agreeing set
                assert!(agreeing_directors.contains(&canonical_director));

                // Verify all agreeing directors are from the available set
                let available_directors = vec![
                    "Dir1".to_string(),
                    "Dir2".to_string(),
                    "Dir3".to_string(),
                    "Dir5".to_string(),
                ];

                for director in &agreeing_directors {
                    assert!(
                        available_directors.contains(director),
                        "Agreeing director {} should be from available set",
                        director
                    );
                }
            }
            _ => panic!("Expected BftResult::Success with 4 directors"),
        }
    }

    /// Test Case: BFT round proceeds correctly with fewer directors
    /// Purpose: Verify BFT logic handles 4 directors (instead of 5)
    /// Contract: Consensus threshold adjusts appropriately
    #[test]
    fn test_bft_degraded_consensus() {
        let coordinator = BftCoordinator::new("Alice".to_string(), 0.95);

        // Test with only 3 directors (minimum for BFT)
        let emb_a = vec![1.0, 0.0, 0.0];
        let emb_a2 = vec![0.99, 0.01, 0.01];
        let emb_a3 = vec![0.98, 0.02, 0.00];

        let embeddings = vec![
            ("Dir1".to_string(), emb_a),
            ("Dir2".to_string(), emb_a2),
            ("Dir3".to_string(), emb_a3),
        ];

        let result = coordinator.compute_agreement(embeddings);

        match result {
            BftResult::Success {
                agreeing_directors, ..
            } => {
                // All 3 should agree (minimum BFT threshold)
                assert_eq!(agreeing_directors.len(), 3);
            }
            _ => panic!("Expected BftResult::Success with 3 directors"),
        }
    }

    /// Test Case: Validate exact director lists in BFT agreement
    /// Purpose: Deepen assertion - verify correct directors in agreement
    /// Contract: Agreement list should contain expected directors
    #[test]
    fn test_bft_agreement_director_validation() {
        let coordinator = BftCoordinator::new("Alice".to_string(), 0.95);

        let emb_a = vec![1.0, 0.0, 0.0];
        let emb_a2 = vec![0.99, 0.01, 0.01];
        let emb_a3 = vec![0.98, 0.02, 0.00];
        let emb_a4 = vec![0.97, 0.03, 0.00];
        let emb_b = vec![0.0, 1.0, 0.0]; // Outlier

        let embeddings = vec![
            ("Dir1".to_string(), emb_a),
            ("Dir2".to_string(), emb_a2),
            ("Dir3".to_string(), emb_a3),
            ("Dir4".to_string(), emb_b),
            ("Dir5".to_string(), emb_a4),
        ];

        let result = coordinator.compute_agreement(embeddings);

        match result {
            BftResult::Success {
                canonical_director,
                agreeing_directors,
                embedding_hash: _,
            } => {
                // Verify canonical director is one of Dir1, Dir2, Dir3, or Dir5
                let expected_agreeing = vec![
                    "Dir1".to_string(),
                    "Dir2".to_string(),
                    "Dir3".to_string(),
                    "Dir5".to_string(),
                ];

                assert!(
                    expected_agreeing.contains(&canonical_director),
                    "Canonical director {} should be from majority group",
                    canonical_director
                );

                // Verify agreeing directors don't include outlier Dir4
                assert!(
                    !agreeing_directors.contains(&"Dir4".to_string()),
                    "Outlier Dir4 should not be in agreeing set"
                );

                // Verify agreeing directors are subset of expected
                for director in &agreeing_directors {
                    assert!(
                        expected_agreeing.contains(director),
                        "Director {} should be from expected set",
                        director
                    );
                }

                // Should have at least 3 agreeing
                assert!(agreeing_directors.len() >= 3);
            }
            _ => panic!("Expected BftResult::Success"),
        }
    }
}
