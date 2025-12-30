//! GossipSub peer scoring configuration
//!
//! Implements topic-based peer scoring with on-chain reputation integration.
//! Peer scores determine mesh membership, message propagation, and graylist enforcement.

use super::reputation_oracle::ReputationOracle;
use super::topics::TopicCategory;
use libp2p::gossipsub::{PeerScoreParams, PeerScoreThresholds, TopicHash, TopicScoreParams};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Gossip threshold - below this, no IHAVE/IWANT exchange
pub const GOSSIP_THRESHOLD: f64 = -10.0;

/// Publish threshold - below this, no message publishing accepted
pub const PUBLISH_THRESHOLD: f64 = -50.0;

/// Graylist threshold - below this, all messages from peer ignored
pub const GRAYLIST_THRESHOLD: f64 = -100.0;

/// Accept PX threshold - minimum score to accept PX peer exchange
pub const ACCEPT_PX_THRESHOLD: f64 = 0.0;

/// Opportunistic graft threshold - minimum score for opportunistic grafting
pub const OPPORTUNISTIC_GRAFT_THRESHOLD: f64 = 5.0;

/// Invalid message penalty (per message)
pub const INVALID_MESSAGE_PENALTY: f64 = -10.0;

/// Invalid message penalty for BFT signals (critical topic)
pub const BFT_INVALID_MESSAGE_PENALTY: f64 = -20.0;

/// Build peer score parameters with NSN topic configuration
///
/// # Arguments
/// * `reputation_oracle` - Oracle for on-chain reputation integration
///
/// # Returns
/// Tuple of (PeerScoreParams, PeerScoreThresholds)
pub fn build_peer_score_params(
    _reputation_oracle: Arc<ReputationOracle>,
) -> (PeerScoreParams, PeerScoreThresholds) {
    let topics = build_topic_score_params();
    let thresholds = build_peer_score_thresholds();

    let params = PeerScoreParams {
        topics,
        app_specific_weight: 1.0,
        ..Default::default()
    };

    (params, thresholds)
}

/// Build topic-specific score parameters for all NSN topics
fn build_topic_score_params() -> HashMap<TopicHash, TopicScoreParams> {
    let mut topics = HashMap::new();

    for category in TopicCategory::all() {
        let topic_hash = category.to_topic().hash();
        let params = build_topic_params(&category);
        topics.insert(topic_hash, params);
    }

    topics
}

/// Build score parameters for a specific topic category
fn build_topic_params(category: &TopicCategory) -> TopicScoreParams {
    let weight = category.weight();

    // Invalid message penalty varies by topic criticality
    let invalid_penalty = match category {
        TopicCategory::BftSignals => BFT_INVALID_MESSAGE_PENALTY,
        TopicCategory::Challenges => -15.0,
        _ => INVALID_MESSAGE_PENALTY,
    };

    TopicScoreParams {
        // Topic weight in overall peer score
        topic_weight: weight,

        // First message deliveries (rewards first delivery of unique messages)
        time_in_mesh_weight: 0.01,
        time_in_mesh_quantum: Duration::from_secs(1),
        time_in_mesh_cap: 3600.0, // 1 hour

        // First message deliveries
        first_message_deliveries_weight: match category {
            TopicCategory::VideoChunks => 1.0,
            TopicCategory::BftSignals => 2.0,
            _ => 0.5,
        },
        first_message_deliveries_decay: 0.9,
        first_message_deliveries_cap: 100.0,

        // Mesh message deliveries (penalty for not delivering - must be negative or 0)
        // This penalizes mesh peers that don't forward messages to us
        mesh_message_deliveries_weight: match category {
            TopicCategory::VideoChunks => -0.5,
            TopicCategory::BftSignals => -1.0,
            _ => 0.0, // disabled for other topics
        },
        mesh_message_deliveries_decay: 0.95,
        mesh_message_deliveries_cap: 50.0,
        mesh_message_deliveries_threshold: 5.0,
        mesh_message_deliveries_window: Duration::from_secs(2),
        mesh_message_deliveries_activation: Duration::from_secs(10),

        // Mesh failure penalty (penalizes peers that don't forward in mesh)
        mesh_failure_penalty_weight: -0.5,
        mesh_failure_penalty_decay: 0.95,

        // Invalid message deliveries (heavy penalty)
        invalid_message_deliveries_weight: invalid_penalty,
        invalid_message_deliveries_decay: 0.5,
    }
}

/// Build peer score thresholds for mesh membership and graylist enforcement
fn build_peer_score_thresholds() -> PeerScoreThresholds {
    PeerScoreThresholds {
        gossip_threshold: GOSSIP_THRESHOLD,
        publish_threshold: PUBLISH_THRESHOLD,
        graylist_threshold: GRAYLIST_THRESHOLD,
        accept_px_threshold: ACCEPT_PX_THRESHOLD,
        opportunistic_graft_threshold: OPPORTUNISTIC_GRAFT_THRESHOLD,
    }
}

/// Application-specific score function that integrates on-chain reputation
///
/// This is called by GossipSub to add custom scoring logic beyond topic scores.
pub async fn compute_app_specific_score(
    peer_id: &libp2p::PeerId,
    reputation_oracle: &ReputationOracle,
) -> f64 {
    // Get normalized GossipSub score from on-chain reputation (0-50)
    reputation_oracle.get_gossipsub_score(peer_id).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;
    use libp2p::PeerId;

    #[test]
    fn test_build_topic_score_params() {
        let topics = build_topic_score_params();

        // Should have 6 topics
        assert_eq!(topics.len(), 6);

        // Verify BFT signals has highest invalid message penalty
        let bft_topic = TopicCategory::BftSignals.to_topic().hash();
        let bft_params = topics.get(&bft_topic).expect("BFT topic should exist");
        assert_eq!(
            bft_params.invalid_message_deliveries_weight,
            BFT_INVALID_MESSAGE_PENALTY
        );
    }

    #[test]
    fn test_topic_params_weights() {
        let recipes_params = build_topic_params(&TopicCategory::Recipes);
        let video_params = build_topic_params(&TopicCategory::VideoChunks);
        let bft_params = build_topic_params(&TopicCategory::BftSignals);
        let challenges_params = build_topic_params(&TopicCategory::Challenges);

        // Verify topic weights match TopicCategory weights
        assert_eq!(recipes_params.topic_weight, 1.0);
        assert_eq!(video_params.topic_weight, 2.0);
        assert_eq!(bft_params.topic_weight, 3.0);
        assert_eq!(challenges_params.topic_weight, 2.5);
    }

    #[test]
    fn test_invalid_message_penalties() {
        let bft_params = build_topic_params(&TopicCategory::BftSignals);
        let challenges_params = build_topic_params(&TopicCategory::Challenges);
        let recipes_params = build_topic_params(&TopicCategory::Recipes);

        // BFT should have harshest penalty
        assert_eq!(bft_params.invalid_message_deliveries_weight, -20.0);

        // Challenges should have medium penalty
        assert_eq!(challenges_params.invalid_message_deliveries_weight, -15.0);

        // Recipes should have standard penalty
        assert_eq!(recipes_params.invalid_message_deliveries_weight, -10.0);
    }

    #[test]
    fn test_peer_score_thresholds() {
        let thresholds = build_peer_score_thresholds();

        assert_eq!(thresholds.gossip_threshold, -10.0);
        assert_eq!(thresholds.publish_threshold, -50.0);
        assert_eq!(thresholds.graylist_threshold, -100.0);
        assert_eq!(thresholds.accept_px_threshold, 0.0);
        assert_eq!(thresholds.opportunistic_graft_threshold, 5.0);
    }

    #[tokio::test]
    async fn test_app_specific_score_integration() {
        let oracle = Arc::new(ReputationOracle::new("ws://localhost:9944".to_string()));

        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        // Set high reputation
        oracle.set_reputation(peer_id, 1000).await;

        let score = compute_app_specific_score(&peer_id, &oracle).await;

        // Should be normalized to max GossipSub bonus (50.0)
        assert!((score - 50.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_app_specific_score_low_reputation() {
        let oracle = Arc::new(ReputationOracle::new("ws://localhost:9944".to_string()));

        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        // Set low reputation
        oracle.set_reputation(peer_id, 100).await;

        let score = compute_app_specific_score(&peer_id, &oracle).await;

        // Should be 5.0 (100/1000 * 50)
        assert!((score - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_mesh_message_deliveries_config() {
        let video_params = build_topic_params(&TopicCategory::VideoChunks);
        let bft_params = build_topic_params(&TopicCategory::BftSignals);

        // Video chunks should penalize non-delivery (negative weight)
        assert_eq!(video_params.mesh_message_deliveries_weight, -0.5);

        // BFT signals should penalize non-delivery even more
        assert_eq!(bft_params.mesh_message_deliveries_weight, -1.0);
    }

    #[test]
    fn test_first_message_deliveries_config() {
        let video_params = build_topic_params(&TopicCategory::VideoChunks);
        let bft_params = build_topic_params(&TopicCategory::BftSignals);
        let recipes_params = build_topic_params(&TopicCategory::Recipes);

        // BFT should have highest first message delivery weight
        assert_eq!(bft_params.first_message_deliveries_weight, 2.0);

        // Video chunks should have medium weight
        assert_eq!(video_params.first_message_deliveries_weight, 1.0);

        // Recipes should have standard weight
        assert_eq!(recipes_params.first_message_deliveries_weight, 0.5);
    }
}
