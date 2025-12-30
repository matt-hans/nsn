//! GossipSub topic definitions for NSN dual-lane architecture
//!
//! Defines all topics for Lane 0 (video generation) and Lane 1 (general compute).
//! Topics use versioned string identifiers following the format: /nsn/<name>/<version>

use libp2p::gossipsub::IdentTopic;
use serde::{Deserialize, Serialize};
use std::fmt;

// Lane 0 topics (video generation with BFT consensus)
/// Recipe broadcast topic - JSON instructions for AI generation
pub const RECIPES_TOPIC: &str = "/nsn/recipes/1.0.0";

/// Video chunks topic - 16MB max chunks for Director→Super-Node distribution
pub const VIDEO_CHUNKS_TOPIC: &str = "/nsn/video/1.0.0";

/// BFT signals topic - Critical CLIP embedding exchange for consensus
pub const BFT_SIGNALS_TOPIC: &str = "/nsn/bft/1.0.0";

/// Validator attestations topic - CLIP verification results
pub const ATTESTATIONS_TOPIC: &str = "/nsn/attestations/1.0.0";

/// Challenges topic - BFT result disputes
pub const CHALLENGES_TOPIC: &str = "/nsn/challenges/1.0.0";

// Lane 1 topics (general compute marketplace)
/// Tasks topic - Arbitrary AI task broadcast for Lane 1 marketplace
pub const TASKS_TOPIC: &str = "/nsn/tasks/1.0.0";

/// Topic category for filtering and configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TopicCategory {
    /// Lane 0: Recipe broadcast
    Recipes,
    /// Lane 0: Video chunk distribution
    VideoChunks,
    /// Lane 0: BFT consensus signals (critical path)
    BftSignals,
    /// Lane 0: Validator attestations
    Attestations,
    /// Lane 0: BFT challenges
    Challenges,
    /// Lane 1: General compute tasks
    Tasks,
}

impl TopicCategory {
    /// Get the topic string for this category
    pub fn as_str(&self) -> &'static str {
        match self {
            TopicCategory::Recipes => RECIPES_TOPIC,
            TopicCategory::VideoChunks => VIDEO_CHUNKS_TOPIC,
            TopicCategory::BftSignals => BFT_SIGNALS_TOPIC,
            TopicCategory::Attestations => ATTESTATIONS_TOPIC,
            TopicCategory::Challenges => CHALLENGES_TOPIC,
            TopicCategory::Tasks => TASKS_TOPIC,
        }
    }

    /// Get the IdentTopic for this category
    pub fn to_topic(&self) -> IdentTopic {
        IdentTopic::new(self.as_str())
    }

    /// Get topic weight for peer scoring (higher = more important)
    pub fn weight(&self) -> f64 {
        match self {
            TopicCategory::BftSignals => 3.0, // Critical - consensus requires low latency
            TopicCategory::Challenges => 2.5, // High - dispute resolution
            TopicCategory::VideoChunks => 2.0, // High - content delivery
            TopicCategory::Attestations => 2.0, // High - validation results
            TopicCategory::Tasks => 1.5,      // Medium - Lane 1 marketplace
            TopicCategory::Recipes => 1.0,    // Normal - broadcast
        }
    }

    /// Whether this topic uses flood publishing (low latency)
    pub fn uses_flood_publish(&self) -> bool {
        matches!(self, TopicCategory::BftSignals)
    }

    /// Maximum message size for this topic in bytes
    pub fn max_message_size(&self) -> usize {
        match self {
            TopicCategory::VideoChunks => 16 * 1024 * 1024, // 16MB
            TopicCategory::Recipes => 1024 * 1024,          // 1MB
            TopicCategory::BftSignals => 64 * 1024,         // 64KB (CLIP embeddings)
            TopicCategory::Attestations => 64 * 1024,       // 64KB
            TopicCategory::Challenges => 128 * 1024,        // 128KB
            TopicCategory::Tasks => 1024 * 1024,            // 1MB
        }
    }

    /// Get all topic categories
    pub fn all() -> Vec<TopicCategory> {
        vec![
            TopicCategory::Recipes,
            TopicCategory::VideoChunks,
            TopicCategory::BftSignals,
            TopicCategory::Attestations,
            TopicCategory::Challenges,
            TopicCategory::Tasks,
        ]
    }

    /// Get all Lane 0 topics
    pub fn lane_0() -> Vec<TopicCategory> {
        vec![
            TopicCategory::Recipes,
            TopicCategory::VideoChunks,
            TopicCategory::BftSignals,
            TopicCategory::Attestations,
            TopicCategory::Challenges,
        ]
    }

    /// Get all Lane 1 topics
    pub fn lane_1() -> Vec<TopicCategory> {
        vec![TopicCategory::Tasks]
    }
}

impl fmt::Display for TopicCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<TopicCategory> for IdentTopic {
    fn from(category: TopicCategory) -> Self {
        category.to_topic()
    }
}

/// Get all NSN topics as IdentTopic instances
pub fn all_topics() -> Vec<IdentTopic> {
    TopicCategory::all()
        .into_iter()
        .map(|cat| cat.to_topic())
        .collect()
}

/// Get Lane 0 topics only
pub fn lane_0_topics() -> Vec<IdentTopic> {
    TopicCategory::lane_0()
        .into_iter()
        .map(|cat| cat.to_topic())
        .collect()
}

/// Get Lane 1 topics only
pub fn lane_1_topics() -> Vec<IdentTopic> {
    TopicCategory::lane_1()
        .into_iter()
        .map(|cat| cat.to_topic())
        .collect()
}

/// Parse topic string to TopicCategory
pub fn parse_topic(topic: &str) -> Option<TopicCategory> {
    match topic {
        RECIPES_TOPIC => Some(TopicCategory::Recipes),
        VIDEO_CHUNKS_TOPIC => Some(TopicCategory::VideoChunks),
        BFT_SIGNALS_TOPIC => Some(TopicCategory::BftSignals),
        ATTESTATIONS_TOPIC => Some(TopicCategory::Attestations),
        CHALLENGES_TOPIC => Some(TopicCategory::Challenges),
        TASKS_TOPIC => Some(TopicCategory::Tasks),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_topics_count() {
        let topics = all_topics();
        assert_eq!(topics.len(), 6, "Should have exactly 6 topics");
    }

    #[test]
    fn test_lane_0_topics_count() {
        let topics = lane_0_topics();
        assert_eq!(topics.len(), 5, "Lane 0 should have 5 topics");
    }

    #[test]
    fn test_lane_1_topics_count() {
        let topics = lane_1_topics();
        assert_eq!(topics.len(), 1, "Lane 1 should have 1 topic");
    }

    #[test]
    fn test_topic_category_strings() {
        assert_eq!(TopicCategory::Recipes.as_str(), "/nsn/recipes/1.0.0");
        assert_eq!(TopicCategory::VideoChunks.as_str(), "/nsn/video/1.0.0");
        assert_eq!(TopicCategory::BftSignals.as_str(), "/nsn/bft/1.0.0");
        assert_eq!(
            TopicCategory::Attestations.as_str(),
            "/nsn/attestations/1.0.0"
        );
        assert_eq!(TopicCategory::Challenges.as_str(), "/nsn/challenges/1.0.0");
        assert_eq!(TopicCategory::Tasks.as_str(), "/nsn/tasks/1.0.0");
    }

    #[test]
    fn test_topic_weights() {
        assert_eq!(TopicCategory::BftSignals.weight(), 3.0);
        assert_eq!(TopicCategory::Challenges.weight(), 2.5);
        assert_eq!(TopicCategory::VideoChunks.weight(), 2.0);
        assert_eq!(TopicCategory::Attestations.weight(), 2.0);
        assert_eq!(TopicCategory::Tasks.weight(), 1.5);
        assert_eq!(TopicCategory::Recipes.weight(), 1.0);
    }

    #[test]
    fn test_flood_publish_flag() {
        assert!(TopicCategory::BftSignals.uses_flood_publish());
        assert!(!TopicCategory::Recipes.uses_flood_publish());
        assert!(!TopicCategory::VideoChunks.uses_flood_publish());
        assert!(!TopicCategory::Attestations.uses_flood_publish());
        assert!(!TopicCategory::Challenges.uses_flood_publish());
        assert!(!TopicCategory::Tasks.uses_flood_publish());
    }

    #[test]
    fn test_max_message_sizes() {
        assert_eq!(
            TopicCategory::VideoChunks.max_message_size(),
            16 * 1024 * 1024
        );
        assert_eq!(TopicCategory::Recipes.max_message_size(), 1024 * 1024);
        assert_eq!(TopicCategory::BftSignals.max_message_size(), 64 * 1024);
        assert_eq!(TopicCategory::Attestations.max_message_size(), 64 * 1024);
        assert_eq!(TopicCategory::Challenges.max_message_size(), 128 * 1024);
        assert_eq!(TopicCategory::Tasks.max_message_size(), 1024 * 1024);
    }

    #[test]
    fn test_parse_topic() {
        assert_eq!(
            parse_topic("/nsn/recipes/1.0.0"),
            Some(TopicCategory::Recipes)
        );
        assert_eq!(
            parse_topic("/nsn/video/1.0.0"),
            Some(TopicCategory::VideoChunks)
        );
        assert_eq!(
            parse_topic("/nsn/bft/1.0.0"),
            Some(TopicCategory::BftSignals)
        );
        assert_eq!(
            parse_topic("/nsn/attestations/1.0.0"),
            Some(TopicCategory::Attestations)
        );
        assert_eq!(
            parse_topic("/nsn/challenges/1.0.0"),
            Some(TopicCategory::Challenges)
        );
        assert_eq!(parse_topic("/nsn/tasks/1.0.0"), Some(TopicCategory::Tasks));
        assert_eq!(parse_topic("/unknown/topic/1.0.0"), None);
    }

    #[test]
    fn test_topic_to_ident_topic() {
        let topic = TopicCategory::BftSignals.to_topic();
        // IdentTopic doesn't have as_str, use hash() to verify topic was created
        let hash = topic.hash();
        assert!(!hash.as_str().is_empty(), "Topic hash should not be empty");
    }

    #[test]
    fn test_topic_display() {
        assert_eq!(format!("{}", TopicCategory::Recipes), "/nsn/recipes/1.0.0");
        assert_eq!(format!("{}", TopicCategory::BftSignals), "/nsn/bft/1.0.0");
    }

    #[test]
    fn test_topic_category_all() {
        let all = TopicCategory::all();
        assert_eq!(all.len(), 6);
        assert!(all.contains(&TopicCategory::Recipes));
        assert!(all.contains(&TopicCategory::VideoChunks));
        assert!(all.contains(&TopicCategory::BftSignals));
        assert!(all.contains(&TopicCategory::Attestations));
        assert!(all.contains(&TopicCategory::Challenges));
        assert!(all.contains(&TopicCategory::Tasks));
    }

    #[test]
    fn test_topic_category_serialization() {
        let category = TopicCategory::BftSignals;

        // Serialize to JSON
        let json = serde_json::to_string(&category).expect("Failed to serialize");

        // Deserialize back
        let deserialized: TopicCategory =
            serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(deserialized, category);
    }
}
