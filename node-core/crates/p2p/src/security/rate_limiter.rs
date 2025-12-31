//! Rate limiter for P2P requests with reputation-based policies

use super::metrics::SecurityMetrics;
use crate::reputation_oracle::ReputationOracle;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, warn};

#[derive(Debug, Error, PartialEq)]
pub enum RateLimitError {
    #[error("Rate limit exceeded for peer {peer_id}: {actual}/{limit} requests")]
    LimitExceeded {
        peer_id: PeerId,
        limit: u32,
        actual: u32,
    },
}

/// Rate limiter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiterConfig {
    /// Maximum requests per minute per peer
    pub max_requests_per_minute: u32,

    /// Time window for rate limiting
    #[serde(with = "humantime_serde")]
    pub rate_limit_window: Duration,

    /// Reputation multiplier for high-reputation peers (e.g., 2.0 = 2× rate limit)
    pub reputation_rate_limit_multiplier: f64,

    /// Minimum reputation score to qualify for bypass (0-1000)
    pub min_reputation_for_bypass: u64,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            max_requests_per_minute: 100,
            rate_limit_window: Duration::from_secs(60),
            reputation_rate_limit_multiplier: 2.0,
            min_reputation_for_bypass: 800,
        }
    }
}

/// Request counter for tracking per-peer request counts
#[derive(Debug, Clone)]
struct RequestCounter {
    count: u32,
    window_start: Instant,
}

/// Rate limiter for P2P requests
pub struct RateLimiter {
    config: RateLimiterConfig,
    request_counts: Arc<RwLock<HashMap<PeerId, RequestCounter>>>,
    reputation_oracle: Option<Arc<ReputationOracle>>,
    metrics: Arc<SecurityMetrics>,
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new(
        config: RateLimiterConfig,
        reputation_oracle: Option<Arc<ReputationOracle>>,
        metrics: Arc<SecurityMetrics>,
    ) -> Self {
        Self {
            config,
            request_counts: Arc::new(RwLock::new(HashMap::new())),
            reputation_oracle,
            metrics,
        }
    }

    /// Check if request from peer is within rate limit
    ///
    /// Returns Ok(()) if allowed, Err(RateLimitError) if exceeded
    pub async fn check_rate_limit(&self, peer_id: &PeerId) -> Result<(), RateLimitError> {
        let mut counts = self.request_counts.write().await;
        let now = Instant::now();

        let counter = counts.entry(*peer_id).or_insert(RequestCounter {
            count: 0,
            window_start: now,
        });

        // Reset window if expired
        if now.duration_since(counter.window_start) > self.config.rate_limit_window {
            counter.count = 0;
            counter.window_start = now;
        }

        // Get rate limit (with reputation multiplier if applicable)
        let limit = self.get_rate_limit_for_peer(peer_id).await;

        // Check limit
        if counter.count >= limit {
            self.metrics
                .rate_limit_violations
                .with_label_values(&[&peer_id.to_string()])
                .inc();

            warn!(
                "Rate limit exceeded for peer {}: {}/{} requests",
                peer_id, counter.count, limit
            );

            return Err(RateLimitError::LimitExceeded {
                peer_id: *peer_id,
                limit,
                actual: counter.count,
            });
        }

        // Increment counter
        counter.count += 1;
        self.metrics.rate_limit_allowed.inc();

        Ok(())
    }

    /// Get rate limit for a specific peer (with reputation multiplier)
    async fn get_rate_limit_for_peer(&self, peer_id: &PeerId) -> u32 {
        let base_limit = self.config.max_requests_per_minute;

        if let Some(oracle) = &self.reputation_oracle {
            let reputation = oracle.get_reputation(peer_id).await;

            // High-reputation peers get multiplier
            if reputation >= self.config.min_reputation_for_bypass {
                let multiplier = self.config.reputation_rate_limit_multiplier;
                let adjusted = (base_limit as f64 * multiplier) as u32;

                debug!(
                    "Peer {} has high reputation ({}), rate limit adjusted: {} -> {}",
                    peer_id, reputation, base_limit, adjusted
                );

                self.metrics.reputation_bypass_applied.inc();
                return adjusted;
            }
        }

        base_limit
    }

    /// Reset rate limit for a peer (for testing or manual override)
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn reset_peer(&self, peer_id: &PeerId) {
        self.request_counts.write().await.remove(peer_id);
    }

    /// Get current request count for a peer (for testing)
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn get_count(&self, peer_id: &PeerId) -> u32 {
        self.request_counts
            .read()
            .await
            .get(peer_id)
            .map(|c| c.count)
            .unwrap_or(0)
    }

    /// Clean up expired entries (periodic background task)
    pub async fn cleanup_expired(&self) {
        let mut counts = self.request_counts.write().await;
        let now = Instant::now();

        counts.retain(|_peer_id, counter| {
            now.duration_since(counter.window_start) < self.config.rate_limit_window * 2
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    fn create_test_rate_limiter(config: RateLimiterConfig) -> RateLimiter {
        let metrics = Arc::new(SecurityMetrics::new_unregistered());
        RateLimiter::new(config, None, metrics)
    }

    fn create_test_peer_id() -> PeerId {
        let keypair = Keypair::generate_ed25519();
        PeerId::from(keypair.public())
    }

    #[tokio::test]
    async fn test_rate_limit_allows_under_limit() {
        let config = RateLimiterConfig {
            max_requests_per_minute: 5,
            rate_limit_window: Duration::from_secs(60),
            ..Default::default()
        };

        let rate_limiter = create_test_rate_limiter(config);
        let peer_id = create_test_peer_id();

        // First 5 requests should succeed
        for i in 0..5 {
            let result = rate_limiter.check_rate_limit(&peer_id).await;
            assert!(
                result.is_ok(),
                "Request {} should be allowed (under limit)",
                i + 1
            );
        }

        // Verify count
        let count = rate_limiter.get_count(&peer_id).await;
        assert_eq!(count, 5, "Request count should be 5");
    }

    #[tokio::test]
    async fn test_rate_limit_rejects_over_limit() {
        let config = RateLimiterConfig {
            max_requests_per_minute: 5,
            rate_limit_window: Duration::from_secs(60),
            ..Default::default()
        };

        let rate_limiter = create_test_rate_limiter(config);
        let peer_id = create_test_peer_id();

        // Exhaust limit
        for _ in 0..5 {
            let _ = rate_limiter.check_rate_limit(&peer_id).await;
        }

        // 6th request should fail
        let result = rate_limiter.check_rate_limit(&peer_id).await;
        assert!(
            result.is_err(),
            "Request should be rejected (limit exceeded)"
        );

        match result {
            Err(RateLimitError::LimitExceeded {
                peer_id: returned_peer,
                limit,
                actual,
            }) => {
                assert_eq!(returned_peer, peer_id, "PeerId should match");
                assert_eq!(limit, 5, "Limit should be 5");
                assert_eq!(actual, 5, "Actual count should be 5");
            }
            _ => panic!("Expected LimitExceeded error"),
        }
    }

    #[tokio::test]
    async fn test_rate_limit_window_reset() {
        let config = RateLimiterConfig {
            max_requests_per_minute: 3,
            rate_limit_window: Duration::from_millis(100), // 100ms window for testing
            ..Default::default()
        };

        let rate_limiter = create_test_rate_limiter(config);
        let peer_id = create_test_peer_id();

        // Exhaust limit
        for _ in 0..3 {
            let _ = rate_limiter.check_rate_limit(&peer_id).await;
        }

        // Should be rejected
        let result = rate_limiter.check_rate_limit(&peer_id).await;
        assert!(result.is_err(), "Should be rejected immediately");

        // Wait for window to expire
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should be allowed again (window reset)
        let result = rate_limiter.check_rate_limit(&peer_id).await;
        assert!(result.is_ok(), "Should be allowed after window reset");

        let count = rate_limiter.get_count(&peer_id).await;
        assert_eq!(count, 1, "Count should reset to 1");
    }

    #[tokio::test]
    async fn test_rate_limit_per_peer_isolation() {
        let config = RateLimiterConfig {
            max_requests_per_minute: 3,
            rate_limit_window: Duration::from_secs(60),
            ..Default::default()
        };

        let rate_limiter = create_test_rate_limiter(config);
        let peer1 = create_test_peer_id();
        let peer2 = create_test_peer_id();

        // Peer 1 exhausts limit
        for _ in 0..3 {
            let _ = rate_limiter.check_rate_limit(&peer1).await;
        }

        // Peer 1 should be rejected
        let result = rate_limiter.check_rate_limit(&peer1).await;
        assert!(result.is_err(), "Peer 1 should be rate limited");

        // Peer 2 should still be allowed
        let result = rate_limiter.check_rate_limit(&peer2).await;
        assert!(
            result.is_ok(),
            "Peer 2 should not be affected by Peer 1's limit"
        );
    }

    #[tokio::test]
    async fn test_rate_limit_with_reputation_bypass() {
        use crate::reputation_oracle::ReputationOracle;

        let config = RateLimiterConfig {
            max_requests_per_minute: 5,
            rate_limit_window: Duration::from_secs(60),
            reputation_rate_limit_multiplier: 2.0,
            min_reputation_for_bypass: 800,
        };

        let oracle = Arc::new(ReputationOracle::new_without_registry(
            "ws://localhost:9944".to_string(),
        ));

        let metrics = Arc::new(SecurityMetrics::new_unregistered());
        let rate_limiter = RateLimiter::new(config, Some(oracle.clone()), metrics);

        let high_rep_peer = create_test_peer_id();

        // Set high reputation (800+)
        oracle.set_reputation(high_rep_peer, 850).await;

        // High-reputation peer should get 2× limit (5 * 2.0 = 10)
        for i in 0..10 {
            let result = rate_limiter.check_rate_limit(&high_rep_peer).await;
            assert!(
                result.is_ok(),
                "High-reputation peer request {} should be allowed (within 2× limit)",
                i + 1
            );
        }

        // 11th request should fail
        let result = rate_limiter.check_rate_limit(&high_rep_peer).await;
        assert!(
            result.is_err(),
            "High-reputation peer should still be limited at 2× rate"
        );
    }

    #[tokio::test]
    async fn test_rate_limit_reputation_bypass_threshold() {
        use crate::reputation_oracle::ReputationOracle;

        let config = RateLimiterConfig {
            max_requests_per_minute: 5,
            rate_limit_window: Duration::from_secs(60),
            reputation_rate_limit_multiplier: 2.0,
            min_reputation_for_bypass: 800,
        };

        let oracle = Arc::new(ReputationOracle::new_without_registry(
            "ws://localhost:9944".to_string(),
        ));

        let metrics = Arc::new(SecurityMetrics::new_unregistered());
        let rate_limiter = RateLimiter::new(config, Some(oracle.clone()), metrics);

        let low_rep_peer = create_test_peer_id();

        // Set reputation below threshold (799)
        oracle.set_reputation(low_rep_peer, 799).await;

        // Low-reputation peer should get base limit (5)
        for _ in 0..5 {
            let _ = rate_limiter.check_rate_limit(&low_rep_peer).await;
        }

        // 6th request should fail (no bypass)
        let result = rate_limiter.check_rate_limit(&low_rep_peer).await;
        assert!(
            result.is_err(),
            "Low-reputation peer should not get bypass (reputation 799 < 800)"
        );
    }

    #[tokio::test]
    async fn test_rate_limit_reset_peer() {
        let config = RateLimiterConfig {
            max_requests_per_minute: 3,
            rate_limit_window: Duration::from_secs(60),
            ..Default::default()
        };

        let rate_limiter = create_test_rate_limiter(config);
        let peer_id = create_test_peer_id();

        // Make some requests
        for _ in 0..3 {
            let _ = rate_limiter.check_rate_limit(&peer_id).await;
        }

        // Should be rate limited
        let result = rate_limiter.check_rate_limit(&peer_id).await;
        assert!(result.is_err(), "Should be rate limited");

        // Reset peer
        rate_limiter.reset_peer(&peer_id).await;

        // Should be allowed again
        let result = rate_limiter.check_rate_limit(&peer_id).await;
        assert!(result.is_ok(), "Should be allowed after reset");

        let count = rate_limiter.get_count(&peer_id).await;
        assert_eq!(count, 1, "Count should be 1 after reset");
    }

    #[tokio::test]
    async fn test_rate_limit_cleanup_expired() {
        let config = RateLimiterConfig {
            max_requests_per_minute: 3,
            rate_limit_window: Duration::from_millis(50), // 50ms window for testing
            ..Default::default()
        };

        let rate_limiter = create_test_rate_limiter(config);
        let peer1 = create_test_peer_id();
        let peer2 = create_test_peer_id();

        // Make requests from both peers
        let _ = rate_limiter.check_rate_limit(&peer1).await;
        let _ = rate_limiter.check_rate_limit(&peer2).await;

        // Wait for windows to expire (2× window duration)
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Cleanup expired entries
        rate_limiter.cleanup_expired().await;

        // Internal state should be cleaned (verify via new requests resetting count)
        let result = rate_limiter.check_rate_limit(&peer1).await;
        assert!(result.is_ok(), "Peer1 should work after cleanup");

        let count = rate_limiter.get_count(&peer1).await;
        assert_eq!(count, 1, "Count should be 1 (fresh window after cleanup)");
    }

    #[tokio::test]
    async fn test_rate_limit_metrics_violations() {
        let config = RateLimiterConfig {
            max_requests_per_minute: 2,
            rate_limit_window: Duration::from_secs(60),
            ..Default::default()
        };

        let rate_limiter = create_test_rate_limiter(config);
        let peer_id = create_test_peer_id();

        // Exhaust limit
        for _ in 0..2 {
            let _ = rate_limiter.check_rate_limit(&peer_id).await;
        }

        let initial_violations = rate_limiter
            .metrics
            .rate_limit_violations
            .with_label_values(&[&peer_id.to_string()])
            .get();

        // Trigger violation
        let _ = rate_limiter.check_rate_limit(&peer_id).await;

        let final_violations = rate_limiter
            .metrics
            .rate_limit_violations
            .with_label_values(&[&peer_id.to_string()])
            .get();

        assert_eq!(
            final_violations,
            initial_violations + 1,
            "Violations metric should increment"
        );
    }

    #[tokio::test]
    async fn test_rate_limit_metrics_allowed() {
        let config = RateLimiterConfig::default();
        let rate_limiter = create_test_rate_limiter(config);
        let peer_id = create_test_peer_id();

        let initial_allowed = rate_limiter.metrics.rate_limit_allowed.get();

        // Make allowed request
        let _ = rate_limiter.check_rate_limit(&peer_id).await;

        let final_allowed = rate_limiter.metrics.rate_limit_allowed.get();

        assert_eq!(
            final_allowed,
            initial_allowed + 1,
            "Allowed metric should increment"
        );
    }

    #[tokio::test]
    async fn test_rate_limit_metrics_reputation_bypass() {
        use crate::reputation_oracle::ReputationOracle;

        let config = RateLimiterConfig {
            min_reputation_for_bypass: 800,
            ..Default::default()
        };

        let oracle = Arc::new(ReputationOracle::new_without_registry(
            "ws://localhost:9944".to_string(),
        ));

        let metrics = Arc::new(SecurityMetrics::new_unregistered());
        let rate_limiter = RateLimiter::new(config, Some(oracle.clone()), metrics.clone());

        let high_rep_peer = create_test_peer_id();
        oracle.set_reputation(high_rep_peer, 900).await;

        let initial_bypass = metrics.reputation_bypass_applied.get();

        // Make request (should trigger bypass)
        let _ = rate_limiter.check_rate_limit(&high_rep_peer).await;

        let final_bypass = metrics.reputation_bypass_applied.get();

        assert_eq!(
            final_bypass,
            initial_bypass + 1,
            "Bypass metric should increment"
        );
    }
}
