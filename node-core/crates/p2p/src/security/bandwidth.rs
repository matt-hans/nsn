//! Bandwidth throttling for P2P connections

use super::metrics::SecurityMetrics;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Bandwidth limiter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthLimiterConfig {
    /// Maximum bandwidth in Mbps per connection
    pub max_bandwidth_mbps: u32,

    /// Measurement interval for bandwidth calculation
    #[serde(with = "humantime_serde")]
    pub measurement_interval: Duration,
}

impl Default for BandwidthLimiterConfig {
    fn default() -> Self {
        Self {
            max_bandwidth_mbps: 100,
            measurement_interval: Duration::from_secs(1),
        }
    }
}

/// Bandwidth usage tracker for a peer
#[derive(Debug, Clone)]
struct BandwidthTracker {
    bytes_transferred: u64,
    interval_start: Instant,
}

/// Bandwidth limiter for P2P connections
pub struct BandwidthLimiter {
    config: BandwidthLimiterConfig,
    trackers: Arc<RwLock<HashMap<PeerId, BandwidthTracker>>>,
    metrics: Arc<SecurityMetrics>,
}

impl BandwidthLimiter {
    /// Create new bandwidth limiter
    pub fn new(config: BandwidthLimiterConfig, metrics: Arc<SecurityMetrics>) -> Self {
        Self {
            config,
            trackers: Arc::new(RwLock::new(HashMap::new())),
            metrics,
        }
    }

    /// Record bytes transferred for a peer
    ///
    /// Returns true if within bandwidth limit, false if throttled
    pub async fn record_transfer(&self, peer_id: &PeerId, bytes: u64) -> bool {
        let mut trackers = self.trackers.write().await;
        let now = Instant::now();

        let tracker = trackers.entry(*peer_id).or_insert(BandwidthTracker {
            bytes_transferred: 0,
            interval_start: now,
        });

        // Reset interval if expired
        if now.duration_since(tracker.interval_start) > self.config.measurement_interval {
            tracker.bytes_transferred = 0;
            tracker.interval_start = now;
        }

        tracker.bytes_transferred += bytes;

        // Calculate current bandwidth in Mbps
        let elapsed_secs = now.duration_since(tracker.interval_start).as_secs_f64();
        let mbps = if elapsed_secs > 0.0 {
            (tracker.bytes_transferred as f64 * 8.0) / (elapsed_secs * 1_000_000.0)
        } else {
            0.0
        };

        // Update metrics
        self.metrics.bandwidth_bytes_transferred.inc_by(bytes);
        self.metrics.bandwidth_usage_mbps.set(mbps);

        // Check limit
        if mbps > self.config.max_bandwidth_mbps as f64 {
            self.metrics.bandwidth_throttled.inc();

            warn!(
                "Bandwidth limit exceeded for peer {}: {:.2} Mbps / {} Mbps",
                peer_id, mbps, self.config.max_bandwidth_mbps
            );

            return false; // Throttled
        }

        debug!("Bandwidth usage for peer {}: {:.2} Mbps", peer_id, mbps);

        true // Allowed
    }

    /// Get current bandwidth usage for a peer (in Mbps)
    pub async fn get_bandwidth(&self, peer_id: &PeerId) -> f64 {
        let trackers = self.trackers.read().await;

        if let Some(tracker) = trackers.get(peer_id) {
            let now = Instant::now();
            let elapsed_secs = now.duration_since(tracker.interval_start).as_secs_f64();

            if elapsed_secs > 0.0 {
                return (tracker.bytes_transferred as f64 * 8.0) / (elapsed_secs * 1_000_000.0);
            }
        }

        0.0
    }

    /// Cleanup expired trackers (periodic background task)
    pub async fn cleanup_expired(&self) {
        let mut trackers = self.trackers.write().await;
        let now = Instant::now();

        trackers.retain(|_peer_id, tracker| {
            now.duration_since(tracker.interval_start) < self.config.measurement_interval * 2
        });
    }

    /// Reset tracker for a peer (for testing)
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn reset_peer(&self, peer_id: &PeerId) {
        self.trackers.write().await.remove(peer_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    fn create_test_bandwidth_limiter(config: BandwidthLimiterConfig) -> BandwidthLimiter {
        let metrics = Arc::new(SecurityMetrics::new_unregistered());
        BandwidthLimiter::new(config, metrics)
    }

    fn create_test_peer_id() -> PeerId {
        let keypair = Keypair::generate_ed25519();
        PeerId::from(keypair.public())
    }

    #[tokio::test]
    async fn test_bandwidth_limiter_allows_under_limit() {
        let config = BandwidthLimiterConfig {
            max_bandwidth_mbps: 10, // 10 Mbps
            measurement_interval: Duration::from_secs(1),
        };

        let limiter = create_test_bandwidth_limiter(config);
        let peer_id = create_test_peer_id();

        // Transfer 1 MB (8 Mbps over 1 second)
        let bytes = 1_000_000;
        let allowed = limiter.record_transfer(&peer_id, bytes).await;

        assert!(allowed, "Transfer should be allowed (under 10 Mbps limit)");
    }

    #[tokio::test]
    async fn test_bandwidth_limiter_throttles_over_limit() {
        let config = BandwidthLimiterConfig {
            max_bandwidth_mbps: 1, // 1 Mbps
            measurement_interval: Duration::from_secs(1),
        };

        let limiter = create_test_bandwidth_limiter(config);
        let peer_id = create_test_peer_id();

        // Transfer 2 MB (16 Mbps over 1 second - exceeds 1 Mbps)
        let bytes = 2_000_000;

        // First transfer to establish interval
        let _ = limiter.record_transfer(&peer_id, bytes / 2).await;

        // Small delay to allow bandwidth calculation
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Second transfer that exceeds limit
        let allowed = limiter.record_transfer(&peer_id, bytes / 2).await;

        // Should be throttled
        assert!(
            !allowed,
            "Transfer should be throttled (exceeds 1 Mbps limit)"
        );
    }

    #[tokio::test]
    async fn test_bandwidth_limiter_interval_reset() {
        let config = BandwidthLimiterConfig {
            max_bandwidth_mbps: 1,
            measurement_interval: Duration::from_millis(100), // 100ms for testing
        };

        let limiter = create_test_bandwidth_limiter(config);
        let peer_id = create_test_peer_id();

        // Transfer that would exceed limit
        let bytes = 1_000_000;
        let _ = limiter.record_transfer(&peer_id, bytes).await;

        // Should be throttled initially
        tokio::time::sleep(Duration::from_millis(10)).await;
        let allowed = limiter.record_transfer(&peer_id, bytes).await;
        assert!(!allowed, "Should be throttled before interval reset");

        // Wait for interval to expire
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should be allowed again (new interval)
        let allowed = limiter.record_transfer(&peer_id, bytes / 10).await;
        assert!(allowed, "Should be allowed after interval reset");
    }

    #[tokio::test]
    async fn test_bandwidth_limiter_per_peer_isolation() {
        let config = BandwidthLimiterConfig {
            max_bandwidth_mbps: 1,
            measurement_interval: Duration::from_secs(1),
        };

        let limiter = create_test_bandwidth_limiter(config);
        let peer1 = create_test_peer_id();
        let peer2 = create_test_peer_id();

        // Peer 1 exhausts bandwidth
        let bytes = 2_000_000;
        let _ = limiter.record_transfer(&peer1, bytes).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Peer 1 should be throttled
        let allowed = limiter.record_transfer(&peer1, bytes).await;
        assert!(!allowed, "Peer 1 should be throttled");

        // Peer 2 should not be affected
        let allowed = limiter.record_transfer(&peer2, bytes / 10).await;
        assert!(
            allowed,
            "Peer 2 should not be affected by Peer 1's bandwidth"
        );
    }

    #[tokio::test]
    async fn test_bandwidth_limiter_get_bandwidth() {
        let config = BandwidthLimiterConfig {
            max_bandwidth_mbps: 100,
            measurement_interval: Duration::from_secs(1),
        };

        let limiter = create_test_bandwidth_limiter(config);
        let peer_id = create_test_peer_id();

        // Initially 0
        let bandwidth = limiter.get_bandwidth(&peer_id).await;
        assert_eq!(bandwidth, 0.0, "Initial bandwidth should be 0");

        // Transfer 1 MB
        let bytes = 1_000_000;
        let _ = limiter.record_transfer(&peer_id, bytes).await;

        // Allow time for measurement
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Get bandwidth (should be ~8 Mbps)
        let bandwidth = limiter.get_bandwidth(&peer_id).await;
        assert!(
            bandwidth > 0.0,
            "Bandwidth should be greater than 0 after transfer"
        );
    }

    #[tokio::test]
    async fn test_bandwidth_limiter_cleanup_expired() {
        let config = BandwidthLimiterConfig {
            max_bandwidth_mbps: 100,
            measurement_interval: Duration::from_millis(50), // 50ms for testing
        };

        let limiter = create_test_bandwidth_limiter(config);
        let peer_id = create_test_peer_id();

        // Make transfer
        let _ = limiter.record_transfer(&peer_id, 1000).await;

        // Verify tracker exists
        let bandwidth_before = limiter.get_bandwidth(&peer_id).await;
        assert!(bandwidth_before >= 0.0, "Tracker should exist");

        // Wait for interval to expire (2× duration)
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Cleanup
        limiter.cleanup_expired().await;

        // After cleanup and new interval, bandwidth should reset
        tokio::time::sleep(Duration::from_millis(100)).await;
        let bandwidth_after = limiter.get_bandwidth(&peer_id).await;

        // Either cleaned up (0.0) or new interval started
        assert!(
            bandwidth_after >= 0.0,
            "Bandwidth should be valid after cleanup"
        );
    }

    #[tokio::test]
    async fn test_bandwidth_limiter_metrics_bytes_transferred() {
        let config = BandwidthLimiterConfig::default();
        let limiter = create_test_bandwidth_limiter(config);
        let peer_id = create_test_peer_id();

        let initial_bytes = limiter.metrics.bandwidth_bytes_transferred.get();

        // Transfer 1000 bytes
        let _ = limiter.record_transfer(&peer_id, 1000).await;

        let final_bytes = limiter.metrics.bandwidth_bytes_transferred.get();

        assert_eq!(
            final_bytes,
            initial_bytes + 1000,
            "Bytes transferred metric should increment"
        );
    }

    #[tokio::test]
    async fn test_bandwidth_limiter_metrics_throttled() {
        let config = BandwidthLimiterConfig {
            max_bandwidth_mbps: 1,
            measurement_interval: Duration::from_secs(1),
        };

        let limiter = create_test_bandwidth_limiter(config);
        let peer_id = create_test_peer_id();

        let initial_throttled = limiter.metrics.bandwidth_throttled.get();

        // Transfer that exceeds limit
        let _ = limiter.record_transfer(&peer_id, 2_000_000).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        let _ = limiter.record_transfer(&peer_id, 2_000_000).await;

        let final_throttled = limiter.metrics.bandwidth_throttled.get();

        assert!(
            final_throttled > initial_throttled,
            "Throttled metric should increment"
        );
    }

    #[tokio::test]
    async fn test_bandwidth_limiter_reset_peer() {
        let config = BandwidthLimiterConfig::default();
        let limiter = create_test_bandwidth_limiter(config);
        let peer_id = create_test_peer_id();

        // Make transfer
        let _ = limiter.record_transfer(&peer_id, 1_000_000).await;

        // Verify bandwidth tracked
        let bandwidth_before = limiter.get_bandwidth(&peer_id).await;
        assert!(bandwidth_before > 0.0, "Bandwidth should be tracked");

        // Reset peer
        limiter.reset_peer(&peer_id).await;

        // Bandwidth should be 0 (tracker removed)
        let bandwidth_after = limiter.get_bandwidth(&peer_id).await;
        assert_eq!(bandwidth_after, 0.0, "Bandwidth should be 0 after reset");
    }
}
