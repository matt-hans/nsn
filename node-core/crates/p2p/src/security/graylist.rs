//! Graylist enforcement for temporarily banning violating peers

use super::metrics::SecurityMetrics;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Graylist configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraylistConfig {
    /// Duration of graylist ban (default: 1 hour)
    #[serde(with = "humantime_serde")]
    pub duration: Duration,

    /// Number of violations before graylisting
    pub threshold_violations: u32,
}

impl Default for GraylistConfig {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(3600), // 1 hour
            threshold_violations: 3,
        }
    }
}

/// Graylist entry with violation tracking
#[derive(Debug, Clone)]
pub struct GraylistEntry {
    pub banned_at: Instant,
    pub reason: String,
    pub violations: u32,
}

/// Graylist for temporarily banning peers
pub struct Graylist {
    config: GraylistConfig,
    graylisted: Arc<RwLock<HashMap<PeerId, GraylistEntry>>>,
    metrics: Arc<SecurityMetrics>,
}

impl Graylist {
    /// Create new graylist
    pub fn new(config: GraylistConfig, metrics: Arc<SecurityMetrics>) -> Self {
        Self {
            config,
            graylisted: Arc::new(RwLock::new(HashMap::new())),
            metrics,
        }
    }

    /// Check if peer is graylisted
    pub async fn is_graylisted(&self, peer_id: &PeerId) -> bool {
        let graylisted = self.graylisted.read().await;

        if let Some(entry) = graylisted.get(peer_id) {
            let now = Instant::now();
            let elapsed = now.duration_since(entry.banned_at);

            if elapsed < self.config.duration {
                self.metrics.graylist_rejections.inc();
                return true;
            }
        }

        false
    }

    /// Add peer to graylist
    pub async fn add(&self, peer_id: PeerId, reason: String) {
        let mut graylisted = self.graylisted.write().await;

        let entry = graylisted.entry(peer_id).or_insert(GraylistEntry {
            banned_at: Instant::now(),
            reason: reason.clone(),
            violations: 0,
        });

        entry.violations += 1;
        entry.banned_at = Instant::now();
        entry.reason = reason.clone();

        warn!(
            "Peer {} graylisted (violations: {}): {}",
            peer_id, entry.violations, reason
        );

        self.metrics.peers_graylisted.inc();
        self.metrics.graylist_size.set(graylisted.len() as f64);
    }

    /// Remove peer from graylist
    pub async fn remove(&self, peer_id: &PeerId) {
        let mut graylisted = self.graylisted.write().await;
        graylisted.remove(peer_id);

        info!("Peer {} removed from graylist", peer_id);

        self.metrics.graylist_size.set(graylisted.len() as f64);
    }

    /// Cleanup expired graylist entries (periodic background task)
    pub async fn cleanup_expired(&self) {
        let mut graylisted = self.graylisted.write().await;
        let now = Instant::now();

        graylisted.retain(|peer_id, entry| {
            let elapsed = now.duration_since(entry.banned_at);
            let keep = elapsed < self.config.duration;

            if !keep {
                info!("Graylist expired for peer {}", peer_id);
            }

            keep
        });

        self.metrics.graylist_size.set(graylisted.len() as f64);
    }

    /// Get graylist size
    pub async fn size(&self) -> usize {
        self.graylisted.read().await.len()
    }

    /// Get time remaining for graylisted peer (for diagnostics)
    pub async fn time_remaining(&self, peer_id: &PeerId) -> Option<Duration> {
        let graylisted = self.graylisted.read().await;

        if let Some(entry) = graylisted.get(peer_id) {
            let now = Instant::now();
            let elapsed = now.duration_since(entry.banned_at);

            if elapsed < self.config.duration {
                return Some(self.config.duration - elapsed);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    fn create_test_graylist(config: GraylistConfig) -> Graylist {
        let metrics = Arc::new(SecurityMetrics::new_unregistered());
        Graylist::new(config, metrics)
    }

    fn create_test_peer_id() -> PeerId {
        let keypair = Keypair::generate_ed25519();
        PeerId::from(keypair.public())
    }

    #[tokio::test]
    async fn test_graylist_add_and_check() {
        let config = GraylistConfig::default();
        let graylist = create_test_graylist(config);
        let peer_id = create_test_peer_id();

        // Initially not graylisted
        assert!(!graylist.is_graylisted(&peer_id).await);

        // Add to graylist
        graylist
            .add(peer_id, "Rate limit violations".to_string())
            .await;

        // Should be graylisted
        assert!(graylist.is_graylisted(&peer_id).await);
    }

    #[tokio::test]
    async fn test_graylist_expiration() {
        let config = GraylistConfig {
            duration: Duration::from_millis(100), // 100ms for testing
            threshold_violations: 3,
        };

        let graylist = create_test_graylist(config);
        let peer_id = create_test_peer_id();

        // Add to graylist
        graylist.add(peer_id, "Test".to_string()).await;

        // Should be graylisted initially
        assert!(graylist.is_graylisted(&peer_id).await);

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should no longer be graylisted
        assert!(!graylist.is_graylisted(&peer_id).await);
    }

    #[tokio::test]
    async fn test_graylist_remove() {
        let config = GraylistConfig::default();
        let graylist = create_test_graylist(config);
        let peer_id = create_test_peer_id();

        // Add and verify
        graylist.add(peer_id, "Test".to_string()).await;
        assert!(graylist.is_graylisted(&peer_id).await);

        // Remove
        graylist.remove(&peer_id).await;

        // Should no longer be graylisted
        assert!(!graylist.is_graylisted(&peer_id).await);
    }

    #[tokio::test]
    async fn test_graylist_violations_increment() {
        let config = GraylistConfig::default();
        let graylist = create_test_graylist(config);
        let peer_id = create_test_peer_id();

        // Add multiple times
        graylist.add(peer_id, "Violation 1".to_string()).await;
        graylist.add(peer_id, "Violation 2".to_string()).await;
        graylist.add(peer_id, "Violation 3".to_string()).await;

        // Check violations count (via graylisted map)
        let graylisted = graylist.graylisted.read().await;
        let entry = graylisted.get(&peer_id).unwrap();

        assert_eq!(entry.violations, 3, "Violations should increment to 3");
    }

    #[tokio::test]
    async fn test_graylist_cleanup_expired() {
        let config = GraylistConfig {
            duration: Duration::from_millis(50), // 50ms for testing
            threshold_violations: 3,
        };

        let graylist = create_test_graylist(config);
        let peer1 = create_test_peer_id();
        let peer2 = create_test_peer_id();

        // Add peers
        graylist.add(peer1, "Test 1".to_string()).await;
        graylist.add(peer2, "Test 2".to_string()).await;

        assert_eq!(graylist.size().await, 2, "Should have 2 graylisted peers");

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Cleanup
        graylist.cleanup_expired().await;

        assert_eq!(
            graylist.size().await,
            0,
            "All peers should be removed after cleanup"
        );
    }

    #[tokio::test]
    async fn test_graylist_time_remaining() {
        let config = GraylistConfig {
            duration: Duration::from_secs(60),
            threshold_violations: 3,
        };

        let graylist = create_test_graylist(config);
        let peer_id = create_test_peer_id();

        // Not graylisted - no time remaining
        assert_eq!(
            graylist.time_remaining(&peer_id).await,
            None,
            "Should be None for non-graylisted peer"
        );

        // Add to graylist
        graylist.add(peer_id, "Test".to_string()).await;

        // Should have time remaining (close to 60 seconds)
        let remaining = graylist.time_remaining(&peer_id).await;
        assert!(remaining.is_some(), "Should have time remaining");

        let remaining_secs = remaining.unwrap().as_secs();
        assert!(
            (59..=60).contains(&remaining_secs),
            "Remaining time should be ~60 seconds, got {}",
            remaining_secs
        );
    }

    #[tokio::test]
    async fn test_graylist_metrics_size() {
        let config = GraylistConfig::default();
        let graylist = create_test_graylist(config);

        // Initially 0
        assert_eq!(graylist.metrics.graylist_size.get(), 0.0);

        // Add peer
        let peer_id = create_test_peer_id();
        graylist.add(peer_id, "Test".to_string()).await;

        // Size should be 1
        assert_eq!(graylist.metrics.graylist_size.get(), 1.0);

        // Remove peer
        graylist.remove(&peer_id).await;

        // Size should be 0
        assert_eq!(graylist.metrics.graylist_size.get(), 0.0);
    }

    #[tokio::test]
    async fn test_graylist_metrics_graylisted() {
        let config = GraylistConfig::default();
        let graylist = create_test_graylist(config);
        let peer_id = create_test_peer_id();

        let initial_count = graylist.metrics.peers_graylisted.get();

        // Add peer
        graylist.add(peer_id, "Test".to_string()).await;

        let final_count = graylist.metrics.peers_graylisted.get();

        assert_eq!(
            final_count,
            initial_count + 1,
            "Graylisted metric should increment"
        );
    }

    #[tokio::test]
    async fn test_graylist_metrics_rejections() {
        let config = GraylistConfig::default();
        let graylist = create_test_graylist(config);
        let peer_id = create_test_peer_id();

        // Add peer
        graylist.add(peer_id, "Test".to_string()).await;

        let initial_rejections = graylist.metrics.graylist_rejections.get();

        // Check if graylisted (triggers rejection metric)
        let _ = graylist.is_graylisted(&peer_id).await;

        let final_rejections = graylist.metrics.graylist_rejections.get();

        assert_eq!(
            final_rejections,
            initial_rejections + 1,
            "Rejections metric should increment"
        );
    }
}
