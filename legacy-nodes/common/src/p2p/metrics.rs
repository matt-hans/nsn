//! P2P network metrics
//!
//! Prometheus metrics for monitoring P2P network health including
//! active connections, peer count, and data transfer statistics.

use prometheus::{IntCounter, IntGauge, Registry, TextEncoder};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetricsError {
    #[error("Prometheus error: {0}")]
    Prometheus(#[from] prometheus::Error),

    #[error("Metrics encoding error")]
    EncodingError,
}

/// P2P network metrics
#[derive(Clone)]
pub struct P2pMetrics {
    /// Number of currently active connections
    pub active_connections: IntGauge,

    /// Number of unique connected peers
    pub connected_peers: IntGauge,

    /// Total bytes sent over P2P network
    pub bytes_sent_total: IntCounter,

    /// Total bytes received over P2P network
    pub bytes_received_total: IntCounter,

    /// Connection establishment successes
    pub connections_established_total: IntCounter,

    /// Connection failures
    pub connections_failed_total: IntCounter,

    /// Connection closed total
    pub connections_closed_total: IntCounter,

    /// Current connection limit
    pub connection_limit: IntGauge,

    // GossipSub metrics
    /// Number of topics subscribed to
    pub gossipsub_topics_subscribed: IntGauge,

    /// Total messages published
    pub gossipsub_messages_published_total: IntCounter,

    /// Total messages received
    pub gossipsub_messages_received_total: IntCounter,

    /// Total invalid messages rejected
    pub gossipsub_invalid_messages_total: IntCounter,

    /// Messages ignored from graylisted peers
    pub gossipsub_graylisted_messages_total: IntCounter,

    /// Current mesh peer count
    pub gossipsub_mesh_peers: IntGauge,

    /// Prometheus registry
    registry: Arc<Registry>,
}

impl P2pMetrics {
    /// Create new P2P metrics instance
    ///
    /// Registers all metrics with a new Prometheus registry.
    pub fn new() -> Result<Self, MetricsError> {
        let registry = Registry::new();

        let active_connections = IntGauge::new(
            "icn_p2p_active_connections",
            "Number of currently active P2P connections",
        )?;
        registry.register(Box::new(active_connections.clone()))?;

        let connected_peers = IntGauge::new(
            "icn_p2p_connected_peers",
            "Number of unique connected peers",
        )?;
        registry.register(Box::new(connected_peers.clone()))?;

        let bytes_sent_total = IntCounter::new(
            "icn_p2p_bytes_sent_total",
            "Total bytes sent over P2P network",
        )?;
        registry.register(Box::new(bytes_sent_total.clone()))?;

        let bytes_received_total = IntCounter::new(
            "icn_p2p_bytes_received_total",
            "Total bytes received over P2P network",
        )?;
        registry.register(Box::new(bytes_received_total.clone()))?;

        let connections_established_total = IntCounter::new(
            "icn_p2p_connections_established_total",
            "Total number of successful connection establishments",
        )?;
        registry.register(Box::new(connections_established_total.clone()))?;

        let connections_failed_total = IntCounter::new(
            "icn_p2p_connections_failed_total",
            "Total number of failed connection attempts",
        )?;
        registry.register(Box::new(connections_failed_total.clone()))?;

        let connections_closed_total = IntCounter::new(
            "icn_p2p_connections_closed_total",
            "Total number of closed connections",
        )?;
        registry.register(Box::new(connections_closed_total.clone()))?;

        let connection_limit = IntGauge::new(
            "icn_p2p_connection_limit",
            "Maximum allowed concurrent connections",
        )?;
        registry.register(Box::new(connection_limit.clone()))?;

        // GossipSub metrics
        let gossipsub_topics_subscribed = IntGauge::new(
            "nsn_gossipsub_topics_subscribed",
            "Number of GossipSub topics currently subscribed to",
        )?;
        registry.register(Box::new(gossipsub_topics_subscribed.clone()))?;

        let gossipsub_messages_published_total = IntCounter::new(
            "nsn_gossipsub_messages_published_total",
            "Total number of messages published to GossipSub topics",
        )?;
        registry.register(Box::new(gossipsub_messages_published_total.clone()))?;

        let gossipsub_messages_received_total = IntCounter::new(
            "nsn_gossipsub_messages_received_total",
            "Total number of messages received from GossipSub topics",
        )?;
        registry.register(Box::new(gossipsub_messages_received_total.clone()))?;

        let gossipsub_invalid_messages_total = IntCounter::new(
            "nsn_gossipsub_invalid_messages_total",
            "Total number of invalid messages rejected by GossipSub",
        )?;
        registry.register(Box::new(gossipsub_invalid_messages_total.clone()))?;

        let gossipsub_graylisted_messages_total = IntCounter::new(
            "nsn_gossipsub_graylisted_messages_total",
            "Total number of messages ignored from graylisted peers",
        )?;
        registry.register(Box::new(gossipsub_graylisted_messages_total.clone()))?;

        let gossipsub_mesh_peers = IntGauge::new(
            "nsn_gossipsub_mesh_peers",
            "Current number of peers in GossipSub mesh",
        )?;
        registry.register(Box::new(gossipsub_mesh_peers.clone()))?;

        Ok(Self {
            active_connections,
            connected_peers,
            bytes_sent_total,
            bytes_received_total,
            connections_established_total,
            connections_failed_total,
            connections_closed_total,
            connection_limit,
            gossipsub_topics_subscribed,
            gossipsub_messages_published_total,
            gossipsub_messages_received_total,
            gossipsub_invalid_messages_total,
            gossipsub_graylisted_messages_total,
            gossipsub_mesh_peers,
            registry: Arc::new(registry),
        })
    }

    /// Encode metrics in Prometheus text format
    ///
    /// # Returns
    /// String containing all metrics in Prometheus exposition format
    pub fn encode(&self) -> Result<String, MetricsError> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();

        encoder
            .encode_to_string(&metric_families)
            .map_err(|_| MetricsError::EncodingError)
    }

    /// Get registry for custom metrics registration
    pub fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }
}

impl Default for P2pMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create default P2pMetrics")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Use a mutex to serialize test execution to avoid Prometheus registry conflicts
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_metrics_creation() {
        let _guard = TEST_LOCK.lock().unwrap();
        let metrics = P2pMetrics::new().expect("Failed to create metrics");

        // Verify initial values
        assert_eq!(metrics.active_connections.get(), 0);
        assert_eq!(metrics.connected_peers.get(), 0);
        assert_eq!(metrics.bytes_sent_total.get(), 0);
        assert_eq!(metrics.bytes_received_total.get(), 0);
    }

    #[test]
    fn test_metrics_update() {
        let _guard = TEST_LOCK.lock().unwrap();
        let metrics = P2pMetrics::new().expect("Failed to create metrics");

        // Update metrics
        metrics.active_connections.set(5);
        metrics.connected_peers.set(3);
        metrics.bytes_sent_total.inc_by(1024);
        metrics.bytes_received_total.inc_by(2048);
        metrics.connections_established_total.inc();
        metrics.connection_limit.set(256);

        // Verify updates
        assert_eq!(metrics.active_connections.get(), 5);
        assert_eq!(metrics.connected_peers.get(), 3);
        assert_eq!(metrics.bytes_sent_total.get(), 1024);
        assert_eq!(metrics.bytes_received_total.get(), 2048);
        assert_eq!(metrics.connections_established_total.get(), 1);
        assert_eq!(metrics.connection_limit.get(), 256);
    }

    #[test]
    fn test_metrics_encoding() {
        let _guard = TEST_LOCK.lock().unwrap();
        let metrics = P2pMetrics::new().expect("Failed to create metrics");

        metrics.active_connections.set(10);
        metrics.connected_peers.set(5);

        let encoded = metrics.encode().expect("Failed to encode metrics");

        // Verify Prometheus format
        assert!(encoded.contains("icn_p2p_active_connections"));
        assert!(encoded.contains("icn_p2p_connected_peers"));
        assert!(encoded.contains("10")); // active_connections value
        assert!(encoded.contains("5")); // connected_peers value
    }

    #[test]
    fn test_metrics_clone() {
        let _guard = TEST_LOCK.lock().unwrap();
        let metrics1 = P2pMetrics::new().expect("Failed to create metrics");
        metrics1.active_connections.set(42);

        let metrics2 = metrics1.clone();

        // Both should reference the same underlying metrics
        assert_eq!(metrics2.active_connections.get(), 42);

        metrics2.active_connections.set(100);
        assert_eq!(metrics1.active_connections.get(), 100);
    }
}
