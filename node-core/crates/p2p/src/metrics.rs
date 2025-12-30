//! P2P Prometheus metrics
//!
//! Provides metrics for monitoring P2P network health, connection status,
//! and message throughput.

use prometheus::{Counter, Gauge, Histogram, HistogramOpts, Registry};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetricsError {
    #[error("Failed to register metric: {0}")]
    Registration(#[from] prometheus::Error),
}

/// P2P metrics collection
#[derive(Debug)]
pub struct P2pMetrics {
    /// Number of currently active connections
    pub active_connections: Gauge,

    /// Number of unique connected peers
    pub connected_peers: Gauge,

    /// Configured connection limit
    pub connection_limit: Gauge,

    /// Total connections established (cumulative)
    pub connections_established_total: Counter,

    /// Total connections closed (cumulative)
    pub connections_closed_total: Counter,

    /// Total connection failures (cumulative)
    pub connections_failed_total: Counter,

    /// Connection duration histogram
    pub connection_duration_seconds: Histogram,

    /// Number of GossipSub messages sent
    pub gossipsub_messages_sent_total: Counter,

    /// Number of GossipSub messages received
    pub gossipsub_messages_received_total: Counter,

    /// Number of GossipSub messages failed to publish
    pub gossipsub_publish_failures_total: Counter,

    /// GossipSub mesh size by topic
    pub gossipsub_mesh_size: Gauge,

    /// Prometheus registry for this metrics instance
    pub registry: Registry,
}

impl P2pMetrics {
    /// Create new P2P metrics with a dedicated registry
    ///
    /// This uses a per-instance registry to avoid conflicts when creating
    /// multiple instances (e.g., in tests running in parallel).
    pub fn new() -> Result<Self, MetricsError> {
        let registry = Registry::new_custom(Some("nsn_p2p".to_string()), None)?;

        let active_connections = Gauge::new(
            "nsn_p2p_active_connections",
            "Number of currently active P2P connections",
        )?;
        registry.register(Box::new(active_connections.clone()))?;

        let connected_peers = Gauge::new(
            "nsn_p2p_connected_peers",
            "Number of unique connected peers",
        )?;
        registry.register(Box::new(connected_peers.clone()))?;

        let connection_limit = Gauge::new(
            "nsn_p2p_connection_limit",
            "Configured maximum number of connections",
        )?;
        registry.register(Box::new(connection_limit.clone()))?;

        let connections_established_total = Counter::new(
            "nsn_p2p_connections_established_total",
            "Total number of connections established",
        )?;
        registry.register(Box::new(connections_established_total.clone()))?;

        let connections_closed_total = Counter::new(
            "nsn_p2p_connections_closed_total",
            "Total number of connections closed",
        )?;
        registry.register(Box::new(connections_closed_total.clone()))?;

        let connections_failed_total = Counter::new(
            "nsn_p2p_connections_failed_total",
            "Total number of connection failures",
        )?;
        registry.register(Box::new(connections_failed_total.clone()))?;

        let connection_duration_seconds = Histogram::with_opts(HistogramOpts::new(
            "nsn_p2p_connection_duration_seconds",
            "Duration of P2P connections in seconds",
        ))?;
        registry.register(Box::new(connection_duration_seconds.clone()))?;

        let gossipsub_messages_sent_total = Counter::new(
            "nsn_p2p_gossipsub_messages_sent_total",
            "Total number of GossipSub messages sent",
        )?;
        registry.register(Box::new(gossipsub_messages_sent_total.clone()))?;

        let gossipsub_messages_received_total = Counter::new(
            "nsn_p2p_gossipsub_messages_received_total",
            "Total number of GossipSub messages received",
        )?;
        registry.register(Box::new(gossipsub_messages_received_total.clone()))?;

        let gossipsub_publish_failures_total = Counter::new(
            "nsn_p2p_gossipsub_publish_failures_total",
            "Total number of GossipSub publish failures",
        )?;
        registry.register(Box::new(gossipsub_publish_failures_total.clone()))?;

        let gossipsub_mesh_size = Gauge::new(
            "nsn_p2p_gossipsub_mesh_size",
            "Current GossipSub mesh size across all topics",
        )?;
        registry.register(Box::new(gossipsub_mesh_size.clone()))?;

        Ok(Self {
            active_connections,
            connected_peers,
            connection_limit,
            connections_established_total,
            connections_closed_total,
            connections_failed_total,
            connection_duration_seconds,
            gossipsub_messages_sent_total,
            gossipsub_messages_received_total,
            gossipsub_publish_failures_total,
            gossipsub_mesh_size,
            registry,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = P2pMetrics::new().expect("Failed to create metrics");

        // Verify initial values
        assert_eq!(metrics.active_connections.get(), 0.0);
        assert_eq!(metrics.connected_peers.get(), 0.0);
        assert_eq!(metrics.connections_established_total.get(), 0.0);
        assert_eq!(metrics.connections_closed_total.get(), 0.0);
        assert_eq!(metrics.connections_failed_total.get(), 0.0);
    }

    #[test]
    fn test_metrics_update() {
        let metrics = P2pMetrics::new().expect("Failed to create metrics");

        // Update metrics
        metrics.active_connections.set(5.0);
        metrics.connected_peers.set(3.0);
        metrics.connections_established_total.inc();
        metrics.connections_failed_total.inc_by(2.0);

        // Verify updates
        assert_eq!(metrics.active_connections.get(), 5.0);
        assert_eq!(metrics.connected_peers.get(), 3.0);
        assert_eq!(metrics.connections_established_total.get(), 1.0);
        assert_eq!(metrics.connections_failed_total.get(), 2.0);
    }
}
