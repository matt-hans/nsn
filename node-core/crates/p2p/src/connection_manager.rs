//! Connection management and limit enforcement
//!
//! Handles connection tracking and enforces per-peer and global connection limits.

use super::behaviour::ConnectionTracker;
use super::config::P2pConfig;
use super::metrics::P2pMetrics;
use libp2p::swarm::{ConnectionId, NetworkBehaviour};
use libp2p::{PeerId, Swarm};
use std::sync::Arc;
use thiserror::Error;
use tracing::{info, warn};

#[derive(Debug, Error)]
pub enum ConnectionError {
    #[error("Connection limit reached: {current}/{max}")]
    LimitReached { current: usize, max: usize },

    #[error("Per-peer connection limit reached for {peer_id}: {current}/{max}")]
    PerPeerLimitReached {
        peer_id: PeerId,
        current: usize,
        max: usize,
    },
}

/// Manages connection lifecycle and enforces limits
pub struct ConnectionManager {
    pub(crate) tracker: ConnectionTracker,
    config: P2pConfig,
    metrics: Arc<P2pMetrics>,
}

impl ConnectionManager {
    /// Create a new connection manager
    pub fn new(config: P2pConfig, metrics: Arc<P2pMetrics>) -> Self {
        Self {
            tracker: ConnectionTracker::new(),
            config,
            metrics,
        }
    }

    /// Get the connection tracker
    pub fn tracker(&self) -> &ConnectionTracker {
        &self.tracker
    }

    /// Handle connection established event
    ///
    /// Returns Err if connection limits are exceeded and connection should be closed
    pub fn handle_connection_established<B: NetworkBehaviour>(
        &mut self,
        peer_id: PeerId,
        connection_id: ConnectionId,
        num_established: std::num::NonZeroU32,
        swarm: &mut Swarm<B>,
    ) -> Result<(), ConnectionError> {
        // Check global connection limit
        if self.tracker.total_connections() >= self.config.max_connections {
            warn!(
                "Connection limit reached ({}/{}), closing connection to {}",
                self.tracker.total_connections(),
                self.config.max_connections,
                peer_id
            );
            let _ = swarm.close_connection(connection_id);
            return Err(ConnectionError::LimitReached {
                current: self.tracker.total_connections(),
                max: self.config.max_connections,
            });
        }

        // Check per-peer connection limit
        if num_established.get() as usize > self.config.max_connections_per_peer {
            warn!(
                "Per-peer connection limit reached for {} ({}/{})",
                peer_id,
                num_established.get(),
                self.config.max_connections_per_peer
            );
            let _ = swarm.close_connection(connection_id);
            return Err(ConnectionError::PerPeerLimitReached {
                peer_id,
                current: num_established.get() as usize,
                max: self.config.max_connections_per_peer,
            });
        }

        // Track connection
        self.tracker.connection_established(peer_id);

        // Update metrics
        self.metrics.connections_established_total.inc();
        self.metrics
            .active_connections
            .set(self.tracker.total_connections() as f64);
        self.metrics
            .connected_peers
            .set(self.tracker.connected_peers() as f64);

        info!(
            "Connected to {} (total: {}, peers: {})",
            peer_id,
            self.tracker.total_connections(),
            self.tracker.connected_peers()
        );

        Ok(())
    }

    /// Handle connection closed event
    pub fn handle_connection_closed(&mut self, peer_id: PeerId) {
        // Untrack connection
        self.tracker.connection_closed(&peer_id);

        // Update metrics
        self.metrics.connections_closed_total.inc();
        self.metrics
            .active_connections
            .set(self.tracker.total_connections() as f64);
        self.metrics
            .connected_peers
            .set(self.tracker.connected_peers() as f64);

        info!(
            "Disconnected from {} (total: {}, peers: {})",
            peer_id,
            self.tracker.total_connections(),
            self.tracker.connected_peers()
        );
    }

    /// Record connection failure
    pub fn handle_connection_failed(&self) {
        self.metrics.connections_failed_total.inc();
    }

    /// Reset all connection tracking (for graceful shutdown)
    pub fn reset(&mut self) {
        self.metrics.active_connections.set(0.0);
        self.metrics.connected_peers.set(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;

    #[test]
    fn test_connection_manager_creation() {
        let (manager, _metrics) = create_test_connection_manager();
        assert_eq!(manager.tracker().total_connections(), 0);
        assert_eq!(manager.tracker().connected_peers(), 0);
    }

    #[test]
    fn test_connection_closed_updates_metrics() {
        let (mut manager, metrics) = create_test_connection_manager();

        let peer_id = PeerId::random();
        manager.tracker.connection_established(peer_id);
        metrics.active_connections.set(1.0);
        metrics.connected_peers.set(1.0);

        manager.handle_connection_closed(peer_id);

        assert_eq!(manager.tracker().total_connections(), 0);
        assert_eq!(manager.tracker().connected_peers(), 0);
        assert_eq!(metrics.active_connections.get(), 0.0);
        assert_eq!(metrics.connected_peers.get(), 0.0);
    }

    #[test]
    fn test_connection_failed_increments_metric() {
        let (manager, metrics) = create_test_connection_manager();

        let initial = metrics.connections_failed_total.get();
        manager.handle_connection_failed();
        assert_eq!(metrics.connections_failed_total.get(), initial + 1.0);
    }

    #[test]
    fn test_global_connection_limit_enforced() {
        use std::num::NonZeroU32;

        let config = test_config_with_limits(3, 4);
        let (mut manager, _metrics) = create_test_connection_manager_with_config(config);

        let keypair = generate_test_keypair();
        let mut swarm = create_test_swarm(&keypair);

        // Establish connections up to limit
        for i in 0..3 {
            let peer_id = PeerId::random();
            let conn_id = libp2p::swarm::ConnectionId::new_unchecked(i);
            let num_established = NonZeroU32::new(1).unwrap();

            let result = manager.handle_connection_established(
                peer_id,
                conn_id,
                num_established,
                &mut swarm,
            );

            assert!(
                result.is_ok(),
                "Connection {} should succeed (under limit)",
                i
            );
        }

        assert_eq!(manager.tracker().total_connections(), 3);

        // Try to exceed limit
        let peer_id = PeerId::random();
        let conn_id = libp2p::swarm::ConnectionId::new_unchecked(3);
        let num_established = NonZeroU32::new(1).unwrap();

        let result =
            manager.handle_connection_established(peer_id, conn_id, num_established, &mut swarm);

        assert!(
            result.is_err(),
            "Connection should be rejected (limit reached)"
        );

        if let Err(ConnectionError::LimitReached { current, max }) = result {
            assert_eq!(current, 3, "Current connections should be 3");
            assert_eq!(max, 3, "Max connections should be 3");
        } else {
            panic!("Expected LimitReached error");
        }

        // Verify total connections stayed at limit
        assert_eq!(
            manager.tracker().total_connections(),
            3,
            "Total connections should not exceed limit"
        );
    }

    #[test]
    fn test_per_peer_connection_limit_enforced() {
        use std::num::NonZeroU32;

        let config = test_config_with_limits(256, 2);
        let (mut manager, _metrics) = create_test_connection_manager_with_config(config);

        let keypair = generate_test_keypair();
        let mut swarm = create_test_swarm(&keypair);

        // Same peer opens 2 connections (should succeed)
        let peer_id = PeerId::random();

        // First connection
        let conn_id_1 = libp2p::swarm::ConnectionId::new_unchecked(0);
        let num_established_1 = NonZeroU32::new(1).unwrap();
        let result = manager.handle_connection_established(
            peer_id,
            conn_id_1,
            num_established_1,
            &mut swarm,
        );
        assert!(result.is_ok(), "First connection should succeed");

        // Second connection (num_established = 2)
        let conn_id_2 = libp2p::swarm::ConnectionId::new_unchecked(1);
        let num_established_2 = NonZeroU32::new(2).unwrap();
        let result = manager.handle_connection_established(
            peer_id,
            conn_id_2,
            num_established_2,
            &mut swarm,
        );
        assert!(result.is_ok(), "Second connection should succeed");

        // Third connection (num_established = 3, exceeds per-peer limit)
        let conn_id_3 = libp2p::swarm::ConnectionId::new_unchecked(2);
        let num_established_3 = NonZeroU32::new(3).unwrap();
        let result = manager.handle_connection_established(
            peer_id,
            conn_id_3,
            num_established_3,
            &mut swarm,
        );

        assert!(
            result.is_err(),
            "Third connection should be rejected (per-peer limit)"
        );

        if let Err(ConnectionError::PerPeerLimitReached {
            peer_id: returned_peer,
            current,
            max,
        }) = result
        {
            assert_eq!(returned_peer, peer_id);
            assert_eq!(current, 3, "Current per-peer connections should be 3");
            assert_eq!(max, 2, "Max per-peer connections should be 2");
        } else {
            panic!("Expected PerPeerLimitReached error");
        }
    }

    #[test]
    fn test_connection_limit_error_messages() {
        // Test error message formatting
        let err = ConnectionError::LimitReached {
            current: 256,
            max: 256,
        };
        assert_eq!(
            err.to_string(),
            "Connection limit reached: 256/256",
            "Global limit error message format"
        );

        let peer_id = PeerId::random();
        let err = ConnectionError::PerPeerLimitReached {
            peer_id,
            current: 3,
            max: 2,
        };
        assert!(
            err.to_string()
                .contains("Per-peer connection limit reached"),
            "Per-peer limit error message"
        );
        assert!(
            err.to_string().contains("3/2"),
            "Error should show current/max"
        );
    }
}
