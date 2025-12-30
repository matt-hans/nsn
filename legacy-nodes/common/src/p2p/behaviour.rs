//! P2P network behaviour
//!
//! Defines the libp2p NetworkBehaviour for NSN nodes with
//! GossipSub messaging and connection management.

use libp2p::gossipsub;
use libp2p::swarm::NetworkBehaviour;
use libp2p::PeerId;
use std::collections::HashMap;

/// P2P network behaviour with GossipSub
///
/// Combines GossipSub for pub/sub messaging with future protocols
/// (Kademlia DHT, Request-Response, etc.)
#[derive(NetworkBehaviour)]
pub struct NsnBehaviour {
    /// GossipSub pub/sub messaging
    pub gossipsub: gossipsub::Behaviour,
}

impl NsnBehaviour {
    /// Create new NSN behaviour with GossipSub
    ///
    /// # Arguments
    /// * `gossipsub` - Configured GossipSub behavior
    pub fn new(gossipsub: gossipsub::Behaviour) -> Self {
        Self { gossipsub }
    }

    /// Create a simple behaviour for testing (uses default GossipSub config)
    #[cfg(test)]
    pub fn new_for_testing(keypair: &libp2p::identity::Keypair) -> Self {
        use libp2p::gossipsub::{ConfigBuilder, MessageAuthenticity};

        let config = ConfigBuilder::default()
            .build()
            .expect("Default config should be valid");

        let gossipsub =
            gossipsub::Behaviour::new(MessageAuthenticity::Signed(keypair.clone()), config)
                .expect("GossipSub creation should succeed");

        Self { gossipsub }
    }
}

/// Connection tracker for managing per-peer connection counts
#[derive(Debug, Clone, Default)]
pub struct ConnectionTracker {
    connections_per_peer: HashMap<PeerId, usize>,
}

impl ConnectionTracker {
    pub fn new() -> Self {
        Self {
            connections_per_peer: HashMap::new(),
        }
    }

    /// Track a new connection to a peer
    pub fn connection_established(&mut self, peer_id: PeerId) {
        *self.connections_per_peer.entry(peer_id).or_insert(0) += 1;
    }

    /// Untrack a closed connection to a peer
    pub fn connection_closed(&mut self, peer_id: &PeerId) {
        if let Some(count) = self.connections_per_peer.get_mut(peer_id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.connections_per_peer.remove(peer_id);
            }
        }
    }

    /// Get number of connections to a specific peer
    pub fn connections_to_peer(&self, peer_id: &PeerId) -> usize {
        self.connections_per_peer.get(peer_id).copied().unwrap_or(0)
    }

    /// Get total number of active connections
    pub fn total_connections(&self) -> usize {
        self.connections_per_peer.values().sum()
    }

    /// Get number of unique connected peers
    pub fn connected_peers(&self) -> usize {
        self.connections_per_peer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    #[test]
    fn test_connection_tracker() {
        let mut tracker = ConnectionTracker::new();

        let keypair1 = Keypair::generate_ed25519();
        let peer1 = PeerId::from(keypair1.public());

        let keypair2 = Keypair::generate_ed25519();
        let peer2 = PeerId::from(keypair2.public());

        // Establish connections
        tracker.connection_established(peer1);
        assert_eq!(tracker.connections_to_peer(&peer1), 1);
        assert_eq!(tracker.total_connections(), 1);
        assert_eq!(tracker.connected_peers(), 1);

        tracker.connection_established(peer2);
        assert_eq!(tracker.connections_to_peer(&peer2), 1);
        assert_eq!(tracker.total_connections(), 2);
        assert_eq!(tracker.connected_peers(), 2);

        // Multiple connections to same peer
        tracker.connection_established(peer1);
        assert_eq!(tracker.connections_to_peer(&peer1), 2);
        assert_eq!(tracker.total_connections(), 3);
        assert_eq!(tracker.connected_peers(), 2);

        // Close connections
        tracker.connection_closed(&peer1);
        assert_eq!(tracker.connections_to_peer(&peer1), 1);
        assert_eq!(tracker.total_connections(), 2);
        assert_eq!(tracker.connected_peers(), 2);

        tracker.connection_closed(&peer1);
        assert_eq!(tracker.connections_to_peer(&peer1), 0);
        assert_eq!(tracker.total_connections(), 1);
        assert_eq!(tracker.connected_peers(), 1);

        tracker.connection_closed(&peer2);
        assert_eq!(tracker.total_connections(), 0);
        assert_eq!(tracker.connected_peers(), 0);
    }

    #[test]
    fn test_connection_closed_idempotent() {
        let mut tracker = ConnectionTracker::new();

        let keypair = Keypair::generate_ed25519();
        let peer = PeerId::from(keypair.public());

        // Close connection that was never opened
        tracker.connection_closed(&peer);
        assert_eq!(tracker.connections_to_peer(&peer), 0);

        // Establish and close
        tracker.connection_established(peer);
        tracker.connection_closed(&peer);
        tracker.connection_closed(&peer); // Second close should be safe
        assert_eq!(tracker.connections_to_peer(&peer), 0);
    }
}
