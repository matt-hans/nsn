//! Swarm event handlers
//!
//! Split swarm event handling into focused, single-purpose functions.

use super::connection_manager::{ConnectionError, ConnectionManager};
use libp2p::swarm::{NetworkBehaviour, SwarmEvent};
use libp2p::{Multiaddr, PeerId, Swarm};
use thiserror::Error;
use tracing::{debug, info, warn};

#[derive(Debug, Error)]
pub enum EventError {
    #[error("Connection error: {0}")]
    Connection(#[from] ConnectionError),
}

/// Handle new listen address event
pub fn handle_new_listen_addr(address: &Multiaddr) {
    info!("Listening on {}", address);
}

/// Handle connection established event
pub fn handle_connection_established<B: NetworkBehaviour>(
    peer_id: PeerId,
    connection_id: libp2p::swarm::ConnectionId,
    num_established: std::num::NonZeroU32,
    connection_manager: &mut ConnectionManager,
    swarm: &mut Swarm<B>,
) -> Result<(), EventError> {
    debug!(
        "Connection established to {} (connection_id: {}, num_established: {})",
        peer_id, connection_id, num_established
    );

    connection_manager
        .handle_connection_established(peer_id, connection_id, num_established, swarm)
        .map_err(EventError::Connection)
}

/// Handle connection closed event
pub fn handle_connection_closed(
    peer_id: PeerId,
    cause: Option<libp2p::swarm::ConnectionError>,
    connection_manager: &mut ConnectionManager,
) {
    debug!("Connection closed to {}: {:?}", peer_id, cause);
    connection_manager.handle_connection_closed(peer_id);
}

/// Handle outgoing connection error event
pub fn handle_outgoing_connection_error(
    peer_id: Option<PeerId>,
    error: &libp2p::swarm::DialError,
    connection_manager: &ConnectionManager,
) {
    warn!("Outgoing connection error to {:?}: {}", peer_id, error);
    connection_manager.handle_connection_failed();
}

/// Handle incoming connection error event
pub fn handle_incoming_connection_error(
    error: &libp2p::swarm::ListenError,
    connection_manager: &ConnectionManager,
) {
    warn!("Incoming connection error: {}", error);
    connection_manager.handle_connection_failed();
}

/// Main swarm event dispatcher
pub fn dispatch_swarm_event<TBehaviourEvent, B: NetworkBehaviour>(
    event: SwarmEvent<TBehaviourEvent>,
    connection_manager: &mut ConnectionManager,
    swarm: &mut Swarm<B>,
) -> Result<(), EventError> {
    match event {
        SwarmEvent::NewListenAddr { address, .. } => {
            handle_new_listen_addr(&address);
            Ok(())
        }

        SwarmEvent::ConnectionEstablished {
            peer_id,
            connection_id,
            num_established,
            ..
        } => handle_connection_established(
            peer_id,
            connection_id,
            num_established,
            connection_manager,
            swarm,
        ),

        SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
            handle_connection_closed(peer_id, cause, connection_manager);
            Ok(())
        }

        SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
            handle_outgoing_connection_error(peer_id, &error, connection_manager);
            Ok(())
        }

        SwarmEvent::IncomingConnectionError { error, .. } => {
            handle_incoming_connection_error(&error, connection_manager);
            Ok(())
        }

        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;

    #[test]
    fn test_handle_new_listen_addr() {
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/8080".parse().unwrap();
        handle_new_listen_addr(&addr); // Should not panic
    }

    #[test]
    fn test_handle_connection_closed() {
        let (mut manager, _metrics) = create_test_connection_manager();
        let peer_id = PeerId::random();

        handle_connection_closed(peer_id, None, &mut manager);
        assert_eq!(manager.tracker().total_connections(), 0);
    }

    #[test]
    fn test_handle_connection_failed() {
        let (manager, metrics) = create_test_connection_manager();
        let initial = metrics.connections_failed_total.get();

        handle_outgoing_connection_error(
            Some(PeerId::random()),
            &libp2p::swarm::DialError::Aborted,
            &manager,
        );

        assert_eq!(metrics.connections_failed_total.get(), initial + 1.0);
    }
}
