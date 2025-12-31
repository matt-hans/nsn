//! Circuit Relay integration
//!
//! Implements libp2p Circuit Relay v2 for NAT traversal when direct connections
//! and STUN/UPnP fail. Relay nodes earn 0.01 NSN/hour for providing relay services.

use libp2p::relay;
use std::time::Duration;

/// Circuit relay reward per hour (NSN tokens)
pub const RELAY_REWARD_PER_HOUR: f64 = 0.01;

/// Configuration for relay server
#[derive(Debug, Clone)]
pub struct RelayServerConfig {
    /// Maximum number of reservations
    pub max_reservations: usize,

    /// Maximum number of active circuits
    pub max_circuits: usize,

    /// Maximum circuits per peer
    pub max_circuits_per_peer: usize,

    /// Reservation duration
    pub reservation_duration: Duration,

    /// Circuit duration
    pub circuit_duration: Duration,
}

impl Default for RelayServerConfig {
    fn default() -> Self {
        Self {
            max_reservations: 128,
            max_circuits: 16,
            max_circuits_per_peer: 4,
            reservation_duration: Duration::from_secs(3600), // 1 hour
            circuit_duration: Duration::from_secs(120),      // 2 minutes
        }
    }
}

impl From<RelayServerConfig> for relay::Config {
    fn from(config: RelayServerConfig) -> Self {
        relay::Config {
            max_reservations: config.max_reservations,
            max_circuits: config.max_circuits,
            max_circuits_per_peer: config.max_circuits_per_peer,
            reservation_duration: config.reservation_duration,
            max_circuit_duration: config.circuit_duration,
            max_circuit_bytes: 1024 * 1024 * 10, // 10MB
            ..Default::default()
        }
    }
}

/// Build libp2p relay server behavior
///
/// # Arguments
/// * `peer_id` - Local peer ID
/// * `config` - Relay server configuration
///
/// # Returns
/// Configured Relay behavior for inclusion in NetworkBehaviour
pub fn build_relay_server(peer_id: libp2p::PeerId, config: RelayServerConfig) -> relay::Behaviour {
    let relay_config: relay::Config = config.into();
    relay::Behaviour::new(peer_id, relay_config)
}

/// Configuration for relay client
#[derive(Debug, Clone)]
pub struct RelayClientConfig {
    /// Maximum number of relay circuits to maintain
    pub max_circuits: usize,
}

impl Default for RelayClientConfig {
    fn default() -> Self {
        Self { max_circuits: 4 }
    }
}

/// Track relay usage for reward calculation
#[derive(Debug, Clone)]
pub struct RelayUsageTracker {
    /// Total relay hours provided
    total_hours: f64,

    /// Total rewards earned (NSN)
    total_rewards: f64,
}

impl RelayUsageTracker {
    pub fn new() -> Self {
        Self {
            total_hours: 0.0,
            total_rewards: 0.0,
        }
    }

    /// Record relay usage duration
    ///
    /// # Arguments
    /// * `duration` - Duration of relay circuit
    ///
    /// # Returns
    /// Reward earned for this duration
    pub fn record_usage(&mut self, duration: Duration) -> f64 {
        let hours = duration.as_secs_f64() / 3600.0;
        let reward = hours * RELAY_REWARD_PER_HOUR;

        self.total_hours += hours;
        self.total_rewards += reward;

        tracing::debug!(
            "Relay usage recorded: {:.4}h = {:.6} NSN (total: {:.2}h, {:.4} NSN)",
            hours,
            reward,
            self.total_hours,
            self.total_rewards
        );

        reward
    }

    /// Get total relay hours provided
    pub fn total_hours(&self) -> f64 {
        self.total_hours
    }

    /// Get total rewards earned
    pub fn total_rewards(&self) -> f64 {
        self.total_rewards
    }
}

impl Default for RelayUsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relay_server_config_default() {
        let config = RelayServerConfig::default();

        assert_eq!(config.max_reservations, 128);
        assert_eq!(config.max_circuits, 16);
        assert_eq!(config.max_circuits_per_peer, 4);
        assert_eq!(config.reservation_duration, Duration::from_secs(3600));
        assert_eq!(config.circuit_duration, Duration::from_secs(120));
    }

    #[test]
    fn test_relay_client_config_default() {
        let config = RelayClientConfig::default();
        assert_eq!(config.max_circuits, 4);
    }

    #[test]
    fn test_relay_usage_tracker() {
        let mut tracker = RelayUsageTracker::new();

        assert_eq!(tracker.total_hours(), 0.0);
        assert_eq!(tracker.total_rewards(), 0.0);

        // Record 1 hour of usage
        let reward = tracker.record_usage(Duration::from_secs(3600));
        assert!((reward - 0.01).abs() < 1e-6); // 0.01 NSN/hour
        assert!((tracker.total_hours() - 1.0).abs() < 1e-6);
        assert!((tracker.total_rewards() - 0.01).abs() < 1e-6);

        // Record 30 minutes
        let reward = tracker.record_usage(Duration::from_secs(1800));
        assert!((reward - 0.005).abs() < 1e-6); // 0.005 NSN for 0.5h
        assert!((tracker.total_hours() - 1.5).abs() < 1e-6);
        assert!((tracker.total_rewards() - 0.015).abs() < 1e-6);
    }

    #[test]
    fn test_relay_reward_constant() {
        assert_eq!(RELAY_REWARD_PER_HOUR, 0.01);
    }

    #[test]
    fn test_build_relay_server() {
        let config = RelayServerConfig::default();
        let peer_id = libp2p::PeerId::random();
        let _relay = build_relay_server(peer_id, config);
        // Just verify it compiles and constructs without panicking
    }
}
