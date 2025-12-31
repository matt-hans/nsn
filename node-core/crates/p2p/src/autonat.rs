//! AutoNat integration
//!
//! Implements libp2p AutoNat for detecting NAT status and reachability.
//! AutoNat uses remote peers to probe connectivity and determine if a node
//! is publicly reachable or behind NAT.

use libp2p::autonat;
use std::time::Duration;

/// AutoNat configuration
#[derive(Debug, Clone)]
pub struct AutoNatConfig {
    /// Retry interval for failed probes
    pub retry_interval: Duration,

    /// Refresh interval for successful probes
    pub refresh_interval: Duration,

    /// Initial boot delay before first probe
    pub boot_delay: Duration,

    /// Throttle period for server responses
    pub throttle_server_period: Duration,

    /// Only allow trusted peers to probe
    pub only_global_ips: bool,
}

impl Default for AutoNatConfig {
    fn default() -> Self {
        Self {
            retry_interval: Duration::from_secs(30),
            refresh_interval: Duration::from_secs(300), // 5 minutes
            boot_delay: Duration::from_secs(5),
            throttle_server_period: Duration::from_secs(1),
            only_global_ips: true,
        }
    }
}

impl From<AutoNatConfig> for autonat::Config {
    fn from(config: AutoNatConfig) -> Self {
        autonat::Config {
            retry_interval: config.retry_interval,
            refresh_interval: config.refresh_interval,
            boot_delay: config.boot_delay,
            throttle_server_period: config.throttle_server_period,
            only_global_ips: config.only_global_ips,
            ..Default::default()
        }
    }
}

/// Build AutoNat behavior
///
/// # Arguments
/// * `peer_id` - Local peer ID
/// * `config` - AutoNat configuration
///
/// # Returns
/// Configured AutoNat behavior for inclusion in NetworkBehaviour
pub fn build_autonat(peer_id: libp2p::PeerId, config: AutoNatConfig) -> autonat::Behaviour {
    let autonat_config: autonat::Config = config.into();
    autonat::Behaviour::new(peer_id, autonat_config)
}

/// NAT status determined by AutoNat
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatStatus {
    /// Node is publicly reachable
    Public,

    /// Node is behind NAT (not publicly reachable)
    Private,

    /// NAT status unknown (not enough probe data)
    Unknown,
}

impl NatStatus {
    /// Check if node is publicly reachable
    pub fn is_public(&self) -> bool {
        matches!(self, NatStatus::Public)
    }

    /// Check if node is behind NAT
    pub fn is_private(&self) -> bool {
        matches!(self, NatStatus::Private)
    }

    /// Check if status is known
    pub fn is_known(&self) -> bool {
        !matches!(self, NatStatus::Unknown)
    }

    /// Convert to string for logging/metrics
    pub fn as_str(&self) -> &'static str {
        match self {
            NatStatus::Public => "public",
            NatStatus::Private => "private",
            NatStatus::Unknown => "unknown",
        }
    }
}

impl From<autonat::NatStatus> for NatStatus {
    fn from(status: autonat::NatStatus) -> Self {
        match status {
            autonat::NatStatus::Public(_) => NatStatus::Public,
            autonat::NatStatus::Private => NatStatus::Private,
            autonat::NatStatus::Unknown => NatStatus::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autonat_config_default() {
        let config = AutoNatConfig::default();

        assert_eq!(config.retry_interval, Duration::from_secs(30));
        assert_eq!(config.refresh_interval, Duration::from_secs(300));
        assert_eq!(config.boot_delay, Duration::from_secs(5));
        assert_eq!(config.throttle_server_period, Duration::from_secs(1));
        assert!(config.only_global_ips);
    }

    #[test]
    fn test_nat_status_predicates() {
        assert!(NatStatus::Public.is_public());
        assert!(!NatStatus::Public.is_private());
        assert!(NatStatus::Public.is_known());

        assert!(!NatStatus::Private.is_public());
        assert!(NatStatus::Private.is_private());
        assert!(NatStatus::Private.is_known());

        assert!(!NatStatus::Unknown.is_public());
        assert!(!NatStatus::Unknown.is_private());
        assert!(!NatStatus::Unknown.is_known());
    }

    #[test]
    fn test_nat_status_as_str() {
        assert_eq!(NatStatus::Public.as_str(), "public");
        assert_eq!(NatStatus::Private.as_str(), "private");
        assert_eq!(NatStatus::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_build_autonat() {
        let config = AutoNatConfig::default();
        let peer_id = libp2p::PeerId::random();
        let _autonat = build_autonat(peer_id, config);
        // Just verify it compiles and constructs without panicking
    }
}
