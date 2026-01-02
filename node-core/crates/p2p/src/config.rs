//! P2P network configuration
//!
//! Defines configuration parameters for libp2p networking including ports,
//! connection limits, timeouts, and keypair paths.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use crate::security::SecureP2pConfig;

/// P2P network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2pConfig {
    /// Port to listen on for QUIC connections
    pub listen_port: u16,

    /// Maximum total number of concurrent connections
    pub max_connections: usize,

    /// Maximum connections per individual peer
    pub max_connections_per_peer: usize,

    /// Connection idle timeout duration
    #[serde(with = "humantime_serde")]
    pub connection_timeout: Duration,

    /// Optional path to persisted keypair file
    /// If None, generates ephemeral keypair
    pub keypair_path: Option<PathBuf>,

    /// Prometheus metrics server port
    pub metrics_port: u16,

    /// Enable UPnP port mapping for NAT traversal
    pub enable_upnp: bool,

    /// Enable circuit relay for NAT traversal
    pub enable_relay: bool,

    /// STUN servers for external IP discovery
    pub stun_servers: Vec<String>,

    /// Enable AutoNat for NAT status detection
    pub enable_autonat: bool,

    /// Security configuration (rate limiting, graylist, DoS detection)
    #[serde(default)]
    pub security: SecureP2pConfig,
}

impl Default for P2pConfig {
    fn default() -> Self {
        Self {
            listen_port: 9000,
            max_connections: 256,
            max_connections_per_peer: 2,
            connection_timeout: Duration::from_secs(30),
            keypair_path: None,
            metrics_port: 9100,
            enable_upnp: true,
            enable_relay: true,
            stun_servers: vec![
                "stun.l.google.com:19302".to_string(),
                "stun1.l.google.com:19302".to_string(),
                "stun2.l.google.com:19302".to_string(),
            ],
            enable_autonat: true,
            security: SecureP2pConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = P2pConfig::default();

        assert_eq!(config.listen_port, 9000);
        assert_eq!(config.max_connections, 256);
        assert_eq!(config.max_connections_per_peer, 2);
        assert_eq!(config.connection_timeout, Duration::from_secs(30));
        assert_eq!(config.metrics_port, 9100);
        assert!(config.keypair_path.is_none());
        assert!(config.enable_upnp);
        assert!(config.enable_relay);
        assert_eq!(config.stun_servers.len(), 3);
        assert!(config.enable_autonat);
    }

    #[test]
    fn test_config_serialization() {
        let config = P2pConfig {
            listen_port: 9001,
            max_connections: 512,
            max_connections_per_peer: 3,
            connection_timeout: Duration::from_secs(60),
            keypair_path: Some(PathBuf::from("/tmp/test.key")),
            metrics_port: 9101,
            enable_upnp: false,
            enable_relay: true,
            stun_servers: vec!["stun.example.com:19302".to_string()],
            enable_autonat: true,
            security: SecureP2pConfig::default(),
        };

        // Serialize to JSON
        let json = serde_json::to_string(&config).expect("Failed to serialize");

        // Deserialize back
        let deserialized: P2pConfig = serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(deserialized.listen_port, 9001);
        assert_eq!(deserialized.max_connections, 512);
        assert_eq!(deserialized.max_connections_per_peer, 3);
        assert_eq!(deserialized.connection_timeout, Duration::from_secs(60));
        assert_eq!(deserialized.metrics_port, 9101);
        assert_eq!(
            deserialized.keypair_path,
            Some(PathBuf::from("/tmp/test.key"))
        );
    }
}
