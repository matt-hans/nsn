//! P2P discovery endpoint types and utilities
//!
//! Provides the `/p2p/info` HTTP endpoint response format and address filtering
//! for browser discovery of WebRTC connection details.

use libp2p::Multiaddr;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Discovery endpoint error types
#[derive(Debug, Error)]
pub enum P2pInfoError {
    #[error("Swarm not initialized")]
    SwarmNotReady,

    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Feature flags for the discovery response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2pFeatures {
    /// Whether WebRTC transport is enabled
    pub webrtc_enabled: bool,

    /// Whether WebSocket transport is enabled
    pub websocket_enabled: bool,

    /// Node role (director, validator, storage, supernode)
    pub role: String,
}

/// Discovery response data payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2pInfoData {
    /// Node's libp2p PeerId
    pub peer_id: String,

    /// Filtered multiaddrs (WebRTC with certhash prioritized)
    pub multiaddrs: Vec<String>,

    /// Supported libp2p protocols
    pub protocols: Vec<String>,

    /// Feature flags
    pub features: P2pFeatures,
}

/// Full discovery response envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2pInfoResponse {
    /// Success flag
    pub success: bool,

    /// Response data (present on success)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<P2pInfoData>,

    /// Error details (present on failure)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<P2pInfoErrorPayload>,
}

/// Error payload for failed responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2pInfoErrorPayload {
    /// Error code (SNAKE_CASE)
    pub code: String,

    /// Human-readable message
    pub message: String,
}

impl P2pInfoResponse {
    /// Create a successful response
    pub fn success(data: P2pInfoData) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    /// Create an error response
    pub fn error(code: &str, message: &str) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(P2pInfoErrorPayload {
                code: code.to_string(),
                message: message.to_string(),
            }),
        }
    }
}

/// Filter multiaddrs for browser consumption.
///
/// Filtering rules (per CONTEXT.md Decision 4):
/// 1. If external_address is configured, return ONLY that address
/// 2. Otherwise, filter out IPv6 link-local addresses (fe80::/10)
/// 3. Filter out unspecified IPs (0.0.0.0, ::)
/// 4. Keep localhost (127.x.x.x) for dev mode
/// 5. Keep RFC1918 private addresses (10.x, 172.16-31.x, 192.168.x)
///
/// # Arguments
/// * `listeners` - Iterator of listening multiaddrs from swarm
/// * `external_addrs` - Iterator of external addresses from swarm
/// * `external_override` - Optional configured external address (replaces all others)
pub fn filter_addresses<'a>(
    listeners: impl Iterator<Item = &'a Multiaddr>,
    external_addrs: impl Iterator<Item = &'a Multiaddr>,
    external_override: Option<&str>,
) -> Vec<String> {
    // If external address is configured, use ONLY that
    if let Some(ext) = external_override {
        if !ext.is_empty() {
            return vec![ext.to_string()];
        }
    }

    // Combine listeners and external addresses, prioritizing external
    let mut addrs: Vec<String> = external_addrs
        .chain(listeners)
        .filter(|addr| !is_link_local_ipv6(addr))
        .filter(|addr| !is_unspecified_ip(addr))
        .map(|addr| addr.to_string())
        .collect();

    // Deduplicate while preserving order
    let mut seen = std::collections::HashSet::new();
    addrs.retain(|addr| seen.insert(addr.clone()));

    addrs
}

/// Check if a multiaddr contains an IPv6 link-local address (fe80::/10)
fn is_link_local_ipv6(addr: &Multiaddr) -> bool {
    use libp2p::multiaddr::Protocol;

    for proto in addr.iter() {
        if let Protocol::Ip6(ip) = proto {
            // fe80::/10 - link-local unicast
            let segments = ip.segments();
            if (segments[0] & 0xffc0) == 0xfe80 {
                return true;
            }
        }
    }
    false
}

/// Check if a multiaddr contains an unspecified IP address (0.0.0.0 or ::)
fn is_unspecified_ip(addr: &Multiaddr) -> bool {
    use libp2p::multiaddr::Protocol;

    for proto in addr.iter() {
        match proto {
            Protocol::Ip4(ip) if ip.is_unspecified() => return true,
            Protocol::Ip6(ip) if ip.is_unspecified() => return true,
            _ => {}
        }
    }
    false
}

/// Check if a multiaddr is a WebRTC address
pub fn is_webrtc_address(addr: &str) -> bool {
    addr.contains("/webrtc") || addr.contains("/webrtc-direct")
}

/// Default protocols list for when swarm protocols aren't available
pub fn default_protocols() -> Vec<String> {
    vec![
        "/nsn/video/1.0.0".to_string(),
        "/ipfs/id/1.0.0".to_string(),
        "/ipfs/ping/1.0.0".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_success_response_format() {
        let data = P2pInfoData {
            peer_id: "12D3KooWExample".to_string(),
            multiaddrs: vec!["/ip4/1.2.3.4/udp/9003/webrtc-direct/certhash/uEiD...".to_string()],
            protocols: vec!["/nsn/video/1.0.0".to_string()],
            features: P2pFeatures {
                webrtc_enabled: true,
                role: "director".to_string(),
            },
        };

        let response = P2pInfoResponse::success(data);
        let json = serde_json::to_string(&response).expect("serialize");

        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"peer_id\""));
        assert!(json.contains("\"multiaddrs\""));
        assert!(json.contains("\"protocols\""));
        assert!(json.contains("\"webrtc_enabled\":true"));
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn test_success_response_has_protocols_field() {
        let data = P2pInfoData {
            peer_id: "12D3KooWExample".to_string(),
            multiaddrs: vec!["/ip4/1.2.3.4/udp/9003/webrtc-direct".to_string()],
            protocols: vec!["/nsn/video/1.0.0".to_string(), "/ipfs/id/1.0.0".to_string()],
            features: P2pFeatures {
                webrtc_enabled: true,
                role: "director".to_string(),
            },
        };

        let response = P2pInfoResponse::success(data);
        let json = serde_json::to_string(&response).expect("serialize");

        // Verify protocols field is present and contains expected values
        assert!(json.contains("\"protocols\""));
        assert!(json.contains("/nsn/video/1.0.0"));
        assert!(json.contains("/ipfs/id/1.0.0"));
    }

    #[test]
    fn test_error_response_format() {
        let response = P2pInfoResponse::error("NODE_INITIALIZING", "Swarm not ready");
        let json = serde_json::to_string(&response).expect("serialize");

        assert!(json.contains("\"success\":false"));
        assert!(json.contains("\"code\":\"NODE_INITIALIZING\""));
        assert!(json.contains("\"message\":\"Swarm not ready\""));
        assert!(!json.contains("\"data\""));
    }

    #[test]
    fn test_node_initializing_error_format() {
        // This tests the error response format for 503 NODE_INITIALIZING
        let response = P2pInfoResponse::error("NODE_INITIALIZING", "Swarm not ready, please retry");
        let json = serde_json::to_string(&response).expect("serialize");

        assert!(json.contains("\"success\":false"));
        assert!(json.contains("\"code\":\"NODE_INITIALIZING\""));
        assert!(json.contains("\"message\":\"Swarm not ready, please retry\""));
        assert!(!json.contains("\"data\""));
    }

    #[test]
    fn test_filter_external_override() {
        let listeners: Vec<Multiaddr> = vec![
            "/ip4/127.0.0.1/tcp/9000".parse().unwrap(),
            "/ip4/192.168.1.5/udp/9003/webrtc-direct".parse().unwrap(),
        ];

        let external: Vec<Multiaddr> = vec![];
        let override_addr = "/ip4/1.2.3.4/udp/9003/webrtc-direct";

        let result = filter_addresses(listeners.iter(), external.iter(), Some(override_addr));

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], override_addr);
    }

    #[test]
    fn test_filter_removes_link_local_ipv6() {
        let listeners: Vec<Multiaddr> = vec![
            "/ip6/fe80::1/tcp/9000".parse().unwrap(),
            "/ip4/127.0.0.1/tcp/9000".parse().unwrap(),
            "/ip6/2001:db8::1/tcp/9000".parse().unwrap(),
        ];

        let result = filter_addresses(listeners.iter(), std::iter::empty(), None);

        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|a| !a.contains("fe80")));
        assert!(result.iter().any(|a| a.contains("127.0.0.1")));
        assert!(result.iter().any(|a| a.contains("2001:db8")));
    }

    #[test]
    fn test_filter_removes_unspecified_ips() {
        let listeners: Vec<Multiaddr> = vec![
            "/ip4/0.0.0.0/udp/9003/webrtc-direct".parse().unwrap(),
            "/ip6/::/tcp/9000".parse().unwrap(),
            "/ip4/127.0.0.1/udp/9003/webrtc-direct".parse().unwrap(),
        ];

        let result = filter_addresses(listeners.iter(), std::iter::empty(), None);

        assert_eq!(result.len(), 1);
        assert!(result[0].contains("127.0.0.1"));
        assert!(!result.iter().any(|addr| addr.contains("0.0.0.0")));
    }

    #[test]
    fn test_filter_keeps_rfc1918() {
        let listeners: Vec<Multiaddr> = vec![
            "/ip4/10.0.0.1/tcp/9000".parse().unwrap(),
            "/ip4/172.16.0.1/tcp/9000".parse().unwrap(),
            "/ip4/192.168.1.1/tcp/9000".parse().unwrap(),
        ];

        let result = filter_addresses(listeners.iter(), std::iter::empty(), None);

        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_filter_deduplicates() {
        let listeners: Vec<Multiaddr> = vec!["/ip4/127.0.0.1/tcp/9000".parse().unwrap()];
        let external: Vec<Multiaddr> = vec!["/ip4/127.0.0.1/tcp/9000".parse().unwrap()];

        let result = filter_addresses(listeners.iter(), external.iter(), None);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_is_webrtc_address() {
        assert!(is_webrtc_address(
            "/ip4/1.2.3.4/udp/9003/webrtc-direct/certhash/uEiD"
        ));
        assert!(is_webrtc_address(
            "/ip4/1.2.3.4/udp/9003/webrtc/certhash/uEiD"
        ));
        assert!(!is_webrtc_address("/ip4/1.2.3.4/tcp/9000"));
        assert!(!is_webrtc_address("/ip4/1.2.3.4/udp/9000/quic-v1"));
    }

    #[test]
    fn test_default_protocols() {
        let protocols = default_protocols();
        assert!(protocols.contains(&"/nsn/video/1.0.0".to_string()));
        assert!(protocols.contains(&"/ipfs/id/1.0.0".to_string()));
    }
}
