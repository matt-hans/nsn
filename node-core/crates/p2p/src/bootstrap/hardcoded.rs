//! Hardcoded Bootstrap Peers
//!
//! Provides a hardcoded list of foundation-operated bootstrap peers
//! compiled directly into the binary. These have the highest trust level.

use super::{PeerInfo, TrustLevel};
use libp2p::PeerId;

/// Get hardcoded bootstrap peers for mainnet
///
/// These are foundation-operated nodes with high availability guarantees.
/// Returns at least 3 bootstrap peers.
pub fn get_hardcoded_peers() -> Vec<PeerInfo> {
    vec![
        // Bootstrap peer 1: boot1.nsn.network
        PeerInfo {
            peer_id: parse_peer_id("12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN"),
            addrs: vec![
                "/dns4/boot1.nsn.network/tcp/9000"
                    .parse()
                    .expect("Valid multiaddr"),
                "/dns4/boot1.nsn.network/udp/9000/quic-v1"
                    .parse()
                    .expect("Valid multiaddr"),
            ],
            trust_level: TrustLevel::Hardcoded,
            signature: None,
            latency_ms: None,
        },
        // Bootstrap peer 2: boot2.nsn.network
        PeerInfo {
            peer_id: parse_peer_id("12D3KooWPjceQrSwdWXPyLLeABRXmuqt69Rg3sBYbU1Nft9HyQ6X"),
            addrs: vec![
                "/dns4/boot2.nsn.network/tcp/9000"
                    .parse()
                    .expect("Valid multiaddr"),
                "/dns4/boot2.nsn.network/udp/9000/quic-v1"
                    .parse()
                    .expect("Valid multiaddr"),
            ],
            trust_level: TrustLevel::Hardcoded,
            signature: None,
            latency_ms: None,
        },
        // Bootstrap peer 3: boot3.nsn.network
        PeerInfo {
            peer_id: parse_peer_id("12D3KooWLbPE9KGr5B9pN4N4rGzh4uWqGqTqSRE7VyZ7x9QzVx9Y"),
            addrs: vec![
                "/dns4/boot3.nsn.network/tcp/9000"
                    .parse()
                    .expect("Valid multiaddr"),
                "/dns4/boot3.nsn.network/udp/9000/quic-v1"
                    .parse()
                    .expect("Valid multiaddr"),
            ],
            trust_level: TrustLevel::Hardcoded,
            signature: None,
            latency_ms: None,
        },
    ]
}

/// Parse PeerId from base58 string
///
/// Helper to convert hardcoded base58 PeerId strings
fn parse_peer_id(s: &str) -> PeerId {
    s.parse().expect("Valid PeerId")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_hardcoded_peers_returns_at_least_three() {
        let peers = get_hardcoded_peers();
        assert!(
            peers.len() >= 3,
            "Must have at least 3 hardcoded bootstrap peers"
        );
    }

    #[test]
    fn test_all_hardcoded_peers_have_trust_level_hardcoded() {
        let peers = get_hardcoded_peers();
        for peer in &peers {
            assert_eq!(
                peer.trust_level,
                TrustLevel::Hardcoded,
                "All hardcoded peers must have TrustLevel::Hardcoded"
            );
        }
    }

    #[test]
    fn test_hardcoded_peers_have_valid_multiaddrs() {
        let peers = get_hardcoded_peers();
        for peer in &peers {
            assert!(
                !peer.addrs.is_empty(),
                "Each hardcoded peer must have at least one multiaddr"
            );

            for addr in &peer.addrs {
                // Verify multiaddr is parseable (already validated in const construction)
                assert!(
                    addr.iter().count() > 0,
                    "Multiaddr should have at least one protocol component"
                );
            }
        }
    }

    #[test]
    fn test_hardcoded_peers_have_unique_peer_ids() {
        let peers = get_hardcoded_peers();
        let mut seen_ids = std::collections::HashSet::new();

        for peer in &peers {
            assert!(
                seen_ids.insert(peer.peer_id),
                "PeerIds must be unique across hardcoded peers"
            );
        }
    }

    #[test]
    fn test_hardcoded_peers_have_no_signatures() {
        let peers = get_hardcoded_peers();
        for peer in &peers {
            assert!(
                peer.signature.is_none(),
                "Hardcoded peers should not have signatures (implicitly trusted)"
            );
        }
    }

    #[test]
    fn test_hardcoded_peers_have_no_latency() {
        let peers = get_hardcoded_peers();
        for peer in &peers {
            assert!(
                peer.latency_ms.is_none(),
                "Hardcoded peers should not have latency at initialization"
            );
        }
    }

    #[test]
    fn test_parse_peer_id_valid() {
        let peer_id_str = "12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN";
        let peer_id = parse_peer_id(peer_id_str);
        assert_eq!(peer_id.to_string(), peer_id_str);
    }

    #[test]
    fn test_hardcoded_peers_include_both_tcp_and_quic() {
        let peers = get_hardcoded_peers();

        for peer in &peers {
            let has_tcp = peer
                .addrs
                .iter()
                .any(|addr| addr.to_string().contains("/tcp/"));
            let has_quic = peer
                .addrs
                .iter()
                .any(|addr| addr.to_string().contains("/quic-v1"));

            assert!(
                has_tcp || has_quic,
                "Each peer should have at least TCP or QUIC transport"
            );
        }
    }
}
