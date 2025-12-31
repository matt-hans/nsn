//! Peer Deduplication and Ranking
//!
//! Deduplicates peers from multiple bootstrap sources and ranks them by:
//! 1. Trust level (primary): Hardcoded > DNS > HTTP > DHT
//! 2. Latency (secondary): Lower latency preferred within same trust level

use super::PeerInfo;
use libp2p::PeerId;
use std::collections::HashMap;
use tracing::debug;

/// Deduplicate and rank peers
///
/// Merges peers from multiple sources, keeping:
/// - Highest trust level for each PeerId
/// - All unique multiaddrs from all sources
/// - Lowest latency measurement if available
///
/// Returns peers sorted by:
/// 1. Trust level (descending): Hardcoded > DNS > HTTP > DHT
/// 2. Latency (ascending): Lower latency first
pub fn deduplicate_and_rank(peers: Vec<PeerInfo>) -> Vec<PeerInfo> {
    let mut deduped: HashMap<PeerId, PeerInfo> = HashMap::new();

    for peer in peers {
        deduped
            .entry(peer.peer_id)
            .and_modify(|existing| {
                // Merge multiaddrs (deduplicate addresses)
                for addr in &peer.addrs {
                    if !existing.addrs.contains(addr) {
                        existing.addrs.push(addr.clone());
                    }
                }

                // Keep highest trust level
                if peer.trust_level > existing.trust_level {
                    existing.trust_level = peer.trust_level;
                    existing.signature = peer.signature.clone();
                }

                // Keep lowest latency
                match (existing.latency_ms, peer.latency_ms) {
                    (Some(existing_lat), Some(new_lat)) if new_lat < existing_lat => {
                        existing.latency_ms = Some(new_lat);
                    }
                    (None, Some(new_lat)) => {
                        existing.latency_ms = Some(new_lat);
                    }
                    _ => {}
                }
            })
            .or_insert(peer);
    }

    let mut result: Vec<_> = deduped.into_values().collect();

    // Sort by trust level (descending), then latency (ascending)
    result.sort_by(|a, b| {
        // Primary: Higher trust level first
        b.trust_level
            .cmp(&a.trust_level)
            // Secondary: Lower latency first (None is treated as infinity)
            .then_with(|| match (a.latency_ms, b.latency_ms) {
                (Some(a_lat), Some(b_lat)) => a_lat.cmp(&b_lat),
                (Some(_), None) => std::cmp::Ordering::Less, // Has latency < no latency
                (None, Some(_)) => std::cmp::Ordering::Greater, // No latency > has latency
                (None, None) => std::cmp::Ordering::Equal,
            })
    });

    debug!(
        "Deduplication complete: {} unique peers (from {} total)",
        result.len(),
        result.len()
    );

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap::TrustLevel;
    use libp2p::{Multiaddr, PeerId};

    #[test]
    fn test_deduplicate_and_rank_removes_duplicates() {
        let peer_id = PeerId::random();
        let addr1: Multiaddr = "/ip4/127.0.0.1/tcp/9000".parse().unwrap();
        let addr2: Multiaddr = "/ip4/127.0.0.1/udp/9000/quic-v1".parse().unwrap();

        let peers = vec![
            PeerInfo {
                peer_id,
                addrs: vec![addr1.clone()],
                trust_level: TrustLevel::DNS,
                signature: None,
                latency_ms: None,
            },
            PeerInfo {
                peer_id,
                addrs: vec![addr2.clone()],
                trust_level: TrustLevel::HTTP,
                signature: None,
                latency_ms: None,
            },
        ];

        let result = deduplicate_and_rank(peers);

        assert_eq!(result.len(), 1, "Should have only one unique peer");
        assert_eq!(result[0].peer_id, peer_id);
        assert_eq!(
            result[0].addrs.len(),
            2,
            "Should merge addrs from both sources"
        );
    }

    #[test]
    fn test_deduplicate_and_rank_keeps_highest_trust_level() {
        let peer_id = PeerId::random();
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/9000".parse().unwrap();

        let peers = vec![
            PeerInfo {
                peer_id,
                addrs: vec![addr.clone()],
                trust_level: TrustLevel::HTTP,
                signature: None,
                latency_ms: None,
            },
            PeerInfo {
                peer_id,
                addrs: vec![addr.clone()],
                trust_level: TrustLevel::Hardcoded,
                signature: None,
                latency_ms: None,
            },
            PeerInfo {
                peer_id,
                addrs: vec![addr],
                trust_level: TrustLevel::DHT,
                signature: None,
                latency_ms: None,
            },
        ];

        let result = deduplicate_and_rank(peers);

        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].trust_level,
            TrustLevel::Hardcoded,
            "Should keep highest trust level"
        );
    }

    #[test]
    fn test_deduplicate_and_rank_sorts_by_trust_level() {
        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        let peer3 = PeerId::random();
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/9000".parse().unwrap();

        let peers = vec![
            PeerInfo {
                peer_id: peer1,
                addrs: vec![addr.clone()],
                trust_level: TrustLevel::HTTP,
                signature: None,
                latency_ms: None,
            },
            PeerInfo {
                peer_id: peer2,
                addrs: vec![addr.clone()],
                trust_level: TrustLevel::Hardcoded,
                signature: None,
                latency_ms: None,
            },
            PeerInfo {
                peer_id: peer3,
                addrs: vec![addr],
                trust_level: TrustLevel::DNS,
                signature: None,
                latency_ms: None,
            },
        ];

        let result = deduplicate_and_rank(peers);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].trust_level, TrustLevel::Hardcoded);
        assert_eq!(result[1].trust_level, TrustLevel::DNS);
        assert_eq!(result[2].trust_level, TrustLevel::HTTP);
    }

    #[test]
    fn test_deduplicate_and_rank_sorts_by_latency_within_trust_level() {
        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        let peer3 = PeerId::random();
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/9000".parse().unwrap();

        let peers = vec![
            PeerInfo {
                peer_id: peer1,
                addrs: vec![addr.clone()],
                trust_level: TrustLevel::DNS,
                signature: None,
                latency_ms: Some(100),
            },
            PeerInfo {
                peer_id: peer2,
                addrs: vec![addr.clone()],
                trust_level: TrustLevel::DNS,
                signature: None,
                latency_ms: Some(50),
            },
            PeerInfo {
                peer_id: peer3,
                addrs: vec![addr],
                trust_level: TrustLevel::DNS,
                signature: None,
                latency_ms: Some(200),
            },
        ];

        let result = deduplicate_and_rank(peers);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].latency_ms, Some(50), "Lowest latency first");
        assert_eq!(result[1].latency_ms, Some(100));
        assert_eq!(result[2].latency_ms, Some(200));
    }

    #[test]
    fn test_deduplicate_and_rank_keeps_lowest_latency() {
        let peer_id = PeerId::random();
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/9000".parse().unwrap();

        let peers = vec![
            PeerInfo {
                peer_id,
                addrs: vec![addr.clone()],
                trust_level: TrustLevel::DNS,
                signature: None,
                latency_ms: Some(100),
            },
            PeerInfo {
                peer_id,
                addrs: vec![addr],
                trust_level: TrustLevel::DNS,
                signature: None,
                latency_ms: Some(50),
            },
        ];

        let result = deduplicate_and_rank(peers);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].latency_ms, Some(50), "Should keep lowest latency");
    }

    #[test]
    fn test_deduplicate_and_rank_peers_without_latency_sorted_last_within_trust_level() {
        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        let peer3 = PeerId::random();
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/9000".parse().unwrap();

        let peers = vec![
            PeerInfo {
                peer_id: peer1,
                addrs: vec![addr.clone()],
                trust_level: TrustLevel::DNS,
                signature: None,
                latency_ms: None,
            },
            PeerInfo {
                peer_id: peer2,
                addrs: vec![addr.clone()],
                trust_level: TrustLevel::DNS,
                signature: None,
                latency_ms: Some(100),
            },
            PeerInfo {
                peer_id: peer3,
                addrs: vec![addr],
                trust_level: TrustLevel::DNS,
                signature: None,
                latency_ms: Some(50),
            },
        ];

        let result = deduplicate_and_rank(peers);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].latency_ms, Some(50));
        assert_eq!(result[1].latency_ms, Some(100));
        assert_eq!(result[2].latency_ms, None, "No latency sorted last");
    }

    #[test]
    fn test_deduplicate_and_rank_merges_addrs_without_duplicates() {
        let peer_id = PeerId::random();
        let addr1: Multiaddr = "/ip4/127.0.0.1/tcp/9000".parse().unwrap();
        let addr2: Multiaddr = "/ip4/127.0.0.1/udp/9000/quic-v1".parse().unwrap();

        let peers = vec![
            PeerInfo {
                peer_id,
                addrs: vec![addr1.clone(), addr2.clone()],
                trust_level: TrustLevel::DNS,
                signature: None,
                latency_ms: None,
            },
            PeerInfo {
                peer_id,
                addrs: vec![addr1.clone()],
                trust_level: TrustLevel::HTTP,
                signature: None,
                latency_ms: None,
            },
        ];

        let result = deduplicate_and_rank(peers);

        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].addrs.len(),
            2,
            "Should have 2 unique addrs (no duplicates)"
        );
        assert!(result[0].addrs.contains(&addr1));
        assert!(result[0].addrs.contains(&addr2));
    }

    #[test]
    fn test_deduplicate_and_rank_empty_input() {
        let peers = vec![];
        let result = deduplicate_and_rank(peers);
        assert_eq!(result.len(), 0);
    }
}
