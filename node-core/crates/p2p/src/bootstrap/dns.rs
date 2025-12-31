//! DNS TXT Record Bootstrap with Signature Verification
//!
//! Resolves DNS TXT records for bootstrap peer discovery.
//! Format: `nsn:peer:<multiaddr>:sig:<hex_signature>`

use super::{signature::verify_signature, BootstrapError, PeerInfo, TrustLevel};
use libp2p::identity::PublicKey;
use libp2p::{multiaddr::Protocol, Multiaddr};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};
use trust_dns_resolver::config::{ResolverConfig, ResolverOpts};
use trust_dns_resolver::TokioAsyncResolver;

/// Resolve DNS TXT records for bootstrap peers
///
/// # Arguments
/// * `dns_seed` - DNS name to query (e.g., "_nsn-bootstrap._tcp.nsn.network")
/// * `trusted_signers` - Set of trusted signer public keys
/// * `require_signatures` - If true, reject unsigned records
/// * `timeout` - DNS query timeout
///
/// # Returns
/// Vector of discovered and verified peers
pub async fn resolve_dns_seed(
    dns_seed: &str,
    trusted_signers: Arc<HashSet<PublicKey>>,
    require_signatures: bool,
    timeout: Duration,
) -> Result<Vec<PeerInfo>, BootstrapError> {
    // Create DNS resolver with custom timeout
    let mut opts = ResolverOpts::default();
    opts.timeout = timeout;

    let resolver = TokioAsyncResolver::tokio(ResolverConfig::default(), opts);

    // Query TXT records
    let response = tokio::time::timeout(timeout, resolver.txt_lookup(dns_seed))
        .await
        .map_err(|_| BootstrapError::DnsResolutionFailed("Timeout".to_string()))?
        .map_err(|e| BootstrapError::DnsResolutionFailed(e.to_string()))?;

    let mut peers = Vec::new();

    for record in response.iter() {
        // Concatenate TXT record data (may be split across multiple strings)
        let txt_data: String = record
            .iter()
            .map(|data| String::from_utf8_lossy(data.as_ref()).to_string())
            .collect();

        debug!("Parsing DNS TXT record: {}", txt_data);

        match parse_dns_record(&txt_data, &trusted_signers, require_signatures) {
            Ok(Some(peer_info)) => {
                peers.push(peer_info);
            }
            Ok(None) => {
                debug!("Skipping non-NSN DNS record: {}", txt_data);
            }
            Err(e) => {
                warn!("Invalid DNS record: {} - error: {}", txt_data, e);
            }
        }
    }

    debug!("Resolved {} peers from DNS seed {}", peers.len(), dns_seed);

    Ok(peers)
}

/// Parse DNS TXT record
///
/// Format: `nsn:peer:<multiaddr>:sig:<hex_signature>`
///
/// # Arguments
/// * `record` - TXT record string
/// * `trusted_signers` - Trusted signer public keys
/// * `require_signatures` - Reject unsigned records if true
fn parse_dns_record(
    record: &str,
    trusted_signers: &HashSet<PublicKey>,
    require_signatures: bool,
) -> Result<Option<PeerInfo>, BootstrapError> {
    // Split by colons
    let parts: Vec<&str> = record.split(':').collect();

    // Format: nsn:peer:<multiaddr>:sig:<hex_signature>
    if parts.len() < 3 {
        return Ok(None); // Not an NSN record
    }

    if parts[0] != "nsn" || parts[1] != "peer" {
        return Ok(None); // Not an NSN peer record
    }

    // Extract multiaddr (may contain colons, so rejoin)
    let multiaddr_end = if parts.len() >= 5 && parts[parts.len() - 2] == "sig" {
        parts.len() - 2
    } else {
        parts.len()
    };

    let multiaddr_str: String = parts[2..multiaddr_end].join(":");
    let multiaddr: Multiaddr = multiaddr_str
        .parse()
        .map_err(|_| BootstrapError::InvalidMultiaddr)?;

    // Extract PeerId from multiaddr (if present)
    let peer_id = multiaddr
        .iter()
        .find_map(|proto| match proto {
            Protocol::P2p(peer_id) => Some(peer_id),
            _ => None,
        })
        .ok_or(BootstrapError::NoPeerIdInMultiaddr)?;

    // Extract and verify signature (if present)
    let signature = if parts.len() >= 5 && parts[parts.len() - 2] == "sig" {
        let sig_hex = parts[parts.len() - 1];
        let sig_bytes = hex::decode(sig_hex).map_err(|_| BootstrapError::InvalidSignature)?;

        // Verify signature
        let message = format!("{}:{}", peer_id, multiaddr_str);
        if !verify_signature(message.as_bytes(), &sig_bytes, trusted_signers) {
            warn!("DNS record signature verification failed for {}", peer_id);
            return Err(BootstrapError::InvalidSignature);
        }

        Some(sig_bytes)
    } else if require_signatures {
        warn!("DNS record missing required signature for {}", peer_id);
        return Err(BootstrapError::InvalidSignature);
    } else {
        None
    };

    Ok(Some(PeerInfo {
        peer_id,
        addrs: vec![multiaddr],
        trust_level: TrustLevel::DNS,
        signature,
        latency_ms: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;
    use libp2p::PeerId;

    #[test]
    fn test_parse_dns_record_valid_without_signature() {
        let record = "nsn:peer:/ip4/127.0.0.1/tcp/9000/p2p/12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN";
        let signers = HashSet::new();

        let result = parse_dns_record(record, &signers, false);
        assert!(result.is_ok());

        let peer_info = result.unwrap();
        assert!(peer_info.is_some());

        let peer_info = peer_info.unwrap();
        assert_eq!(peer_info.trust_level, TrustLevel::DNS);
        assert_eq!(peer_info.addrs.len(), 1);
        assert!(peer_info.signature.is_none());
    }

    #[test]
    fn test_parse_dns_record_missing_signature_when_required() {
        let record = "nsn:peer:/ip4/127.0.0.1/tcp/9000/p2p/12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN";
        let signers = HashSet::new();

        let result = parse_dns_record(record, &signers, true);
        assert!(result.is_err());
        assert!(matches!(result, Err(BootstrapError::InvalidSignature)));
    }

    #[test]
    fn test_parse_dns_record_with_valid_signature() {
        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());
        let multiaddr_str = format!("/ip4/127.0.0.1/tcp/9000/p2p/{}", peer_id);

        // Sign the message
        let message = format!("{}:{}", peer_id, multiaddr_str);
        let signature = keypair
            .sign(message.as_bytes())
            .expect("Signing should succeed");
        let sig_hex = hex::encode(&signature);

        let record = format!("nsn:peer:{}:sig:{}", multiaddr_str, sig_hex);

        let mut signers = HashSet::new();
        signers.insert(keypair.public());

        let result = parse_dns_record(&record, &signers, true);
        assert!(result.is_ok());

        let peer_info = result.unwrap().expect("Should have peer info");
        assert_eq!(peer_info.peer_id, peer_id);
        assert!(peer_info.signature.is_some());
    }

    #[test]
    fn test_parse_dns_record_with_invalid_signature() {
        let peer_id_str = "12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN";
        let multiaddr_str = format!("/ip4/127.0.0.1/tcp/9000/p2p/{}", peer_id_str);

        // Invalid signature (just zeros)
        let invalid_sig = hex::encode(vec![0u8; 64]);

        let record = format!("nsn:peer:{}:sig:{}", multiaddr_str, invalid_sig);

        let keypair = Keypair::generate_ed25519();
        let mut signers = HashSet::new();
        signers.insert(keypair.public());

        let result = parse_dns_record(&record, &signers, true);
        assert!(result.is_err());
        assert!(matches!(result, Err(BootstrapError::InvalidSignature)));
    }

    #[test]
    fn test_parse_dns_record_invalid_format() {
        let record = "not:a:valid:nsn:record";
        let signers = HashSet::new();

        let result = parse_dns_record(record, &signers, false);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // Should skip non-NSN records
    }

    #[test]
    fn test_parse_dns_record_missing_peer_id_in_multiaddr() {
        let record = "nsn:peer:/ip4/127.0.0.1/tcp/9000"; // No /p2p/<peer_id>
        let signers = HashSet::new();

        let result = parse_dns_record(record, &signers, false);
        assert!(result.is_err());
        assert!(matches!(result, Err(BootstrapError::NoPeerIdInMultiaddr)));
    }

    #[test]
    fn test_parse_dns_record_invalid_multiaddr() {
        let record = "nsn:peer:invalid_multiaddr:sig:abcd";
        let signers = HashSet::new();

        let result = parse_dns_record(record, &signers, false);
        assert!(result.is_err());
        assert!(matches!(result, Err(BootstrapError::InvalidMultiaddr)));
    }

    #[test]
    fn test_parse_dns_record_multiaddr_with_colons() {
        // IPv6 multiaddrs contain colons
        let record =
            "nsn:peer:/ip6/::1/tcp/9000/p2p/12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN";
        let signers = HashSet::new();

        let result = parse_dns_record(record, &signers, false);
        assert!(result.is_ok());

        let peer_info = result.unwrap().expect("Should have peer info");
        assert_eq!(peer_info.addrs.len(), 1);
        assert!(peer_info.addrs[0].to_string().contains("::1"));
    }

    // Note: Actual DNS resolution tests require running DNS server or mocking
    // These are covered in integration tests with test DNS setup
}
