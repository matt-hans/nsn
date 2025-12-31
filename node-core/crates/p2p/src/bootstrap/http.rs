//! HTTP(S) Bootstrap Manifest with Signature Verification
//!
//! Fetches signed peer manifests from HTTPS endpoints.
//! JSON format with Ed25519 signature verification.

use super::{signature::verify_signature, BootstrapError, PeerInfo, TrustLevel};
use libp2p::identity::PublicKey;
use libp2p::{Multiaddr, PeerId};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};

/// Peer manifest JSON structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerManifest {
    /// List of peers
    pub peers: Vec<ManifestPeer>,
    /// Ed25519 signature of the peers array (hex-encoded)
    pub signature: String,
    /// Signer's public key (hex-encoded protobuf)
    pub signer: String,
}

/// Individual peer in manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestPeer {
    /// Peer ID (base58)
    pub peer_id: String,
    /// Multiaddresses
    pub addrs: Vec<String>,
}

/// Fetch peers from HTTPS endpoint
///
/// # Arguments
/// * `endpoint` - HTTPS URL (e.g., "https://bootstrap.nsn.network/peers.json")
/// * `trusted_signers` - Set of trusted signer public keys
/// * `require_signatures` - Reject unsigned manifests if true
/// * `timeout` - HTTP request timeout
///
/// # Returns
/// Vector of discovered and verified peers
pub async fn fetch_http_peers(
    endpoint: &str,
    trusted_signers: Arc<HashSet<PublicKey>>,
    require_signatures: bool,
    timeout: Duration,
) -> Result<Vec<PeerInfo>, BootstrapError> {
    // Create HTTP client with TLS
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| BootstrapError::HttpFetchFailed(e.to_string()))?;

    // Fetch manifest
    debug!("Fetching bootstrap manifest from {}", endpoint);
    let response = client
        .get(endpoint)
        .send()
        .await
        .map_err(|e| BootstrapError::HttpFetchFailed(e.to_string()))?;

    if !response.status().is_success() {
        return Err(BootstrapError::HttpFetchFailed(format!(
            "HTTP {}: {}",
            response.status(),
            response.status().canonical_reason().unwrap_or("Unknown")
        )));
    }

    let manifest: PeerManifest = response
        .json()
        .await
        .map_err(|e| BootstrapError::JsonParseFailed(e.to_string()))?;

    debug!(
        "Fetched manifest with {} peers, signer: {}",
        manifest.peers.len(),
        manifest.signer
    );

    // Verify manifest signature
    if require_signatures || !manifest.signature.is_empty() {
        verify_manifest_signature(&manifest, &trusted_signers)?;
    }

    // Parse peers
    let mut peers = Vec::new();
    for peer_entry in manifest.peers {
        match parse_manifest_peer(peer_entry, manifest.signature.clone()) {
            Ok(peer_info) => {
                peers.push(peer_info);
            }
            Err(e) => {
                warn!("Invalid peer in manifest: {}", e);
            }
        }
    }

    debug!("Parsed {} peers from HTTP manifest", peers.len());

    Ok(peers)
}

/// Verify manifest signature
fn verify_manifest_signature(
    manifest: &PeerManifest,
    trusted_signers: &HashSet<PublicKey>,
) -> Result<(), BootstrapError> {
    // Decode signer public key
    let signer_bytes =
        hex::decode(&manifest.signer).map_err(|_| BootstrapError::UntrustedSigner)?;
    let signer_pubkey = PublicKey::try_decode_protobuf(&signer_bytes)
        .map_err(|_| BootstrapError::UntrustedSigner)?;

    // Check if signer is trusted
    if !trusted_signers.contains(&signer_pubkey) {
        warn!("Manifest signed by untrusted signer: {}", manifest.signer);
        return Err(BootstrapError::UntrustedSigner);
    }

    // Decode signature
    let signature_bytes =
        hex::decode(&manifest.signature).map_err(|_| BootstrapError::InvalidSignature)?;

    // Serialize peers array for signing (canonical JSON)
    let message =
        serde_json::to_vec(&manifest.peers).map_err(|_| BootstrapError::JsonSerializeFailed)?;

    // Verify signature
    if !verify_signature(&message, &signature_bytes, trusted_signers) {
        warn!("Manifest signature verification failed");
        return Err(BootstrapError::InvalidManifestSignature);
    }

    debug!("Manifest signature verified successfully");

    Ok(())
}

/// Parse individual peer from manifest
fn parse_manifest_peer(
    peer_entry: ManifestPeer,
    signature: String,
) -> Result<PeerInfo, BootstrapError> {
    // Parse PeerId
    let peer_id: PeerId = peer_entry
        .peer_id
        .parse()
        .map_err(|_| BootstrapError::InvalidPeerId)?;

    // Parse multiaddrs
    let addrs: Vec<Multiaddr> = peer_entry
        .addrs
        .iter()
        .filter_map(|a| a.parse().ok())
        .collect();

    if addrs.is_empty() {
        return Err(BootstrapError::InvalidMultiaddr);
    }

    let signature_bytes = if !signature.is_empty() {
        Some(hex::decode(&signature).map_err(|_| BootstrapError::InvalidSignature)?)
    } else {
        None
    };

    Ok(PeerInfo {
        peer_id,
        addrs,
        trust_level: TrustLevel::HTTP,
        signature: signature_bytes,
        latency_ms: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    #[test]
    fn test_parse_manifest_peer_valid() {
        let peer_entry = ManifestPeer {
            peer_id: "12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN".to_string(),
            addrs: vec![
                "/ip4/127.0.0.1/tcp/9000".to_string(),
                "/ip4/127.0.0.1/udp/9000/quic-v1".to_string(),
            ],
        };

        let result = parse_manifest_peer(peer_entry, "".to_string());
        assert!(result.is_ok());

        let peer_info = result.unwrap();
        assert_eq!(peer_info.trust_level, TrustLevel::HTTP);
        assert_eq!(peer_info.addrs.len(), 2);
        assert!(peer_info.signature.is_none());
    }

    #[test]
    fn test_parse_manifest_peer_invalid_peer_id() {
        let peer_entry = ManifestPeer {
            peer_id: "invalid_peer_id".to_string(),
            addrs: vec!["/ip4/127.0.0.1/tcp/9000".to_string()],
        };

        let result = parse_manifest_peer(peer_entry, "".to_string());
        assert!(result.is_err());
        assert!(matches!(result, Err(BootstrapError::InvalidPeerId)));
    }

    #[test]
    fn test_parse_manifest_peer_invalid_addrs() {
        let peer_entry = ManifestPeer {
            peer_id: "12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN".to_string(),
            addrs: vec!["invalid_multiaddr".to_string()],
        };

        let result = parse_manifest_peer(peer_entry, "".to_string());
        assert!(result.is_err());
        assert!(matches!(result, Err(BootstrapError::InvalidMultiaddr)));
    }

    #[test]
    fn test_parse_manifest_peer_empty_addrs() {
        let peer_entry = ManifestPeer {
            peer_id: "12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN".to_string(),
            addrs: vec![],
        };

        let result = parse_manifest_peer(peer_entry, "".to_string());
        assert!(result.is_err());
        assert!(matches!(result, Err(BootstrapError::InvalidMultiaddr)));
    }

    #[test]
    fn test_parse_manifest_peer_with_signature() {
        let sig_hex = hex::encode(vec![1, 2, 3, 4]);
        let peer_entry = ManifestPeer {
            peer_id: "12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN".to_string(),
            addrs: vec!["/ip4/127.0.0.1/tcp/9000".to_string()],
        };

        let result = parse_manifest_peer(peer_entry, sig_hex.clone());
        assert!(result.is_ok());

        let peer_info = result.unwrap();
        assert!(peer_info.signature.is_some());
        assert_eq!(peer_info.signature.unwrap(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_verify_manifest_signature_untrusted_signer() {
        let keypair = Keypair::generate_ed25519();
        let public_key_protobuf = keypair.public().encode_protobuf();

        let peers = vec![];
        let message = serde_json::to_vec(&peers).unwrap();
        let signature = keypair.sign(&message).unwrap();

        let manifest = PeerManifest {
            peers,
            signature: hex::encode(&signature),
            signer: hex::encode(&public_key_protobuf),
        };

        let trusted_signers = HashSet::new(); // Empty - no trusted signers

        let result = verify_manifest_signature(&manifest, &trusted_signers);
        assert!(result.is_err());
        assert!(matches!(result, Err(BootstrapError::UntrustedSigner)));
    }

    #[test]
    fn test_verify_manifest_signature_invalid_signature() {
        let keypair = Keypair::generate_ed25519();
        let public_key_protobuf = keypair.public().encode_protobuf();

        let peers = vec![];
        let invalid_signature = hex::encode(vec![0u8; 64]);

        let manifest = PeerManifest {
            peers,
            signature: invalid_signature,
            signer: hex::encode(&public_key_protobuf),
        };

        let mut trusted_signers = HashSet::new();
        trusted_signers.insert(keypair.public());

        let result = verify_manifest_signature(&manifest, &trusted_signers);
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(BootstrapError::InvalidManifestSignature)
        ));
    }

    #[test]
    fn test_verify_manifest_signature_valid() {
        let keypair = Keypair::generate_ed25519();
        let public_key_protobuf = keypair.public().encode_protobuf();

        let peers = vec![ManifestPeer {
            peer_id: "12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN".to_string(),
            addrs: vec!["/ip4/127.0.0.1/tcp/9000".to_string()],
        }];

        let message = serde_json::to_vec(&peers).unwrap();
        let signature = keypair.sign(&message).unwrap();

        let manifest = PeerManifest {
            peers,
            signature: hex::encode(&signature),
            signer: hex::encode(&public_key_protobuf),
        };

        let mut trusted_signers = HashSet::new();
        trusted_signers.insert(keypair.public());

        let result = verify_manifest_signature(&manifest, &trusted_signers);
        assert!(result.is_ok());
    }

    // Note: Actual HTTP fetch tests require running HTTP server or mocking
    // These are covered in integration tests with test HTTP setup
}
