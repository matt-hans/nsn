//! HTTP(S) Bootstrap Manifest with Signature Verification
//!
//! Fetches signed peer manifests from HTTPS endpoints.
//! JSON format with Ed25519 signature verification.

use super::signature::{verify_signature_quorum, TrustedSignerSet};
use super::{BootstrapError, PeerInfo, TrustLevel};
use libp2p::{Multiaddr, PeerId};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, warn};

/// Peer manifest JSON structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerManifest {
    /// List of peers
    pub peers: Vec<ManifestPeer>,
    /// Single Ed25519 signature (hex) for legacy manifests
    #[serde(default)]
    pub signature: Option<String>,
    /// Single signer public key (hex protobuf) for legacy manifests
    #[serde(default)]
    pub signer: Option<String>,
    /// Multiple signatures (hex)
    #[serde(default)]
    pub signatures: Option<Vec<String>>,
    /// Optional signer public keys (hex protobuf)
    #[serde(default)]
    pub signers: Option<Vec<String>>,
    /// Optional signer epoch (on-chain rotation version)
    #[serde(default)]
    pub epoch: Option<u64>,
}

impl PeerManifest {
    fn signature_list(&self) -> Vec<String> {
        if let Some(signatures) = &self.signatures {
            return signatures.clone();
        }
        self.signature
            .as_ref()
            .map(|sig| vec![sig.clone()])
            .unwrap_or_default()
    }
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
/// * `transparency_log` - Optional transparency log path
///
/// # Returns
/// Vector of discovered and verified peers
pub async fn fetch_http_peers(
    endpoint: &str,
    trusted_signers: Arc<TrustedSignerSet>,
    require_signatures: bool,
    timeout: Duration,
    transparency_log: Option<&Path>,
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

    debug!("Fetched manifest with {} peers", manifest.peers.len());

    let signatures = manifest.signature_list();

    // Verify manifest signatures
    let verified_signers = if require_signatures || !signatures.is_empty() {
        Some(verify_manifest_signature(&manifest, &trusted_signers)?)
    } else {
        None
    };

    if let (Some(path), Some(verified)) = (transparency_log, &verified_signers) {
        if let Err(err) = record_manifest_transparency(path, endpoint, &manifest, verified) {
            warn!("Failed to record transparency log: {}", err);
        }
    }

    // Parse peers
    let mut peers = Vec::new();
    let signature_bytes = signatures.get(0).and_then(|sig| hex::decode(sig).ok());

    for peer_entry in manifest.peers {
        match parse_manifest_peer(peer_entry, signature_bytes.clone()) {
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

/// Verify manifest signatures
fn verify_manifest_signature(
    manifest: &PeerManifest,
    trusted_signers: &TrustedSignerSet,
) -> Result<HashSet<libp2p::identity::PublicKey>, BootstrapError> {
    let signature_list = manifest.signature_list();
    if signature_list.is_empty() {
        return Err(BootstrapError::InvalidSignature);
    }

    let mut signatures = Vec::new();
    for sig_hex in signature_list {
        let sig_bytes = hex::decode(sig_hex).map_err(|_| BootstrapError::InvalidSignature)?;
        signatures.push(sig_bytes);
    }

    // Serialize peers array for signing (canonical JSON)
    let message =
        serde_json::to_vec(&manifest.peers).map_err(|_| BootstrapError::JsonSerializeFailed)?;

    let verified = verify_signature_quorum(&message, &signatures, trusted_signers)?;
    debug!("Manifest signatures verified successfully");

    Ok(verified)
}

fn record_manifest_transparency(
    path: &Path,
    endpoint: &str,
    manifest: &PeerManifest,
    verified_signers: &HashSet<libp2p::identity::PublicKey>,
) -> Result<(), BootstrapError> {
    let message =
        serde_json::to_vec(&manifest.peers).map_err(|_| BootstrapError::JsonSerializeFailed)?;
    let hash = blake3::hash(&message).to_hex().to_string();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let signers_hex: Vec<String> = verified_signers
        .iter()
        .map(|pk| hex::encode(pk.encode_protobuf()))
        .collect();

    let entry = serde_json::json!({
        "ts": timestamp,
        "endpoint": endpoint,
        "manifest_hash": hash,
        "peer_count": manifest.peers.len(),
        "signers": signers_hex,
        "signature_count": manifest.signature_list().len(),
        "epoch": manifest.epoch,
    });

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| BootstrapError::HttpFetchFailed(e.to_string()))?;
    writeln!(file, "{}", entry.to_string())
        .map_err(|e| BootstrapError::HttpFetchFailed(e.to_string()))?;
    Ok(())
}

/// Parse individual peer from manifest
fn parse_manifest_peer(
    peer_entry: ManifestPeer,
    signature: Option<Vec<u8>>,
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

    Ok(PeerInfo {
        peer_id,
        addrs,
        trust_level: TrustLevel::HTTP,
        signature,
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

        let result = parse_manifest_peer(peer_entry, None);
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

        let result = parse_manifest_peer(peer_entry, None);
        assert!(result.is_err());
        assert!(matches!(result, Err(BootstrapError::InvalidPeerId)));
    }

    #[test]
    fn test_parse_manifest_peer_invalid_addrs() {
        let peer_entry = ManifestPeer {
            peer_id: "12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN".to_string(),
            addrs: vec!["invalid_multiaddr".to_string()],
        };

        let result = parse_manifest_peer(peer_entry, None);
        assert!(result.is_err());
        assert!(matches!(result, Err(BootstrapError::InvalidMultiaddr)));
    }

    #[test]
    fn test_verify_manifest_signature_valid() {
        let keypair = Keypair::generate_ed25519();
        let public_key = keypair.public();

        let peer_entry = ManifestPeer {
            peer_id: "12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN".to_string(),
            addrs: vec!["/ip4/127.0.0.1/tcp/9000".to_string()],
        };

        let message = serde_json::to_vec(&vec![peer_entry.clone()]).unwrap();
        let signature = keypair.sign(&message).unwrap();

        let manifest = PeerManifest {
            peers: vec![peer_entry],
            signature: Some(hex::encode(signature)),
            signer: Some(hex::encode(public_key.encode_protobuf())),
            signatures: None,
            signers: None,
            epoch: None,
        };

        let mut active = HashSet::new();
        active.insert(public_key);
        let signers = TrustedSignerSet::new(active, HashSet::new(), 1).unwrap();

        let result = verify_manifest_signature(&manifest, &signers);
        assert!(result.is_ok());
    }
}
