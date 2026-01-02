//! Multi-Layer Bootstrap Protocol with Signed Manifests
//!
//! Implements trust-tiered peer discovery:
//! - Layer 1: Hardcoded bootstrap peers (highest trust, compiled into binary)
//! - Layer 2: DNS TXT record seeds with Ed25519 signatures
//! - Layer 3: HTTPS JSON endpoints with signed peer manifests
//! - Layer 4: Kademlia DHT walk (after connecting to initial peers)
//!
//! # Security Model
//!
//! - Hardcoded peers: Implicitly trusted (foundation-operated)
//! - DNS/HTTP: Ed25519 signature verification against trusted signers
//! - DHT: Lowest trust, used after ≥3 bootstrap peers connected
//! - Deduplication: Merge peers from multiple sources, keep highest trust level
//! - Ranking: Trust level (primary), latency (secondary)

mod dht_walk;
mod dns;
mod hardcoded;
mod http;
mod ranking;
mod signature;

pub use dht_walk::*;
pub use dns::*;
pub use hardcoded::*;
pub use http::*;
pub use ranking::*;
pub use signature::*;

use libp2p::{Multiaddr, PeerId};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Trust level for bootstrap sources
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    /// DHT-discovered peers (lowest trust)
    DHT = 1,
    /// HTTP manifest peers
    HTTP = 2,
    /// DNS seed peers
    DNS = 3,
    /// Hardcoded peers (highest trust)
    Hardcoded = 4,
}

/// Information about a discovered peer
#[derive(Debug, Clone)]
pub struct PeerInfo {
    /// Peer ID
    pub peer_id: PeerId,
    /// Known multiaddresses
    pub addrs: Vec<Multiaddr>,
    /// Trust level of this peer
    pub trust_level: TrustLevel,
    /// Signature (if from signed source)
    pub signature: Option<Vec<u8>>,
    /// Measured latency (if pinged)
    pub latency_ms: Option<u64>,
}

impl PeerInfo {
    /// Create message bytes for signature verification
    pub fn signing_message(&self) -> Vec<u8> {
        let mut msg = self.peer_id.to_bytes();
        for addr in &self.addrs {
            msg.extend_from_slice(addr.to_string().as_bytes());
        }
        msg
    }
}

#[derive(Debug, Error)]
pub enum BootstrapError {
    #[error("DNS resolution failed: {0}")]
    DnsResolutionFailed(String),

    #[error("HTTP fetch failed: {0}")]
    HttpFetchFailed(String),

    #[error("DHT discovery failed: {0}")]
    DhtDiscoveryFailed(String),

    #[error("Invalid multiaddr")]
    InvalidMultiaddr,

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("No PeerId in multiaddr")]
    NoPeerIdInMultiaddr,

    #[error("JSON parse failed: {0}")]
    JsonParseFailed(String),

    #[error("JSON serialize failed")]
    JsonSerializeFailed,

    #[error("Untrusted signer")]
    UntrustedSigner,

    #[error("Invalid manifest signature")]
    InvalidManifestSignature,

    #[error("Invalid PeerId")]
    InvalidPeerId,

    #[error("All bootstrap sources failed")]
    AllSourcesFailed,

    #[error("No trusted signers configured")]
    NoTrustedSigners,

    #[error("Invalid signer quorum")]
    InvalidSignerQuorum,

    #[error("Chain signer fetch failed: {0}")]
    ChainSignerFetchFailed(String),
}

/// Bootstrap protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapConfig {
    /// DNS seeds (e.g., "_nsn-bootstrap._tcp.nsn.network")
    pub dns_seeds: Vec<String>,

    /// HTTP endpoints (e.g., "https://bootstrap.nsn.network/peers.json")
    pub http_endpoints: Vec<String>,

    /// Require signed manifests for DNS/HTTP
    pub require_signed_manifests: bool,

    /// Minimum peers needed before DHT walk
    pub min_peers_for_dht: usize,

    /// HTTP request timeout
    #[serde(with = "humantime_serde")]
    pub http_timeout: Duration,

    /// DNS query timeout
    #[serde(with = "humantime_serde")]
    pub dns_timeout: Duration,

    /// Trusted signer configuration
    #[serde(default)]
    pub signer_config: SignerConfig,

    /// Optional transparency log for HTTP manifests
    #[serde(default)]
    pub transparency_log_path: Option<PathBuf>,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            dns_seeds: vec!["_nsn-bootstrap._tcp.nsn.network".to_string()],
            http_endpoints: vec![
                "https://bootstrap.nsn.network/peers.json".to_string(),
                "https://backup.nsn.network/peers.json".to_string(),
            ],
            require_signed_manifests: true,
            min_peers_for_dht: 3,
            http_timeout: Duration::from_secs(10),
            dns_timeout: Duration::from_secs(5),
            signer_config: SignerConfig::default(),
            transparency_log_path: std::env::var("NSN_BOOTSTRAP_TRANSPARENCY_LOG")
                .ok()
                .map(PathBuf::from),
        }
    }
}

/// Multi-layer bootstrap protocol
pub struct BootstrapProtocol {
    /// Configuration
    config: BootstrapConfig,

    /// Trusted signer public keys
    trusted_signers: Arc<TrustedSignerSet>,

    /// Bootstrap metrics (optional)
    #[allow(dead_code)] // Reserved for future metrics integration
    metrics: Option<Arc<super::metrics::P2pMetrics>>,
}

impl BootstrapProtocol {
    /// Create new bootstrap protocol with explicit signer set
    pub fn new(
        config: BootstrapConfig,
        trusted_signers: TrustedSignerSet,
        metrics: Option<Arc<super::metrics::P2pMetrics>>,
    ) -> Self {
        let trusted_signers = Arc::new(trusted_signers);

        Self {
            config,
            trusted_signers,
            metrics,
        }
    }

    /// Discover peers from all bootstrap sources
    ///
    /// Tries layers in order: Hardcoded → DNS → HTTP → DHT
    /// Returns deduplicated and ranked peer list
    pub async fn discover_peers(&self) -> Result<Vec<PeerInfo>, BootstrapError> {
        let mut discovered = Vec::new();
        let mut had_success = false;

        // Layer 1: Hardcoded peers
        info!("Bootstrap Layer 1: Hardcoded peers");
        let hardcoded_peers = get_hardcoded_peers();
        if !hardcoded_peers.is_empty() {
            discovered.extend(hardcoded_peers.clone());
            info!("Discovered {} hardcoded peers", hardcoded_peers.len());
            had_success = true;
        }

        // Layer 2: DNS seeds
        info!("Bootstrap Layer 2: DNS seeds");
        for dns_seed in &self.config.dns_seeds {
            match resolve_dns_seed(
                dns_seed,
                self.trusted_signers.clone(),
                self.config.require_signed_manifests,
                self.config.dns_timeout,
            )
            .await
            {
                Ok(peers) if !peers.is_empty() => {
                    info!("DNS seed {} returned {} peers", dns_seed, peers.len());
                    discovered.extend(peers);
                    had_success = true;
                }
                Ok(_) => {
                    debug!("DNS seed {} returned no peers", dns_seed);
                }
                Err(e) => {
                    warn!("DNS seed {} failed: {}", dns_seed, e);
                }
            }
        }

        // Layer 3: HTTP endpoints
        info!("Bootstrap Layer 3: HTTP endpoints");
        for endpoint in &self.config.http_endpoints {
            match fetch_http_peers(
                endpoint,
                self.trusted_signers.clone(),
                self.config.require_signed_manifests,
                self.config.http_timeout,
                self.config.transparency_log_path.as_deref(),
            )
            .await
            {
                Ok(peers) if !peers.is_empty() => {
                    info!("HTTP endpoint {} returned {} peers", endpoint, peers.len());
                    discovered.extend(peers);
                    had_success = true;
                }
                Ok(_) => {
                    debug!("HTTP endpoint {} returned no peers", endpoint);
                }
                Err(e) => {
                    warn!("HTTP endpoint {} failed: {}", endpoint, e);
                }
            }
        }

        // Layer 4: DHT walk (only if enough peers discovered)
        if discovered.len() >= self.config.min_peers_for_dht {
            info!(
                "Bootstrap Layer 4: DHT walk (have {} peers)",
                discovered.len()
            );
            // DHT walk requires connection to bootstrap peers first
            // This is a placeholder - actual implementation requires DHT integration
            debug!("DHT walk would be performed after connecting to bootstrap peers");
        } else {
            debug!(
                "Skipping DHT walk (need {} peers, have {})",
                self.config.min_peers_for_dht,
                discovered.len()
            );
        }

        if !had_success {
            return Err(BootstrapError::AllSourcesFailed);
        }

        // Deduplicate and rank
        let final_peers = deduplicate_and_rank(discovered);

        info!(
            "Bootstrap complete: {} unique peers discovered",
            final_peers.len()
        );

        Ok(final_peers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;
    use std::collections::HashSet;

    fn test_signers() -> TrustedSignerSet {
        let keypair = Keypair::generate_ed25519();
        let mut active = HashSet::new();
        active.insert(keypair.public());
        TrustedSignerSet::new(active, HashSet::new(), 1).expect("valid test signers")
    }

    #[tokio::test]
    async fn test_bootstrap_protocol_creation() {
        let config = BootstrapConfig::default();
        let protocol = BootstrapProtocol::new(config, test_signers(), None);

        assert!(!protocol.config.dns_seeds.is_empty());
        assert!(!protocol.config.http_endpoints.is_empty());
        assert!(protocol.config.require_signed_manifests);
    }

    #[tokio::test]
    async fn test_bootstrap_discovers_hardcoded_peers() {
        let config = BootstrapConfig {
            dns_seeds: vec![],      // Disable DNS
            http_endpoints: vec![], // Disable HTTP
            ..Default::default()
        };

        let protocol = BootstrapProtocol::new(config, test_signers(), None);
        let peers = protocol
            .discover_peers()
            .await
            .expect("Should discover hardcoded peers");

        assert!(!peers.is_empty(), "Should have at least 1 hardcoded peer");
        assert!(peers.iter().all(|p| p.trust_level == TrustLevel::Hardcoded));
    }

    #[test]
    fn test_trust_level_ordering() {
        assert!(TrustLevel::Hardcoded > TrustLevel::DNS);
        assert!(TrustLevel::DNS > TrustLevel::HTTP);
        assert!(TrustLevel::HTTP > TrustLevel::DHT);
    }

    #[test]
    fn test_peer_info_signing_message() {
        let peer_id = PeerId::random();
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/9000".parse().unwrap();

        let peer_info = PeerInfo {
            peer_id,
            addrs: vec![addr.clone()],
            trust_level: TrustLevel::Hardcoded,
            signature: None,
            latency_ms: None,
        };

        let message = peer_info.signing_message();
        assert!(!message.is_empty());
        assert!(message.len() > peer_id.to_bytes().len());
    }
}
