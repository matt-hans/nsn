//! Ed25519 Signature Verification for Bootstrap Manifests
//!
//! Provides signature verification for DNS and HTTP bootstrap sources
//! using Ed25519 public keys from trusted signers.

use super::BootstrapError;
use libp2p::identity::PublicKey;
use parity_scale_codec::Decode;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::env;
use std::time::Duration;
use subxt::{dynamic::storage, OnlineClient, PolkadotConfig};

/// Source of trusted signer data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignerSource {
    /// Use static signer list (config/env).
    Static,
    /// Fetch signer list from chain storage.
    Chain,
}

impl SignerSource {
    fn from_env() -> Option<Self> {
        env::var("NSN_BOOTSTRAP_SIGNER_SOURCE")
            .ok()
            .and_then(|value| match value.trim().to_lowercase().as_str() {
                "static" => Some(SignerSource::Static),
                "chain" => Some(SignerSource::Chain),
                _ => None,
            })
    }
}

/// Configuration for trusted signer resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignerConfig {
    /// Trusted signer public keys (hex-encoded protobuf bytes).
    pub trusted_signers_hex: Vec<String>,
    /// Revoked signer public keys (hex-encoded protobuf bytes).
    pub revoked_signers_hex: Vec<String>,
    /// Required signature quorum.
    pub quorum: usize,
    /// Signer source preference.
    pub source: SignerSource,
    /// Chain fetch timeout.
    #[serde(with = "humantime_serde")]
    pub chain_timeout: Duration,
}

impl Default for SignerConfig {
    fn default() -> Self {
        let trusted_signers_hex = parse_env_list("NSN_BOOTSTRAP_TRUSTED_SIGNERS");
        let revoked_signers_hex = parse_env_list("NSN_BOOTSTRAP_REVOKED_SIGNERS");
        let quorum = env::var("NSN_BOOTSTRAP_SIGNER_QUORUM")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1);
        let source = SignerSource::from_env().unwrap_or(SignerSource::Chain);

        Self {
            trusted_signers_hex,
            revoked_signers_hex,
            quorum,
            source,
            chain_timeout: Duration::from_secs(5),
        }
    }
}

/// Trusted signer set with quorum and revocations.
#[derive(Debug, Clone)]
pub struct TrustedSignerSet {
    /// Active trusted signers.
    pub active: HashSet<PublicKey>,
    /// Revoked signers.
    pub revoked: HashSet<PublicKey>,
    /// Required quorum.
    pub quorum: usize,
}

impl TrustedSignerSet {
    pub fn new(
        active: HashSet<PublicKey>,
        revoked: HashSet<PublicKey>,
        quorum: usize,
    ) -> Result<Self, BootstrapError> {
        let active_filtered: HashSet<PublicKey> =
            active.difference(&revoked).cloned().collect();
        if active_filtered.is_empty() {
            return Err(BootstrapError::NoTrustedSigners);
        }
        if quorum == 0 || quorum > active_filtered.len() {
            return Err(BootstrapError::InvalidSignerQuorum);
        }
        Ok(Self {
            active: active_filtered,
            revoked,
            quorum,
        })
    }
}

/// Resolve trusted signers based on config and optional chain source.
pub async fn resolve_trusted_signers(
    config: &SignerConfig,
    rpc_url: &str,
) -> Result<TrustedSignerSet, BootstrapError> {
    let config = effective_config(config);

    match config.source {
        SignerSource::Static => load_from_hex(&config),
        SignerSource::Chain => {
            match tokio::time::timeout(config.chain_timeout, fetch_signers_from_chain(rpc_url)).await
            {
                Ok(Ok(chain_set)) => Ok(chain_set),
                Ok(Err(err)) => {
                    if !config.trusted_signers_hex.is_empty() {
                        tracing::warn!(
                            error = %err,
                            "Failed to fetch chain signers, falling back to static config"
                        );
                        load_from_hex(&config)
                    } else {
                        Err(BootstrapError::ChainSignerFetchFailed(format!("{}", err)))
                    }
                }
                Err(err) => {
                    if !config.trusted_signers_hex.is_empty() {
                        tracing::warn!(
                            error = %err,
                            "Chain signer fetch timed out, falling back to static config"
                        );
                        load_from_hex(&config)
                    } else {
                        Err(BootstrapError::ChainSignerFetchFailed(format!("{}", err)))
                    }
                }
            }
        }
    }
}

fn effective_config(config: &SignerConfig) -> SignerConfig {
    if !config.trusted_signers_hex.is_empty()
        || env::var("NSN_BOOTSTRAP_TRUSTED_SIGNERS").is_ok()
    {
        return config.clone();
    }
    SignerConfig::default()
}

fn load_from_hex(config: &SignerConfig) -> Result<TrustedSignerSet, BootstrapError> {
    let active = parse_signers(&config.trusted_signers_hex)?;
    let revoked = parse_signers(&config.revoked_signers_hex)?;
    TrustedSignerSet::new(active, revoked, config.quorum)
}

fn parse_env_list(key: &str) -> Vec<String> {
    env::var(key)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|item| item.trim())
                .filter(|item| !item.is_empty())
                .map(|item| item.to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn parse_signers(keys: &[String]) -> Result<HashSet<PublicKey>, BootstrapError> {
    let mut signers = HashSet::new();
    for key in keys {
        let key_bytes = hex::decode(key).map_err(|_| BootstrapError::UntrustedSigner)?;
        let public_key = PublicKey::try_decode_protobuf(&key_bytes)
            .map_err(|_| BootstrapError::UntrustedSigner)?;
        signers.insert(public_key);
    }
    Ok(signers)
}

async fn fetch_signers_from_chain(rpc_url: &str) -> Result<TrustedSignerSet, BootstrapError> {
    let client = OnlineClient::<PolkadotConfig>::from_url(rpc_url)
        .await
        .map_err(|e| BootstrapError::ChainSignerFetchFailed(e.to_string()))?;

    let storage_at = client
        .storage()
        .at_latest()
        .await
        .map_err(|e| BootstrapError::ChainSignerFetchFailed(e.to_string()))?;

    let trusted_query =
        storage("NsnBootstrap", "TrustedSigners", Vec::<subxt::dynamic::Value>::new());
    let trusted_value = storage_at
        .fetch(&trusted_query)
        .await
        .map_err(|e| BootstrapError::ChainSignerFetchFailed(e.to_string()))?
        .ok_or_else(|| {
            BootstrapError::ChainSignerFetchFailed("NsnBootstrap::TrustedSigners missing".to_string())
        })?;
    let trusted = decode_value::<Vec<Vec<u8>>>(trusted_value.encoded())?;

    let revoked_query =
        storage("NsnBootstrap", "RevokedSigners", Vec::<subxt::dynamic::Value>::new());
    let revoked_value = storage_at
        .fetch(&revoked_query)
        .await
        .map_err(|e| BootstrapError::ChainSignerFetchFailed(e.to_string()))?
        .ok_or_else(|| {
            BootstrapError::ChainSignerFetchFailed("NsnBootstrap::RevokedSigners missing".to_string())
        })?;
    let revoked = decode_value::<Vec<Vec<u8>>>(revoked_value.encoded())?;

    let quorum_query =
        storage("NsnBootstrap", "SignerQuorum", Vec::<subxt::dynamic::Value>::new());
    let quorum_value = storage_at
        .fetch(&quorum_query)
        .await
        .map_err(|e| BootstrapError::ChainSignerFetchFailed(e.to_string()))?
        .ok_or_else(|| {
            BootstrapError::ChainSignerFetchFailed("NsnBootstrap::SignerQuorum missing".to_string())
        })?;
    let quorum = decode_value::<u32>(quorum_value.encoded())?;

    let active_keys = parse_bytes(&trusted)?;
    let revoked_keys = parse_bytes(&revoked)?;

    TrustedSignerSet::new(active_keys, revoked_keys, quorum as usize)
}

fn decode_value<T: Decode>(bytes: &[u8]) -> Result<T, BootstrapError> {
    Decode::decode(&mut &bytes[..]).map_err(|e| BootstrapError::ChainSignerFetchFailed(e.to_string()))
}

fn parse_bytes(raw: &[Vec<u8>]) -> Result<HashSet<PublicKey>, BootstrapError> {
    let mut out = HashSet::new();
    for key_bytes in raw {
        let public_key = PublicKey::try_decode_protobuf(key_bytes)
            .map_err(|_| BootstrapError::UntrustedSigner)?;
        out.insert(public_key);
    }
    Ok(out)
}

/// Verify signatures against trusted signers with quorum.
pub fn verify_signature_quorum(
    message: &[u8],
    signatures: &[Vec<u8>],
    trusted_signers: &TrustedSignerSet,
) -> Result<HashSet<PublicKey>, BootstrapError> {
    if signatures.is_empty() {
        return Err(BootstrapError::InvalidSignature);
    }

    let mut matched = HashSet::new();
    for signature in signatures {
        for signer in &trusted_signers.active {
            if signer.verify(message, signature) {
                matched.insert(signer.clone());
                break;
            }
        }
    }

    if matched.len() < trusted_signers.quorum {
        return Err(BootstrapError::InvalidManifestSignature);
    }

    Ok(matched)
}

/// Verify signature on message (any trusted signer).
///
/// # Arguments
/// * `message` - Message bytes that were signed
/// * `signature` - Signature bytes
/// * `trusted_signers` - Set of trusted public keys
///
/// # Returns
/// `true` if signature is valid from any trusted signer
pub fn verify_signature(
    message: &[u8],
    signature: &[u8],
    trusted_signers: &TrustedSignerSet,
) -> bool {
    trusted_signers
        .active
        .iter()
        .any(|pk| pk.verify(message, signature))
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    fn test_signer_set() -> TrustedSignerSet {
        let keypair = Keypair::generate_ed25519();
        let mut active = HashSet::new();
        active.insert(keypair.public());
        TrustedSignerSet::new(active, HashSet::new(), 1).expect("valid test signer set")
    }

    #[test]
    fn test_verify_signature_valid() {
        let keypair = Keypair::generate_ed25519();
        let public_key = keypair.public();

        let message = b"test message";
        let signature = keypair.sign(message).expect("Signing should succeed");

        let mut active = HashSet::new();
        active.insert(public_key);
        let signers = TrustedSignerSet::new(active, HashSet::new(), 1).unwrap();

        assert!(verify_signature(message, &signature, &signers));
    }

    #[test]
    fn test_verify_signature_invalid_signature() {
        let keypair = Keypair::generate_ed25519();
        let public_key = keypair.public();

        let message = b"test message";
        let invalid_signature = vec![0u8; 64];

        let mut active = HashSet::new();
        active.insert(public_key);
        let signers = TrustedSignerSet::new(active, HashSet::new(), 1).unwrap();

        assert!(!verify_signature(message, &invalid_signature, &signers));
    }

    #[test]
    fn test_verify_signature_quorum() {
        let keypair1 = Keypair::generate_ed25519();
        let keypair2 = Keypair::generate_ed25519();

        let message = b"test message";
        let sig1 = keypair1.sign(message).expect("Signing should succeed");
        let sig2 = keypair2.sign(message).expect("Signing should succeed");

        let mut active = HashSet::new();
        active.insert(keypair1.public());
        active.insert(keypair2.public());
        let signers = TrustedSignerSet::new(active, HashSet::new(), 2).unwrap();

        let matched = verify_signature_quorum(message, &[sig1, sig2], &signers).unwrap();
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn test_verify_signature_quorum_insufficient() {
        let signers = test_signer_set();
        let message = b"test message";
        let signature = vec![0u8; 64];
        let result = verify_signature_quorum(message, &[signature], &signers);
        assert!(result.is_err());
    }
}
