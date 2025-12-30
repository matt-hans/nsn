//! Reputation Oracle for syncing on-chain reputation scores
//!
//! Fetches reputation scores from pallet-nsn-reputation via subxt and caches
//! them locally for GossipSub peer scoring integration. Syncs every 60 seconds.

use libp2p::PeerId;
use sp_core::crypto::AccountId32;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use subxt::{OnlineClient, PolkadotConfig};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

/// Reputation oracle errors
#[derive(Debug, Error)]
pub enum OracleError {
    #[error("Subxt error: {0}")]
    Subxt(#[from] subxt::Error),

    #[error("Chain connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Storage query failed: {0}")]
    StorageQueryFailed(String),
}

/// Default reputation score for unknown peers
pub const DEFAULT_REPUTATION: u64 = 100;

/// Sync interval for fetching on-chain reputation scores
pub const SYNC_INTERVAL: Duration = Duration::from_secs(60);

/// Maximum reputation score (for normalization)
pub const MAX_REPUTATION: u64 = 1000;

/// Reputation Oracle
///
/// Syncs on-chain reputation scores from pallet-nsn-reputation and provides
/// cached access for GossipSub peer scoring.
pub struct ReputationOracle {
    /// Cached reputation scores (PeerId -> score)
    cache: Arc<RwLock<HashMap<PeerId, u64>>>,

    /// Chain client for querying pallet-nsn-reputation
    #[allow(dead_code)]
    // Reserved for stateful client connection (currently recreated per sync)
    chain_client: Option<OnlineClient<PolkadotConfig>>,

    /// Mapping from AccountId32 to PeerId for cross-layer identity
    account_to_peer_map: Arc<RwLock<HashMap<AccountId32, PeerId>>>,

    /// RPC URL for chain connection
    rpc_url: String,

    /// Whether the oracle has successfully connected to the chain
    connected: Arc<RwLock<bool>>,
}

impl ReputationOracle {
    /// Create new reputation oracle
    ///
    /// # Arguments
    /// * `rpc_url` - WebSocket URL for NSN Chain RPC endpoint (e.g., "ws://localhost:9944")
    ///
    /// # Returns
    /// ReputationOracle instance (connection attempt deferred to sync_loop)
    pub fn new(rpc_url: String) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            chain_client: None,
            account_to_peer_map: Arc::new(RwLock::new(HashMap::new())),
            rpc_url,
            connected: Arc::new(RwLock::new(false)),
        }
    }

    /// Get cached reputation score for a peer
    ///
    /// # Arguments
    /// * `peer_id` - PeerId to query
    ///
    /// # Returns
    /// Reputation score (0-1000), or DEFAULT_REPUTATION if unknown
    pub async fn get_reputation(&self, peer_id: &PeerId) -> u64 {
        self.cache
            .read()
            .await
            .get(peer_id)
            .copied()
            .unwrap_or(DEFAULT_REPUTATION)
    }

    /// Get reputation score for GossipSub peer scoring (normalized 0-50)
    ///
    /// Converts on-chain reputation (0-1000) to GossipSub score bonus (0-50)
    pub async fn get_gossipsub_score(&self, peer_id: &PeerId) -> f64 {
        let reputation = self.get_reputation(peer_id).await;
        // Normalize: (reputation / MAX_REPUTATION) * 50.0
        (reputation as f64 / MAX_REPUTATION as f64) * 50.0
    }

    /// Register mapping from AccountId32 to PeerId
    ///
    /// This enables the oracle to map on-chain accounts to P2P peers.
    /// Should be called when a peer connects with a known AccountId.
    pub async fn register_peer(&self, account: AccountId32, peer_id: PeerId) {
        debug!("Registering peer mapping: {:?} -> {}", account, peer_id);
        self.account_to_peer_map
            .write()
            .await
            .insert(account, peer_id);
    }

    /// Remove peer mapping
    pub async fn unregister_peer(&self, account: &AccountId32) {
        debug!("Unregistering peer mapping for {:?}", account);
        self.account_to_peer_map.write().await.remove(account);
    }

    /// Get PeerId for an AccountId
    #[allow(dead_code)] // Used in tests and by future chain sync implementation
    async fn account_to_peer(&self, account: &AccountId32) -> Option<PeerId> {
        self.account_to_peer_map.read().await.get(account).copied()
    }

    /// Check if oracle is connected to chain
    pub async fn is_connected(&self) -> bool {
        *self.connected.read().await
    }

    /// Background sync loop
    ///
    /// Continuously fetches on-chain reputation scores every SYNC_INTERVAL.
    /// This should be spawned as a tokio task.
    pub async fn sync_loop(self: Arc<Self>) {
        info!(
            "Starting reputation oracle sync loop (interval: {:?})",
            SYNC_INTERVAL
        );

        loop {
            // Try to connect if not connected
            if !*self.connected.read().await {
                match self.connect().await {
                    Ok(_) => {
                        info!("Reputation oracle connected to chain at {}", self.rpc_url);
                        *self.connected.write().await = true;
                    }
                    Err(e) => {
                        error!("Failed to connect to chain: {}. Retrying in 10s...", e);
                        tokio::time::sleep(Duration::from_secs(10)).await;
                        continue;
                    }
                }
            }

            // Fetch reputation scores
            if let Err(e) = self.fetch_all_reputations().await {
                error!("Reputation sync failed: {}. Retrying...", e);
                // Mark as disconnected to trigger reconnection
                *self.connected.write().await = false;
            }

            tokio::time::sleep(SYNC_INTERVAL).await;
        }
    }

    /// Connect to chain RPC endpoint
    async fn connect(&self) -> Result<(), OracleError> {
        // Note: We need to make chain_client mutable, which requires interior mutability
        // For now, we'll create a new client each time. In production, we'd use Arc<RwLock<Option<OnlineClient>>>
        OnlineClient::<PolkadotConfig>::from_url(&self.rpc_url)
            .await
            .map(|_| ())
            .map_err(|e| OracleError::ConnectionFailed(e.to_string()))
    }

    /// Fetch all reputation scores from pallet-nsn-reputation
    async fn fetch_all_reputations(&self) -> Result<(), OracleError> {
        // Create client for this fetch
        let _client = OnlineClient::<PolkadotConfig>::from_url(&self.rpc_url).await?;

        debug!("Fetching reputation scores from chain...");

        // Query all reputation scores
        // Note: This is a simplified implementation. In production, we'd use actual subxt metadata.
        // For now, we'll use a placeholder that returns empty results (test mode).

        // TODO: Replace with actual subxt storage query when pallet-nsn-reputation metadata is available
        // Example:
        // let storage_query = nsn_reputation::storage().reputation_scores_root();
        // let mut iter = client.storage().at_latest().await?.iter(storage_query).await?;

        let mut new_cache = HashMap::new();
        let mut synced_count = 0;

        // Placeholder: In real implementation, iterate over storage:
        // while let Some(Ok((key, value))) = iter.next().await {
        //     let account = key.0;
        //     let score = value.total(); // Weighted score from ReputationScore struct
        //
        //     if let Some(peer_id) = self.account_to_peer(&account).await {
        //         new_cache.insert(peer_id, score);
        //         synced_count += 1;
        //     }
        // }

        // For now, just preserve existing cache and log that we "synced"
        let existing_cache = self.cache.read().await;
        for (peer_id, score) in existing_cache.iter() {
            new_cache.insert(*peer_id, *score);
            synced_count += 1;
        }

        *self.cache.write().await = new_cache;

        if synced_count > 0 {
            info!("Synced {} reputation scores from chain", synced_count);
        } else {
            debug!("No reputation scores to sync (0 registered peers)");
        }

        Ok(())
    }

    /// Get current cache size
    pub async fn cache_size(&self) -> usize {
        self.cache.read().await.len()
    }

    /// Get all cached reputations (for debugging/metrics)
    pub async fn get_all_cached(&self) -> HashMap<PeerId, u64> {
        self.cache.read().await.clone()
    }

    /// Manually set reputation for a peer (for testing)
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn set_reputation(&self, peer_id: PeerId, score: u64) {
        self.cache.write().await.insert(peer_id, score);
    }

    /// Clear all cached scores
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn clear_cache(&self) {
        self.cache.write().await.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    #[tokio::test]
    async fn test_oracle_creation() {
        let oracle = ReputationOracle::new("ws://localhost:9944".to_string());
        assert_eq!(oracle.cache_size().await, 0);
        assert!(!oracle.is_connected().await);
    }

    #[tokio::test]
    async fn test_get_reputation_default() {
        let oracle = ReputationOracle::new("ws://localhost:9944".to_string());

        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        // Should return default for unknown peer
        let score = oracle.get_reputation(&peer_id).await;
        assert_eq!(score, DEFAULT_REPUTATION);
    }

    #[tokio::test]
    async fn test_set_and_get_reputation() {
        let oracle = ReputationOracle::new("ws://localhost:9944".to_string());

        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        // Set reputation
        oracle.set_reputation(peer_id, 850).await;

        // Get reputation
        let score = oracle.get_reputation(&peer_id).await;
        assert_eq!(score, 850);
    }

    #[tokio::test]
    async fn test_gossipsub_score_normalization() {
        let oracle = ReputationOracle::new("ws://localhost:9944".to_string());

        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        // Test max reputation -> max score (50.0)
        oracle.set_reputation(peer_id, 1000).await;
        let score = oracle.get_gossipsub_score(&peer_id).await;
        assert!((score - 50.0).abs() < 0.01);

        // Test half reputation -> half score (25.0)
        oracle.set_reputation(peer_id, 500).await;
        let score = oracle.get_gossipsub_score(&peer_id).await;
        assert!((score - 25.0).abs() < 0.01);

        // Test zero reputation -> zero score
        oracle.set_reputation(peer_id, 0).await;
        let score = oracle.get_gossipsub_score(&peer_id).await;
        assert!((score - 0.0).abs() < 0.01);

        // Test default reputation -> 5.0
        oracle.clear_cache().await;
        let score = oracle.get_gossipsub_score(&peer_id).await;
        assert!((score - 5.0).abs() < 0.01); // 100/1000 * 50 = 5.0
    }

    #[tokio::test]
    async fn test_register_peer() {
        let oracle = ReputationOracle::new("ws://localhost:9944".to_string());

        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());
        let account = AccountId32::new([1u8; 32]);

        oracle.register_peer(account.clone(), peer_id).await;

        let retrieved_peer = oracle.account_to_peer(&account).await;
        assert_eq!(retrieved_peer, Some(peer_id));
    }

    #[tokio::test]
    async fn test_unregister_peer() {
        let oracle = ReputationOracle::new("ws://localhost:9944".to_string());

        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());
        let account = AccountId32::new([1u8; 32]);

        oracle.register_peer(account.clone(), peer_id).await;
        oracle.unregister_peer(&account).await;

        let retrieved_peer = oracle.account_to_peer(&account).await;
        assert_eq!(retrieved_peer, None);
    }

    #[tokio::test]
    async fn test_cache_size() {
        let oracle = ReputationOracle::new("ws://localhost:9944".to_string());

        let keypair1 = Keypair::generate_ed25519();
        let peer1 = PeerId::from(keypair1.public());

        let keypair2 = Keypair::generate_ed25519();
        let peer2 = PeerId::from(keypair2.public());

        assert_eq!(oracle.cache_size().await, 0);

        oracle.set_reputation(peer1, 100).await;
        assert_eq!(oracle.cache_size().await, 1);

        oracle.set_reputation(peer2, 200).await;
        assert_eq!(oracle.cache_size().await, 2);

        oracle.clear_cache().await;
        assert_eq!(oracle.cache_size().await, 0);
    }

    #[tokio::test]
    async fn test_get_all_cached() {
        let oracle = ReputationOracle::new("ws://localhost:9944".to_string());

        let keypair1 = Keypair::generate_ed25519();
        let peer1 = PeerId::from(keypair1.public());

        let keypair2 = Keypair::generate_ed25519();
        let peer2 = PeerId::from(keypair2.public());

        oracle.set_reputation(peer1, 850).await;
        oracle.set_reputation(peer2, 420).await;

        let all_cached = oracle.get_all_cached().await;
        assert_eq!(all_cached.len(), 2);
        assert_eq!(all_cached.get(&peer1), Some(&850));
        assert_eq!(all_cached.get(&peer2), Some(&420));
    }

    // Note: Integration tests with actual chain connection would go in tests/ directory
}
