//! Reputation Oracle for syncing on-chain reputation scores
//!
//! Fetches reputation scores from pallet-nsn-reputation via subxt and caches
//! them locally for GossipSub peer scoring integration. Syncs every 60 seconds.

use libp2p::PeerId;
use serde::Deserialize;
use sp_core::crypto::AccountId32;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use subxt::ext::scale_value;
use subxt::{dynamic::storage, OnlineClient, PolkadotConfig};
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

#[derive(Debug, Deserialize)]
struct ReputationScore {
    director_score: u64,
    validator_score: u64,
    seeder_score: u64,
    last_activity: u64,
}

impl ReputationScore {
    fn total(&self) -> u64 {
        let director_weighted = self.director_score.saturating_mul(50);
        let validator_weighted = self.validator_score.saturating_mul(30);
        let seeder_weighted = self.seeder_score.saturating_mul(20);

        director_weighted
            .saturating_add(validator_weighted)
            .saturating_add(seeder_weighted)
            .saturating_div(100)
    }
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
        let client = OnlineClient::<PolkadotConfig>::from_url(&self.rpc_url).await?;

        debug!("Fetching reputation scores from chain...");

        let mut new_cache = HashMap::new();
        let mut synced_count = 0;

        let storage_query = storage("NsnReputation", "ReputationScores", vec![]);
        let mut iter = client
            .storage()
            .at_latest()
            .await?
            .iter(storage_query)
            .await?;

        while let Some(result) = iter.next().await {
            let key_value = result.map_err(|e| OracleError::StorageQueryFailed(e.to_string()))?;
            let account_value = key_value
                .keys
                .get(0)
                .ok_or_else(|| OracleError::StorageQueryFailed("Missing account key".into()))?;
            let account: AccountId32 = scale_value::serde::from_value(account_value.clone())
                .map_err(|e| OracleError::StorageQueryFailed(format!("Key decode failed: {e}")))?;
            let value = key_value.value.to_value().map_err(|e| {
                OracleError::StorageQueryFailed(format!("Value decode failed: {e}"))
            })?;
            let score: ReputationScore = scale_value::serde::from_value(value).map_err(|e| {
                OracleError::StorageQueryFailed(format!("Score decode failed: {e}"))
            })?;

            if let Some(peer_id) = self.account_to_peer(&account).await {
                new_cache.insert(peer_id, score.total());
                synced_count += 1;
            }
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

    #[tokio::test]
    async fn test_reputation_oracle_rpc_failure_handling() {
        // Test with invalid RPC URL to trigger connection failure
        let oracle = Arc::new(ReputationOracle::new("ws://invalid-host:9999".to_string()));

        // Verify initial state
        assert!(!oracle.is_connected().await, "Should start disconnected");

        // Test connect() with invalid URL
        let result = oracle.connect().await;
        assert!(result.is_err(), "Connection to invalid host should fail");

        // Verify connection state remains disconnected
        assert!(
            !oracle.is_connected().await,
            "Should remain disconnected after failed connect"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_reputation_oracle_concurrent_access() {
        let oracle = Arc::new(ReputationOracle::new("ws://localhost:9944".to_string()));

        // Generate test peers
        let mut peers = Vec::new();
        for _ in 0..10 {
            let keypair = Keypair::generate_ed25519();
            let peer_id = PeerId::from(keypair.public());
            peers.push(peer_id);
        }

        // Populate cache
        for (i, peer_id) in peers.iter().enumerate() {
            oracle.set_reputation(*peer_id, (i as u64 + 1) * 100).await;
        }

        // Spawn multiple concurrent tasks reading scores
        let mut handles = Vec::new();

        for _ in 0..20 {
            let oracle_clone = oracle.clone();
            let peers_clone = peers.clone();

            let handle = tokio::spawn(async move {
                for peer_id in peers_clone.iter() {
                    // Concurrent get_reputation calls
                    let score = oracle_clone.get_reputation(peer_id).await;
                    assert!(score > 0, "Score should be non-zero");

                    // Concurrent get_gossipsub_score calls
                    let gossip_score = oracle_clone.get_gossipsub_score(peer_id).await;
                    assert!(
                        gossip_score >= 0.0 && gossip_score <= 50.0,
                        "GossipSub score should be in range [0, 50]"
                    );
                }

                // Test concurrent cache_size access
                let size = oracle_clone.cache_size().await;
                assert!(size > 0, "Cache should not be empty");
            });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.expect("Task should complete without panic");
        }

        // Verify cache integrity after concurrent access
        assert_eq!(
            oracle.cache_size().await,
            10,
            "Cache should still have 10 entries"
        );

        for (i, peer_id) in peers.iter().enumerate() {
            let score = oracle.get_reputation(peer_id).await;
            assert_eq!(
                score,
                (i as u64 + 1) * 100,
                "Reputation should be unchanged after concurrent reads"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_reputation_oracle_concurrent_write_access() {
        let oracle = Arc::new(ReputationOracle::new("ws://localhost:9944".to_string()));

        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        // Spawn multiple concurrent tasks writing to the same peer
        let mut handles = Vec::new();

        for i in 0..10 {
            let oracle_clone = oracle.clone();
            let score = (i + 1) * 100;

            let handle = tokio::spawn(async move {
                oracle_clone.set_reputation(peer_id, score).await;
            });

            handles.push(handle);
        }

        // Wait for all writes to complete
        for handle in handles {
            handle.await.expect("Task should complete without panic");
        }

        // Verify final state is valid (one of the written values)
        let final_score = oracle.get_reputation(&peer_id).await;
        assert!(
            final_score % 100 == 0 && final_score <= 1000,
            "Final score should be one of the written values"
        );
    }

    #[tokio::test]
    async fn test_sync_loop_connection_recovery() {
        // Test that sync_loop handles connection failures gracefully
        let oracle = Arc::new(ReputationOracle::new("ws://invalid-host:9999".to_string()));

        // Spawn sync_loop with timeout (it will retry connection every 10s)
        let oracle_clone = oracle.clone();
        let sync_handle = tokio::spawn(async move {
            oracle_clone.sync_loop().await;
        });

        // Give it time to attempt connection
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify it remains disconnected (can't connect to invalid host)
        assert!(
            !oracle.is_connected().await,
            "Should remain disconnected with invalid RPC URL"
        );

        // Abort the sync loop (it would run forever)
        sync_handle.abort();
    }

    // Note: Integration tests with actual chain connection would go in tests/ directory
}
