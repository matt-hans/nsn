//! Storage cleanup for expired pinning deals

use crate::chain_client::ChainClient;
use crate::metrics;
use crate::storage::Storage;
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};

pub struct StorageCleanup {
    cleanup_interval_blocks: u64,
    chain_client: Arc<ChainClient>,
    storage: Arc<Storage>,
}

impl StorageCleanup {
    pub fn new(
        cleanup_interval_blocks: u64,
        chain_client: Arc<ChainClient>,
        storage: Arc<Storage>,
    ) -> Self {
        Self {
            cleanup_interval_blocks,
            chain_client,
            storage,
        }
    }

    /// Start storage cleanup loop
    ///
    /// Runs every cleanup_interval_blocks and removes expired content
    pub async fn start(self) -> crate::error::Result<()> {
        info!(
            "Storage cleanup task started (interval: {} blocks)",
            self.cleanup_interval_blocks
        );

        // Convert blocks to approximate time (assuming 6s per block)
        let interval_secs = self.cleanup_interval_blocks * 6;
        let mut cleanup_timer = interval(Duration::from_secs(interval_secs));

        loop {
            cleanup_timer.tick().await;

            debug!("Running storage cleanup");

            // Get current finalized block
            match self.chain_client.get_finalized_block().await {
                Ok(current_block) => {
                    if let Err(e) = self.cleanup_expired_content(current_block).await {
                        error!("Storage cleanup failed: {}", e);
                    }
                }
                Err(e) => {
                    // In offline mode, this is expected
                    if self.chain_client.is_connected() {
                        error!("Failed to get current block: {}", e);
                    } else {
                        debug!("Skipping cleanup (chain not connected): {}", e);
                    }
                }
            }
        }
    }

    /// Clean up expired content
    ///
    /// Queries on-chain PinningDeals and deletes shards for expired deals
    async fn cleanup_expired_content(&self, current_block: u64) -> crate::error::Result<()> {
        debug!("Cleanup check at block {}", current_block);

        // Query all pinning deals from chain
        let deals = self.chain_client.get_pinning_deals().await?;

        if deals.is_empty() {
            debug!("No pinning deals found");
            return Ok(());
        }

        info!("Checking {} pinning deals for expiration", deals.len());

        let mut deleted_count = 0;
        let mut deleted_bytes = 0u64;

        for deal in deals {
            // Check if deal has expired
            if deal.expires_at < current_block {
                info!(
                    "Pinning deal {} expired at block {} (current: {}), deleting shards for CID: {}",
                    deal.deal_id, deal.expires_at, current_block, deal.cid
                );

                // Calculate shard sizes before deletion for metrics
                match self.calculate_shard_sizes(&deal.cid).await {
                    Ok(size) => {
                        deleted_bytes += size;
                    }
                    Err(e) => {
                        warn!("Failed to calculate shard sizes for {}: {}", deal.cid, e);
                    }
                }

                // Delete shards from storage
                match self.storage.delete_shards(&deal.cid).await {
                    Ok(()) => {
                        deleted_count += 1;
                        info!(
                            "Deleted shards for expired deal {}: CID={}",
                            deal.deal_id, deal.cid
                        );
                    }
                    Err(e) => {
                        error!("Failed to delete shards for CID {}: {}", deal.cid, e);
                    }
                }

                // TODO: Remove DHT manifest
                // This requires P2P service reference
                // For now, we log that this step is needed
                debug!("TODO: Remove DHT manifest for CID {}", deal.cid);
            }
        }

        // Update metrics
        if deleted_count > 0 {
            // Note: SHARD_COUNT tracks total shards (typically 14 per deal)
            metrics::SHARD_COUNT.sub((deleted_count * 14) as i64);
            metrics::BYTES_STORED.sub(deleted_bytes as i64);

            info!(
                "Cleanup completed: deleted {} expired deals ({} shards, {} bytes)",
                deleted_count,
                deleted_count * 14,
                deleted_bytes
            );
        } else {
            debug!("Cleanup completed: no expired deals found");
        }

        Ok(())
    }

    /// Calculate total size of shards for a CID
    ///
    /// Returns total bytes across all shards (typically 14 shards)
    async fn calculate_shard_sizes(&self, cid: &str) -> crate::error::Result<u64> {
        let mut total_size = 0u64;

        // Assume 14 shards (10 data + 4 parity)
        for shard_index in 0..14 {
            match self.storage.get_shard(cid, shard_index).await {
                Ok(shard_data) => {
                    total_size += shard_data.len() as u64;
                }
                Err(_) => {
                    // Shard may not exist (already deleted or never stored)
                    break;
                }
            }
        }

        Ok(total_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_cleanup_creation() {
        let tmp_dir = tempdir().unwrap();
        let storage = Arc::new(Storage::new(tmp_dir.path().to_path_buf()));

        let (chain_client, _rx) = ChainClient::connect("ws://127.0.0.1:9944".to_string())
            .await
            .unwrap();
        let chain_client = Arc::new(chain_client);

        let cleanup = StorageCleanup::new(1000, chain_client, storage);

        // Just verify creation succeeds
        assert_eq!(cleanup.cleanup_interval_blocks, 1000);
    }
}
