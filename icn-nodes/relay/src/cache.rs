//! LRU cache with disk persistence
//!
//! Implements a Least Recently Used (LRU) cache for video shards with:
//! - 1TB capacity limit (configurable)
//! - Disk-backed persistence
//! - Automatic eviction when full
//! - Manifest-based state persistence across restarts

use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use tokio::fs;
use tracing::{debug, info, warn};

/// Shard key for cache lookup
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShardKey {
    pub cid: String,
    pub shard_index: usize,
}

impl ShardKey {
    pub fn new(cid: String, shard_index: usize) -> Self {
        Self { cid, shard_index }
    }

    /// Generate hash for shard filename
    pub fn hash(&self) -> String {
        format!("{}_{:02}", self.cid, self.shard_index)
    }
}

/// Cache manifest for persistence
#[derive(Debug, Serialize, Deserialize)]
struct CacheManifest {
    /// Map of shard key hash to (CID, shard_index, size_bytes, last_access_timestamp)
    entries: HashMap<String, (String, usize, u64, i64)>,
    total_size_bytes: u64,
}

/// LRU cache with disk persistence
pub struct ShardCache {
    /// LRU cache (key -> file path)
    cache: LruCache<ShardKey, PathBuf>,
    /// Cache root directory
    cache_dir: PathBuf,
    /// Maximum cache size in bytes
    max_size_bytes: u64,
    /// Current cache size in bytes
    current_size: u64,
    /// Manifest file path
    manifest_path: PathBuf,
}

impl ShardCache {
    /// Create new shard cache
    ///
    /// # Arguments
    /// * `cache_dir` - Root directory for cached shards
    /// * `max_size_gb` - Maximum cache size in GB (e.g., 1000 for 1TB)
    ///
    /// # Returns
    /// ShardCache instance with loaded manifest (if exists)
    pub async fn new(cache_dir: PathBuf, max_size_gb: u64) -> crate::error::Result<Self> {
        // Ensure cache directory exists
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir).await?;
        }

        let max_size_bytes = max_size_gb * 1_000_000_000; // GB to bytes
        let manifest_path = cache_dir.join("cache_manifest.json");

        // Initialize LRU cache with reasonable entry limit (10k shards)
        let cache = LruCache::new(NonZeroUsize::new(10_000).unwrap());

        let mut shard_cache = Self {
            cache,
            cache_dir,
            max_size_bytes,
            current_size: 0,
            manifest_path,
        };

        // Load existing manifest if present
        shard_cache.load_manifest().await?;

        info!(
            "Shard cache initialized: max {} GB, current {} MB",
            max_size_gb,
            shard_cache.current_size / 1_000_000
        );

        Ok(shard_cache)
    }

    /// Get shard from cache
    ///
    /// # Arguments
    /// * `key` - Shard key (CID + index)
    ///
    /// # Returns
    /// Shard data if cached, None otherwise
    pub async fn get(&mut self, key: &ShardKey) -> Option<Vec<u8>> {
        if let Some(path) = self.cache.get(key) {
            // Cache hit - read from disk
            match fs::read(path).await {
                Ok(data) => {
                    debug!("Cache HIT: {} ({} bytes)", key.hash(), data.len());
                    Some(data)
                }
                Err(e) => {
                    warn!("Cache read error for {}: {}", key.hash(), e);
                    // Remove invalid entry
                    self.cache.pop(key);
                    None
                }
            }
        } else {
            debug!("Cache MISS: {}", key.hash());
            None
        }
    }

    /// Put shard into cache
    ///
    /// # Arguments
    /// * `key` - Shard key (CID + index)
    /// * `data` - Shard bytes
    ///
    /// # Returns
    /// Result indicating success/failure
    pub async fn put(&mut self, key: ShardKey, data: Vec<u8>) -> crate::error::Result<()> {
        let shard_size = data.len() as u64;

        // Evict if needed
        while self.current_size + shard_size > self.max_size_bytes {
            if let Some((old_key, old_path)) = self.cache.pop_lru() {
                match fs::metadata(&old_path).await {
                    Ok(metadata) => {
                        let old_size = metadata.len();
                        if let Err(e) = fs::remove_file(&old_path).await {
                            warn!("Failed to delete evicted shard {}: {}", old_key.hash(), e);
                        } else {
                            debug!("Evicted shard: {} ({} bytes)", old_key.hash(), old_size);
                            self.current_size = self.current_size.saturating_sub(old_size);
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Failed to get metadata for evicted shard {}: {}",
                            old_key.hash(),
                            e
                        );
                    }
                }
            } else {
                // Cache empty but still can't fit
                return Err(crate::error::RelayError::CacheEvictionFailed(format!(
                    "Shard size {} exceeds max cache size {}",
                    shard_size, self.max_size_bytes
                )));
            }
        }

        // Write shard to disk
        let shard_filename = format!("{}.bin", key.hash());
        let path = self.cache_dir.join(&shard_filename);
        fs::write(&path, &data).await?;

        // Update cache
        self.cache.put(key.clone(), path);
        self.current_size += shard_size;

        debug!("Cached shard: {} ({} bytes)", key.hash(), shard_size);

        Ok(())
    }

    /// Load cache manifest from disk
    async fn load_manifest(&mut self) -> crate::error::Result<()> {
        if !self.manifest_path.exists() {
            debug!("No existing cache manifest found");
            return Ok(());
        }

        let manifest_data = fs::read_to_string(&self.manifest_path).await?;
        let manifest: CacheManifest = serde_json::from_str(&manifest_data)?;

        info!(
            "Loading cache manifest: {} entries, {} MB",
            manifest.entries.len(),
            manifest.total_size_bytes / 1_000_000
        );

        // Rebuild LRU cache from manifest (sorted by last_access_timestamp)
        let mut entries: Vec<_> = manifest.entries.into_iter().collect();
        entries.sort_by_key(|(_, (_, _, _, timestamp))| *timestamp);

        for (hash, (cid, shard_index, size_bytes, _timestamp)) in entries {
            let key = ShardKey::new(cid, shard_index);
            let path = self.cache_dir.join(format!("{}.bin", hash));

            // Verify file exists
            if path.exists() {
                self.cache.put(key, path);
                self.current_size += size_bytes;
            } else {
                warn!("Manifest entry {} missing from disk, skipping", hash);
            }
        }

        info!(
            "Cache manifest loaded: {} entries, {} MB",
            self.cache.len(),
            self.current_size / 1_000_000
        );

        Ok(())
    }

    /// Save cache manifest to disk
    ///
    /// Called during graceful shutdown to persist cache state
    pub async fn save_manifest(&self) -> crate::error::Result<()> {
        let mut entries = HashMap::new();
        let timestamp = chrono::Utc::now().timestamp();

        // Iterate through cache (LRU maintains access order)
        for (key, path) in self.cache.iter() {
            if let Ok(metadata) = fs::metadata(path).await {
                let size_bytes = metadata.len();
                let hash = key.hash();
                entries.insert(
                    hash,
                    (key.cid.clone(), key.shard_index, size_bytes, timestamp),
                );
            }
        }

        let manifest = CacheManifest {
            entries,
            total_size_bytes: self.current_size,
        };

        let manifest_json = serde_json::to_string_pretty(&manifest)?;
        fs::write(&self.manifest_path, manifest_json).await?;

        info!(
            "Cache manifest saved: {} entries, {} MB",
            manifest.entries.len(),
            manifest.total_size_bytes / 1_000_000
        );

        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.cache.len(),
            size_bytes: self.current_size,
            max_size_bytes: self.max_size_bytes,
            utilization_percent: (self.current_size as f64 / self.max_size_bytes as f64 * 100.0)
                as u32,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub size_bytes: u64,
    pub max_size_bytes: u64,
    pub utilization_percent: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Test Case: Cache put and get
    /// Purpose: Verify basic cache operations
    /// Contract: Put shard, then get returns same data
    #[tokio::test]
    async fn test_cache_put_get() {
        let tmp_dir = tempdir().unwrap();
        let mut cache = ShardCache::new(tmp_dir.path().to_path_buf(), 1)
            .await
            .unwrap();

        let key = ShardKey::new("bafytest123".to_string(), 0);
        let data = vec![1, 2, 3, 4, 5];

        // Put shard
        cache.put(key.clone(), data.clone()).await.unwrap();

        // Get shard
        let result = cache.get(&key).await;
        assert!(result.is_some(), "Should retrieve cached shard");
        assert_eq!(result.unwrap(), data);
    }

    /// Test Case: Cache miss
    /// Purpose: Verify behavior for non-existent key
    /// Contract: Returns None for uncached shard
    #[tokio::test]
    async fn test_cache_miss() {
        let tmp_dir = tempdir().unwrap();
        let mut cache = ShardCache::new(tmp_dir.path().to_path_buf(), 1)
            .await
            .unwrap();

        let key = ShardKey::new("nonexistent".to_string(), 99);

        let result = cache.get(&key).await;
        assert!(result.is_none(), "Should return None for cache miss");
    }

    /// Test Case: LRU eviction
    /// Purpose: Verify eviction when cache full
    /// Contract: Oldest shard evicted when size limit reached
    #[tokio::test]
    async fn test_cache_lru_eviction() {
        let tmp_dir = tempdir().unwrap();
        // Small cache: 1 KB
        let mut cache = ShardCache::new(tmp_dir.path().to_path_buf(), 0)
            .await
            .unwrap();
        cache.max_size_bytes = 1_000; // 1 KB for testing

        // Add shards until full
        let shard1 = ShardKey::new("cid1".to_string(), 0);
        let shard2 = ShardKey::new("cid2".to_string(), 0);
        let shard3 = ShardKey::new("cid3".to_string(), 0);

        let data = vec![0u8; 400]; // 400 bytes each

        cache.put(shard1.clone(), data.clone()).await.unwrap();
        cache.put(shard2.clone(), data.clone()).await.unwrap();
        cache.put(shard3.clone(), data.clone()).await.unwrap(); // This should evict shard1

        // shard1 should be evicted (LRU)
        assert!(
            cache.get(&shard1).await.is_none(),
            "shard1 should be evicted"
        );
        assert!(cache.get(&shard2).await.is_some(), "shard2 should remain");
        assert!(cache.get(&shard3).await.is_some(), "shard3 should remain");
    }

    /// Test Case: Cache persistence across restarts
    /// Purpose: Verify manifest save/load
    /// Contract: Cache state preserved after restart
    #[tokio::test]
    async fn test_cache_persistence() {
        let tmp_dir = tempdir().unwrap();
        let cache_path = tmp_dir.path().to_path_buf();

        // Create cache and add shard
        {
            let mut cache = ShardCache::new(cache_path.clone(), 1).await.unwrap();
            let key = ShardKey::new("persistent_cid".to_string(), 5);
            let data = vec![9, 8, 7, 6, 5];

            cache.put(key.clone(), data.clone()).await.unwrap();
            cache.save_manifest().await.unwrap();
        }

        // Restart cache (new instance)
        {
            let mut cache = ShardCache::new(cache_path.clone(), 1).await.unwrap();
            let key = ShardKey::new("persistent_cid".to_string(), 5);

            // Should load from manifest
            let result = cache.get(&key).await;
            assert!(result.is_some(), "Should load shard from manifest");
            assert_eq!(result.unwrap(), vec![9, 8, 7, 6, 5]);
        }
    }

    /// Test Case: Shard too large for cache
    /// Purpose: Verify error handling for oversized shard
    /// Contract: Returns CacheEvictionFailed error
    #[tokio::test]
    async fn test_cache_shard_too_large() {
        let tmp_dir = tempdir().unwrap();
        let mut cache = ShardCache::new(tmp_dir.path().to_path_buf(), 0)
            .await
            .unwrap();
        cache.max_size_bytes = 100; // Tiny cache

        let key = ShardKey::new("huge_shard".to_string(), 0);
        let data = vec![0u8; 200]; // 200 bytes > 100 byte limit

        let result = cache.put(key, data).await;
        assert!(result.is_err(), "Should reject oversized shard");

        match result.unwrap_err() {
            crate::error::RelayError::CacheEvictionFailed(_) => {
                // Expected error
            }
            e => panic!("Unexpected error: {:?}", e),
        }
    }

    /// Test Case: ShardKey hash generation
    /// Purpose: Verify consistent hash for filenames
    #[test]
    fn test_shard_key_hash() {
        let key1 = ShardKey::new("bafy123".to_string(), 7);
        let key2 = ShardKey::new("bafy123".to_string(), 7);
        let key3 = ShardKey::new("bafy456".to_string(), 7);

        assert_eq!(key1.hash(), key2.hash(), "Same key should hash identically");
        assert_ne!(
            key1.hash(),
            key3.hash(),
            "Different CIDs should hash differently"
        );
        assert_eq!(key1.hash(), "bafy123_07");
    }

    /// Test Case: Cache stats
    /// Purpose: Verify statistics reporting
    #[tokio::test]
    async fn test_cache_stats() {
        let tmp_dir = tempdir().unwrap();
        let mut cache = ShardCache::new(tmp_dir.path().to_path_buf(), 1)
            .await
            .unwrap();

        let key1 = ShardKey::new("stats1".to_string(), 0);
        let key2 = ShardKey::new("stats2".to_string(), 1);
        let data = vec![0u8; 1000]; // 1 KB each

        cache.put(key1, data.clone()).await.unwrap();
        cache.put(key2, data.clone()).await.unwrap();

        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.size_bytes, 2000);
        // With 1GB cache and 2KB used, utilization rounds to 0%
        assert!(stats.size_bytes > 0);
    }
}
