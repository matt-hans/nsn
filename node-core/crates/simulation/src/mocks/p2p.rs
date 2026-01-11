//! Mock P2P service and chunk publisher for testing.

use std::collections::VecDeque;

use nsn_lane0::PublishResult;
use nsn_types::VideoChunkHeader;

/// A recorded publish event for verification.
#[derive(Debug, Clone)]
pub struct PublishEvent {
    /// Slot number
    pub slot: u64,
    /// Content identifier
    pub content_id: String,
    /// Number of chunks published
    pub num_chunks: usize,
    /// Total bytes published
    pub total_bytes: usize,
}

/// Mock chunk publisher for testing P2P video distribution.
///
/// Tracks all published video chunks for verification.
///
/// # Example
///
/// ```rust,ignore
/// use nsn_simulation::mocks::MockChunkPublisher;
///
/// let mut publisher = MockChunkPublisher::new();
/// let headers = publisher.publish_video(1, "QmTest", &video_data).await?;
/// assert_eq!(publisher.publish_count(), 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MockChunkPublisher {
    /// Published events (for verification)
    pub published: VecDeque<PublishEvent>,
    /// All chunk headers created
    pub all_headers: VecDeque<VideoChunkHeader>,
    /// Chunk size for splitting (default 1MB)
    chunk_size: usize,
    /// Whether to fail publishes
    fail_mode: bool,
}

impl MockChunkPublisher {
    /// Create a new mock publisher.
    pub fn new() -> Self {
        Self {
            published: VecDeque::new(),
            all_headers: VecDeque::new(),
            chunk_size: 1024 * 1024, // 1MB default
            fail_mode: false,
        }
    }

    /// Configure chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Configure failure mode.
    pub fn with_fail_mode(mut self, fail: bool) -> Self {
        self.fail_mode = fail;
        self
    }

    /// Get the number of publish events.
    pub fn publish_count(&self) -> usize {
        self.published.len()
    }

    /// Get total chunks published across all events.
    pub fn total_chunks_published(&self) -> usize {
        self.published.iter().map(|e| e.num_chunks).sum()
    }

    /// Get publish events for a specific slot.
    pub fn events_for_slot(&self, slot: u64) -> Vec<&PublishEvent> {
        self.published.iter().filter(|e| e.slot == slot).collect()
    }

    /// Clear published events.
    pub fn clear(&mut self) {
        self.published.clear();
        self.all_headers.clear();
    }

    /// Publish video to P2P network.
    pub async fn publish_video(
        &mut self,
        slot: u64,
        content_id: &str,
        video_data: &[u8],
    ) -> PublishResult<Vec<VideoChunkHeader>> {
        if self.fail_mode {
            return Err(nsn_lane0::PublishError::P2pFailed(
                "Mock failure mode enabled".to_string(),
            ));
        }

        let num_chunks = video_data.len().div_ceil(self.chunk_size);

        self.published.push_back(PublishEvent {
            slot,
            content_id: content_id.to_string(),
            num_chunks,
            total_bytes: video_data.len(),
        });

        // Create mock headers
        let headers: Vec<VideoChunkHeader> = (0..num_chunks as u32)
            .map(|i| {
                let header = VideoChunkHeader {
                    version: 1,
                    slot,
                    content_id: content_id.to_string(),
                    chunk_index: i,
                    total_chunks: num_chunks as u32,
                    timestamp_ms: 0,
                    is_keyframe: i == 0,
                    payload_hash: [0u8; 32],
                };
                self.all_headers.push_back(header.clone());
                header
            })
            .collect();

        Ok(headers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_publish_video() {
        let mut publisher = MockChunkPublisher::new();

        let video_data = vec![0u8; 2 * 1024 * 1024]; // 2MB
        let headers = publisher
            .publish_video(1, "QmTest", &video_data)
            .await
            .unwrap();

        assert_eq!(headers.len(), 2); // 2 chunks for 2MB
        assert_eq!(publisher.publish_count(), 1);
        assert_eq!(publisher.total_chunks_published(), 2);
    }

    #[tokio::test]
    async fn test_multiple_publishes() {
        let mut publisher = MockChunkPublisher::new();

        for slot in 1..=5 {
            let video_data = vec![0u8; 1024 * 1024];
            publisher
                .publish_video(slot, &format!("Qm{}", slot), &video_data)
                .await
                .unwrap();
        }

        assert_eq!(publisher.publish_count(), 5);
        assert_eq!(publisher.total_chunks_published(), 5);
    }

    #[tokio::test]
    async fn test_fail_mode() {
        let mut publisher = MockChunkPublisher::new().with_fail_mode(true);

        let video_data = vec![0u8; 1024];
        let result = publisher.publish_video(1, "QmTest", &video_data).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_custom_chunk_size() {
        let mut publisher = MockChunkPublisher::new().with_chunk_size(512 * 1024); // 512KB

        let video_data = vec![0u8; 1024 * 1024]; // 1MB
        let headers = publisher
            .publish_video(1, "QmTest", &video_data)
            .await
            .unwrap();

        assert_eq!(headers.len(), 2); // 1MB / 512KB = 2 chunks
    }

    #[tokio::test]
    async fn test_events_for_slot() {
        let mut publisher = MockChunkPublisher::new();

        let video_data = vec![0u8; 1024 * 1024];
        publisher
            .publish_video(1, "QmTest1", &video_data)
            .await
            .unwrap();
        publisher
            .publish_video(2, "QmTest2", &video_data)
            .await
            .unwrap();
        publisher
            .publish_video(1, "QmTest3", &video_data)
            .await
            .unwrap();

        let slot1_events = publisher.events_for_slot(1);
        assert_eq!(slot1_events.len(), 2);
    }
}
