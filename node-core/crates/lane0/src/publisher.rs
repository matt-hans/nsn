//! Video chunk publisher for Lane 0 distribution.
//!
//! Handles chunking, signing, and publishing video content to the P2P network.
//! Uses the existing `nsn_p2p::video` module for chunk creation and distribution.

use std::time::Duration;

use libp2p::identity::Keypair;
use tokio::sync::mpsc;
use tracing::{debug, info};

use nsn_p2p::{
    build_video_chunks, publish_video_chunks, ServiceCommand, VideoChunkConfig, VideoPublishReport,
};
use nsn_types::VideoChunkHeader;

use crate::error::{PublishError, PublishResult};

/// Default chunk size for video distribution (1 MiB).
pub const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;

/// Default acknowledgment timeout for publishing.
pub const DEFAULT_ACK_TIMEOUT_MS: u64 = 2000;

/// Configuration for the chunk publisher.
#[derive(Debug, Clone)]
pub struct PublisherConfig {
    /// Chunk size in bytes.
    pub chunk_size: usize,
    /// Keyframe interval (0 = only first chunk is keyframe).
    pub keyframe_interval: u32,
    /// Timeout for publish acknowledgments in milliseconds.
    pub ack_timeout_ms: u64,
}

impl Default for PublisherConfig {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            keyframe_interval: 0,
            ack_timeout_ms: DEFAULT_ACK_TIMEOUT_MS,
        }
    }
}

/// Video chunk publisher for Lane 0.
///
/// Chunks video data, signs each chunk with the node's keypair,
/// and publishes them to the P2P GossipSub network.
pub struct ChunkPublisher {
    /// Node keypair for signing chunks.
    keypair: Keypair,
    /// Configuration.
    config: PublisherConfig,
    /// P2P command sender.
    p2p_tx: mpsc::UnboundedSender<ServiceCommand>,
}

impl ChunkPublisher {
    /// Create a new chunk publisher.
    ///
    /// # Arguments
    ///
    /// * `keypair` - Node identity keypair for signing chunks
    /// * `config` - Publisher configuration
    /// * `p2p_tx` - Channel to P2P service for publishing
    pub fn new(
        keypair: Keypair,
        config: PublisherConfig,
        p2p_tx: mpsc::UnboundedSender<ServiceCommand>,
    ) -> Self {
        Self {
            keypair,
            config,
            p2p_tx,
        }
    }

    /// Get the publisher configuration.
    pub fn config(&self) -> &PublisherConfig {
        &self.config
    }

    /// Publish video data for a slot.
    ///
    /// Chunks the video, signs each chunk, and publishes to P2P.
    /// Returns headers for all published chunks.
    ///
    /// # Arguments
    ///
    /// * `slot` - Slot number for the video
    /// * `content_id` - Content identifier (CID or hash)
    /// * `video_data` - Raw video bytes to publish
    pub async fn publish_video(
        &self,
        slot: u64,
        content_id: &str,
        video_data: &[u8],
    ) -> PublishResult<Vec<VideoChunkHeader>> {
        if video_data.is_empty() {
            return Err(PublishError::EmptyVideo);
        }

        if content_id.is_empty() {
            return Err(PublishError::InvalidContentId("empty content ID".to_string()));
        }

        let chunk_config = VideoChunkConfig {
            chunk_size: self.config.chunk_size,
            keyframe_interval: self.config.keyframe_interval,
            ..Default::default()
        };

        debug!(
            slot,
            content_id,
            video_size = video_data.len(),
            chunk_size = chunk_config.chunk_size,
            "Building video chunks"
        );

        // Build signed chunks using P2P module
        let chunks = build_video_chunks(content_id, slot, video_data, &self.keypair, &chunk_config)
            .map_err(|e| PublishError::ChunkingFailed(e.to_string()))?;

        let num_chunks = chunks.len();
        debug!(slot, num_chunks, "Publishing video chunks");

        // Publish all chunks
        let report = publish_video_chunks(
            &self.p2p_tx,
            chunks.clone(),
            Duration::from_millis(self.config.ack_timeout_ms),
        )
        .await
        .map_err(|e| PublishError::P2pFailed(e.to_string()))?;

        self.log_publish_report(slot, &report);

        // Extract headers from published chunks
        let headers: Vec<VideoChunkHeader> = chunks.into_iter().map(|c| c.header).collect();

        info!(
            slot,
            chunks = headers.len(),
            content_id,
            "Video published successfully"
        );

        Ok(headers)
    }

    /// Log the publish report for observability.
    fn log_publish_report(&self, slot: u64, report: &VideoPublishReport) {
        info!(
            slot,
            total = report.total_chunks,
            published = report.published,
            failed = report.failed,
            max_ack_ms = report.max_ack_ms,
            avg_ack_ms = report.avg_ack_ms,
            "Video publish report"
        );
    }

    /// Calculate the expected number of chunks for given video size.
    pub fn estimate_chunks(&self, video_size: usize) -> usize {
        if video_size == 0 || self.config.chunk_size == 0 {
            return 0;
        }
        video_size.div_ceil(self.config.chunk_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PublisherConfig::default();
        assert_eq!(config.chunk_size, 1024 * 1024);
        assert_eq!(config.keyframe_interval, 0);
        assert_eq!(config.ack_timeout_ms, 2000);
    }

    #[test]
    fn test_estimate_chunks() {
        let keypair = Keypair::generate_ed25519();
        let (tx, _rx) = mpsc::unbounded_channel();
        let publisher = ChunkPublisher::new(keypair, PublisherConfig::default(), tx);

        // Exact multiple
        assert_eq!(publisher.estimate_chunks(1024 * 1024), 1);
        assert_eq!(publisher.estimate_chunks(2 * 1024 * 1024), 2);

        // Partial chunk
        assert_eq!(publisher.estimate_chunks(1024 * 1024 + 1), 2);

        // Edge cases
        assert_eq!(publisher.estimate_chunks(0), 0);
    }

    #[test]
    fn test_estimate_chunks_custom_size() {
        let keypair = Keypair::generate_ed25519();
        let (tx, _rx) = mpsc::unbounded_channel();
        let config = PublisherConfig {
            chunk_size: 1000,
            ..Default::default()
        };
        let publisher = ChunkPublisher::new(keypair, config, tx);

        assert_eq!(publisher.estimate_chunks(1000), 1);
        assert_eq!(publisher.estimate_chunks(1001), 2);
        assert_eq!(publisher.estimate_chunks(3000), 3);
    }
}
