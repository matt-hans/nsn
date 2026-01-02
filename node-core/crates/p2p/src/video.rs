//! Video chunking, signing, and validation for GossipSub distribution.
//!
//! Provides helpers to chunk a video payload, sign each chunk with a libp2p
//! keypair, and validate inbound chunks for integrity checks and latency
//! measurement.

use crate::topics::TopicCategory;
use libp2p::identity::{Keypair, PublicKey};
use nsn_types::{VideoChunk, VideoChunkHeader};
use parity_scale_codec::{Decode, Encode};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::oneshot;
use tokio::time::{timeout, Duration, Instant};

use crate::service::ServiceCommand;

/// Default chunk size for video distribution (1 MiB).
pub const DEFAULT_CHUNK_SIZE_BYTES: usize = 1024 * 1024;

/// Video chunk format version.
pub const VIDEO_CHUNK_VERSION: u16 = 1;

/// Configuration for video chunking.
#[derive(Debug, Clone)]
pub struct VideoChunkConfig {
    /// Chunk size in bytes.
    pub chunk_size: usize,
    /// Keyframe interval in chunks (0 = only first chunk is keyframe).
    pub keyframe_interval: u32,
    /// Message format version.
    pub version: u16,
}

impl Default for VideoChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE_BYTES,
            keyframe_interval: 0,
            version: VIDEO_CHUNK_VERSION,
        }
    }
}

/// Report for a publish operation across multiple chunks.
#[derive(Debug, Clone)]
pub struct VideoPublishReport {
    pub total_chunks: u32,
    pub published: u32,
    pub failed: u32,
    pub max_ack_ms: u64,
    pub avg_ack_ms: u64,
}

/// Video chunking/signing errors.
#[derive(Debug, Error)]
pub enum VideoChunkError {
    #[error("content id is empty")]
    EmptyContentId,
    #[error("payload is empty")]
    EmptyPayload,
    #[error("invalid chunk size")]
    InvalidChunkSize,
    #[error("signature failed")]
    SigningFailed,
    #[error("decode failed: {0}")]
    DecodeFailed(String),
    #[error("invalid signer public key")]
    InvalidSigner,
    #[error("signature verification failed")]
    InvalidSignature,
    #[error("payload hash mismatch")]
    PayloadHashMismatch,
    #[error("publish failed: {0}")]
    PublishFailed(String),
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn signing_payload(header: &VideoChunkHeader) -> Vec<u8> {
    header.encode()
}

/// Build signed video chunks from a payload.
pub fn build_video_chunks(
    content_id: &str,
    slot: u64,
    payload: &[u8],
    keypair: &Keypair,
    config: &VideoChunkConfig,
) -> Result<Vec<VideoChunk>, VideoChunkError> {
    if content_id.trim().is_empty() {
        return Err(VideoChunkError::EmptyContentId);
    }
    if payload.is_empty() {
        return Err(VideoChunkError::EmptyPayload);
    }
    if config.chunk_size == 0 {
        return Err(VideoChunkError::InvalidChunkSize);
    }

    let max_allowed = TopicCategory::VideoChunks.max_message_size();
    if config.chunk_size > max_allowed {
        return Err(VideoChunkError::InvalidChunkSize);
    }

    let total_chunks = ((payload.len() + config.chunk_size - 1) / config.chunk_size) as u32;
    let signer = keypair.public().encode_protobuf();

    let mut chunks = Vec::with_capacity(total_chunks as usize);
    for chunk_index in 0..total_chunks {
        let start = (chunk_index as usize) * config.chunk_size;
        let end = std::cmp::min(start + config.chunk_size, payload.len());
        let slice = &payload[start..end];
        let payload_hash = *blake3::hash(slice).as_bytes();
        let timestamp_ms = now_ms();
        let is_keyframe = chunk_index == 0
            || (config.keyframe_interval > 0 && chunk_index % config.keyframe_interval == 0);

        let header = VideoChunkHeader {
            version: config.version,
            slot,
            content_id: content_id.to_string(),
            chunk_index,
            total_chunks,
            timestamp_ms,
            is_keyframe,
            payload_hash,
        };

        let signature = keypair
            .sign(&signing_payload(&header))
            .map_err(|_| VideoChunkError::SigningFailed)?;

        let chunk = VideoChunk {
            header,
            payload: slice.to_vec(),
            signer: signer.clone(),
            signature,
        };

        let encoded_len = chunk.encode().len();
        if encoded_len > max_allowed {
            return Err(VideoChunkError::InvalidChunkSize);
        }

        chunks.push(chunk);
    }

    Ok(chunks)
}

/// Decode a video chunk from bytes.
pub fn decode_video_chunk(data: &[u8]) -> Result<VideoChunk, VideoChunkError> {
    VideoChunk::decode(&mut &data[..]).map_err(|err| VideoChunkError::DecodeFailed(err.to_string()))
}

/// Verify a video chunk signature and payload hash.
pub fn verify_video_chunk(chunk: &VideoChunk) -> Result<(), VideoChunkError> {
    let computed = *blake3::hash(&chunk.payload).as_bytes();
    if computed != chunk.header.payload_hash {
        return Err(VideoChunkError::PayloadHashMismatch);
    }

    let public_key = PublicKey::try_decode_protobuf(&chunk.signer)
        .map_err(|_| VideoChunkError::InvalidSigner)?;

    if !public_key.verify(&signing_payload(&chunk.header), &chunk.signature) {
        return Err(VideoChunkError::InvalidSignature);
    }

    Ok(())
}

/// Compute latency (ms) from the chunk's timestamp to now.
pub fn chunk_latency_ms(chunk: &VideoChunk) -> u64 {
    now_ms().saturating_sub(chunk.header.timestamp_ms)
}

/// Publish video chunks over GossipSub using the P2P service command channel.
pub async fn publish_video_chunks(
    cmd_tx: &tokio::sync::mpsc::UnboundedSender<ServiceCommand>,
    chunks: Vec<VideoChunk>,
    ack_timeout: Duration,
) -> Result<VideoPublishReport, VideoChunkError> {
    if chunks.is_empty() {
        return Err(VideoChunkError::EmptyPayload);
    }

    let total_chunks = chunks.len() as u32;
    let mut published = 0u32;
    let failed = 0u32;
    let mut max_ack_ms = 0u64;
    let mut ack_total_ms = 0u64;

    for chunk in chunks {
        let encoded = chunk.encode();
        let (tx, rx) = oneshot::channel();
        cmd_tx
            .send(ServiceCommand::Publish(
                TopicCategory::VideoChunks,
                encoded,
                tx,
            ))
            .map_err(|err| VideoChunkError::PublishFailed(err.to_string()))?;

        let start = Instant::now();
        match timeout(ack_timeout, rx).await {
            Ok(Ok(Ok(_message_id))) => {
                let elapsed = start.elapsed().as_millis() as u64;
                published += 1;
                ack_total_ms += elapsed;
                if elapsed > max_ack_ms {
                    max_ack_ms = elapsed;
                }
            }
            Ok(Ok(Err(err))) => {
                return Err(VideoChunkError::PublishFailed(err.to_string()));
            }
            Ok(Err(_)) => {
                return Err(VideoChunkError::PublishFailed(
                    "publish channel dropped".to_string(),
                ));
            }
            Err(_) => {
                return Err(VideoChunkError::PublishFailed(
                    "publish timeout".to_string(),
                ));
            }
        }
    }

    let avg_ack_ms = if published > 0 {
        ack_total_ms / published as u64
    } else {
        0
    };

    Ok(VideoPublishReport {
        total_chunks,
        published,
        failed,
        max_ack_ms,
        avg_ack_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    #[test]
    fn test_build_and_verify_chunks() {
        let keypair = Keypair::generate_ed25519();
        let payload = vec![1u8; 1024 * 4];
        let config = VideoChunkConfig {
            chunk_size: 1024,
            keyframe_interval: 2,
            version: VIDEO_CHUNK_VERSION,
        };

        let chunks = build_video_chunks("QmTest", 7, &payload, &keypair, &config).expect("chunks");
        assert_eq!(chunks.len(), 4);
        assert!(chunks[0].header.is_keyframe);
        assert!(chunks[2].header.is_keyframe);

        for chunk in &chunks {
            verify_video_chunk(chunk).expect("valid chunk");
            let encoded = chunk.encode();
            let decoded = decode_video_chunk(&encoded).expect("decode");
            assert_eq!(decoded, *chunk);
        }
    }
}
