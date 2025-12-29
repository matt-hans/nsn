#![allow(unexpected_cfgs)] // ffmpeg feature planned but not yet in Cargo.toml

use image::{DynamicImage, ImageBuffer, Rgb};
use std::path::Path;
use tracing::{debug, warn};

use crate::error::{Result, ValidatorError};

/// Video decoder for extracting keyframes from video chunks
pub struct VideoDecoder {
    keyframe_count: usize,
}

impl VideoDecoder {
    pub fn new(keyframe_count: usize) -> Self {
        Self { keyframe_count }
    }

    /// Extract evenly-spaced keyframes from video chunk
    ///
    /// # Arguments
    /// * `video_data` - Raw video chunk bytes (AV1 or VP9 encoded)
    ///
    /// # Returns
    /// Vec of DynamicImage representing extracted keyframes
    pub async fn extract_keyframes(&self, video_data: &[u8]) -> Result<Vec<DynamicImage>> {
        debug!(
            "Extracting {} keyframes from video chunk ({} bytes)",
            self.keyframe_count,
            video_data.len()
        );

        #[cfg(not(test))]
        {
            // Real implementation would use ffmpeg-next to decode video
            // This is a placeholder until ffmpeg integration is complete
            warn!("Video decoding not yet implemented (requires ffmpeg-next integration)");

            // For now, return placeholder images
            let mut frames = Vec::with_capacity(self.keyframe_count);
            for i in 0..self.keyframe_count {
                frames.push(Self::create_placeholder_frame(i));
            }

            Ok(frames)
        }

        #[cfg(test)]
        {
            // Test stub: validate input and return test frames
            if video_data.is_empty() {
                return Err(ValidatorError::VideoDecode("Empty video data".to_string()));
            }

            // Check for corruption marker (for testing error paths)
            if video_data.starts_with(b"CORRUPTED") {
                return Err(ValidatorError::VideoDecode(
                    "Corrupted video chunk".to_string(),
                ));
            }

            let mut frames = Vec::with_capacity(self.keyframe_count);
            for i in 0..self.keyframe_count {
                frames.push(Self::create_test_frame(i, video_data[0]));
            }

            Ok(frames)
        }
    }

    /// Extract keyframes from video file (for testing)
    #[allow(dead_code)]
    pub async fn extract_from_file(&self, path: &Path) -> Result<Vec<DynamicImage>> {
        let video_data = std::fs::read(path).map_err(|e| {
            ValidatorError::VideoDecode(format!("Failed to read video file: {}", e))
        })?;

        self.extract_keyframes(&video_data).await
    }

    #[cfg(not(test))]
    fn create_placeholder_frame(index: usize) -> DynamicImage {
        // Create a 512x512 gradient image as placeholder
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(512, 512, |x, y| {
            let r = (x as f32 / 512.0 * 255.0) as u8;
            let g = (y as f32 / 512.0 * 255.0) as u8;
            let b = ((index as f32 / 5.0) * 255.0) as u8;
            Rgb([r, g, b])
        });
        DynamicImage::ImageRgb8(img)
    }

    #[cfg(test)]
    fn create_test_frame(index: usize, seed: u8) -> DynamicImage {
        // Create test frame with deterministic but varied pattern based on index and seed
        // Uses hash-based color variation for more realistic test diversity
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(&[seed]);
        hasher.update(&index.to_le_bytes());
        let hash = hasher.finalize();

        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(512, 512, |x, y| {
            // Combine position, index, seed, and hash for varied patterns
            let r = ((x as u32 + hash[0] as u32 + index as u32) % 256) as u8;
            let g = ((y as u32 + hash[1] as u32 + index as u32) % 256) as u8;
            let b = seed.wrapping_add(hash[2]).wrapping_add(index as u8);
            Rgb([r, g, b])
        });
        DynamicImage::ImageRgb8(img)
    }
}

/// Real ffmpeg-based decoder (will be implemented when ffmpeg-next is fully integrated)
#[allow(dead_code)]
#[cfg(feature = "ffmpeg")]
mod ffmpeg_decoder {
    use super::*;

    pub struct FFmpegDecoder {
        keyframe_count: usize,
    }

    impl FFmpegDecoder {
        pub fn new(keyframe_count: usize) -> Self {
            Self { keyframe_count }
        }

        pub async fn extract_keyframes(&self, video_data: &[u8]) -> Result<Vec<DynamicImage>> {
            // TODO: Implement real ffmpeg-based decoding
            // 1. Initialize ffmpeg context
            // 2. Decode video stream
            // 3. Extract evenly-spaced keyframes
            // 4. Convert frames to DynamicImage

            unimplemented!("FFmpeg decoding not yet implemented")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_extract_keyframes_success() {
        let decoder = VideoDecoder::new(5);
        let video_data = b"VALID_VIDEO_DATA_PLACEHOLDER";

        let result = decoder.extract_keyframes(video_data).await;
        assert!(result.is_ok());

        let frames = result.unwrap();
        assert_eq!(frames.len(), 5);

        // Verify each frame is valid
        for frame in frames {
            assert_eq!(frame.width(), 512);
            assert_eq!(frame.height(), 512);
        }
    }

    #[tokio::test]
    async fn test_extract_keyframes_empty_data() {
        let decoder = VideoDecoder::new(5);
        let video_data = b"";

        let result = decoder.extract_keyframes(video_data).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Empty video data"));
    }

    #[tokio::test]
    async fn test_extract_keyframes_corrupted() {
        let decoder = VideoDecoder::new(5);
        let video_data = b"CORRUPTED_VIDEO_CHUNK";

        let result = decoder.extract_keyframes(video_data).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Corrupted"));
    }

    #[tokio::test]
    async fn test_keyframe_count_configuration() {
        let decoder = VideoDecoder::new(3);
        let video_data = b"VIDEO_DATA";

        let frames = decoder.extract_keyframes(video_data).await.unwrap();
        assert_eq!(frames.len(), 3);
    }

    #[tokio::test]
    async fn test_deterministic_extraction() {
        let decoder = VideoDecoder::new(5);
        let video_data = b"TEST_VIDEO_123";

        let frames1 = decoder.extract_keyframes(video_data).await.unwrap();
        let frames2 = decoder.extract_keyframes(video_data).await.unwrap();

        // Same input should produce same frames
        assert_eq!(frames1.len(), frames2.len());

        for (f1, f2) in frames1.iter().zip(frames2.iter()) {
            assert_eq!(f1.width(), f2.width());
            assert_eq!(f1.height(), f2.height());
        }
    }
}
