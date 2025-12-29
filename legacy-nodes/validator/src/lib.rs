//! ICN Validator Node
//!
//! Semantic verification layer for director-generated content using CLIP (Contrastive Language-Image Pretraining).
//!
//! ## Overview
//!
//! Validators perform three core functions:
//! 1. **Content Verification**: Run CLIP-ViT-B-32 + CLIP-ViT-L-14 dual ensemble on video frames
//! 2. **Attestation Generation**: Sign verification results with Ed25519 keypair
//! 3. **Challenge Participation**: Provide attestations during BFT dispute resolution
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐    ┌─────────────┐   ┌────────────┐
//! │   Config    │───▶│    Main     │───▶│  Metrics   │
//! │   Loader    │    │   Runtime   │    │  Server    │
//! └─────────────┘    └─────────────┘   └────────────┘
//!                           │
//!        ┌──────────────────┼──────────────────┐
//!        ▼                  ▼                  ▼
//! ┌─────────────┐    ┌─────────────┐   ┌────────────┐
//! │    Chain    │    │    CLIP     │   │    P2P     │
//! │   Client    │    │   Engine    │   │  Service   │
//! │  (subxt)    │    │  (ONNX RT)  │   │ (libp2p)   │
//! └─────────────┘    └─────────────┘   └────────────┘
//! ```
//!
//! ## Example Usage
//!
//! ```no_run
//! use icn_validator::{ValidatorConfig, ValidatorNode};
//! use std::path::Path;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load configuration
//!     let config = ValidatorConfig::from_file(Path::new("config/validator.toml"))?;
//!
//!     // Create and start validator node
//!     let validator = ValidatorNode::new(config).await?;
//!     validator.run().await?;
//!
//!     Ok(())
//! }
//! ```

pub mod attestation;
pub mod chain_client;
pub mod challenge_monitor;
pub mod clip_engine;
pub mod config;
pub mod error;
pub mod metrics;
pub mod p2p_service;
pub mod video_decoder;

// Re-export key types
pub use attestation::{derive_peer_id, load_keypair, Attestation};
pub use chain_client::ChainClient;
pub use challenge_monitor::ChallengeMonitor;
pub use clip_engine::ClipEngine;
pub use config::{ChallengeConfig, ClipConfig, MetricsConfig, P2PConfig, ValidatorConfig};
pub use error::{Result, ValidatorError};
pub use metrics::ValidatorMetrics;
pub use p2p_service::P2PService;
pub use video_decoder::VideoDecoder;

use ed25519_dalek::SigningKey;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, instrument};

/// Main validator node runtime
pub struct ValidatorNode {
    config: ValidatorConfig,
    signing_key: SigningKey,
    validator_id: String,
    clip_engine: Arc<ClipEngine>,
    video_decoder: Arc<VideoDecoder>,
    p2p_service: Arc<RwLock<P2PService>>,
    chain_client: Arc<ChainClient>,
    metrics: Arc<ValidatorMetrics>,
}

impl ValidatorNode {
    /// Create a new validator node
    pub async fn new(config: ValidatorConfig) -> Result<Self> {
        info!("Initializing ICN Validator Node");

        // Load Ed25519 keypair
        let signing_key = load_keypair(&config.keypair_path)?;
        let validator_id = derive_peer_id(&signing_key);
        info!("Validator ID: {}", validator_id);

        // Initialize CLIP engine
        let clip_engine = Arc::new(ClipEngine::new(&config.models_dir, config.clip.clone())?);

        // Initialize video decoder
        let video_decoder = Arc::new(VideoDecoder::new(config.clip.keyframe_count));

        // Initialize P2P service
        let p2p_service = Arc::new(RwLock::new(P2PService::new(config.p2p.clone())?));

        // Initialize chain client
        let chain_client = Arc::new(ChainClient::new(config.chain_endpoint.clone()).await?);

        // Initialize metrics
        let metrics = Arc::new(ValidatorMetrics::new()?);

        info!("Validator node initialized successfully");

        Ok(Self {
            config,
            signing_key,
            validator_id,
            clip_engine,
            video_decoder,
            p2p_service,
            chain_client,
            metrics,
        })
    }

    /// Run the validator node
    pub async fn run(self) -> Result<()> {
        info!("Starting ICN Validator Node");

        // Start metrics server
        let metrics_clone = self.metrics.clone();
        let metrics_config = self.config.metrics.clone();
        tokio::spawn(async move {
            if let Err(e) = Self::run_metrics_server(metrics_clone, metrics_config).await {
                tracing::error!("Metrics server error: {}", e);
            }
        });

        // Start P2P service
        {
            let mut p2p = self.p2p_service.write().await;
            p2p.start().await?;
            p2p.subscribe_video_chunks().await?;
            p2p.subscribe_challenges().await?;
        }

        // Start challenge monitor
        if self.config.challenge.enabled {
            let monitor =
                ChallengeMonitor::new(self.config.challenge.clone(), (*self.chain_client).clone());

            tokio::spawn(async move {
                if let Err(e) = monitor.start().await {
                    tracing::error!("Challenge monitor error: {}", e);
                }
            });
        }

        info!("Validator node running");

        // Main event loop
        self.event_loop().await
    }

    async fn event_loop(&self) -> Result<()> {
        use tokio::sync::mpsc;
        use tracing::warn;

        // Create channel for video chunk reception
        let (_tx, mut rx) = mpsc::channel::<(u64, Vec<u8>, String)>(100);

        // Spawn task to receive video chunks from P2P
        let _p2p_service = self.p2p_service.clone();
        tokio::spawn(async move {
            // In real implementation, this would listen to libp2p GossipSub
            // and forward chunks to the channel. For now, stub.
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            }
        });

        info!("Event loop started, waiting for video chunks");

        // Main event loop: process incoming video chunks
        loop {
            tokio::select! {
                // Receive video chunk from P2P
                Some((slot, video_data, prompt)) = rx.recv() => {
                    info!("Received video chunk for slot {}", slot);

                    // Validate chunk
                    match self.validate_chunk(slot, &video_data, &prompt).await {
                        Ok(attestation) => {
                            info!("Successfully validated slot {}: score={:.4}, passed={}",
                                  slot, attestation.clip_score, attestation.passed);
                        }
                        Err(e) => {
                            warn!("Failed to validate slot {}: {}", slot, e);
                        }
                    }
                }

                // Graceful shutdown signal (would be added in real implementation)
                _ = tokio::signal::ctrl_c() => {
                    info!("Shutdown signal received, stopping event loop");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Validate a video chunk
    #[instrument(skip(self, video_data, prompt))]
    pub async fn validate_chunk(
        &self,
        slot: u64,
        video_data: &[u8],
        prompt: &str,
    ) -> Result<Attestation> {
        let start = std::time::Instant::now();

        // Extract keyframes
        let frames = match self.video_decoder.extract_keyframes(video_data).await {
            Ok(f) => f,
            Err(e) => {
                self.metrics.record_frame_error();
                return Err(e);
            }
        };

        // Run CLIP inference
        let clip_start = std::time::Instant::now();
        let score = match self.clip_engine.compute_score(&frames, prompt).await {
            Ok(s) => s,
            Err(e) => {
                self.metrics.record_clip_error();
                return Err(e);
            }
        };
        let clip_duration = clip_start.elapsed().as_secs_f64();
        self.metrics.record_clip_inference(clip_duration);

        // Create and sign attestation
        let attestation = Attestation::new(
            slot,
            self.validator_id.clone(),
            score,
            self.config.clip.threshold,
        )?
        .sign(&self.signing_key)?;

        // Record metrics
        let total_duration = start.elapsed().as_secs_f64();
        self.metrics
            .record_validation(score, attestation.passed, total_duration);

        // Broadcast attestation
        {
            let mut p2p = self.p2p_service.write().await;
            p2p.publish_attestation(&attestation).await?;
        }
        self.metrics.record_attestation();

        info!(
            "Validated slot {}: score={:.4}, passed={}, duration={:.2}s",
            slot, score, attestation.passed, total_duration
        );

        Ok(attestation)
    }

    async fn run_metrics_server(
        metrics: Arc<ValidatorMetrics>,
        config: MetricsConfig,
    ) -> Result<()> {
        use http_body_util::Full;
        use hyper::body::Bytes;
        use hyper::server::conn::http1;
        use hyper::service::service_fn;
        use hyper::{Request, Response};
        use hyper_util::rt::TokioIo;
        use prometheus::Encoder;
        use tokio::net::TcpListener;

        let addr = format!("{}:{}", config.listen_address, config.port);
        let listener = TcpListener::bind(&addr).await.map_err(|e| {
            ValidatorError::Metrics(format!("Failed to bind metrics server: {}", e))
        })?;

        info!("Metrics server listening on http://{}/metrics", addr);

        loop {
            let (stream, _) = listener.accept().await.map_err(|e| {
                ValidatorError::Metrics(format!("Failed to accept connection: {}", e))
            })?;

            let io = TokioIo::new(stream);
            let metrics_clone = metrics.clone();

            tokio::spawn(async move {
                let service = service_fn(move |_req: Request<hyper::body::Incoming>| {
                    let metrics = metrics_clone.clone();
                    async move {
                        let mut buffer = vec![];
                        let encoder = prometheus::TextEncoder::new();
                        let metric_families = metrics.registry().gather();
                        encoder
                            .encode(&metric_families, &mut buffer)
                            .expect("Failed to encode metrics - buffer should always be writable");

                        Ok::<_, hyper::Error>(Response::new(Full::new(Bytes::from(buffer))))
                    }
                });

                if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                    tracing::error!("Error serving connection: {:?}", err);
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use tempfile::tempdir;

    async fn create_test_config() -> ValidatorConfig {
        let temp_dir = tempdir().unwrap();

        // Create temporary keypair (persist to avoid cleanup)
        // Base64 encode 32 bytes of 42 (0x2A): [42; 32]
        let keypair_path = temp_dir.path().join("test_keypair.json");
        let secret_bytes = vec![42u8; 32];
        let secret_b64 = base64::engine::general_purpose::STANDARD.encode(&secret_bytes);
        let keypair_json = format!(r#"{{"secretKey":"{}"}}"#, secret_b64);
        std::fs::write(&keypair_path, keypair_json).unwrap();

        // Create temporary model files (empty for test)
        let models_dir = temp_dir.path().to_path_buf();
        std::fs::write(models_dir.join("clip-b32.onnx"), b"").unwrap();
        std::fs::write(models_dir.join("clip-l14.onnx"), b"").unwrap();

        // Keep temp_dir alive by leaking it (acceptable for tests)
        let models_dir = Box::leak(Box::new(temp_dir)).path().to_path_buf();
        let keypair_path = models_dir.join("test_keypair.json");

        ValidatorConfig {
            chain_endpoint: "ws://localhost:9944".to_string(),
            keypair_path,
            models_dir,
            clip: ClipConfig {
                model_b32_path: "clip-b32.onnx".to_string(),
                model_l14_path: "clip-l14.onnx".to_string(),
                b32_weight: 0.4,
                l14_weight: 0.6,
                threshold: 0.75,
                keyframe_count: 5,
                inference_timeout_secs: 5,
            },
            p2p: P2PConfig {
                listen_addresses: vec!["/ip4/127.0.0.1/tcp/0".to_string()],
                bootstrap_peers: vec![],
                max_peers: 50,
            },
            metrics: MetricsConfig {
                listen_address: "127.0.0.1".to_string(),
                port: 0, // Random port for tests
            },
            challenge: ChallengeConfig {
                enabled: false, // Disable for tests
                response_buffer_blocks: 40,
                poll_interval_secs: 6,
            },
        }
    }

    // Note: Full integration tests are skipped in unit tests due to
    // global Prometheus registry conflicts. These are tested in integration tests.

    #[tokio::test]
    #[ignore] // Run with --ignored flag or in integration tests
    async fn test_validator_node_creation() {
        let config = create_test_config().await;
        let result = ValidatorNode::new(config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    #[ignore] // Run with --ignored flag or in integration tests
    async fn test_validate_chunk_success() {
        let config = create_test_config().await;
        let validator = ValidatorNode::new(config).await.unwrap();

        let video_data = b"TEST_VIDEO_DATA";
        let prompt = "scientist in lab coat";

        let result = validator.validate_chunk(100, video_data, prompt).await;
        assert!(result.is_ok());

        let attestation = result.unwrap();
        assert_eq!(attestation.slot, 100);
        assert!(attestation.clip_score >= 0.0 && attestation.clip_score <= 1.0);
        assert!(!attestation.signature.is_empty());
    }
}
