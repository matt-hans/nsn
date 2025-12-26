//! Integration tests for ICN Validator Node
//!
//! These tests verify end-to-end functionality of the validator node,
//! including video chunk reception, CLIP inference, and attestation generation.

use base64::Engine;
use icn_validator::{
    ChallengeConfig, ClipConfig, MetricsConfig, P2PConfig, ValidatorConfig, ValidatorNode,
};
use tempfile::tempdir;

/// Create a test configuration with temporary directories
fn create_integration_test_config() -> ValidatorConfig {
    let temp_dir = tempdir().expect("Failed to create temp dir");

    // Create temporary keypair
    let keypair_path = temp_dir.path().join("test_keypair.json");
    let secret_bytes = vec![42u8; 32];
    let secret_b64 = base64::engine::general_purpose::STANDARD.encode(&secret_bytes);
    let keypair_json = format!(r#"{{"secretKey":"{}"}}"#, secret_b64);
    std::fs::write(&keypair_path, keypair_json).expect("Failed to write keypair");

    // Create temporary model files
    let models_dir = temp_dir.path().to_path_buf();
    std::fs::write(models_dir.join("clip-b32.onnx"), b"").expect("Failed to write model");
    std::fs::write(models_dir.join("clip-l14.onnx"), b"").expect("Failed to write model");

    // Leak temp_dir to keep it alive (acceptable for tests)
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
            port: 0, // Random port
        },
        challenge: ChallengeConfig {
            enabled: false,
            response_buffer_blocks: 40,
            poll_interval_secs: 6,
        },
    }
}

#[tokio::test]
#[ignore] // Run with --ignored flag (requires no global Prometheus registry conflicts)
async fn test_validator_node_full_lifecycle() {
    let config = create_integration_test_config();
    let validator = ValidatorNode::new(config)
        .await
        .expect("Failed to create validator node");

    // Test 1: Validate a video chunk
    let video_data = b"TEST_VIDEO_DATA";
    let prompt = "scientist in lab coat";

    let attestation = validator
        .validate_chunk(100, video_data, prompt)
        .await
        .expect("Failed to validate chunk");

    // Verify attestation
    assert_eq!(attestation.slot, 100);
    assert!(attestation.clip_score >= 0.0 && attestation.clip_score <= 1.0);
    assert!(!attestation.signature.is_empty());
    assert!(!attestation.validator_id.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_video_chunk_validation_flow() {
    let config = create_integration_test_config();
    let validator = ValidatorNode::new(config)
        .await
        .expect("Failed to create validator node");

    // Test with multiple video chunks
    let test_cases = vec![
        (200, b"VIDEO_CHUNK_1".as_slice(), "test prompt 1"),
        (201, b"VIDEO_CHUNK_2".as_slice(), "test prompt 2"),
        (202, b"VIDEO_CHUNK_3".as_slice(), "test prompt 3"),
    ];

    for (slot, video_data, prompt) in test_cases {
        let result = validator.validate_chunk(slot, video_data, prompt).await;
        assert!(
            result.is_ok(),
            "Failed to validate slot {}: {:?}",
            slot,
            result.err()
        );

        let attestation = result.unwrap();
        assert_eq!(attestation.slot, slot);
        assert!(attestation.clip_score >= 0.0 && attestation.clip_score <= 1.0);
    }
}

#[tokio::test]
#[ignore]
async fn test_attestation_generation() {
    let config = create_integration_test_config();
    let validator = ValidatorNode::new(config)
        .await
        .expect("Failed to create validator node");

    let video_data = b"ATTESTATION_TEST_VIDEO";
    let prompt = "test attestation generation";

    let attestation = validator
        .validate_chunk(300, video_data, prompt)
        .await
        .expect("Failed to generate attestation");

    // Verify attestation structure
    assert_eq!(attestation.slot, 300);
    assert!(!attestation.validator_id.is_empty());
    assert!(!attestation.signature.is_empty());

    // Score should be in valid range
    assert!(attestation.clip_score >= 0.0 && attestation.clip_score <= 1.0);

    // Passed flag should be consistent with threshold
    // In test mode: B-32=0.82, L-14=0.85, ensemble=0.838, threshold=0.75
    // So it should pass
    assert!(attestation.passed);
}

#[tokio::test]
#[ignore]
async fn test_challenge_detection_integration() {
    let mut config = create_integration_test_config();
    config.challenge.enabled = true;

    let _validator = ValidatorNode::new(config)
        .await
        .expect("Failed to create validator node");

    // Note: Full challenge flow requires chain client integration
    // This test verifies the validator can be created with challenges enabled
    // Actual challenge handling tested in unit tests
}
