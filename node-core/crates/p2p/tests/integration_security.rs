//! Integration tests for P2P security layer
//!
//! Tests comprehensive security scenarios including rate limiting,
//! bandwidth throttling, graylist enforcement, and DoS detection.

use libp2p::identity::Keypair;
use libp2p::PeerId;
use nsn_p2p::security::{
    BandwidthLimiter, BandwidthLimiterConfig, DosDetector, DosDetectorConfig, Graylist,
    GraylistConfig, RateLimiter, RateLimiterConfig, SecureP2pConfig, SecurityMetrics,
};
// Note: ReputationOracle integration tested separately in unit tests with test-helpers feature
use prometheus::Registry;
use std::sync::Arc;
use std::time::Duration;

fn create_test_peer_id() -> PeerId {
    let keypair = Keypair::generate_ed25519();
    PeerId::from(keypair.public())
}

#[tokio::test]
async fn test_security_layer_integration() {
    // Create security metrics
    let registry = Registry::new();
    let metrics = Arc::new(SecurityMetrics::new(&registry).expect("Failed to create metrics"));

    // Create security components with default config
    let config = SecureP2pConfig::default();

    let rate_limiter = RateLimiter::new(config.rate_limiter.clone(), None, metrics.clone());
    let bandwidth_limiter =
        BandwidthLimiter::new(config.bandwidth_limiter.clone(), metrics.clone());
    let graylist = Graylist::new(config.graylist.clone(), metrics.clone());
    let dos_detector = DosDetector::new(config.dos_detector.clone(), metrics.clone());

    let peer_id = create_test_peer_id();

    // Test rate limiting
    for _ in 0..100 {
        let result = rate_limiter.check_rate_limit(&peer_id).await;
        assert!(result.is_ok(), "First 100 requests should be allowed");
    }

    // 101st request should be rejected
    let result = rate_limiter.check_rate_limit(&peer_id).await;
    assert!(result.is_err(), "101st request should be rate limited");

    // Add to graylist after violation
    graylist
        .add(peer_id, "Rate limit violations".to_string())
        .await;

    // Check graylist status
    assert!(
        graylist.is_graylisted(&peer_id).await,
        "Peer should be graylisted"
    );

    // Test bandwidth tracking
    let allowed = bandwidth_limiter.record_transfer(&peer_id, 1_000_000).await;
    assert!(allowed, "Bandwidth transfer should be allowed");

    // Test DoS detection (connection flood)
    // Default threshold is 50, so record 51 attempts
    for _ in 0..51 {
        dos_detector.record_connection_attempt().await;
    }

    let flood_detected = dos_detector.detect_connection_flood().await;
    assert!(
        flood_detected,
        "Connection flood should be detected (51 > 50)"
    );
}

#[tokio::test]
async fn test_rate_limit_without_reputation() {
    // Test rate limiting without reputation oracle (base case)
    let registry = Registry::new();
    let metrics = Arc::new(SecurityMetrics::new(&registry).expect("Failed to create metrics"));

    // Create rate limiter without reputation oracle
    let config = RateLimiterConfig {
        max_requests_per_minute: 50,
        rate_limit_window: Duration::from_secs(60),
        reputation_rate_limit_multiplier: 2.0,
        min_reputation_for_bypass: 800,
    };

    let rate_limiter = RateLimiter::new(config, None, metrics);

    let peer_id = create_test_peer_id();

    // Without reputation oracle, should get base limit (50 requests)
    for i in 0..50 {
        let result = rate_limiter.check_rate_limit(&peer_id).await;
        assert!(
            result.is_ok(),
            "Request {} should be allowed (within base limit)",
            i + 1
        );
    }

    // 51st request should be rejected
    let result = rate_limiter.check_rate_limit(&peer_id).await;
    assert!(
        result.is_err(),
        "Request should be rejected at base limit (50)"
    );
}

#[tokio::test]
async fn test_graylist_workflow_integration() {
    let registry = Registry::new();
    let metrics = Arc::new(SecurityMetrics::new(&registry).expect("Failed to create metrics"));

    let config = GraylistConfig {
        duration: Duration::from_millis(200), // 200ms for testing
        threshold_violations: 3,
    };

    let graylist = Graylist::new(config, metrics.clone());
    let rate_limiter = RateLimiter::new(
        RateLimiterConfig {
            max_requests_per_minute: 5,
            rate_limit_window: Duration::from_secs(60),
            ..Default::default()
        },
        None,
        metrics.clone(),
    );

    let peer_id = create_test_peer_id();

    // Exhaust rate limit
    for _ in 0..5 {
        let _ = rate_limiter.check_rate_limit(&peer_id).await;
    }

    // Violate rate limit 3 times
    for _ in 0..3 {
        let result = rate_limiter.check_rate_limit(&peer_id).await;
        assert!(result.is_err(), "Should be rate limited");

        // Add to graylist on each violation
        graylist
            .add(peer_id, "Rate limit violation".to_string())
            .await;
    }

    // Peer should be graylisted
    assert!(graylist.is_graylisted(&peer_id).await);

    // Wait for graylist to expire
    tokio::time::sleep(Duration::from_millis(250)).await;

    // Cleanup expired entries
    graylist.cleanup_expired().await;

    // Peer should no longer be graylisted
    assert!(!graylist.is_graylisted(&peer_id).await);
}

#[tokio::test]
async fn test_dos_detection_integration() {
    let registry = Registry::new();
    let metrics = Arc::new(SecurityMetrics::new(&registry).expect("Failed to create metrics"));

    let config = DosDetectorConfig {
        connection_flood_threshold: 10,
        detection_window: Duration::from_secs(5),
        message_spam_threshold: 50,
    };

    let detector = DosDetector::new(config, metrics);

    // Simulate connection flood (15 connections in short time)
    for _ in 0..15 {
        detector.record_connection_attempt().await;
    }

    // Should detect flood
    assert!(
        detector.detect_connection_flood().await,
        "Connection flood should be detected (15 > 10)"
    );

    // Simulate message spam (60 messages)
    for _ in 0..60 {
        detector.record_message_attempt().await;
    }

    // Should detect spam
    assert!(
        detector.detect_message_spam().await,
        "Message spam should be detected (60 > 50)"
    );
}

#[tokio::test]
async fn test_bandwidth_throttling_integration() {
    let registry = Registry::new();
    let metrics = Arc::new(SecurityMetrics::new(&registry).expect("Failed to create metrics"));

    let config = BandwidthLimiterConfig {
        max_bandwidth_mbps: 10,
        measurement_interval: Duration::from_secs(1),
    };

    let limiter = BandwidthLimiter::new(config, metrics);
    let peer_id = create_test_peer_id();

    // Transfer 1 MB (8 Mbps over 1 second - under limit)
    let allowed = limiter.record_transfer(&peer_id, 1_000_000).await;
    assert!(allowed, "Transfer should be allowed (under 10 Mbps)");

    // Small delay for bandwidth calculation
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Transfer 5 MB more (would exceed limit if within same interval)
    let _ = limiter.record_transfer(&peer_id, 5_000_000).await;

    // Get current bandwidth
    let bandwidth = limiter.get_bandwidth(&peer_id).await;
    assert!(
        bandwidth > 0.0,
        "Bandwidth should be tracked, got {} Mbps",
        bandwidth
    );
}

#[tokio::test]
async fn test_metrics_integration() {
    let registry = Registry::new();
    let metrics = Arc::new(SecurityMetrics::new(&registry).expect("Failed to create metrics"));

    let rate_limiter = RateLimiter::new(
        RateLimiterConfig {
            max_requests_per_minute: 2,
            ..Default::default()
        },
        None,
        metrics.clone(),
    );

    let peer_id = create_test_peer_id();

    // Make allowed requests
    let initial_allowed = metrics.rate_limit_allowed.get();
    for _ in 0..2 {
        let _ = rate_limiter.check_rate_limit(&peer_id).await;
    }
    let final_allowed = metrics.rate_limit_allowed.get();
    assert_eq!(
        final_allowed,
        initial_allowed + 2,
        "Allowed metric should increment by 2"
    );

    // Make rejected request
    let initial_violations = metrics
        .rate_limit_violations
        .with_label_values(&[&peer_id.to_string()])
        .get();

    let _ = rate_limiter.check_rate_limit(&peer_id).await;

    let final_violations = metrics
        .rate_limit_violations
        .with_label_values(&[&peer_id.to_string()])
        .get();

    assert_eq!(
        final_violations,
        initial_violations + 1,
        "Violations metric should increment by 1"
    );
}
