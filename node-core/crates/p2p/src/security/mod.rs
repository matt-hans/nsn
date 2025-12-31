//! Security module for P2P networking
//!
//! Provides rate limiting, bandwidth throttling, graylist enforcement,
//! and DoS attack detection for NSN P2P nodes.

mod bandwidth;
mod dos_detection;
mod graylist;
mod metrics;
mod rate_limiter;

pub use bandwidth::{BandwidthLimiter, BandwidthLimiterConfig};
pub use dos_detection::{DosDetector, DosDetectorConfig};
pub use graylist::{Graylist, GraylistConfig, GraylistEntry};
pub use metrics::SecurityMetrics;
pub use rate_limiter::{RateLimitError, RateLimiter, RateLimiterConfig};

use serde::{Deserialize, Serialize};

/// Comprehensive security configuration for P2P networking
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecureP2pConfig {
    /// Rate limiting configuration
    pub rate_limiter: RateLimiterConfig,

    /// Bandwidth throttling configuration
    pub bandwidth_limiter: BandwidthLimiterConfig,

    /// Graylist enforcement configuration
    pub graylist: GraylistConfig,

    /// DoS detection configuration
    pub dos_detector: DosDetectorConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_secure_p2p_config_defaults() {
        let config = SecureP2pConfig::default();

        // Verify rate limiter defaults
        assert_eq!(
            config.rate_limiter.max_requests_per_minute, 100,
            "Default rate limit should be 100 req/min"
        );

        // Verify bandwidth defaults
        assert_eq!(
            config.bandwidth_limiter.max_bandwidth_mbps, 100,
            "Default bandwidth limit should be 100 Mbps"
        );

        // Verify graylist defaults
        assert_eq!(
            config.graylist.duration,
            Duration::from_secs(3600),
            "Default graylist duration should be 1 hour"
        );

        // Verify DoS detection defaults
        assert_eq!(
            config.dos_detector.connection_flood_threshold, 50,
            "Default connection flood threshold should be 50"
        );
    }

    #[test]
    fn test_secure_p2p_config_serialization() {
        let config = SecureP2pConfig::default();

        // Serialize to JSON
        let json = serde_json::to_string(&config).expect("Failed to serialize");

        // Deserialize back
        let deserialized: SecureP2pConfig =
            serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(
            deserialized.rate_limiter.max_requests_per_minute,
            config.rate_limiter.max_requests_per_minute
        );
        assert_eq!(
            deserialized.bandwidth_limiter.max_bandwidth_mbps,
            config.bandwidth_limiter.max_bandwidth_mbps
        );
    }
}
