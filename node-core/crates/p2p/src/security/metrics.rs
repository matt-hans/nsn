//! Prometheus metrics for P2P security events

use prometheus::{Gauge, Histogram, HistogramOpts, IntCounter, IntCounterVec, Opts, Registry};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetricsError {
    #[error("Prometheus error: {0}")]
    Prometheus(#[from] prometheus::Error),
}

/// Security-specific Prometheus metrics
pub struct SecurityMetrics {
    // Rate limiting metrics
    pub rate_limit_violations: IntCounterVec,
    pub rate_limit_allowed: IntCounter,
    pub reputation_bypass_applied: IntCounter,

    // Bandwidth metrics
    pub bandwidth_throttled: IntCounter,
    pub bandwidth_bytes_transferred: IntCounter,
    pub bandwidth_usage_mbps: Gauge,

    // Connection metrics
    pub connection_rejections: IntCounter,
    pub connection_timeouts: IntCounter,
    pub per_peer_limit_violations: IntCounter,

    // Graylist metrics
    pub peers_graylisted: IntCounter,
    pub graylist_rejections: IntCounter,
    pub graylist_size: Gauge,

    // DoS detection metrics
    pub dos_attacks_detected: IntCounter,
    pub connection_flood_detected: IntCounter,
    pub message_spam_detected: IntCounter,

    // Security event processing time
    pub security_check_duration: Histogram,
}

impl SecurityMetrics {
    /// Create new security metrics and register with Prometheus
    pub fn new(registry: &Registry) -> Result<Self, MetricsError> {
        // Rate limiting
        let rate_limit_violations = IntCounterVec::new(
            Opts::new(
                "nsn_p2p_rate_limit_violations_total",
                "Total rate limit violations by peer",
            ),
            &["peer_id"],
        )?;

        let rate_limit_allowed = IntCounter::with_opts(Opts::new(
            "nsn_p2p_rate_limit_allowed_total",
            "Total allowed requests (within rate limit)",
        ))?;

        let reputation_bypass_applied = IntCounter::with_opts(Opts::new(
            "nsn_p2p_reputation_bypass_applied_total",
            "Total high-reputation peer rate limit bypasses",
        ))?;

        // Bandwidth
        let bandwidth_throttled = IntCounter::with_opts(Opts::new(
            "nsn_p2p_bandwidth_throttled_total",
            "Total bandwidth throttling events",
        ))?;

        let bandwidth_bytes_transferred = IntCounter::with_opts(Opts::new(
            "nsn_p2p_bandwidth_bytes_transferred_total",
            "Total bytes transferred across all connections",
        ))?;

        let bandwidth_usage_mbps = Gauge::with_opts(Opts::new(
            "nsn_p2p_bandwidth_usage_mbps",
            "Current bandwidth usage in Mbps",
        ))?;

        // Connections
        let connection_rejections = IntCounter::with_opts(Opts::new(
            "nsn_p2p_connection_rejections_total",
            "Total connection rejections (max connections reached)",
        ))?;

        let connection_timeouts = IntCounter::with_opts(Opts::new(
            "nsn_p2p_connection_timeouts_total",
            "Total connection timeouts due to inactivity",
        ))?;

        let per_peer_limit_violations = IntCounter::with_opts(Opts::new(
            "nsn_p2p_per_peer_limit_violations_total",
            "Total per-peer connection limit violations",
        ))?;

        // Graylist
        let peers_graylisted = IntCounter::with_opts(Opts::new(
            "nsn_p2p_peers_graylisted_total",
            "Total peers added to graylist",
        ))?;

        let graylist_rejections = IntCounter::with_opts(Opts::new(
            "nsn_p2p_graylist_rejections_total",
            "Total connection attempts from graylisted peers",
        ))?;

        let graylist_size = Gauge::with_opts(Opts::new(
            "nsn_p2p_graylist_size",
            "Current number of graylisted peers",
        ))?;

        // DoS detection
        let dos_attacks_detected = IntCounter::with_opts(Opts::new(
            "nsn_p2p_dos_attacks_detected_total",
            "Total DoS attacks detected",
        ))?;

        let connection_flood_detected = IntCounter::with_opts(Opts::new(
            "nsn_p2p_connection_flood_detected_total",
            "Total connection flood attacks detected",
        ))?;

        let message_spam_detected = IntCounter::with_opts(Opts::new(
            "nsn_p2p_message_spam_detected_total",
            "Total message spam attacks detected",
        ))?;

        // Performance
        let security_check_duration = Histogram::with_opts(HistogramOpts::new(
            "nsn_p2p_security_check_duration_seconds",
            "Duration of security check operations",
        ))?;

        // Register all metrics
        registry.register(Box::new(rate_limit_violations.clone()))?;
        registry.register(Box::new(rate_limit_allowed.clone()))?;
        registry.register(Box::new(reputation_bypass_applied.clone()))?;
        registry.register(Box::new(bandwidth_throttled.clone()))?;
        registry.register(Box::new(bandwidth_bytes_transferred.clone()))?;
        registry.register(Box::new(bandwidth_usage_mbps.clone()))?;
        registry.register(Box::new(connection_rejections.clone()))?;
        registry.register(Box::new(connection_timeouts.clone()))?;
        registry.register(Box::new(per_peer_limit_violations.clone()))?;
        registry.register(Box::new(peers_graylisted.clone()))?;
        registry.register(Box::new(graylist_rejections.clone()))?;
        registry.register(Box::new(graylist_size.clone()))?;
        registry.register(Box::new(dos_attacks_detected.clone()))?;
        registry.register(Box::new(connection_flood_detected.clone()))?;
        registry.register(Box::new(message_spam_detected.clone()))?;
        registry.register(Box::new(security_check_duration.clone()))?;

        Ok(Self {
            rate_limit_violations,
            rate_limit_allowed,
            reputation_bypass_applied,
            bandwidth_throttled,
            bandwidth_bytes_transferred,
            bandwidth_usage_mbps,
            connection_rejections,
            connection_timeouts,
            per_peer_limit_violations,
            peers_graylisted,
            graylist_rejections,
            graylist_size,
            dos_attacks_detected,
            connection_flood_detected,
            message_spam_detected,
            security_check_duration,
        })
    }

    /// Create metrics without registry (for testing)
    #[cfg(test)]
    pub fn new_unregistered() -> Self {
        Self {
            rate_limit_violations: IntCounterVec::new(
                Opts::new("test_rate_limit_violations", "test"),
                &["peer_id"],
            )
            .unwrap(),
            rate_limit_allowed: IntCounter::new("test_rate_limit_allowed", "test").unwrap(),
            reputation_bypass_applied: IntCounter::new("test_reputation_bypass_applied", "test")
                .unwrap(),
            bandwidth_throttled: IntCounter::new("test_bandwidth_throttled", "test").unwrap(),
            bandwidth_bytes_transferred: IntCounter::new(
                "test_bandwidth_bytes_transferred",
                "test",
            )
            .unwrap(),
            bandwidth_usage_mbps: Gauge::new("test_bandwidth_usage_mbps", "test").unwrap(),
            connection_rejections: IntCounter::new("test_connection_rejections", "test").unwrap(),
            connection_timeouts: IntCounter::new("test_connection_timeouts", "test").unwrap(),
            per_peer_limit_violations: IntCounter::new("test_per_peer_limit_violations", "test")
                .unwrap(),
            peers_graylisted: IntCounter::new("test_peers_graylisted", "test").unwrap(),
            graylist_rejections: IntCounter::new("test_graylist_rejections", "test").unwrap(),
            graylist_size: Gauge::new("test_graylist_size", "test").unwrap(),
            dos_attacks_detected: IntCounter::new("test_dos_attacks_detected", "test").unwrap(),
            connection_flood_detected: IntCounter::new("test_connection_flood_detected", "test")
                .unwrap(),
            message_spam_detected: IntCounter::new("test_message_spam_detected", "test").unwrap(),
            security_check_duration: Histogram::with_opts(HistogramOpts::new(
                "test_security_check_duration",
                "test",
            ))
            .unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_metrics_creation() {
        let registry = Registry::new();
        let metrics = SecurityMetrics::new(&registry).expect("Failed to create metrics");

        // Verify initial values
        assert_eq!(
            metrics.rate_limit_allowed.get(),
            0,
            "Initial rate limit allowed should be 0"
        );
        assert_eq!(
            metrics.graylist_size.get(),
            0.0,
            "Initial graylist size should be 0"
        );
        assert_eq!(
            metrics.dos_attacks_detected.get(),
            0,
            "Initial DoS attacks should be 0"
        );
    }

    #[test]
    fn test_security_metrics_unregistered() {
        let metrics = SecurityMetrics::new_unregistered();

        // Test counter increments
        metrics.rate_limit_allowed.inc();
        assert_eq!(metrics.rate_limit_allowed.get(), 1);

        metrics.peers_graylisted.inc();
        assert_eq!(metrics.peers_graylisted.get(), 1);

        // Test gauge set
        metrics.graylist_size.set(5.0);
        assert_eq!(metrics.graylist_size.get(), 5.0);
    }

    #[test]
    fn test_rate_limit_violations_per_peer() {
        let metrics = SecurityMetrics::new_unregistered();

        // Increment violations for specific peer
        metrics
            .rate_limit_violations
            .with_label_values(&["peer123"])
            .inc();
        metrics
            .rate_limit_violations
            .with_label_values(&["peer123"])
            .inc();

        assert_eq!(
            metrics
                .rate_limit_violations
                .with_label_values(&["peer123"])
                .get(),
            2,
            "Peer violations should be 2"
        );

        // Different peer
        metrics
            .rate_limit_violations
            .with_label_values(&["peer456"])
            .inc();

        assert_eq!(
            metrics
                .rate_limit_violations
                .with_label_values(&["peer456"])
                .get(),
            1,
            "Different peer violations should be 1"
        );
    }
}
