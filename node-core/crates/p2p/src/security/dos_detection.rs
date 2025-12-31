//! DoS attack detection with pattern recognition

use super::metrics::SecurityMetrics;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{error, warn};

/// DoS detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DosDetectorConfig {
    /// Connection flood threshold (connections per detection window)
    pub connection_flood_threshold: u32,

    /// Detection window duration
    #[serde(with = "humantime_serde")]
    pub detection_window: Duration,

    /// Message spam threshold (messages per detection window)
    pub message_spam_threshold: u32,
}

impl Default for DosDetectorConfig {
    fn default() -> Self {
        Self {
            connection_flood_threshold: 50,
            detection_window: Duration::from_secs(10),
            message_spam_threshold: 1000,
        }
    }
}

/// DoS detector for attack pattern recognition
pub struct DosDetector {
    config: DosDetectorConfig,
    connection_attempts: Arc<RwLock<VecDeque<Instant>>>,
    message_attempts: Arc<RwLock<VecDeque<Instant>>>,
    metrics: Arc<SecurityMetrics>,
}

impl DosDetector {
    /// Create new DoS detector
    pub fn new(config: DosDetectorConfig, metrics: Arc<SecurityMetrics>) -> Self {
        Self {
            config,
            connection_attempts: Arc::new(RwLock::new(VecDeque::new())),
            message_attempts: Arc::new(RwLock::new(VecDeque::new())),
            metrics,
        }
    }

    /// Record connection attempt
    pub async fn record_connection_attempt(&self) {
        let mut attempts = self.connection_attempts.write().await;
        attempts.push_back(Instant::now());

        // Keep only recent attempts (sliding window)
        let cutoff = Instant::now() - self.config.detection_window * 2;
        while let Some(&oldest) = attempts.front() {
            if oldest < cutoff {
                attempts.pop_front();
            } else {
                break;
            }
        }
    }

    /// Detect connection flood attack
    ///
    /// Returns true if attack detected
    pub async fn detect_connection_flood(&self) -> bool {
        let attempts = self.connection_attempts.read().await;
        let now = Instant::now();

        // Count attempts within detection window
        let recent_attempts = attempts
            .iter()
            .filter(|&&t| now.duration_since(t) < self.config.detection_window)
            .count();

        if recent_attempts as u32 > self.config.connection_flood_threshold {
            error!(
                "DoS attack detected: {} connection attempts in {}s",
                recent_attempts,
                self.config.detection_window.as_secs()
            );

            self.metrics.dos_attacks_detected.inc();
            self.metrics.connection_flood_detected.inc();

            return true;
        }

        false
    }

    /// Record message attempt
    pub async fn record_message_attempt(&self) {
        let mut attempts = self.message_attempts.write().await;
        attempts.push_back(Instant::now());

        // Keep only recent attempts
        let cutoff = Instant::now() - self.config.detection_window * 2;
        while let Some(&oldest) = attempts.front() {
            if oldest < cutoff {
                attempts.pop_front();
            } else {
                break;
            }
        }
    }

    /// Detect message spam attack
    ///
    /// Returns true if attack detected
    pub async fn detect_message_spam(&self) -> bool {
        let attempts = self.message_attempts.read().await;
        let now = Instant::now();

        // Count attempts within detection window
        let recent_attempts = attempts
            .iter()
            .filter(|&&t| now.duration_since(t) < self.config.detection_window)
            .count();

        if recent_attempts as u32 > self.config.message_spam_threshold {
            warn!(
                "Message spam detected: {} messages in {}s",
                recent_attempts,
                self.config.detection_window.as_secs()
            );

            self.metrics.dos_attacks_detected.inc();
            self.metrics.message_spam_detected.inc();

            return true;
        }

        false
    }

    /// Get current connection attempt rate (attempts per second)
    pub async fn get_connection_rate(&self) -> f64 {
        let attempts = self.connection_attempts.read().await;
        let now = Instant::now();

        let recent_attempts = attempts
            .iter()
            .filter(|&&t| now.duration_since(t) < self.config.detection_window)
            .count();

        recent_attempts as f64 / self.config.detection_window.as_secs_f64()
    }

    /// Get current message attempt rate (messages per second)
    pub async fn get_message_rate(&self) -> f64 {
        let attempts = self.message_attempts.read().await;
        let now = Instant::now();

        let recent_attempts = attempts
            .iter()
            .filter(|&&t| now.duration_since(t) < self.config.detection_window)
            .count();

        recent_attempts as f64 / self.config.detection_window.as_secs_f64()
    }

    /// Reset all detection state (for testing)
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn reset(&self) {
        self.connection_attempts.write().await.clear();
        self.message_attempts.write().await.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dos_detector(config: DosDetectorConfig) -> DosDetector {
        let metrics = Arc::new(SecurityMetrics::new_unregistered());
        DosDetector::new(config, metrics)
    }

    #[tokio::test]
    async fn test_dos_detector_no_attack_initially() {
        let config = DosDetectorConfig::default();
        let detector = create_test_dos_detector(config);

        // No attack initially
        assert!(!detector.detect_connection_flood().await);
        assert!(!detector.detect_message_spam().await);
    }

    #[tokio::test]
    async fn test_dos_detector_connection_flood() {
        let config = DosDetectorConfig {
            connection_flood_threshold: 5,
            detection_window: Duration::from_secs(10),
            message_spam_threshold: 1000,
        };

        let detector = create_test_dos_detector(config);

        // Record 6 connection attempts (exceeds threshold of 5)
        for _ in 0..6 {
            detector.record_connection_attempt().await;
        }

        // Should detect flood
        assert!(
            detector.detect_connection_flood().await,
            "Should detect connection flood (6 > 5)"
        );
    }

    #[tokio::test]
    async fn test_dos_detector_connection_flood_under_threshold() {
        let config = DosDetectorConfig {
            connection_flood_threshold: 10,
            detection_window: Duration::from_secs(10),
            message_spam_threshold: 1000,
        };

        let detector = create_test_dos_detector(config);

        // Record 5 connection attempts (under threshold of 10)
        for _ in 0..5 {
            detector.record_connection_attempt().await;
        }

        // Should NOT detect flood
        assert!(
            !detector.detect_connection_flood().await,
            "Should NOT detect flood (5 <= 10)"
        );
    }

    #[tokio::test]
    async fn test_dos_detector_message_spam() {
        let config = DosDetectorConfig {
            connection_flood_threshold: 50,
            detection_window: Duration::from_secs(10),
            message_spam_threshold: 10,
        };

        let detector = create_test_dos_detector(config);

        // Record 11 message attempts (exceeds threshold of 10)
        for _ in 0..11 {
            detector.record_message_attempt().await;
        }

        // Should detect spam
        assert!(
            detector.detect_message_spam().await,
            "Should detect message spam (11 > 10)"
        );
    }

    #[tokio::test]
    async fn test_dos_detector_window_expiration() {
        let config = DosDetectorConfig {
            connection_flood_threshold: 5,
            detection_window: Duration::from_millis(100), // 100ms for testing
            message_spam_threshold: 1000,
        };

        let detector = create_test_dos_detector(config);

        // Record 6 attempts
        for _ in 0..6 {
            detector.record_connection_attempt().await;
        }

        // Should detect flood initially
        assert!(detector.detect_connection_flood().await);

        // Wait for window to expire
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should NOT detect flood (window expired)
        assert!(
            !detector.detect_connection_flood().await,
            "Should NOT detect flood after window expiration"
        );
    }

    #[tokio::test]
    async fn test_dos_detector_connection_rate() {
        let config = DosDetectorConfig {
            connection_flood_threshold: 50,
            detection_window: Duration::from_secs(10),
            message_spam_threshold: 1000,
        };

        let detector = create_test_dos_detector(config);

        // Initially 0
        let rate = detector.get_connection_rate().await;
        assert_eq!(rate, 0.0, "Initial rate should be 0");

        // Record 10 attempts
        for _ in 0..10 {
            detector.record_connection_attempt().await;
        }

        // Rate should be ~1 per second (10 attempts / 10 seconds)
        let rate = detector.get_connection_rate().await;
        assert!(
            (rate - 1.0).abs() < 0.1,
            "Rate should be ~1/sec, got {}",
            rate
        );
    }

    #[tokio::test]
    async fn test_dos_detector_message_rate() {
        let config = DosDetectorConfig {
            connection_flood_threshold: 50,
            detection_window: Duration::from_secs(1), // 1 second for easier calculation
            message_spam_threshold: 1000,
        };

        let detector = create_test_dos_detector(config);

        // Record 50 messages
        for _ in 0..50 {
            detector.record_message_attempt().await;
        }

        // Rate should be ~50 per second
        let rate = detector.get_message_rate().await;
        assert!(
            (rate - 50.0).abs() < 5.0,
            "Rate should be ~50/sec, got {}",
            rate
        );
    }

    #[tokio::test]
    async fn test_dos_detector_reset() {
        let config = DosDetectorConfig {
            connection_flood_threshold: 5,
            detection_window: Duration::from_secs(10),
            message_spam_threshold: 1000,
        };

        let detector = create_test_dos_detector(config);

        // Record attempts
        for _ in 0..10 {
            detector.record_connection_attempt().await;
            detector.record_message_attempt().await;
        }

        // Verify rates are non-zero
        assert!(detector.get_connection_rate().await > 0.0);
        assert!(detector.get_message_rate().await > 0.0);

        // Reset
        detector.reset().await;

        // Rates should be 0
        assert_eq!(detector.get_connection_rate().await, 0.0);
        assert_eq!(detector.get_message_rate().await, 0.0);
    }

    #[tokio::test]
    async fn test_dos_detector_metrics_flood_detected() {
        let config = DosDetectorConfig {
            connection_flood_threshold: 3,
            detection_window: Duration::from_secs(10),
            message_spam_threshold: 1000,
        };

        let detector = create_test_dos_detector(config);

        let initial_dos = detector.metrics.dos_attacks_detected.get();
        let initial_flood = detector.metrics.connection_flood_detected.get();

        // Trigger flood
        for _ in 0..5 {
            detector.record_connection_attempt().await;
        }

        let _ = detector.detect_connection_flood().await;

        let final_dos = detector.metrics.dos_attacks_detected.get();
        let final_flood = detector.metrics.connection_flood_detected.get();

        assert_eq!(
            final_dos,
            initial_dos + 1,
            "DoS attacks metric should increment"
        );
        assert_eq!(
            final_flood,
            initial_flood + 1,
            "Flood detected metric should increment"
        );
    }

    #[tokio::test]
    async fn test_dos_detector_metrics_spam_detected() {
        let config = DosDetectorConfig {
            connection_flood_threshold: 50,
            detection_window: Duration::from_secs(10),
            message_spam_threshold: 5,
        };

        let detector = create_test_dos_detector(config);

        let initial_dos = detector.metrics.dos_attacks_detected.get();
        let initial_spam = detector.metrics.message_spam_detected.get();

        // Trigger spam
        for _ in 0..10 {
            detector.record_message_attempt().await;
        }

        let _ = detector.detect_message_spam().await;

        let final_dos = detector.metrics.dos_attacks_detected.get();
        let final_spam = detector.metrics.message_spam_detected.get();

        assert_eq!(
            final_dos,
            initial_dos + 1,
            "DoS attacks metric should increment"
        );
        assert_eq!(
            final_spam,
            initial_spam + 1,
            "Spam detected metric should increment"
        );
    }
}
