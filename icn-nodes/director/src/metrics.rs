use prometheus::{Counter, Gauge, Histogram, Registry};
use std::sync::Arc;

/// Metrics collector for director node
#[cfg_attr(feature = "stub", allow(dead_code))]
#[derive(Clone)]
pub struct Metrics {
    registry: Arc<Registry>,

    // Slot metrics
    pub current_slot: Gauge,
    pub elected_slots_total: Counter,
    pub missed_slots_total: Counter,

    // BFT metrics
    pub bft_rounds_success: Counter,
    pub bft_rounds_failed: Counter,
    pub bft_round_duration: Histogram,

    // P2P metrics
    pub connected_peers: Gauge,

    // Chain metrics
    pub chain_latest_block: Gauge,
    pub chain_disconnects: Counter,
}

impl Metrics {
    pub fn new() -> crate::error::Result<Self> {
        let registry = Registry::new();

        let current_slot = Gauge::new("icn_director_current_slot", "Current slot number")?;
        let elected_slots_total = Counter::new(
            "icn_director_elected_slots_total",
            "Total number of slots elected as director",
        )?;
        let missed_slots_total = Counter::new(
            "icn_director_missed_slots_total",
            "Total number of slots missed (deadline passed)",
        )?;
        let bft_rounds_success = Counter::new(
            "icn_bft_rounds_success_total",
            "Total successful BFT rounds",
        )?;
        let bft_rounds_failed =
            Counter::new("icn_bft_rounds_failed_total", "Total failed BFT rounds")?;
        let bft_round_duration = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "icn_bft_round_duration_seconds",
                "BFT round duration in seconds",
            )
            .buckets(vec![1.0, 2.0, 5.0, 10.0, 20.0, 30.0]),
        )?;
        let connected_peers =
            Gauge::new("icn_p2p_connected_peers", "Number of connected P2P peers")?;
        let chain_latest_block = Gauge::new(
            "icn_chain_latest_block",
            "Latest finalized block number from chain",
        )?;
        let chain_disconnects = Counter::new(
            "icn_chain_disconnects_total",
            "Total chain RPC disconnections",
        )?;

        registry.register(Box::new(current_slot.clone()))?;
        registry.register(Box::new(elected_slots_total.clone()))?;
        registry.register(Box::new(missed_slots_total.clone()))?;
        registry.register(Box::new(bft_rounds_success.clone()))?;
        registry.register(Box::new(bft_rounds_failed.clone()))?;
        registry.register(Box::new(bft_round_duration.clone()))?;
        registry.register(Box::new(connected_peers.clone()))?;
        registry.register(Box::new(chain_latest_block.clone()))?;
        registry.register(Box::new(chain_disconnects.clone()))?;

        Ok(Self {
            registry: Arc::new(registry),
            current_slot,
            elected_slots_total,
            missed_slots_total,
            bft_rounds_success,
            bft_rounds_failed,
            bft_round_duration,
            connected_peers,
            chain_latest_block,
            chain_disconnects,
        })
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new().expect("Failed to create metrics")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::Encoder;

    /// Test Case: Metrics registry creation
    /// Purpose: Verify metrics registry is created successfully
    /// Contract: Registry should contain all expected metrics
    #[test]
    fn test_metrics_registry_creation() {
        let metrics = Metrics::new().expect("Failed to create metrics");

        // Verify registry exists
        let registry = metrics.registry();

        // Verify metrics are registered
        let metric_families = registry.gather();

        // Should have at least 9 metrics registered
        assert!(
            metric_families.len() >= 9,
            "Expected at least 9 metrics, got {}",
            metric_families.len()
        );

        // Verify specific metrics exist by name
        let metric_names: Vec<String> = metric_families
            .iter()
            .map(|mf| mf.get_name().to_string())
            .collect();

        let expected_metrics = vec![
            "icn_director_current_slot",
            "icn_director_elected_slots_total",
            "icn_director_missed_slots_total",
            "icn_bft_rounds_success_total",
            "icn_bft_rounds_failed_total",
            "icn_bft_round_duration_seconds",
            "icn_p2p_connected_peers",
            "icn_chain_latest_block",
            "icn_chain_disconnects_total",
        ];

        for expected in expected_metrics {
            assert!(
                metric_names.contains(&expected.to_string()),
                "Missing metric: {}",
                expected
            );
        }
    }

    /// Test Case: Metrics HTTP endpoint format
    /// Purpose: Verify metrics are exported in Prometheus text format
    /// Contract: Should produce valid Prometheus exposition format
    #[test]
    fn test_metrics_prometheus_format() {
        let metrics = Metrics::new().expect("Failed to create metrics");

        // Set some metric values
        metrics.current_slot.set(100.0);
        metrics.elected_slots_total.inc();
        metrics.bft_rounds_success.inc();
        metrics.connected_peers.set(15.0);
        metrics.chain_latest_block.set(5000.0);

        // Encode metrics to Prometheus text format
        let encoder = prometheus::TextEncoder::new();
        let metric_families = metrics.registry().gather();
        let mut buffer = Vec::new();

        encoder
            .encode(&metric_families, &mut buffer)
            .expect("Failed to encode metrics");

        let output = String::from_utf8(buffer).expect("Failed to convert to string");

        // Verify Prometheus format
        assert!(output.contains("# HELP icn_director_current_slot"));
        assert!(output.contains("# TYPE icn_director_current_slot gauge"));
        assert!(output.contains("icn_director_current_slot 100"));

        assert!(output.contains("# HELP icn_director_elected_slots_total"));
        assert!(output.contains("# TYPE icn_director_elected_slots_total counter"));
        assert!(output.contains("icn_director_elected_slots_total 1"));

        assert!(output.contains("# HELP icn_p2p_connected_peers"));
        assert!(output.contains("icn_p2p_connected_peers 15"));

        assert!(output.contains("# HELP icn_chain_latest_block"));
        assert!(output.contains("icn_chain_latest_block 5000"));
    }

    /// Test Case: BFT metrics counters
    /// Purpose: Verify BFT success/failure counters increment correctly
    /// Contract: Counters should increase monotonically
    #[test]
    fn test_metrics_bft_rounds_counter() {
        let metrics = Metrics::new().expect("Failed to create metrics");

        // Initial values should be 0
        let initial_success = metrics.bft_rounds_success.get();
        let initial_failed = metrics.bft_rounds_failed.get();

        assert_eq!(initial_success, 0.0);
        assert_eq!(initial_failed, 0.0);

        // Increment success counter
        metrics.bft_rounds_success.inc();
        metrics.bft_rounds_success.inc();
        metrics.bft_rounds_success.inc();

        assert_eq!(metrics.bft_rounds_success.get(), 3.0);
        assert_eq!(metrics.bft_rounds_failed.get(), 0.0);

        // Increment failure counter
        metrics.bft_rounds_failed.inc();

        assert_eq!(metrics.bft_rounds_success.get(), 3.0);
        assert_eq!(metrics.bft_rounds_failed.get(), 1.0);

        // Counters should only increase, never decrease
        // (This is enforced by Prometheus Counter type)
    }

    /// Test Case: Slot metrics tracking
    /// Purpose: Verify slot-related metrics update correctly
    /// Contract: Current slot, elected count, missed count should track properly
    #[test]
    fn test_metrics_slot_tracking() {
        let metrics = Metrics::new().expect("Failed to create metrics");

        // Update current slot
        metrics.current_slot.set(50.0);
        assert_eq!(metrics.current_slot.get(), 50.0);

        metrics.current_slot.set(51.0);
        assert_eq!(metrics.current_slot.get(), 51.0);

        // Track elected slots
        metrics.elected_slots_total.inc();
        metrics.elected_slots_total.inc();
        assert_eq!(metrics.elected_slots_total.get(), 2.0);

        // Track missed slots
        metrics.missed_slots_total.inc();
        assert_eq!(metrics.missed_slots_total.get(), 1.0);
    }

    /// Test Case: BFT round duration histogram
    /// Purpose: Verify histogram records durations in correct buckets
    /// Contract: Should use predefined buckets: 1.0, 2.0, 5.0, 10.0, 20.0, 30.0
    #[test]
    fn test_metrics_bft_duration_histogram() {
        let metrics = Metrics::new().expect("Failed to create metrics");

        // Record some BFT round durations
        metrics.bft_round_duration.observe(0.5); // < 1s
        metrics.bft_round_duration.observe(3.0); // 2-5s bucket
        metrics.bft_round_duration.observe(8.0); // 5-10s bucket
        metrics.bft_round_duration.observe(15.0); // 10-20s bucket

        // Gather metrics
        let metric_families = metrics.registry().gather();

        // Find the histogram metric
        let histogram = metric_families
            .iter()
            .find(|mf| mf.get_name() == "icn_bft_round_duration_seconds")
            .expect("Histogram not found");

        // Verify it's a histogram type
        assert_eq!(
            histogram.get_field_type(),
            prometheus::proto::MetricType::HISTOGRAM
        );

        // Verify we have observations
        let histogram_data = histogram.get_metric()[0].get_histogram();
        assert_eq!(histogram_data.get_sample_count(), 4);
    }

    /// Test Case: Chain metrics tracking
    /// Purpose: Verify chain-related metrics update correctly
    /// Contract: Latest block and disconnect counter should track properly
    #[test]
    fn test_metrics_chain_tracking() {
        let metrics = Metrics::new().expect("Failed to create metrics");

        // Update latest block
        metrics.chain_latest_block.set(1000.0);
        assert_eq!(metrics.chain_latest_block.get(), 1000.0);

        metrics.chain_latest_block.set(1050.0);
        assert_eq!(metrics.chain_latest_block.get(), 1050.0);

        // Track disconnections
        assert_eq!(metrics.chain_disconnects.get(), 0.0);

        metrics.chain_disconnects.inc();
        metrics.chain_disconnects.inc();

        assert_eq!(metrics.chain_disconnects.get(), 2.0);
    }

    /// Test Case: P2P peer count gauge
    /// Purpose: Verify peer count gauge updates correctly
    /// Contract: Gauge should reflect current connected peer count
    #[test]
    fn test_metrics_p2p_peer_count() {
        let metrics = Metrics::new().expect("Failed to create metrics");

        // Initial peer count
        metrics.connected_peers.set(0.0);
        assert_eq!(metrics.connected_peers.get(), 0.0);

        // Peers connect
        metrics.connected_peers.set(5.0);
        assert_eq!(metrics.connected_peers.get(), 5.0);

        metrics.connected_peers.set(10.0);
        assert_eq!(metrics.connected_peers.get(), 10.0);

        // Peers disconnect (gauge can decrease)
        metrics.connected_peers.set(7.0);
        assert_eq!(metrics.connected_peers.get(), 7.0);
    }

    /// Test Case: Metrics HTTP endpoint on port 9100
    /// Purpose: Verify metrics would be served on correct port
    /// Contract: Integration with HTTP server on port 9100
    /// Note: This is a documentation test - actual HTTP server tested in integration
    #[test]
    #[ignore] // Requires HTTP server infrastructure
    fn test_metrics_http_endpoint() {
        // This test documents the expected integration:
        // 1. Metrics registry is passed to HTTP server
        // 2. Server listens on port 9100 (from config.metrics_port)
        // 3. GET /metrics returns Prometheus text format
        // 4. Response has Content-Type: text/plain; version=0.0.4

        // Actual implementation would use:
        // let addr = ([0, 0, 0, 0], config.metrics_port).into();
        // let listener = TcpListener::bind(addr).await?;
        // Route: GET /metrics -> metrics.registry().gather()

        let metrics = Metrics::new().expect("Failed to create metrics");

        // Verify registry can be accessed for HTTP endpoint
        let registry = metrics.registry();
        let metric_families = registry.gather();

        // Verify we have metrics to serve
        assert!(!metric_families.is_empty());

        // Verify encoding works (what HTTP endpoint would return)
        let encoder = prometheus::TextEncoder::new();
        let mut buffer = Vec::new();
        encoder
            .encode(&metric_families, &mut buffer)
            .expect("Encode failed");

        // Verify non-empty response
        assert!(!buffer.is_empty());
    }
}
