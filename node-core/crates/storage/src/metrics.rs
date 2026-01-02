//! Prometheus metrics for storage operations

use prometheus::{Histogram, HistogramOpts, IntCounterVec, Opts, Registry};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetricsError {
    #[error("Prometheus error: {0}")]
    Prometheus(#[from] prometheus::Error),
}

pub struct StorageMetrics {
    pub operations_total: IntCounterVec,
    pub operations_failed_total: IntCounterVec,
    pub operation_duration_seconds: Histogram,
}

impl StorageMetrics {
    pub fn new(registry: &Registry) -> Result<Self, MetricsError> {
        let operations_total = IntCounterVec::new(
            Opts::new(
                "nsn_storage_operations_total",
                "Total number of storage operations",
            ),
            &["operation"],
        )?;

        let operations_failed_total = IntCounterVec::new(
            Opts::new(
                "nsn_storage_operations_failed_total",
                "Total number of failed storage operations",
            ),
            &["operation"],
        )?;

        let operation_duration_seconds = Histogram::with_opts(HistogramOpts::new(
            "nsn_storage_operation_duration_seconds",
            "Duration of storage operations",
        ))?;

        registry.register(Box::new(operations_total.clone()))?;
        registry.register(Box::new(operations_failed_total.clone()))?;
        registry.register(Box::new(operation_duration_seconds.clone()))?;

        Ok(Self {
            operations_total,
            operations_failed_total,
            operation_duration_seconds,
        })
    }

    #[cfg(test)]
    pub fn new_unregistered() -> Self {
        Self {
            operations_total: IntCounterVec::new(
                Opts::new("test_storage_operations_total", "test"),
                &["operation"],
            )
            .unwrap(),
            operations_failed_total: IntCounterVec::new(
                Opts::new("test_storage_operations_failed_total", "test"),
                &["operation"],
            )
            .unwrap(),
            operation_duration_seconds: Histogram::with_opts(HistogramOpts::new(
                "test_storage_operation_duration_seconds",
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
    fn test_storage_metrics_creation() {
        let registry = Registry::new();
        let metrics = StorageMetrics::new(&registry).expect("metrics");

        let initial = metrics.operations_total.with_label_values(&["put"]).get();
        assert_eq!(initial, 0);
    }
}
