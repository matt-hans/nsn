//! Prometheus metrics for Regional Relay

use bytes::Bytes;
use http_body_util::Full;
use hyper::{Request, Response};
use lazy_static::lazy_static;
use prometheus::{
    opts, register_counter, register_gauge, register_histogram, Counter, Gauge, Histogram,
};

lazy_static! {
    /// Cache hit counter
    pub static ref CACHE_HITS: Counter =
        register_counter!(opts!("icn_relay_cache_hits_total", "Total cache hits")).unwrap();

    /// Cache miss counter
    pub static ref CACHE_MISSES: Counter =
        register_counter!(opts!("icn_relay_cache_misses_total", "Total cache misses")).unwrap();

    /// Upstream fetch counter
    pub static ref UPSTREAM_FETCHES: Counter =
        register_counter!(opts!("icn_relay_upstream_fetches_total", "Total upstream Super-Node fetches")).unwrap();

    /// Cache eviction counter
    pub static ref CACHE_EVICTIONS: Counter =
        register_counter!(opts!("icn_relay_cache_evictions_total", "Total cache evictions")).unwrap();

    /// Bytes served to viewers
    pub static ref BYTES_SERVED: Counter =
        register_counter!(opts!("icn_relay_bytes_served_total", "Total bytes served to viewers")).unwrap();

    /// Active viewer connections
    pub static ref VIEWER_CONNECTIONS: Gauge =
        register_gauge!(opts!("icn_relay_viewer_connections", "Active viewer connections")).unwrap();

    /// Cache size in bytes
    pub static ref CACHE_SIZE_BYTES: Gauge =
        register_gauge!(opts!("icn_relay_cache_size_bytes", "Current cache size in bytes")).unwrap();

    /// Cache utilization percentage
    pub static ref CACHE_UTILIZATION: Gauge =
        register_gauge!(opts!("icn_relay_cache_utilization_percent", "Cache utilization percentage")).unwrap();

    /// Shard serve latency histogram
    pub static ref SHARD_SERVE_LATENCY: Histogram =
        register_histogram!("icn_relay_shard_serve_latency_seconds", "Shard serve latency in seconds").unwrap();

    /// Upstream fetch latency histogram
    pub static ref UPSTREAM_FETCH_LATENCY: Histogram =
        register_histogram!("icn_relay_upstream_fetch_latency_seconds", "Upstream fetch latency in seconds").unwrap();
}

/// Start Prometheus metrics HTTP server
///
/// Serves metrics on `/metrics` endpoint
pub async fn start_metrics_server(port: u16) -> crate::error::Result<()> {
    use hyper::{server::conn::http1, service::service_fn};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).await?;

    tracing::info!("Metrics server listening on http://{}/metrics", addr);

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);

        tokio::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(io, service_fn(metrics_handler))
                .await
            {
                tracing::error!("Metrics server error: {:?}", err);
            }
        });
    }
}

async fn metrics_handler(
    _req: Request<hyper::body::Incoming>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    use prometheus::{Encoder, TextEncoder};

    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();

    encoder
        .encode(&metric_families, &mut buffer)
        .map_err(|e| {
            tracing::error!("Failed to encode metrics: {}", e);
        })
        .ok();

    Ok(Response::new(Full::new(Bytes::from(buffer))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_initialization() {
        // Test that metrics are registered
        CACHE_HITS.inc();
        assert!(CACHE_HITS.get() > 0.0);

        CACHE_MISSES.inc();
        assert!(CACHE_MISSES.get() > 0.0);
    }
}
