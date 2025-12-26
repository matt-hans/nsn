//! Prometheus metrics for Super-Node

use prometheus::{
    register_int_counter, register_int_gauge, IntCounter, IntGauge, Registry, TextEncoder,
};

lazy_static::lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    pub static ref SHARD_COUNT: IntGauge = register_int_gauge!(
        "icn_super_node_shard_count",
        "Total number of shards stored"
    ).unwrap();

    pub static ref BYTES_STORED: IntGauge = register_int_gauge!(
        "icn_super_node_bytes_stored",
        "Total bytes stored"
    ).unwrap();

    pub static ref AUDIT_SUCCESS_TOTAL: IntCounter = register_int_counter!(
        "icn_super_node_audit_success_total",
        "Total successful audit responses"
    ).unwrap();

    pub static ref AUDIT_FAILURE_TOTAL: IntCounter = register_int_counter!(
        "icn_super_node_audit_failure_total",
        "Total failed audit responses"
    ).unwrap();
}

/// Get Prometheus metrics as text
pub fn get_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    encoder.encode_to_string(&metric_families).unwrap()
}

use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use std::convert::Infallible;
use std::net::SocketAddr;
use tokio::net::TcpListener;

/// Handle metrics HTTP request
async fn handle_metrics(
    _req: Request<hyper::body::Incoming>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let metrics_text = get_metrics();
    Ok(Response::new(Full::new(Bytes::from(metrics_text))))
}

/// Start metrics HTTP server
pub async fn start_metrics_server(port: u16) -> crate::error::Result<()> {
    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse().map_err(|e| {
        crate::error::SuperNodeError::Internal(format!("Invalid metrics address: {}", e))
    })?;

    tracing::info!("Metrics server listening on http://{}/metrics", addr);

    tokio::spawn(async move {
        let listener = match TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("Metrics server bind failed: {}", e);
                return;
            }
        };

        loop {
            let (stream, _) = match listener.accept().await {
                Ok(conn) => conn,
                Err(e) => {
                    tracing::warn!("Metrics server accept error: {}", e);
                    continue;
                }
            };

            let io = TokioIo::new(stream);

            tokio::spawn(async move {
                if let Err(e) = http1::Builder::new()
                    .serve_connection(io, service_fn(handle_metrics))
                    .await
                {
                    tracing::debug!("Metrics connection error: {}", e);
                }
            });
        }
    });

    Ok(())
}
