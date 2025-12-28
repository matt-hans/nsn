//! QUIC server for serving viewers
//!
//! Accepts viewer connections via QUIC (WebTransport-compatible) and serves cached shards.
//! Based on Super-Node QUIC server pattern but optimized for cache lookups.

use crate::cache::{ShardCache, ShardKey};
use crate::metrics::{
    BYTES_SERVED, CACHE_HITS, CACHE_MISSES, SHARD_SERVE_LATENCY, UPSTREAM_FETCHES,
    UPSTREAM_FETCH_LATENCY, VIEWER_CONNECTIONS,
};
use crate::upstream_client::UpstreamClient;
use governor::{DefaultDirectRateLimiter, Quota, RateLimiter};
use quinn::{Endpoint, ServerConfig};
use rcgen::generate_simple_self_signed;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

/// Connection limiting configuration
#[derive(Clone)]
pub struct ConnectionLimits {
    /// Global rate limiter (connections per second across all IPs)
    global_limiter: Arc<DefaultDirectRateLimiter>,
    /// Per-IP rate limiters
    per_ip_limiters: Arc<Mutex<HashMap<IpAddr, Arc<DefaultDirectRateLimiter>>>>,
    /// Max connections per IP per second
    per_ip_rate: u32,
}

impl ConnectionLimits {
    /// Create new connection limits
    pub fn new(global_rate: u32, per_ip_rate: u32) -> Self {
        let global_limiter = Arc::new(RateLimiter::direct(Quota::per_second(
            NonZeroU32::new(global_rate).expect("global_rate must be > 0"),
        )));

        Self {
            global_limiter,
            per_ip_limiters: Arc::new(Mutex::new(HashMap::new())),
            per_ip_rate,
        }
    }

    /// Check if connection should be allowed
    async fn check(&self, ip: IpAddr) -> bool {
        // Check global limit first
        if self.global_limiter.check().is_err() {
            return false;
        }

        // Check per-IP limit
        let mut limiters = self.per_ip_limiters.lock().await;
        let limiter = limiters.entry(ip).or_insert_with(|| {
            Arc::new(RateLimiter::direct(Quota::per_second(
                NonZeroU32::new(self.per_ip_rate).expect("per_ip_rate must be > 0"),
            )))
        });

        limiter.check().is_ok()
    }
}

/// Configuration for QUIC server
#[derive(Clone)]
pub struct QuicServerConfig {
    pub require_auth: bool,
    pub auth_tokens: Arc<Vec<String>>,
}

impl QuicServerConfig {
    /// Create config requiring authentication
    pub fn with_auth(tokens: Vec<String>) -> Self {
        Self {
            require_auth: true,
            auth_tokens: Arc::new(tokens),
        }
    }

    /// Create config without authentication (dev mode)
    pub fn no_auth() -> Self {
        Self {
            require_auth: false,
            auth_tokens: Arc::new(Vec::new()),
        }
    }
}

/// QUIC server for viewer shard requests
pub struct QuicServer {
    endpoint: Endpoint,
    cache: Arc<Mutex<ShardCache>>,
    upstream_client: Arc<UpstreamClient>,
    super_node_addresses: Vec<String>,
    connection_limits: ConnectionLimits,
    config: QuicServerConfig,
}

impl QuicServer {
    /// Create new QUIC server for viewers
    pub async fn new(
        port: u16,
        cache: Arc<Mutex<ShardCache>>,
        upstream_client: Arc<UpstreamClient>,
        super_node_addresses: Vec<String>,
        global_conn_rate: u32,
        per_ip_conn_rate: u32,
        config: QuicServerConfig,
    ) -> crate::error::Result<Self> {
        // Generate self-signed certificate for TLS
        let cert = generate_simple_self_signed(vec!["icn-relay".to_string()]).map_err(|e| {
            crate::error::RelayError::QuicTransport(format!("Cert generation failed: {}", e))
        })?;

        let key = PrivateKeyDer::Pkcs8(cert.key_pair.serialize_der().into());
        let cert_der = CertificateDer::from(cert.cert);

        // Configure TLS
        let mut server_config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![cert_der], key)
            .map_err(|e| {
                crate::error::RelayError::QuicTransport(format!("TLS config error: {}", e))
            })?;

        server_config.alpn_protocols = vec![b"icn-relay/1".to_vec()];

        let mut quinn_server_config = ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(server_config).map_err(|e| {
                crate::error::RelayError::QuicTransport(format!("QUIC server config error: {}", e))
            })?,
        ));

        // Configure transport parameters
        let mut transport_config = quinn::TransportConfig::default();
        transport_config.max_concurrent_bidi_streams(200u32.into()); // More concurrent viewers
        transport_config.max_concurrent_uni_streams(200u32.into());
        transport_config.max_idle_timeout(Some(quinn::IdleTimeout::from(quinn::VarInt::from_u32(
            60_000,
        )))); // 60 seconds

        quinn_server_config.transport_config(Arc::new(transport_config));

        // Bind to address
        let addr: SocketAddr = format!("0.0.0.0:{}", port).parse().map_err(|e| {
            crate::error::RelayError::QuicTransport(format!("Invalid socket address: {}", e))
        })?;

        let endpoint = Endpoint::server(quinn_server_config, addr).map_err(|e| {
            crate::error::RelayError::QuicTransport(format!("Endpoint creation failed: {}", e))
        })?;

        info!(
            "QUIC server (viewers) listening on port {} (global limit: {}/s, per-IP: {}/s, auth: {})",
            port, global_conn_rate, per_ip_conn_rate, config.require_auth
        );

        Ok(Self {
            endpoint,
            cache,
            upstream_client,
            super_node_addresses,
            connection_limits: ConnectionLimits::new(global_conn_rate, per_ip_conn_rate),
            config,
        })
    }

    /// Run QUIC server event loop
    pub async fn run(&self) -> crate::error::Result<()> {
        loop {
            match self.endpoint.accept().await {
                Some(incoming) => {
                    let remote_addr = incoming.remote_address();
                    let connection_limits = self.connection_limits.clone();

                    // Check rate limits before accepting connection
                    if !connection_limits.check(remote_addr.ip()).await {
                        warn!(
                            "Rate limit exceeded for {}, rejecting connection",
                            remote_addr
                        );
                        continue; // Drop the connection
                    }

                    let cache = Arc::clone(&self.cache);
                    let upstream_client = Arc::clone(&self.upstream_client);
                    let super_node_addresses = self.super_node_addresses.clone();

                    let server_config = self.config.clone();

                    tokio::spawn(async move {
                        match incoming.await {
                            Ok(connection) => {
                                if let Err(e) = Self::handle_connection(
                                    connection,
                                    cache,
                                    upstream_client,
                                    super_node_addresses,
                                    server_config,
                                )
                                .await
                                {
                                    error!("Connection handling error: {}", e);
                                }
                            }
                            Err(e) => {
                                error!("Incoming connection error: {}", e);
                            }
                        }
                    });
                }
                None => {
                    warn!("QUIC endpoint closed");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle incoming viewer connection
    async fn handle_connection(
        connection: quinn::Connection,
        cache: Arc<Mutex<ShardCache>>,
        upstream_client: Arc<UpstreamClient>,
        super_node_addresses: Vec<String>,
        server_config: QuicServerConfig,
    ) -> crate::error::Result<()> {
        let remote_addr = connection.remote_address();
        debug!("Accepted viewer connection from {}", remote_addr);

        // Increment active connections
        VIEWER_CONNECTIONS.inc();

        let result = async {
            loop {
                match connection.accept_bi().await {
                    Ok((send, recv)) => {
                        let cache = Arc::clone(&cache);
                        let upstream_client = Arc::clone(&upstream_client);
                        let super_node_addresses = super_node_addresses.clone();
                        let config = server_config.clone();

                        tokio::spawn(async move {
                            if let Err(e) = Self::handle_stream(
                                send,
                                recv,
                                cache,
                                upstream_client,
                                super_node_addresses,
                                config,
                            )
                            .await
                            {
                                warn!("Stream handling error: {}", e);
                            }
                        });
                    }
                    Err(quinn::ConnectionError::ApplicationClosed(_)) => {
                        debug!("Connection closed by viewer: {}", remote_addr);
                        break;
                    }
                    Err(e) => {
                        error!("Accept stream error: {}", e);
                        break;
                    }
                }
            }
            Ok(())
        }
        .await;

        // Decrement active connections on disconnect
        VIEWER_CONNECTIONS.dec();

        result
    }

    /// Handle bidirectional stream for shard request
    async fn handle_stream(
        mut send: quinn::SendStream,
        mut recv: quinn::RecvStream,
        cache: Arc<Mutex<ShardCache>>,
        upstream_client: Arc<UpstreamClient>,
        super_node_addresses: Vec<String>,
        server_config: QuicServerConfig,
    ) -> crate::error::Result<()> {
        let stream_start = Instant::now();

        // Read and parse request
        let (_request, cid, shard_index) =
            Self::parse_request(&mut send, &mut recv, &server_config).await?;

        let key = ShardKey::new(cid.clone(), shard_index);

        // Fetch shard data (from cache or upstream)
        let shard_data = Self::fetch_shard_data(
            cache,
            upstream_client,
            super_node_addresses,
            &cid,
            shard_index,
            &key,
            &mut send,
        )
        .await?;

        // Send shard to viewer and record metrics
        Self::send_shard_and_record_metrics(&mut send, shard_data, &key, stream_start).await?;

        Ok(())
    }

    /// Parse and authenticate incoming request
    async fn parse_request(
        send: &mut quinn::SendStream,
        recv: &mut quinn::RecvStream,
        server_config: &QuicServerConfig,
    ) -> crate::error::Result<(String, String, usize)> {
        // Read request
        let request_buf = recv.read_to_end(512).await.map_err(|e| {
            crate::error::RelayError::QuicTransport(format!("Request read error: {}", e))
        })?;

        let request = String::from_utf8_lossy(&request_buf).to_string();
        debug!("Viewer request: {}", request);

        // Check authentication if required
        if server_config.require_auth {
            Self::authenticate_request(send, &request, server_config).await?;
        }

        // Parse shard request (format: "GET /shards/<CID>/shard_<N>.bin")
        if let Some((cid, shard_index)) = Self::parse_shard_request(&request) {
            Ok((request, cid, shard_index))
        } else {
            warn!("Invalid shard request: {}", request);
            let error_msg = "ERROR: Invalid request format\n";
            send.write_all(error_msg.as_bytes()).await.ok();
            send.finish().ok();
            Err(crate::error::RelayError::InvalidRequest(
                "Invalid request format".to_string(),
            ))
        }
    }

    /// Authenticate request using token
    async fn authenticate_request(
        send: &mut quinn::SendStream,
        request: &str,
        server_config: &QuicServerConfig,
    ) -> crate::error::Result<()> {
        let lines: Vec<&str> = request.lines().collect();
        if lines.is_empty() {
            let error_msg = "ERROR: Missing authentication\n";
            send.write_all(error_msg.as_bytes()).await.ok();
            send.finish().ok();
            return Err(crate::error::RelayError::Unauthorized(
                "No auth header".to_string(),
            ));
        }

        // First line should be "AUTH <token>"
        let auth_line = lines[0];
        if !auth_line.starts_with("AUTH ") {
            let error_msg = "ERROR: Missing AUTH header\n";
            send.write_all(error_msg.as_bytes()).await.ok();
            send.finish().ok();
            return Err(crate::error::RelayError::Unauthorized(
                "No AUTH header".to_string(),
            ));
        }

        let token = auth_line.trim_start_matches("AUTH ").trim();
        if !server_config.auth_tokens.contains(&token.to_string()) {
            let error_msg = "ERROR: Invalid auth token\n";
            send.write_all(error_msg.as_bytes()).await.ok();
            send.finish().ok();
            return Err(crate::error::RelayError::Unauthorized(format!(
                "Invalid token: {}",
                token
            )));
        }

        debug!("Viewer authenticated successfully");
        Ok(())
    }

    /// Fetch shard data from cache or upstream
    async fn fetch_shard_data(
        cache: Arc<Mutex<ShardCache>>,
        upstream_client: Arc<UpstreamClient>,
        super_node_addresses: Vec<String>,
        cid: &str,
        shard_index: usize,
        key: &ShardKey,
        send: &mut quinn::SendStream,
    ) -> crate::error::Result<Vec<u8>> {
        // Check cache first
        let cached_data = cache.lock().await.get(key).await;

        if let Some(data) = cached_data {
            debug!("Cache HIT: serving {} from cache", key.hash());
            CACHE_HITS.inc();
            return Ok(data);
        }

        // Cache MISS - fetch from upstream
        debug!("Cache MISS: fetching {} from upstream", key.hash());
        CACHE_MISSES.inc();

        let mut last_error = None;
        for super_node_addr in &super_node_addresses {
            let fetch_start = Instant::now();

            match upstream_client
                .fetch_shard(super_node_addr, cid, shard_index)
                .await
            {
                Ok(data) => {
                    UPSTREAM_FETCHES.inc();
                    UPSTREAM_FETCH_LATENCY.observe(fetch_start.elapsed().as_secs_f64());

                    // Validate shard
                    if let Err(e) = Self::validate_shard(&data, cid, shard_index, send).await {
                        warn!("Shard validation failed: {}", e);
                        return Err(e);
                    }

                    // Cache the fetched shard
                    if let Err(e) = cache.lock().await.put(key.clone(), data.clone()).await {
                        warn!("Failed to cache shard {}: {}", key.hash(), e);
                    }
                    return Ok(data);
                }
                Err(e) => {
                    warn!("Upstream fetch from {} failed: {}", super_node_addr, e);
                    last_error = Some(e);
                }
            }
        }

        // All upstream fetches failed
        let error_msg = format!(
            "ERROR: Failed to fetch shard from all Super-Nodes: {:?}\n",
            last_error
        );
        send.write_all(error_msg.as_bytes()).await.ok();
        send.finish().ok();
        Err(crate::error::RelayError::UpstreamFetchFailed(format!(
            "{:?}",
            last_error
        )))
    }

    /// Validate fetched shard data
    async fn validate_shard(
        data: &[u8],
        cid: &str,
        shard_index: usize,
        send: &mut quinn::SendStream,
    ) -> crate::error::Result<()> {
        // Basic shard validation (size and non-empty check)
        if data.is_empty() {
            warn!(
                "Received empty shard data for CID={}, index={}",
                cid, shard_index
            );
            let error_msg = "ERROR: Received empty shard from upstream\n";
            send.write_all(error_msg.as_bytes()).await.ok();
            send.finish().ok();
            return Err(crate::error::RelayError::InvalidShard(
                "Empty shard data".to_string(),
            ));
        }

        // Reasonable size check: shards should be between 100 bytes and 10MB
        if data.len() < 100 || data.len() > 10 * 1024 * 1024 {
            warn!(
                "WARNING: Shard size {} bytes is outside expected range (100 - 10MB) for CID={}, index={}",
                data.len(), cid, shard_index
            );
        }

        Ok(())
    }

    /// Send shard to viewer and record metrics
    async fn send_shard_and_record_metrics(
        send: &mut quinn::SendStream,
        shard_data: Vec<u8>,
        key: &ShardKey,
        stream_start: Instant,
    ) -> crate::error::Result<()> {
        // Send shard to viewer
        send.write_all(&shard_data).await.map_err(|e| {
            crate::error::RelayError::QuicTransport(format!("Shard write error: {}", e))
        })?;
        send.finish().map_err(|e| {
            crate::error::RelayError::QuicTransport(format!("Stream finish error: {}", e))
        })?;

        // Update metrics
        BYTES_SERVED.inc_by(shard_data.len() as f64);
        SHARD_SERVE_LATENCY.observe(stream_start.elapsed().as_secs_f64());

        debug!("Served shard: {} ({} bytes)", key.hash(), shard_data.len());
        Ok(())
    }

    /// Verify shard hash using SHA-256
    ///
    /// # Arguments
    /// * `shard_data` - Shard bytes
    /// * `expected_hash` - Expected hash from manifest (hex encoded)
    ///
    /// # Returns
    /// Ok(()) if hash matches, Err otherwise
    #[allow(dead_code)] // TODO: Will be used once manifest integration is complete
    fn verify_shard_hash(shard_data: &[u8], expected_hash: &str) -> crate::error::Result<()> {
        let mut hasher = Sha256::new();
        hasher.update(shard_data);
        let computed_hash = hasher.finalize();
        let computed_hex = hex::encode(computed_hash);

        if computed_hex != expected_hash {
            return Err(crate::error::RelayError::ShardHashMismatch(
                expected_hash.to_string(),
                computed_hex,
            ));
        }

        Ok(())
    }

    /// Parse shard request (same format as Super-Node)
    fn parse_shard_request(request: &str) -> Option<(String, usize)> {
        let parts: Vec<&str> = request.split_whitespace().collect();
        if parts.len() < 2 || parts[0] != "GET" {
            return None;
        }

        let path = parts[1];
        let path_parts: Vec<&str> = path.split('/').collect();

        if path_parts.len() < 4 || path_parts[1] != "shards" {
            return None;
        }

        let cid = path_parts[2].to_string();
        let shard_filename = path_parts[3];

        if let Some(index_str) = shard_filename.strip_prefix("shard_") {
            if let Some(index_str) = index_str.strip_suffix(".bin") {
                if let Ok(index) = index_str.parse::<usize>() {
                    return Some((cid, index));
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_shard_request_valid() {
        let request = "GET /shards/bafytest123/shard_05.bin";
        let result = QuicServer::parse_shard_request(request);

        assert!(result.is_some());
        let (cid, index) = result.unwrap();
        assert_eq!(cid, "bafytest123");
        assert_eq!(index, 5);
    }

    #[test]
    fn test_parse_shard_request_invalid() {
        assert!(QuicServer::parse_shard_request("POST /shards/x/shard_0.bin").is_none());
        assert!(QuicServer::parse_shard_request("GET /invalid/path").is_none());
    }
}
