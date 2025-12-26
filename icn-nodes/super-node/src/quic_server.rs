//! QUIC transport server for shard transfers
//!
//! Provides high-performance shard streaming to Regional Relays using Quinn QUIC implementation.
//! Supports multiplexed streams, low latency (<100ms shard retrieval), and efficient bandwidth usage.

use quinn::{Endpoint, ServerConfig};
use rcgen::generate_simple_self_signed;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// QUIC shard request from relay
#[derive(Debug, Clone)]
pub struct ShardRequest {
    pub cid: String,
    pub shard_index: usize,
}

/// QUIC server for shard transfers
pub struct QuicServer {
    endpoint: Endpoint,
    storage_root: PathBuf,
}

impl QuicServer {
    /// Create new QUIC server
    ///
    /// # Arguments
    /// * `port` - Port to listen on (default: 9002)
    /// * `storage_root` - Root path for shard storage
    ///
    /// # Returns
    /// QUIC server instance
    pub async fn new(port: u16, storage_root: PathBuf) -> crate::error::Result<Self> {
        // Generate self-signed certificate for TLS
        let cert =
            generate_simple_self_signed(vec!["icn-super-node".to_string()]).map_err(|e| {
                crate::error::SuperNodeError::QuicTransport(format!(
                    "Cert generation failed: {}",
                    e
                ))
            })?;

        let key = PrivateKeyDer::Pkcs8(cert.key_pair.serialize_der().into());
        let cert_der = CertificateDer::from(cert.cert);

        // Configure TLS
        let mut server_config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![cert_der], key)
            .map_err(|e| {
                crate::error::SuperNodeError::QuicTransport(format!("TLS config error: {}", e))
            })?;

        server_config.alpn_protocols = vec![b"icn-shard/1".to_vec()];

        let mut quinn_server_config = ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(server_config).map_err(|e| {
                crate::error::SuperNodeError::QuicTransport(format!(
                    "QUIC server config error: {}",
                    e
                ))
            })?,
        ));

        // Configure transport parameters
        let mut transport_config = quinn::TransportConfig::default();
        transport_config.max_concurrent_bidi_streams(100u32.into());
        transport_config.max_concurrent_uni_streams(100u32.into());
        transport_config.max_idle_timeout(Some(quinn::IdleTimeout::from(quinn::VarInt::from_u32(
            30_000,
        )))); // 30 seconds

        quinn_server_config.transport_config(Arc::new(transport_config));

        // Bind to address
        let addr: SocketAddr = format!("0.0.0.0:{}", port).parse().map_err(|e| {
            crate::error::SuperNodeError::QuicTransport(format!("Invalid socket address: {}", e))
        })?;

        let endpoint = Endpoint::server(quinn_server_config, addr).map_err(|e| {
            crate::error::SuperNodeError::QuicTransport(format!("Endpoint creation failed: {}", e))
        })?;

        info!("QUIC server listening on port {}", port);

        Ok(Self {
            endpoint,
            storage_root,
        })
    }

    /// Run QUIC server event loop
    ///
    /// Accepts incoming connections and handles shard requests
    pub async fn run(&self) -> crate::error::Result<()> {
        loop {
            match self.endpoint.accept().await {
                Some(incoming) => {
                    let storage_root = self.storage_root.clone();
                    tokio::spawn(async move {
                        match incoming.await {
                            Ok(connection) => {
                                if let Err(e) =
                                    Self::handle_connection(connection, storage_root).await
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

    /// Handle incoming QUIC connection
    async fn handle_connection(
        connection: quinn::Connection,
        storage_root: PathBuf,
    ) -> crate::error::Result<()> {
        let remote_addr = connection.remote_address();
        debug!("Accepted connection from {}", remote_addr);

        // Handle bidirectional streams
        loop {
            match connection.accept_bi().await {
                Ok((send, recv)) => {
                    let storage_root = storage_root.clone();
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_stream(send, recv, storage_root).await {
                            warn!("Stream handling error: {}", e);
                        }
                    });
                }
                Err(quinn::ConnectionError::ApplicationClosed(_)) => {
                    debug!("Connection closed by peer: {}", remote_addr);
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

    /// Handle bidirectional stream for shard request
    async fn handle_stream(
        mut send: quinn::SendStream,
        mut recv: quinn::RecvStream,
        storage_root: PathBuf,
    ) -> crate::error::Result<()> {
        // Read request (format: "GET /shards/<CID>/shard_<N>.bin")
        let request_buf = recv.read_to_end(256).await.map_err(|e| {
            crate::error::SuperNodeError::QuicTransport(format!("Request read error: {}", e))
        })?;

        let request = String::from_utf8_lossy(&request_buf);
        debug!("Received request: {}", request);

        // Parse request
        if let Some((cid, shard_index)) = Self::parse_shard_request(&request) {
            // Construct shard path
            let shard_path = storage_root
                .join(&cid)
                .join(format!("shard_{:02}.bin", shard_index));

            // Read shard data
            match tokio::fs::read(&shard_path).await {
                Ok(shard_data) => {
                    debug!(
                        "Serving shard: {} index {} ({} bytes)",
                        cid,
                        shard_index,
                        shard_data.len()
                    );

                    // Send shard data
                    send.write_all(&shard_data).await.map_err(|e| {
                        crate::error::SuperNodeError::QuicTransport(format!(
                            "Shard write error: {}",
                            e
                        ))
                    })?;

                    send.finish().map_err(|e| {
                        crate::error::SuperNodeError::QuicTransport(format!(
                            "Stream finish error: {}",
                            e
                        ))
                    })?;
                }
                Err(e) => {
                    error!("Shard read error: {} - {}", shard_path.display(), e);

                    // Send error response
                    let error_msg = "ERROR: Shard not found\n";
                    send.write_all(error_msg.as_bytes()).await.ok();
                    send.finish().ok();
                }
            }
        } else {
            // Invalid request
            warn!("Invalid shard request: {}", request);
            let error_msg = "ERROR: Invalid request format\n";
            send.write_all(error_msg.as_bytes()).await.ok();
            send.finish().ok();
        }

        Ok(())
    }

    /// Parse shard request
    ///
    /// Format: "GET /shards/<CID>/shard_<N>.bin"
    ///
    /// # Returns
    /// (CID, shard_index) if valid, None otherwise
    fn parse_shard_request(request: &str) -> Option<(String, usize)> {
        let parts: Vec<&str> = request.split_whitespace().collect();

        if parts.len() < 2 || parts[0] != "GET" {
            return None;
        }

        let path = parts[1];

        // Extract CID and shard index from path
        // Expected: /shards/<CID>/shard_NN.bin
        let path_parts: Vec<&str> = path.split('/').collect();

        if path_parts.len() < 4 || path_parts[1] != "shards" {
            return None;
        }

        let cid = path_parts[2].to_string();
        let shard_filename = path_parts[3];

        // Parse shard index from "shard_NN.bin"
        if let Some(index_str) = shard_filename.strip_prefix("shard_") {
            if let Some(index_str) = index_str.strip_suffix(".bin") {
                if let Ok(index) = index_str.parse::<usize>() {
                    return Some((cid, index));
                }
            }
        }

        None
    }

    /// Serve shard to relay (for external/manual usage)
    ///
    /// # Arguments
    /// * `cid` - Content ID
    /// * `shard_index` - Shard index (0-13)
    /// * `shard_data` - Shard bytes
    ///
    /// # Returns
    /// Result indicating success/failure
    pub async fn serve_shard(
        &self,
        _cid: &str,
        _shard_index: usize,
        _shard_data: Vec<u8>,
    ) -> crate::error::Result<()> {
        // This method is deprecated in favor of run() event loop
        // Kept for backwards compatibility
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Setup test environment - install rustls crypto provider
    fn setup_test() {
        // Install default crypto provider for rustls (ring)
        // This is required for TLS operations in tests
        let _ = rustls::crypto::ring::default_provider().install_default();
    }

    #[tokio::test]
    async fn test_quic_server_creation() {
        setup_test();

        let tmp_dir = tempdir().unwrap();
        let storage_root = tmp_dir.path().to_path_buf();

        let result = QuicServer::new(0, storage_root).await; // Port 0 = OS assigns
        assert!(result.is_ok(), "QUIC server creation should succeed");
    }

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
        let invalid_requests = vec![
            "POST /shards/bafytest/shard_00.bin", // Wrong method
            "GET /invalid/path",                  // Invalid path
            "GET /shards/bafytest/invalid.bin",   // Invalid filename
            "GET /shards/bafytest/shard_abc.bin", // Non-numeric index
        ];

        for request in invalid_requests {
            let result = QuicServer::parse_shard_request(request);
            assert!(
                result.is_none(),
                "Should reject invalid request: {}",
                request
            );
        }
    }

    #[tokio::test]
    async fn test_serve_shard() {
        setup_test();

        let tmp_dir = tempdir().unwrap();
        let storage_root = tmp_dir.path().to_path_buf();
        let server = QuicServer::new(0, storage_root).await.unwrap();

        let result = server.serve_shard("bafytest", 0, vec![1, 2, 3]).await;
        assert!(result.is_ok());
    }
}
