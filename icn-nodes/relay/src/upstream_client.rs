//! QUIC client for fetching shards from Super-Nodes
//!
//! Connects to upstream Super-Nodes via QUIC to retrieve video shards not present in local cache.

use quinn::{ClientConfig, Endpoint};
use rustls::RootCertStore;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{debug, info};

/// QUIC client for upstream shard fetching
pub struct UpstreamClient {
    endpoint: Endpoint,
}

impl UpstreamClient {
    /// Create new upstream QUIC client with proper TLS verification
    ///
    /// # Arguments
    /// * `dev_mode` - If true, skip certificate verification (for local development).
    ///   Requires `dev-mode` feature to be enabled at compile time.
    pub fn new(dev_mode: bool) -> crate::error::Result<Self> {
        let crypto = if dev_mode {
            #[cfg(feature = "dev-mode")]
            {
                info!("Upstream QUIC client: DEV MODE - skipping certificate verification");
                // For development: accept self-signed certificates
                rustls::ClientConfig::builder()
                    .dangerous()
                    .with_custom_certificate_verifier(SkipServerVerification::new())
                    .with_no_client_auth()
            }
            #[cfg(not(feature = "dev-mode"))]
            {
                return Err(crate::error::RelayError::Config(
                    "dev_mode requested but 'dev-mode' feature not enabled at compile time. \
                     Rebuild with --features dev-mode to enable insecure TLS verification skip."
                        .to_string(),
                ));
            }
        } else {
            // Production: use WebPKI root certificates
            let mut root_store = RootCertStore::empty();
            root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

            rustls::ClientConfig::builder()
                .with_root_certificates(root_store)
                .with_no_client_auth()
        };

        let client_config = ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(crypto).map_err(|e| {
                crate::error::RelayError::QuicTransport(format!("Client config error: {}", e))
            })?,
        ));

        let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap()).map_err(|e| {
            crate::error::RelayError::QuicTransport(format!("Endpoint creation failed: {}", e))
        })?;

        endpoint.set_default_client_config(client_config);

        info!("Upstream QUIC client initialized (dev_mode: {})", dev_mode);

        Ok(Self { endpoint })
    }

    /// Fetch shard from Super-Node
    ///
    /// # Arguments
    /// * `super_node_addr` - Super-Node address (e.g., "127.0.0.1:9002")
    /// * `cid` - Content ID
    /// * `shard_index` - Shard index (0-13)
    ///
    /// # Returns
    /// Shard bytes if successful
    pub async fn fetch_shard(
        &self,
        super_node_addr: &str,
        cid: &str,
        shard_index: usize,
    ) -> crate::error::Result<Vec<u8>> {
        let socket_addr: SocketAddr = super_node_addr.parse().map_err(|e| {
            crate::error::RelayError::Upstream(format!(
                "Invalid address {}: {}",
                super_node_addr, e
            ))
        })?;

        debug!(
            "Fetching shard from {}: CID={}, index={}",
            super_node_addr, cid, shard_index
        );

        // Connect to Super-Node
        let connection = self
            .endpoint
            .connect(socket_addr, "icn")
            .map_err(|e| crate::error::RelayError::Upstream(format!("Connect error: {}", e)))?
            .await
            .map_err(|e| crate::error::RelayError::Upstream(format!("Connection failed: {}", e)))?;

        // Open bidirectional stream
        let (mut send, mut recv) = connection
            .open_bi()
            .await
            .map_err(|e| crate::error::RelayError::Upstream(format!("Stream open error: {}", e)))?;

        // Send request
        let request = format!("GET /shards/{}/shard_{:02}.bin", cid, shard_index);
        send.write_all(request.as_bytes())
            .await
            .map_err(|e| crate::error::RelayError::Upstream(format!("Write error: {}", e)))?;
        send.finish()
            .map_err(|e| crate::error::RelayError::Upstream(format!("Finish error: {}", e)))?;

        // Read response
        let data = recv
            .read_to_end(10 * 1024 * 1024)
            .await
            .map_err(|e| crate::error::RelayError::Upstream(format!("Read error: {}", e)))?;

        // Check for error response
        if data.starts_with(b"ERROR") {
            return Err(crate::error::RelayError::ShardNotFound(
                cid.to_string(),
                shard_index,
            ));
        }

        debug!(
            "Fetched shard: CID={}, index={}, size={} bytes",
            cid,
            shard_index,
            data.len()
        );

        Ok(data)
    }
}

/// Skip server certificate verification (for self-signed certs)
/// ONLY available when 'dev-mode' feature is enabled
/// WARNING: This is INSECURE and should NEVER be used in production
#[cfg(feature = "dev-mode")]
#[derive(Debug)]
struct SkipServerVerification(Arc<rustls::crypto::CryptoProvider>);

#[cfg(feature = "dev-mode")]
impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self(Arc::new(rustls::crypto::ring::default_provider())))
    }
}

#[cfg(feature = "dev-mode")]
impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "dev-mode")]
    #[tokio::test]
    async fn test_upstream_client_creation_dev_mode() {
        // Install default crypto provider for test
        let _ = rustls::crypto::ring::default_provider().install_default();

        let client = UpstreamClient::new(true);
        assert!(client.is_ok(), "Should create upstream client in dev mode");
    }

    #[cfg(not(feature = "dev-mode"))]
    #[tokio::test]
    async fn test_upstream_client_dev_mode_disabled_without_feature() {
        // Install default crypto provider for test
        let _ = rustls::crypto::ring::default_provider().install_default();

        let client = UpstreamClient::new(true);
        assert!(
            client.is_err(),
            "Should fail to create dev mode client without dev-mode feature"
        );
    }

    #[tokio::test]
    async fn test_upstream_client_creation_production_mode() {
        // Install default crypto provider for test
        let _ = rustls::crypto::ring::default_provider().install_default();

        let client = UpstreamClient::new(false);
        assert!(
            client.is_ok(),
            "Should create upstream client in production mode with WebPKI"
        );
    }

    // TODO: Integration tests with mock Super-Node QUIC server
}
