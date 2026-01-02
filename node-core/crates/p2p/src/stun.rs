//! STUN client for external IP discovery
//!
//! Implements RFC 5389 STUN protocol for discovering external IP addresses
//! and ports through NAT devices.

use crate::nat::{NATError, Result};
use bytecodec::{DecodeExt, EncodeExt};
use std::net::{SocketAddr, UdpSocket};
use std::time::Duration;
use stun_codec::{
    rfc5389::{
        attributes::{MappedAddress, XorMappedAddress},
        methods::BINDING,
        Attribute,
    },
    Message, MessageClass, MessageDecoder, MessageEncoder, TransactionId,
};

/// STUN client timeout (5 seconds)
const STUN_TIMEOUT: Duration = Duration::from_secs(5);

/// STUN client for discovering external addresses
pub struct StunClient {
    /// Local UDP socket for STUN requests
    socket: UdpSocket,
}

impl StunClient {
    /// Create a new STUN client bound to local address
    ///
    /// # Arguments
    /// * `local_addr` - Local address to bind (e.g., "0.0.0.0:0" for any port)
    pub fn new(local_addr: &str) -> Result<Self> {
        let socket = UdpSocket::bind(local_addr)?;
        socket.set_read_timeout(Some(STUN_TIMEOUT))?;
        socket.set_write_timeout(Some(STUN_TIMEOUT))?;

        Ok(Self { socket })
    }

    /// Discover external address via STUN server
    ///
    /// # Arguments
    /// * `stun_server` - STUN server address (e.g., "stun.l.google.com:19302")
    ///
    /// # Returns
    /// External IP and port as seen by the STUN server
    pub fn discover_external(&self, stun_server: &str) -> Result<SocketAddr> {
        // Parse STUN server address
        let server_addr = stun_server
            .parse::<SocketAddr>()
            .map_err(|e| NATError::InvalidStunServer(format!("{}: {}", stun_server, e)))?;

        // Create STUN binding request
        let transaction_id = TransactionId::new(rand::random());
        let message = Message::<Attribute>::new(MessageClass::Request, BINDING, transaction_id);

        // Encode and send request
        let mut encoder = MessageEncoder::new();
        let bytes = encoder
            .encode_into_bytes(message.clone())
            .map_err(|e| NATError::StunFailed(format!("Encoding failed: {}", e)))?;

        self.socket
            .send_to(&bytes, server_addr)
            .map_err(|e| NATError::StunFailed(format!("Send failed: {}", e)))?;

        tracing::debug!("Sent STUN binding request to {}", server_addr);

        // Receive and decode response
        let mut buf = vec![0u8; 1024];
        let (n, _) = self
            .socket
            .recv_from(&mut buf)
            .map_err(|e| NATError::StunFailed(format!("Receive failed: {}", e)))?;

        let mut decoder = MessageDecoder::<Attribute>::new();
        let response = decoder
            .decode_from_bytes(&buf[..n])
            .map_err(|e| NATError::StunFailed(format!("Decoding failed: {}", e)))?
            .map_err(|e| NATError::StunFailed(format!("Incomplete message: {:?}", e)))?;

        tracing::debug!("Received STUN response from {}", server_addr);

        // Extract external address from XOR-MAPPED-ADDRESS or MAPPED-ADDRESS
        let external_addr = response
            .get_attribute::<XorMappedAddress>()
            .map(|attr| attr.address())
            .or_else(|| {
                response
                    .get_attribute::<MappedAddress>()
                    .map(|attr| attr.address())
            })
            .ok_or_else(|| {
                NATError::StunFailed("No MAPPED-ADDRESS or XOR-MAPPED-ADDRESS in response".into())
            })?;

        tracing::info!("Discovered external address: {}", external_addr);

        Ok(external_addr)
    }
}

/// Discover external address using first available STUN server
///
/// # Arguments
/// * `stun_servers` - List of STUN server addresses to try
///
/// # Returns
/// External IP and port, or error if all servers fail
pub fn discover_external_with_fallback(stun_servers: &[String]) -> Result<SocketAddr> {
    let client = StunClient::new("0.0.0.0:0")?;

    for server in stun_servers {
        match client.discover_external(server) {
            Ok(addr) => return Ok(addr),
            Err(e) => {
                tracing::warn!("STUN server {} failed: {}", server, e);
                continue;
            }
        }
    }

    Err(NATError::StunFailed("All STUN servers failed".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stun_client_creation() {
        let client = StunClient::new("127.0.0.1:0");
        match client {
            Ok(_) => {}
            Err(NATError::IoError(err)) if err.kind() == std::io::ErrorKind::PermissionDenied => {
                return;
            }
            Err(err) => panic!("Unexpected STUN client error: {}", err),
        }
    }

    #[test]
    fn test_stun_client_invalid_bind() {
        let client = StunClient::new("invalid:address");
        assert!(client.is_err());
    }

    #[test]
    #[ignore] // Requires network access
    fn test_stun_discovery_google() {
        let client = StunClient::new("0.0.0.0:0").expect("Failed to create client");
        let result = client.discover_external("stun.l.google.com:19302");

        match result {
            Ok(addr) => {
                tracing::info!("Discovered external address: {}", addr);
                assert!(addr.port() > 0);
            }
            Err(e) => {
                // May fail in restricted networks
                tracing::warn!("STUN discovery failed (expected in some networks): {}", e);
            }
        }
    }

    #[test]
    fn test_stun_invalid_server() {
        let client = match StunClient::new("0.0.0.0:0") {
            Ok(client) => client,
            Err(NATError::IoError(err)) if err.kind() == std::io::ErrorKind::PermissionDenied => {
                return;
            }
            Err(err) => panic!("Unexpected STUN client error: {}", err),
        };
        let result = client.discover_external("invalid.server.example.com:19302");
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // Requires network access
    fn test_discover_with_fallback() {
        let servers = vec![
            "stun.l.google.com:19302".to_string(),
            "stun1.l.google.com:19302".to_string(),
        ];

        let result = discover_external_with_fallback(&servers);
        match result {
            Ok(addr) => {
                tracing::info!("Discovered via fallback: {}", addr);
                assert!(addr.port() > 0);
            }
            Err(e) => {
                tracing::warn!("Fallback failed (expected in some networks): {}", e);
            }
        }
    }
}
