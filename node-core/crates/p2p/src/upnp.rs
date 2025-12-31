//! UPnP/IGD port mapping
//!
//! Implements automatic port forwarding using UPnP Internet Gateway Device (IGD) protocol.
//! Allows nodes behind NAT routers to automatically configure port forwarding.

use crate::nat::{NATError, Result};
use igd_next::{Gateway, PortMappingProtocol, SearchOptions};
use std::net::{IpAddr, Ipv4Addr, SocketAddrV4};
use std::time::Duration;

/// UPnP gateway discovery timeout (5 seconds)
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(5);

/// Default lease duration for port mappings (0 = infinite)
const DEFAULT_LEASE_DURATION: u32 = 0;

/// UPnP port mapper for automatic NAT traversal
pub struct UpnpMapper {
    /// Discovered IGD gateway
    gateway: Gateway,
}

impl UpnpMapper {
    /// Discover UPnP gateway on the network
    ///
    /// # Returns
    /// UPnP mapper instance if gateway found, error otherwise
    pub fn discover() -> Result<Self> {
        tracing::debug!("Searching for UPnP gateway...");

        let search_options = SearchOptions {
            timeout: Some(DISCOVERY_TIMEOUT),
            ..Default::default()
        };

        let gateway = igd_next::search_gateway(search_options)
            .map_err(|e| NATError::UPnPFailed(format!("Gateway discovery failed: {}", e)))?;

        tracing::info!("Discovered UPnP gateway");

        Ok(Self { gateway })
    }

    /// Get external IP address from gateway
    pub fn external_ip(&self) -> Result<Ipv4Addr> {
        let ip = self
            .gateway
            .get_external_ip()
            .map_err(|e| NATError::UPnPFailed(format!("Failed to get external IP: {}", e)))?;

        match ip {
            IpAddr::V4(ipv4) => Ok(ipv4),
            IpAddr::V6(_) => Err(NATError::UPnPFailed(
                "Gateway returned IPv6 address, but IPv4 expected".into(),
            )),
        }
    }

    /// Add port mapping for a local port
    ///
    /// # Arguments
    /// * `protocol` - Protocol (TCP or UDP)
    /// * `local_port` - Local port to map
    /// * `description` - Human-readable description for the mapping
    ///
    /// # Returns
    /// External port number
    pub fn add_port_mapping(
        &self,
        protocol: PortMappingProtocol,
        local_port: u16,
        description: &str,
    ) -> Result<u16> {
        let local_addr = SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, local_port);

        tracing::debug!(
            "Adding UPnP port mapping: {}:{} -> {}",
            protocol_name(protocol),
            local_port,
            description
        );

        self.gateway
            .add_port(
                protocol,
                local_port,
                std::net::SocketAddr::V4(local_addr),
                DEFAULT_LEASE_DURATION,
                description,
            )
            .map_err(|e| NATError::UPnPFailed(format!("Failed to add port mapping: {}", e)))?;

        tracing::info!(
            "UPnP port mapping added: {}:{} ({})",
            protocol_name(protocol),
            local_port,
            description
        );

        Ok(local_port)
    }

    /// Remove port mapping
    ///
    /// # Arguments
    /// * `protocol` - Protocol (TCP or UDP)
    /// * `external_port` - External port to remove
    pub fn remove_port_mapping(
        &self,
        protocol: PortMappingProtocol,
        external_port: u16,
    ) -> Result<()> {
        tracing::debug!(
            "Removing UPnP port mapping: {}:{}",
            protocol_name(protocol),
            external_port
        );

        self.gateway
            .remove_port(protocol, external_port)
            .map_err(|e| NATError::UPnPFailed(format!("Failed to remove port mapping: {}", e)))?;

        tracing::info!(
            "UPnP port mapping removed: {}:{}",
            protocol_name(protocol),
            external_port
        );

        Ok(())
    }

    /// Add port mapping for both TCP and UDP
    ///
    /// # Arguments
    /// * `local_port` - Local port to map
    /// * `description` - Human-readable description
    ///
    /// # Returns
    /// (TCP external port, UDP external port)
    pub fn add_port_mapping_both(&self, local_port: u16, description: &str) -> Result<(u16, u16)> {
        let tcp_port = self.add_port_mapping(
            PortMappingProtocol::TCP,
            local_port,
            &format!("{} (TCP)", description),
        )?;
        let udp_port = self.add_port_mapping(
            PortMappingProtocol::UDP,
            local_port,
            &format!("{} (UDP)", description),
        )?;

        Ok((tcp_port, udp_port))
    }
}

/// Helper to convert protocol enum to string
fn protocol_name(protocol: PortMappingProtocol) -> &'static str {
    match protocol {
        PortMappingProtocol::TCP => "TCP",
        PortMappingProtocol::UDP => "UDP",
    }
}

/// Attempt to set up UPnP port mapping for P2P port
///
/// # Arguments
/// * `port` - Local P2P port to map
///
/// # Returns
/// External IP and port if successful, error otherwise
pub fn setup_p2p_port_mapping(port: u16) -> Result<(Ipv4Addr, u16, u16)> {
    let mapper = UpnpMapper::discover()?;
    let external_ip = mapper.external_ip()?;
    let (tcp_port, udp_port) = mapper.add_port_mapping_both(port, "NSN P2P")?;

    Ok((external_ip, tcp_port, udp_port))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_name() {
        assert_eq!(protocol_name(PortMappingProtocol::TCP), "TCP");
        assert_eq!(protocol_name(PortMappingProtocol::UDP), "UDP");
    }

    #[test]
    #[ignore] // Requires UPnP-capable router on network
    fn test_upnp_discovery() {
        match UpnpMapper::discover() {
            Ok(mapper) => {
                tracing::info!("UPnP gateway discovered successfully");

                match mapper.external_ip() {
                    Ok(ip) => tracing::info!("External IP: {}", ip),
                    Err(e) => tracing::warn!("Failed to get external IP: {}", e),
                }
            }
            Err(e) => {
                tracing::info!(
                    "UPnP discovery failed (expected without UPnP router): {}",
                    e
                );
            }
        }
    }

    #[test]
    #[ignore] // Requires UPnP-capable router
    fn test_port_mapping() {
        let mapper = match UpnpMapper::discover() {
            Ok(m) => m,
            Err(e) => {
                tracing::info!("Skipping test: {}", e);
                return;
            }
        };

        // Try to map a high port (less likely to conflict)
        let test_port = 19000;

        match mapper.add_port_mapping(PortMappingProtocol::UDP, test_port, "NSN Test") {
            Ok(port) => {
                tracing::info!("Port mapping added: {}", port);

                // Clean up
                let _ = mapper.remove_port_mapping(PortMappingProtocol::UDP, port);
            }
            Err(e) => {
                tracing::warn!("Port mapping failed: {}", e);
            }
        }
    }
}
