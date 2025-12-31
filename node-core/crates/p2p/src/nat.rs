//! NAT traversal orchestration
//!
//! Implements priority-based NAT traversal strategy:
//! Direct → STUN → UPnP → Circuit Relay → TURN
//!
//! Each strategy has a 10-second timeout before falling back to the next method.

use libp2p::{Multiaddr, PeerId};
use std::time::Duration;
use thiserror::Error;

/// NAT traversal timeout per strategy (10 seconds)
pub const STRATEGY_TIMEOUT: Duration = Duration::from_secs(10);

/// Maximum retry attempts per strategy
pub const MAX_RETRY_ATTEMPTS: u32 = 3;

/// Initial retry delay (2 seconds)
pub const INITIAL_RETRY_DELAY: Duration = Duration::from_secs(2);

/// Connection strategy for NAT traversal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionStrategy {
    /// Direct TCP/QUIC connection (no NAT or port forwarding configured)
    Direct,
    /// STUN-based UDP hole punching
    STUN,
    /// UPnP automatic port mapping
    UPnP,
    /// libp2p Circuit Relay (via relay node)
    CircuitRelay,
    /// TURN relay (ultimate fallback)
    TURN,
}

impl ConnectionStrategy {
    /// Convert strategy to string for metrics
    pub fn as_str(&self) -> &'static str {
        match self {
            ConnectionStrategy::Direct => "direct",
            ConnectionStrategy::STUN => "stun",
            ConnectionStrategy::UPnP => "upnp",
            ConnectionStrategy::CircuitRelay => "circuit_relay",
            ConnectionStrategy::TURN => "turn",
        }
    }

    /// Get all strategies in priority order
    pub fn all_in_order() -> Vec<Self> {
        vec![
            ConnectionStrategy::Direct,
            ConnectionStrategy::STUN,
            ConnectionStrategy::UPnP,
            ConnectionStrategy::CircuitRelay,
            ConnectionStrategy::TURN,
        ]
    }
}

/// NAT traversal errors
#[derive(Debug, Error)]
pub enum NATError {
    #[error("All connection strategies failed")]
    AllStrategiesFailed,

    #[error("Strategy timeout after {0:?}")]
    Timeout(Duration),

    #[error("Failed to dial peer: {0}")]
    DialFailed(String),

    #[error("STUN discovery failed: {0}")]
    StunFailed(String),

    #[error("UPnP port mapping failed: {0}")]
    UPnPFailed(String),

    #[error("No circuit relay nodes available")]
    NoRelaysAvailable,

    #[error("Invalid multiaddr format")]
    InvalidMultiaddr,

    #[error("No TURN servers configured")]
    NoTurnServers,

    #[error("TURN relay not implemented yet")]
    TurnNotImplemented,

    #[error("Invalid STUN server address: {0}")]
    InvalidStunServer(String),

    #[error("Network I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// NAT traversal result
pub type Result<T> = std::result::Result<T, NATError>;

/// Configuration for NAT traversal stack
#[derive(Debug, Clone)]
pub struct NATConfig {
    /// STUN servers for external IP discovery
    pub stun_servers: Vec<String>,

    /// Enable UPnP port mapping
    pub enable_upnp: bool,

    /// Enable circuit relay
    pub enable_relay: bool,

    /// Enable TURN relay (future)
    pub enable_turn: bool,

    /// Circuit relay reward per hour (NSN tokens)
    pub relay_reward_per_hour: f64,

    /// Maximum number of relay circuits
    pub max_relay_circuits: usize,
}

impl Default for NATConfig {
    fn default() -> Self {
        Self {
            stun_servers: vec![
                "stun.l.google.com:19302".to_string(),
                "stun1.l.google.com:19302".to_string(),
                "stun2.l.google.com:19302".to_string(),
            ],
            enable_upnp: true,
            enable_relay: true,
            enable_turn: false,
            relay_reward_per_hour: 0.01, // 0.01 NSN/hour
            max_relay_circuits: 16,
        }
    }
}

/// NAT status detected by AutoNat
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NATStatus {
    /// No NAT detected (public IP)
    Public,
    /// Full-cone NAT (STUN works well)
    FullCone,
    /// Symmetric NAT (STUN may fail)
    Symmetric,
    /// Unknown (AutoNat not run yet)
    Unknown,
}

/// NAT traversal orchestrator
///
/// Attempts connection strategies in priority order with automatic retry logic:
/// Direct → STUN → UPnP → Circuit Relay → TURN
pub struct NATTraversalStack {
    /// Enabled connection strategies
    strategies: Vec<ConnectionStrategy>,

    /// NAT configuration
    config: NATConfig,
}

impl NATTraversalStack {
    /// Create new NAT traversal stack with default configuration
    pub fn new() -> Self {
        Self {
            strategies: ConnectionStrategy::all_in_order(),
            config: NATConfig::default(),
        }
    }

    /// Create NAT traversal stack with custom configuration
    pub fn with_config(config: NATConfig) -> Self {
        let mut strategies = vec![ConnectionStrategy::Direct];

        // Add STUN if servers configured
        if !config.stun_servers.is_empty() {
            strategies.push(ConnectionStrategy::STUN);
        }

        // Add UPnP if enabled
        if config.enable_upnp {
            strategies.push(ConnectionStrategy::UPnP);
        }

        // Add circuit relay if enabled
        if config.enable_relay {
            strategies.push(ConnectionStrategy::CircuitRelay);
        }

        // Add TURN if enabled (future)
        if config.enable_turn {
            strategies.push(ConnectionStrategy::TURN);
        }

        Self { strategies, config }
    }

    /// Establish connection using NAT traversal strategies
    ///
    /// Tries each strategy in priority order until one succeeds or all fail.
    ///
    /// # Arguments
    /// * `target` - Target peer ID to connect to
    /// * `target_addr` - Known multiaddress of target (may not be reachable directly)
    ///
    /// # Returns
    /// Success result with strategy used, or error if all strategies fail
    pub async fn establish_connection(
        &self,
        target: &PeerId,
        target_addr: &Multiaddr,
    ) -> Result<ConnectionStrategy> {
        tracing::info!("Attempting NAT traversal to peer {}", target);

        for strategy in &self.strategies {
            tracing::debug!("Trying strategy: {:?}", strategy);

            match self
                .try_strategy_with_retry(strategy, target, target_addr)
                .await
            {
                Ok(()) => {
                    tracing::info!("Connected via {:?}", strategy);
                    return Ok(*strategy);
                }
                Err(e) => {
                    tracing::warn!("Strategy {:?} failed: {}", strategy, e);
                    continue;
                }
            }
        }

        Err(NATError::AllStrategiesFailed)
    }

    /// Try a strategy with timeout and retry logic
    async fn try_strategy_with_retry(
        &self,
        strategy: &ConnectionStrategy,
        target: &PeerId,
        addr: &Multiaddr,
    ) -> Result<()> {
        let mut delay = INITIAL_RETRY_DELAY;

        for attempt in 1..=MAX_RETRY_ATTEMPTS {
            match self.try_strategy_with_timeout(strategy, target, addr).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if attempt < MAX_RETRY_ATTEMPTS {
                        tracing::debug!(
                            "Attempt {}/{} failed for {:?}: {}. Retrying in {:?}",
                            attempt,
                            MAX_RETRY_ATTEMPTS,
                            strategy,
                            e,
                            delay
                        );
                        tokio::time::sleep(delay).await;
                        delay *= 2; // Exponential backoff
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(NATError::AllStrategiesFailed)
    }

    /// Try a strategy with timeout wrapper
    async fn try_strategy_with_timeout(
        &self,
        strategy: &ConnectionStrategy,
        target: &PeerId,
        addr: &Multiaddr,
    ) -> Result<()> {
        tokio::time::timeout(STRATEGY_TIMEOUT, self.try_strategy(strategy, target, addr))
            .await
            .map_err(|_| NATError::Timeout(STRATEGY_TIMEOUT))?
    }

    /// Execute a specific connection strategy
    async fn try_strategy(
        &self,
        strategy: &ConnectionStrategy,
        target: &PeerId,
        addr: &Multiaddr,
    ) -> Result<()> {
        match strategy {
            ConnectionStrategy::Direct => self.dial_direct(target, addr).await,
            ConnectionStrategy::STUN => self.stun_hole_punch(target, addr).await,
            ConnectionStrategy::UPnP => self.upnp_port_map(target, addr).await,
            ConnectionStrategy::CircuitRelay => self.dial_via_circuit_relay(target).await,
            ConnectionStrategy::TURN => self.dial_via_turn(target, addr).await,
        }
    }

    /// Attempt direct connection (no NAT traversal)
    async fn dial_direct(&self, target: &PeerId, _addr: &Multiaddr) -> Result<()> {
        tracing::debug!("Attempting direct dial to {}", target);
        // NOTE: Actual dial logic would integrate with libp2p Swarm
        // For now, this is a placeholder that would be implemented when
        // integrating with the full P2P stack
        Err(NATError::DialFailed(
            "Direct dial not implemented (requires Swarm integration)".into(),
        ))
    }

    /// Use STUN for hole punching
    async fn stun_hole_punch(&self, target: &PeerId, _addr: &Multiaddr) -> Result<()> {
        tracing::debug!("Attempting STUN hole punching for {}", target);

        // Discover external address via STUN
        use crate::stun::discover_external_with_fallback;
        let external_addr = discover_external_with_fallback(&self.config.stun_servers)?;

        tracing::info!("STUN discovered external address: {}", external_addr);

        // NOTE: Full hole punching would require coordination with target peer
        // This is a simplified implementation
        Err(NATError::StunFailed(
            "STUN hole punching coordination not implemented (requires DHT)".into(),
        ))
    }

    /// Use UPnP for port mapping
    async fn upnp_port_map(&self, target: &PeerId, _addr: &Multiaddr) -> Result<()> {
        tracing::debug!("Attempting UPnP port mapping for {}", target);

        // Discover UPnP gateway and create port mapping
        use crate::upnp::setup_p2p_port_mapping;

        // Use port 9000 as default P2P port
        let (external_ip, tcp_port, udp_port) = setup_p2p_port_mapping(9000)?;

        tracing::info!(
            "UPnP mapping established: {} (TCP: {}, UDP: {})",
            external_ip,
            tcp_port,
            udp_port
        );

        // NOTE: Would advertise external address to DHT and retry direct dial
        // This is a simplified implementation
        Err(NATError::UPnPFailed(
            "UPnP port mapping created but DHT advertisement not implemented".into(),
        ))
    }

    /// Use circuit relay as fallback
    async fn dial_via_circuit_relay(&self, target: &PeerId) -> Result<()> {
        tracing::debug!("Attempting circuit relay for {}", target);

        // NOTE: Would query DHT for relay nodes and use libp2p circuit relay
        // This requires integration with the full P2P stack
        Err(NATError::NoRelaysAvailable)
    }

    /// Use TURN relay as ultimate fallback
    async fn dial_via_turn(&self, target: &PeerId, _addr: &Multiaddr) -> Result<()> {
        tracing::warn!("TURN relay requested for {} (degraded performance)", target);

        // TURN is intentionally not implemented in MVP
        // Will be added in Phase 2 when needed for severely restricted networks
        Err(NATError::TurnNotImplemented)
    }
}

impl Default for NATTraversalStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_strategy_ordering() {
        let strategies = ConnectionStrategy::all_in_order();

        assert_eq!(strategies.len(), 5);
        assert_eq!(strategies[0], ConnectionStrategy::Direct);
        assert_eq!(strategies[1], ConnectionStrategy::STUN);
        assert_eq!(strategies[2], ConnectionStrategy::UPnP);
        assert_eq!(strategies[3], ConnectionStrategy::CircuitRelay);
        assert_eq!(strategies[4], ConnectionStrategy::TURN);
    }

    #[test]
    fn test_connection_strategy_as_str() {
        assert_eq!(ConnectionStrategy::Direct.as_str(), "direct");
        assert_eq!(ConnectionStrategy::STUN.as_str(), "stun");
        assert_eq!(ConnectionStrategy::UPnP.as_str(), "upnp");
        assert_eq!(ConnectionStrategy::CircuitRelay.as_str(), "circuit_relay");
        assert_eq!(ConnectionStrategy::TURN.as_str(), "turn");
    }

    #[test]
    fn test_nat_config_defaults() {
        let config = NATConfig::default();

        assert_eq!(config.stun_servers.len(), 3);
        assert!(config.enable_upnp);
        assert!(config.enable_relay);
        assert!(!config.enable_turn);
        assert_eq!(config.relay_reward_per_hour, 0.01);
        assert_eq!(config.max_relay_circuits, 16);
    }

    #[test]
    fn test_strategy_timeout_constant() {
        assert_eq!(STRATEGY_TIMEOUT, Duration::from_secs(10));
    }

    #[test]
    fn test_nat_traversal_stack_creation() {
        let stack = NATTraversalStack::new();
        assert_eq!(stack.strategies.len(), 5);
    }

    #[test]
    fn test_nat_traversal_stack_with_config() {
        let config = NATConfig {
            enable_upnp: false,
            enable_relay: false,
            ..Default::default()
        };

        let stack = NATTraversalStack::with_config(config);
        // Should only have Direct and STUN (STUN servers are present by default)
        assert_eq!(stack.strategies.len(), 2);
        assert_eq!(stack.strategies[0], ConnectionStrategy::Direct);
        assert_eq!(stack.strategies[1], ConnectionStrategy::STUN);
    }

    #[test]
    fn test_retry_constants() {
        assert_eq!(MAX_RETRY_ATTEMPTS, 3);
        assert_eq!(INITIAL_RETRY_DELAY, Duration::from_secs(2));
    }

    #[tokio::test]
    async fn test_nat_traversal_all_strategies_fail() {
        let stack = NATTraversalStack::new();
        let target = PeerId::random();
        let addr = "/ip4/127.0.0.1/tcp/9000".parse().unwrap();

        let result = stack.establish_connection(&target, &addr).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), NATError::AllStrategiesFailed));
    }
}
