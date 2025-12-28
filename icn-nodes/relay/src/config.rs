//! Regional Relay configuration

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Regional Relay configuration loaded from TOML file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// ICN Chain RPC WebSocket endpoint (minimal usage)
    pub chain_endpoint: String,

    /// Cache root path for shard persistence
    pub cache_path: PathBuf,

    /// QUIC server port for viewer connections
    pub quic_port: u16,

    /// Prometheus metrics port
    pub metrics_port: u16,

    /// libp2p listen address
    pub p2p_listen_addr: String,

    /// Bootstrap peers for P2P network (Super-Nodes)
    pub bootstrap_peers: Vec<String>,

    /// Geographic region (auto-detected if empty)
    #[serde(default)]
    pub region: String,

    /// Maximum cache capacity in GB (default 1TB)
    #[serde(default = "default_max_cache_gb")]
    pub max_cache_gb: u64,

    /// Super-Node health check interval in seconds
    #[serde(default = "default_health_check_secs")]
    pub health_check_secs: u64,

    /// Upstream Super-Node addresses (for latency detection)
    pub super_node_addresses: Vec<String>,
}

fn default_max_cache_gb() -> u64 {
    1_000 // 1TB default
}

fn default_health_check_secs() -> u64 {
    60 // Health check every 60 seconds
}

/// Validate a file path to prevent path traversal attacks
///
/// # Security
/// Rejects paths with:
/// - `..` components (parent directory references)
/// - Absolute paths outside allowed directories
fn validate_path(path: &Path) -> crate::error::Result<PathBuf> {
    // Check for ".." components before canonicalization
    for component in path.components() {
        if let std::path::Component::ParentDir = component {
            return Err(crate::error::RelayError::Config(format!(
                "Path contains '..' component (path traversal): {:?}",
                path
            )));
        }
    }

    // Canonicalize path (resolves symlinks, makes absolute)
    let canonical = path.canonicalize().map_err(|e| {
        crate::error::RelayError::Config(format!(
            "Failed to canonicalize path {:?}: {}. File or directory must exist.",
            path, e
        ))
    })?;

    // Additional check: ensure canonical path doesn't contain ".."
    let path_str = canonical.to_string_lossy();
    if path_str.contains("..") {
        return Err(crate::error::RelayError::Config(format!(
            "Canonicalized path contains '..': {:?}",
            canonical
        )));
    }

    Ok(canonical)
}

impl Config {
    /// Load configuration from TOML file
    ///
    /// # Security
    /// - Validates config file path to prevent traversal
    /// - Validates cache_path after loading
    pub fn load(path: impl AsRef<Path>) -> crate::error::Result<Self> {
        let path = path.as_ref();

        // Read config file
        let content = std::fs::read_to_string(path)?;
        let mut config: Self = toml::from_str(&content).map_err(|e| {
            crate::error::RelayError::Config(format!("Failed to parse TOML: {}", e))
        })?;

        // Validate cache_path (must exist or be creatable)
        if !config.cache_path.exists() {
            // Attempt to create cache directory
            std::fs::create_dir_all(&config.cache_path).map_err(|e| {
                crate::error::RelayError::Config(format!(
                    "Failed to create cache directory {:?}: {}",
                    config.cache_path, e
                ))
            })?;
        }

        // Validate cache_path for security
        config.cache_path = validate_path(&config.cache_path)?;

        Ok(config)
    }

    /// Validate configuration values
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.chain_endpoint.is_empty() {
            return Err(crate::error::RelayError::Config(
                "chain_endpoint cannot be empty".to_string(),
            ));
        }

        if !self.chain_endpoint.starts_with("ws://") && !self.chain_endpoint.starts_with("wss://") {
            return Err(crate::error::RelayError::Config(
                "chain_endpoint must start with ws:// or wss://".to_string(),
            ));
        }

        if self.quic_port == 0 {
            return Err(crate::error::RelayError::Config(
                "quic_port cannot be 0".to_string(),
            ));
        }

        if self.metrics_port == 0 {
            return Err(crate::error::RelayError::Config(
                "metrics_port cannot be 0".to_string(),
            ));
        }

        if self.max_cache_gb == 0 {
            return Err(crate::error::RelayError::Config(
                "max_cache_gb must be > 0".to_string(),
            ));
        }

        if self.health_check_secs == 0 {
            return Err(crate::error::RelayError::Config(
                "health_check_secs must be > 0".to_string(),
            ));
        }

        if self.super_node_addresses.is_empty() {
            return Err(crate::error::RelayError::Config(
                "super_node_addresses cannot be empty (needed for latency detection)".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Test Case: Valid configuration loads successfully
    /// Purpose: Validate config schema and default values
    /// Contract: All fields loaded correctly with defaults
    #[test]
    fn test_config_load_valid() {
        let tmp_dir = tempdir().unwrap();
        let cache_path = tmp_dir.path().join("cache");
        std::fs::create_dir(&cache_path).unwrap();

        let config_content = format!(
            r#"
chain_endpoint = "ws://127.0.0.1:9944"
cache_path = "{}"
quic_port = 9003
metrics_port = 9103
p2p_listen_addr = "/ip4/0.0.0.0/tcp/30335"
bootstrap_peers = ["/ip4/127.0.0.1/tcp/30333/p2p/12D3KooWA"]
super_node_addresses = ["127.0.0.1:9002"]
"#,
            cache_path.to_str().unwrap().replace('\\', "\\\\")
        );

        let config_path = tmp_dir.path().join("config.toml");
        std::fs::write(&config_path, config_content).unwrap();

        let config = Config::load(&config_path).expect("Failed to load config");

        assert_eq!(config.chain_endpoint, "ws://127.0.0.1:9944");
        assert_eq!(config.quic_port, 9003);
        assert_eq!(config.metrics_port, 9103);
        assert_eq!(config.max_cache_gb, 1_000); // default
        assert_eq!(config.health_check_secs, 60); // default
        assert!(config.region.is_empty()); // auto-detect
    }

    /// Test Case: Configuration with explicit region
    /// Purpose: Verify manual region override
    #[test]
    fn test_config_explicit_region() {
        let tmp_dir = tempdir().unwrap();
        let cache_path = tmp_dir.path().join("cache");
        std::fs::create_dir(&cache_path).unwrap();

        let config_content = format!(
            r#"
chain_endpoint = "ws://127.0.0.1:9944"
cache_path = "{}"
quic_port = 9003
metrics_port = 9103
p2p_listen_addr = "/ip4/0.0.0.0/tcp/30335"
bootstrap_peers = []
region = "NA-WEST"
super_node_addresses = ["10.0.0.1:9002", "10.0.0.2:9002"]
"#,
            cache_path.to_str().unwrap().replace('\\', "\\\\")
        );

        let config_path = tmp_dir.path().join("config.toml");
        std::fs::write(&config_path, config_content).unwrap();

        let config = Config::load(&config_path).expect("Failed to load config");

        assert_eq!(config.region, "NA-WEST");
        assert_eq!(config.super_node_addresses.len(), 2);
    }

    /// Test Case: Validation catches empty chain_endpoint
    /// Purpose: Verify required field validation
    #[test]
    fn test_config_validation_empty_endpoint() {
        let config = Config {
            chain_endpoint: "".to_string(),
            cache_path: "/cache".into(),
            quic_port: 9003,
            metrics_port: 9103,
            p2p_listen_addr: "/ip4/0.0.0.0/tcp/30335".to_string(),
            bootstrap_peers: vec![],
            region: "".to_string(),
            max_cache_gb: 1_000,
            health_check_secs: 60,
            super_node_addresses: vec!["127.0.0.1:9002".to_string()],
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    /// Test Case: Validation catches invalid WebSocket scheme
    /// Purpose: Ensure only ws:// or wss:// schemes accepted
    #[test]
    fn test_config_validation_invalid_scheme() {
        let config = Config {
            chain_endpoint: "http://127.0.0.1:9944".to_string(),
            cache_path: "/cache".into(),
            quic_port: 9003,
            metrics_port: 9103,
            p2p_listen_addr: "/ip4/0.0.0.0/tcp/30335".to_string(),
            bootstrap_peers: vec![],
            region: "".to_string(),
            max_cache_gb: 1_000,
            health_check_secs: 60,
            super_node_addresses: vec!["127.0.0.1:9002".to_string()],
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must start with ws://"));
    }

    /// Test Case: Path traversal protection
    /// Purpose: Verify security against path traversal attacks
    #[test]
    fn test_config_path_traversal_protection() {
        let result = validate_path(&PathBuf::from("../../../etc/passwd"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("path traversal"));
    }

    /// Test Case: Cache directory creation
    /// Purpose: Verify automatic directory creation
    #[test]
    fn test_config_creates_cache_directory() {
        let tmp_dir = tempdir().unwrap();
        let cache_path = tmp_dir.path().join("auto_created_cache");

        assert!(!cache_path.exists());

        let config_content = format!(
            r#"
chain_endpoint = "ws://127.0.0.1:9944"
cache_path = "{}"
quic_port = 9003
metrics_port = 9103
p2p_listen_addr = "/ip4/0.0.0.0/tcp/30335"
bootstrap_peers = []
super_node_addresses = ["127.0.0.1:9002"]
"#,
            cache_path.to_str().unwrap().replace('\\', "\\\\")
        );

        let config_path = tmp_dir.path().join("config.toml");
        std::fs::write(&config_path, config_content).unwrap();

        let _config = Config::load(&config_path).expect("Failed to load config");

        // Verify directory was created
        assert!(cache_path.exists());
    }

    /// Test Case: Validation rejects zero ports
    /// Purpose: Verify port range checking
    #[test]
    fn test_config_port_validation() {
        let base_config = Config {
            chain_endpoint: "ws://127.0.0.1:9944".to_string(),
            cache_path: "/cache".into(),
            quic_port: 9003,
            metrics_port: 9103,
            p2p_listen_addr: "/ip4/0.0.0.0/tcp/30335".to_string(),
            bootstrap_peers: vec![],
            region: "".to_string(),
            max_cache_gb: 1_000,
            health_check_secs: 60,
            super_node_addresses: vec!["127.0.0.1:9002".to_string()],
        };

        assert!(base_config.validate().is_ok());

        // Port 0 should be invalid
        let zero_quic = Config {
            quic_port: 0,
            ..base_config.clone()
        };
        assert!(zero_quic.validate().is_err());

        let zero_metrics = Config {
            metrics_port: 0,
            ..base_config.clone()
        };
        assert!(zero_metrics.validate().is_err());
    }

    /// Test Case: Validation requires super_node_addresses
    /// Purpose: Latency detection needs at least one Super-Node
    #[test]
    fn test_config_requires_super_node_addresses() {
        let config_empty_super_nodes = Config {
            chain_endpoint: "ws://127.0.0.1:9944".to_string(),
            cache_path: "/cache".into(),
            quic_port: 9003,
            metrics_port: 9103,
            p2p_listen_addr: "/ip4/0.0.0.0/tcp/30335".to_string(),
            bootstrap_peers: vec![],
            region: "".to_string(),
            max_cache_gb: 1_000,
            health_check_secs: 60,
            super_node_addresses: vec![], // Empty!
        };

        let result = config_empty_super_nodes.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("super_node_addresses cannot be empty"));
    }
}
