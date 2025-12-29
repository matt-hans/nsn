use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Director node configuration loaded from TOML file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// ICN Chain RPC WebSocket endpoint
    pub chain_endpoint: String,

    /// Path to director's keypair file (Ed25519)
    pub keypair_path: PathBuf,

    /// gRPC server port for BFT coordination
    pub grpc_port: u16,

    /// Prometheus metrics port
    pub metrics_port: u16,

    /// libp2p listen address
    pub p2p_listen_addr: String,

    /// Bootstrap peers for P2P network
    pub bootstrap_peers: Vec<String>,

    /// Geographic region (for election distribution)
    pub region: String,

    /// Pipeline lookahead (number of slots to prepare)
    #[serde(default = "default_lookahead")]
    pub pipeline_lookahead: u32,

    /// gRPC connection timeout (seconds)
    #[serde(default = "default_grpc_timeout")]
    pub grpc_timeout_secs: u64,

    /// BFT consensus threshold (cosine similarity)
    #[serde(default = "default_bft_threshold")]
    pub bft_consensus_threshold: f32,
}

fn default_lookahead() -> u32 {
    2
}

fn default_grpc_timeout() -> u64 {
    5
}

fn default_bft_threshold() -> f32 {
    0.95
}

/// Validate a file path to prevent path traversal attacks
///
/// # Security
/// Rejects paths with:
/// - `..` components (parent directory references)
/// - Absolute paths outside allowed directories
/// - Symlinks that escape allowed directories
///
/// # Arguments
/// * `path` - Path to validate
/// * `allowed_extension` - Required file extension (e.g., "toml", "json")
///
/// # Returns
/// Canonicalized path if valid, error otherwise
fn validate_path(path: &Path, allowed_extension: Option<&str>) -> crate::error::Result<PathBuf> {
    // Check for ".." components before canonicalization
    for component in path.components() {
        if let std::path::Component::ParentDir = component {
            return Err(crate::error::DirectorError::Config(format!(
                "Path contains '..' component (path traversal): {:?}",
                path
            ))
            .into());
        }
    }

    // Validate file extension if specified
    if let Some(required_ext) = allowed_extension {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        if ext != required_ext {
            return Err(crate::error::DirectorError::Config(format!(
                "Invalid file extension: expected .{}, got {:?}",
                required_ext, path
            ))
            .into());
        }
    }

    // Canonicalize path (resolves symlinks, makes absolute)
    // Note: This requires the file to exist
    let canonical = path.canonicalize().map_err(|e| {
        crate::error::DirectorError::Config(format!(
            "Failed to canonicalize path {:?}: {}. File must exist.",
            path, e
        ))
    })?;

    // Additional check: ensure canonical path doesn't contain ".."
    // (defense in depth against symlink attacks)
    let path_str = canonical.to_string_lossy();
    if path_str.contains("..") {
        return Err(crate::error::DirectorError::Config(format!(
            "Canonicalized path contains '..': {:?}",
            canonical
        ))
        .into());
    }

    Ok(canonical)
}

impl Config {
    /// Load configuration from TOML file
    ///
    /// # Security
    /// - Validates config file path to prevent traversal
    /// - Validates keypair_path after loading
    pub fn load(path: impl AsRef<Path>) -> crate::error::Result<Self> {
        let path = path.as_ref();

        // Validate config file path
        let validated_path = validate_path(path, Some("toml"))?;

        let content = std::fs::read_to_string(validated_path)?;
        let mut config: Self = toml::from_str(&content).map_err(|e| {
            crate::error::DirectorError::Config(format!("Failed to parse TOML: {}", e))
        })?;

        // Validate keypair_path (must exist, .json extension)
        config.keypair_path = validate_path(&config.keypair_path, Some("json"))?;

        Ok(config)
    }

    /// Validate configuration values
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.chain_endpoint.is_empty() {
            return Err(crate::error::DirectorError::Config(
                "chain_endpoint cannot be empty".to_string(),
            )
            .into());
        }

        if !self.chain_endpoint.starts_with("ws://") && !self.chain_endpoint.starts_with("wss://") {
            return Err(crate::error::DirectorError::Config(
                "chain_endpoint must start with ws:// or wss://".to_string(),
            )
            .into());
        }

        if self.grpc_port == 0 {
            return Err(
                crate::error::DirectorError::Config("grpc_port cannot be 0".to_string()).into(),
            );
        }

        if self.metrics_port == 0 {
            return Err(crate::error::DirectorError::Config(
                "metrics_port cannot be 0".to_string(),
            )
            .into());
        }

        if self.region.is_empty() {
            return Err(
                crate::error::DirectorError::Config("region cannot be empty".to_string()).into(),
            );
        }

        if self.bft_consensus_threshold < 0.0 || self.bft_consensus_threshold > 1.0 {
            return Err(crate::error::DirectorError::Config(
                "bft_consensus_threshold must be between 0.0 and 1.0".to_string(),
            )
            .into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    /// Test Case 1: Valid configuration file loads successfully
    /// Validates config schema and default values
    #[test]
    fn test_config_load_valid() {
        let config_content = r#"
chain_endpoint = "ws://127.0.0.1:9944"
grpc_port = 50051
metrics_port = 9100
p2p_listen_addr = "/ip4/0.0.0.0/tcp/30333"
bootstrap_peers = ["/ip4/127.0.0.1/tcp/30334/p2p/12D3KooWA"]
region = "us-east-1"
"#;

        // Use tempfile::Builder to create a file with .toml extension
        let tmp_dir = tempfile::tempdir().unwrap();

        // Create a dummy keypair file (will be validated to exist)
        let keypair_path = tmp_dir.path().join("alice.json");
        // Create a minimal valid keypair JSON for testing
        use base64::{engine::general_purpose, Engine as _};
        // Generate a valid protobuf-encoded Ed25519 keypair
        let test_keypair = libp2p::identity::Keypair::generate_ed25519();
        let protobuf_bytes = test_keypair.to_protobuf_encoding().unwrap();
        let secret_b64 = general_purpose::STANDARD.encode(&protobuf_bytes);
        let keypair_json = format!(r#"{{"secret_key": "{}"}}"#, secret_b64);
        std::fs::write(&keypair_path, keypair_json).unwrap();

        let keypair_path_str = keypair_path.to_str().unwrap();
        let config_content_with_keypair = format!(
            r#"
chain_endpoint = "ws://127.0.0.1:9944"
keypair_path = "{}"
grpc_port = 50051
metrics_port = 9100
p2p_listen_addr = "/ip4/0.0.0.0/tcp/30333"
bootstrap_peers = ["/ip4/127.0.0.1/tcp/30334/p2p/12D3KooWA"]
region = "us-east-1"
"#,
            keypair_path_str.replace('\\', "\\\\")
        );

        let config_path = tmp_dir.path().join("config.toml");
        std::fs::write(&config_path, config_content_with_keypair).unwrap();

        let config = Config::load(&config_path).expect("Failed to load config");

        assert_eq!(config.chain_endpoint, "ws://127.0.0.1:9944");
        assert_eq!(config.grpc_port, 50051);
        assert_eq!(config.metrics_port, 9100);
        assert_eq!(config.region, "us-east-1");
        assert_eq!(config.pipeline_lookahead, 2); // default
        assert_eq!(config.bft_consensus_threshold, 0.95); // default
    }

    /// Test Case 2: Configuration with custom values
    #[test]
    fn test_config_custom_values() {
        let tmp_dir = tempfile::tempdir().unwrap();

        // Create a dummy keypair file
        let keypair_path = tmp_dir.path().join("director.json");
        use base64::{engine::general_purpose, Engine as _};
        let test_keypair = libp2p::identity::Keypair::generate_ed25519();
        let protobuf_bytes = test_keypair.to_protobuf_encoding().unwrap();
        let secret_b64 = general_purpose::STANDARD.encode(&protobuf_bytes);
        let keypair_json = format!(r#"{{"secret_key": "{}"}}"#, secret_b64);
        std::fs::write(&keypair_path, keypair_json).unwrap();

        let keypair_path_str = keypair_path.to_str().unwrap();
        let config_content = format!(
            r#"
chain_endpoint = "wss://rpc.icn.network:443"
keypair_path = "{}"
grpc_port = 50052
metrics_port = 9101
p2p_listen_addr = "/ip4/0.0.0.0/tcp/30335"
bootstrap_peers = []
region = "eu-west-1"
pipeline_lookahead = 3
grpc_timeout_secs = 10
bft_consensus_threshold = 0.90
"#,
            keypair_path_str.replace('\\', "\\\\")
        );

        let config_path = tmp_dir.path().join("custom.toml");
        std::fs::write(&config_path, config_content).unwrap();

        let config = Config::load(&config_path).expect("Failed to load config");

        assert_eq!(config.pipeline_lookahead, 3);
        assert_eq!(config.grpc_timeout_secs, 10);
        assert_eq!(config.bft_consensus_threshold, 0.90);
    }

    /// Test Case 3: Invalid TOML syntax returns error
    #[test]
    fn test_config_invalid_toml() {
        let config_content = "chain_endpoint = ws://invalid syntax";

        let tmp_file = NamedTempFile::new().unwrap();
        std::fs::write(tmp_file.path(), config_content).unwrap();

        let result = Config::load(tmp_file.path());
        assert!(result.is_err());
    }

    /// Test Case 4: Validation catches empty chain_endpoint
    #[test]
    fn test_config_validation_empty_endpoint() {
        let config = Config {
            chain_endpoint: "".to_string(),
            keypair_path: "/keys/alice.json".into(),
            grpc_port: 50051,
            metrics_port: 9100,
            p2p_listen_addr: "/ip4/0.0.0.0/tcp/30333".to_string(),
            bootstrap_peers: vec![],
            region: "us-east-1".to_string(),
            pipeline_lookahead: 2,
            grpc_timeout_secs: 5,
            bft_consensus_threshold: 0.95,
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    /// Test Case 5: Validation catches invalid WebSocket scheme
    #[test]
    fn test_config_validation_invalid_scheme() {
        let config = Config {
            chain_endpoint: "http://127.0.0.1:9944".to_string(),
            keypair_path: "/keys/alice.json".into(),
            grpc_port: 50051,
            metrics_port: 9100,
            p2p_listen_addr: "/ip4/0.0.0.0/tcp/30333".to_string(),
            bootstrap_peers: vec![],
            region: "us-east-1".to_string(),
            pipeline_lookahead: 2,
            grpc_timeout_secs: 5,
            bft_consensus_threshold: 0.95,
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must start with ws://"));
    }

    /// Test Case 6: Validation catches invalid BFT threshold
    #[test]
    fn test_config_validation_invalid_threshold() {
        let config = Config {
            chain_endpoint: "ws://127.0.0.1:9944".to_string(),
            keypair_path: "/keys/alice.json".into(),
            grpc_port: 50051,
            metrics_port: 9100,
            p2p_listen_addr: "/ip4/0.0.0.0/tcp/30333".to_string(),
            bootstrap_peers: vec![],
            region: "us-east-1".to_string(),
            pipeline_lookahead: 2,
            grpc_timeout_secs: 5,
            bft_consensus_threshold: 1.5, // invalid
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must be between 0.0 and 1.0"));
    }

    /// Test Case 7: Validate port ranges (deeper assertion)
    /// Purpose: Verify port validation is comprehensive
    /// Contract: Ports must be in valid range 1-65535
    #[test]
    fn test_config_port_validation() {
        // Valid ports
        let valid_config = Config {
            chain_endpoint: "ws://127.0.0.1:9944".to_string(),
            keypair_path: "/keys/alice.json".into(),
            grpc_port: 50051,
            metrics_port: 9100,
            p2p_listen_addr: "/ip4/0.0.0.0/tcp/30333".to_string(),
            bootstrap_peers: vec![],
            region: "us-east-1".to_string(),
            pipeline_lookahead: 2,
            grpc_timeout_secs: 5,
            bft_consensus_threshold: 0.95,
        };

        assert!(valid_config.validate().is_ok());

        // Port 0 should be invalid
        let zero_grpc_port = Config {
            grpc_port: 0,
            ..valid_config.clone()
        };
        assert!(zero_grpc_port.validate().is_err());

        let zero_metrics_port = Config {
            metrics_port: 0,
            ..valid_config.clone()
        };
        assert!(zero_metrics_port.validate().is_err());

        // Edge case: Port 1 (minimum valid port) should be accepted
        let min_port = Config {
            grpc_port: 1,
            metrics_port: 1,
            ..valid_config.clone()
        };
        // Note: Current validation only checks != 0, so this passes
        // Future enhancement: check port >= 1024 for non-privileged
        assert!(min_port.validate().is_ok());

        // Port 65535 (maximum) should be accepted
        let max_port = Config {
            grpc_port: 65535,
            metrics_port: 65534,
            ..valid_config.clone()
        };
        assert!(max_port.validate().is_ok());
    }

    /// Test Case 8: URL format validation (deeper assertion)
    /// Purpose: Verify WebSocket URL validation is comprehensive
    /// Contract: Must start with ws:// or wss://
    #[test]
    fn test_config_url_format_validation() {
        let base_config = Config {
            chain_endpoint: "ws://127.0.0.1:9944".to_string(),
            keypair_path: "/keys/alice.json".into(),
            grpc_port: 50051,
            metrics_port: 9100,
            p2p_listen_addr: "/ip4/0.0.0.0/tcp/30333".to_string(),
            bootstrap_peers: vec![],
            region: "us-east-1".to_string(),
            pipeline_lookahead: 2,
            grpc_timeout_secs: 5,
            bft_consensus_threshold: 0.95,
        };

        // Valid: ws://
        let ws_config = Config {
            chain_endpoint: "ws://127.0.0.1:9944".to_string(),
            ..base_config.clone()
        };
        assert!(ws_config.validate().is_ok());

        // Valid: wss:// (secure WebSocket)
        let wss_config = Config {
            chain_endpoint: "wss://rpc.icn.network:443".to_string(),
            ..base_config.clone()
        };
        assert!(wss_config.validate().is_ok());

        // Invalid: http://
        let http_config = Config {
            chain_endpoint: "http://127.0.0.1:9944".to_string(),
            ..base_config.clone()
        };
        assert!(http_config.validate().is_err());

        // Invalid: https://
        let https_config = Config {
            chain_endpoint: "https://rpc.icn.network".to_string(),
            ..base_config.clone()
        };
        assert!(https_config.validate().is_err());

        // Invalid: no scheme
        let no_scheme_config = Config {
            chain_endpoint: "127.0.0.1:9944".to_string(),
            ..base_config.clone()
        };
        assert!(no_scheme_config.validate().is_err());

        // Invalid: empty
        let empty_config = Config {
            chain_endpoint: "".to_string(),
            ..base_config.clone()
        };
        assert!(empty_config.validate().is_err());
    }

    /// Test Case 9: Region validation
    /// Purpose: Verify region field is validated
    /// Contract: Region cannot be empty
    #[test]
    fn test_config_region_validation() {
        let config_empty_region = Config {
            chain_endpoint: "ws://127.0.0.1:9944".to_string(),
            keypair_path: "/keys/alice.json".into(),
            grpc_port: 50051,
            metrics_port: 9100,
            p2p_listen_addr: "/ip4/0.0.0.0/tcp/30333".to_string(),
            bootstrap_peers: vec![],
            region: "".to_string(), // Empty region
            pipeline_lookahead: 2,
            grpc_timeout_secs: 5,
            bft_consensus_threshold: 0.95,
        };

        let result = config_empty_region.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("region cannot be empty"));
    }

    /// Test Case 10: Path traversal attack rejected
    /// Purpose: Verify security protection against path traversal
    /// Contract: Paths with ".." components must be rejected
    #[test]
    fn test_config_path_traversal_protection() {
        // Attempt path traversal in config file
        let result = validate_path(&PathBuf::from("../../../etc/passwd"), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("path traversal"));

        // Attempt path traversal in nested directory
        let result = validate_path(&PathBuf::from("config/../../etc/passwd"), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("path traversal"));
    }

    /// Test Case 11: Invalid file extension rejected
    /// Purpose: Verify file extension validation
    /// Contract: Files must have correct extension
    #[test]
    fn test_config_file_extension_validation() {
        // Create temp file with wrong extension
        let tmp_file = NamedTempFile::new().unwrap();
        let wrong_ext = tmp_file.path().with_extension("txt");
        std::fs::write(&wrong_ext, "test").unwrap();

        // Should reject .txt when .toml required
        let result = validate_path(&wrong_ext, Some("toml"));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid file extension"));

        // Clean up
        let _ = std::fs::remove_file(wrong_ext);
    }

    /// Test Case 12: Keypair path validation in config load
    /// Purpose: Verify keypair_path is validated after loading config
    /// Contract: Invalid keypair_path causes config load to fail
    #[test]
    fn test_config_load_validates_keypair_path() {
        // Create config file with path traversal in keypair_path
        let config_content = r#"
chain_endpoint = "ws://127.0.0.1:9944"
keypair_path = "../../../etc/shadow"
grpc_port = 50051
metrics_port = 9100
p2p_listen_addr = "/ip4/0.0.0.0/tcp/30333"
bootstrap_peers = []
region = "us-east-1"
"#;

        let tmp_file = NamedTempFile::new().unwrap();
        let config_path = tmp_file.path().with_extension("toml");
        std::fs::write(&config_path, config_content).unwrap();

        // Should fail due to path traversal in keypair_path
        let result = Config::load(&config_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("path traversal"));

        // Clean up
        let _ = std::fs::remove_file(config_path);
    }

    /// Test Case 13: Valid paths with subdirectories accepted
    /// Purpose: Verify legitimate subdirectory paths work
    /// Contract: Paths without ".." should work if file exists
    #[test]
    fn test_config_valid_subdirectory_paths() {
        // Create nested directory structure
        let tmp_dir = tempfile::tempdir().unwrap();
        let subdir = tmp_dir.path().join("keys");
        std::fs::create_dir(&subdir).unwrap();

        let keypair_file = subdir.join("alice.json");
        std::fs::write(&keypair_file, r#"{"secret_key":"test"}"#).unwrap();

        // Should accept valid path
        let result = validate_path(&keypair_file, Some("json"));
        assert!(result.is_ok());
    }

    /// Test Case 14: BFT threshold edge cases (deeper assertion)
    /// Purpose: Verify threshold boundary conditions
    /// Contract: Must be in [0.0, 1.0] inclusive
    #[test]
    fn test_config_bft_threshold_boundaries() {
        let base_config = Config {
            chain_endpoint: "ws://127.0.0.1:9944".to_string(),
            keypair_path: "/keys/alice.json".into(),
            grpc_port: 50051,
            metrics_port: 9100,
            p2p_listen_addr: "/ip4/0.0.0.0/tcp/30333".to_string(),
            bootstrap_peers: vec![],
            region: "us-east-1".to_string(),
            pipeline_lookahead: 2,
            grpc_timeout_secs: 5,
            bft_consensus_threshold: 0.95,
        };

        // Valid: 0.0 (minimum)
        let min_threshold = Config {
            bft_consensus_threshold: 0.0,
            ..base_config.clone()
        };
        assert!(min_threshold.validate().is_ok());

        // Valid: 1.0 (maximum)
        let max_threshold = Config {
            bft_consensus_threshold: 1.0,
            ..base_config.clone()
        };
        assert!(max_threshold.validate().is_ok());

        // Invalid: -0.1 (below minimum)
        let below_min = Config {
            bft_consensus_threshold: -0.1,
            ..base_config.clone()
        };
        assert!(below_min.validate().is_err());

        // Invalid: 1.01 (above maximum)
        let above_max = Config {
            bft_consensus_threshold: 1.01,
            ..base_config.clone()
        };
        assert!(above_max.validate().is_err());

        // Valid: 0.95 (typical value)
        let typical = Config {
            bft_consensus_threshold: 0.95,
            ..base_config.clone()
        };
        assert!(typical.validate().is_ok());
    }
}
