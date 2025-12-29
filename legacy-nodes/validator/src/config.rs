use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::error::Result;

/// Validator node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorConfig {
    /// Chain WebSocket endpoint
    pub chain_endpoint: String,

    /// Path to validator Ed25519 keypair JSON
    pub keypair_path: PathBuf,

    /// Directory containing CLIP ONNX models
    pub models_dir: PathBuf,

    /// CLIP configuration
    pub clip: ClipConfig,

    /// P2P networking configuration
    pub p2p: P2PConfig,

    /// Metrics server configuration
    pub metrics: MetricsConfig,

    /// Challenge participation configuration
    pub challenge: ChallengeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipConfig {
    /// Path to CLIP-ViT-B-32 ONNX model (relative to models_dir)
    pub model_b32_path: String,

    /// Path to CLIP-ViT-L-14 ONNX model (relative to models_dir)
    pub model_l14_path: String,

    /// Weight for B-32 model in ensemble (default: 0.4)
    #[serde(default = "default_b32_weight")]
    pub b32_weight: f32,

    /// Weight for L-14 model in ensemble (default: 0.6)
    #[serde(default = "default_l14_weight")]
    pub l14_weight: f32,

    /// Minimum CLIP score threshold for passing validation (default: 0.75)
    #[serde(default = "default_threshold")]
    pub threshold: f32,

    /// Number of keyframes to extract for validation (default: 5)
    #[serde(default = "default_keyframe_count")]
    pub keyframe_count: usize,

    /// Maximum inference timeout in seconds (default: 5)
    #[serde(default = "default_inference_timeout")]
    pub inference_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2PConfig {
    /// libp2p listen addresses
    #[serde(default = "default_listen_addresses")]
    pub listen_addresses: Vec<String>,

    /// Bootstrap peer multiaddrs
    #[serde(default)]
    pub bootstrap_peers: Vec<String>,

    /// Maximum number of peers
    #[serde(default = "default_max_peers")]
    pub max_peers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Prometheus metrics server listen address
    #[serde(default = "default_metrics_address")]
    pub listen_address: String,

    /// Metrics server port (default: 9101)
    #[serde(default = "default_metrics_port")]
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeConfig {
    /// Enable challenge participation (default: true)
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Challenge response buffer in blocks (default: 40 of 50 block period)
    #[serde(default = "default_challenge_buffer")]
    pub response_buffer_blocks: u32,

    /// Poll interval for checking pending challenges in seconds (default: 6)
    #[serde(default = "default_poll_interval")]
    pub poll_interval_secs: u64,
}

// Default value functions
fn default_b32_weight() -> f32 {
    0.4
}

fn default_l14_weight() -> f32 {
    0.6
}

fn default_threshold() -> f32 {
    0.75
}

fn default_keyframe_count() -> usize {
    5
}

fn default_inference_timeout() -> u64 {
    5
}

fn default_listen_addresses() -> Vec<String> {
    vec!["/ip4/0.0.0.0/tcp/0".to_string()]
}

fn default_max_peers() -> usize {
    50
}

fn default_metrics_address() -> String {
    "0.0.0.0".to_string()
}

fn default_metrics_port() -> u16 {
    9101
}

fn default_true() -> bool {
    true
}

fn default_challenge_buffer() -> u32 {
    40
}

fn default_poll_interval() -> u64 {
    6
}

impl ValidatorConfig {
    /// Load configuration from TOML file
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: ValidatorConfig = toml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<()> {
        use crate::error::ValidatorError;

        // Validate CLIP weights sum to 1.0
        let weight_sum = self.clip.b32_weight + self.clip.l14_weight;
        if (weight_sum - 1.0).abs() > 0.001 {
            return Err(ValidatorError::Config(format!(
                "CLIP weights must sum to 1.0, got {}",
                weight_sum
            )));
        }

        // Validate threshold range
        if self.clip.threshold < 0.0 || self.clip.threshold > 1.0 {
            return Err(ValidatorError::Config(format!(
                "CLIP threshold must be in range [0.0, 1.0], got {}",
                self.clip.threshold
            )));
        }

        // Validate keyframe count
        if self.clip.keyframe_count == 0 {
            return Err(ValidatorError::Config(
                "Keyframe count must be > 0".to_string(),
            ));
        }

        // Validate model paths exist
        let b32_path = self.models_dir.join(&self.clip.model_b32_path);
        if !b32_path.exists() {
            return Err(ValidatorError::Config(format!(
                "CLIP B-32 model not found at {:?}",
                b32_path
            )));
        }

        let l14_path = self.models_dir.join(&self.clip.model_l14_path);
        if !l14_path.exists() {
            return Err(ValidatorError::Config(format!(
                "CLIP L-14 model not found at {:?}",
                l14_path
            )));
        }

        // Validate keypair path exists
        if !self.keypair_path.exists() {
            return Err(ValidatorError::Config(format!(
                "Keypair file not found at {:?}",
                self.keypair_path
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_defaults() {
        let config = ClipConfig {
            model_b32_path: "clip-b32.onnx".to_string(),
            model_l14_path: "clip-l14.onnx".to_string(),
            b32_weight: default_b32_weight(),
            l14_weight: default_l14_weight(),
            threshold: default_threshold(),
            keyframe_count: default_keyframe_count(),
            inference_timeout_secs: default_inference_timeout(),
        };

        assert_eq!(config.b32_weight, 0.4);
        assert_eq!(config.l14_weight, 0.6);
        assert_eq!(config.threshold, 0.75);
        assert_eq!(config.keyframe_count, 5);
    }

    #[test]
    fn test_weight_validation_fails() {
        let mut temp_keypair = NamedTempFile::new().unwrap();
        writeln!(temp_keypair, "{{}}").unwrap();

        let mut temp_model_b32 = NamedTempFile::new().unwrap();
        writeln!(temp_model_b32, "").unwrap();

        let mut temp_model_l14 = NamedTempFile::new().unwrap();
        writeln!(temp_model_l14, "").unwrap();

        let models_dir = temp_model_b32.path().parent().unwrap().to_path_buf();

        let config = ValidatorConfig {
            chain_endpoint: "ws://localhost:9944".to_string(),
            keypair_path: temp_keypair.path().to_path_buf(),
            models_dir: models_dir.clone(),
            clip: ClipConfig {
                model_b32_path: temp_model_b32
                    .path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
                model_l14_path: temp_model_l14
                    .path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
                b32_weight: 0.5,
                l14_weight: 0.6, // Sum > 1.0
                threshold: 0.75,
                keyframe_count: 5,
                inference_timeout_secs: 5,
            },
            p2p: P2PConfig {
                listen_addresses: default_listen_addresses(),
                bootstrap_peers: vec![],
                max_peers: 50,
            },
            metrics: MetricsConfig {
                listen_address: default_metrics_address(),
                port: 9101,
            },
            challenge: ChallengeConfig {
                enabled: true,
                response_buffer_blocks: 40,
                poll_interval_secs: 6,
            },
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("sum to 1.0"));
    }

    #[test]
    fn test_threshold_validation_fails() {
        let mut temp_keypair = NamedTempFile::new().unwrap();
        writeln!(temp_keypair, "{{}}").unwrap();

        let mut temp_model_b32 = NamedTempFile::new().unwrap();
        writeln!(temp_model_b32, "").unwrap();

        let mut temp_model_l14 = NamedTempFile::new().unwrap();
        writeln!(temp_model_l14, "").unwrap();

        let models_dir = temp_model_b32.path().parent().unwrap().to_path_buf();

        let config = ValidatorConfig {
            chain_endpoint: "ws://localhost:9944".to_string(),
            keypair_path: temp_keypair.path().to_path_buf(),
            models_dir: models_dir.clone(),
            clip: ClipConfig {
                model_b32_path: temp_model_b32
                    .path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
                model_l14_path: temp_model_l14
                    .path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
                b32_weight: 0.4,
                l14_weight: 0.6,
                threshold: 1.5, // Invalid threshold
                keyframe_count: 5,
                inference_timeout_secs: 5,
            },
            p2p: P2PConfig {
                listen_addresses: default_listen_addresses(),
                bootstrap_peers: vec![],
                max_peers: 50,
            },
            metrics: MetricsConfig {
                listen_address: default_metrics_address(),
                port: 9101,
            },
            challenge: ChallengeConfig {
                enabled: true,
                response_buffer_blocks: 40,
                poll_interval_secs: 6,
            },
        };

        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("threshold must be in range"));
    }
}
