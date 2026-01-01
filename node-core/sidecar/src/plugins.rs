//! Plugin registry and policy enforcement for the sidecar.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::Deserialize;

use crate::error::{SidecarError, SidecarResult};

/// Plugin resource requirements declared in the manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct PluginResources {
    /// Required VRAM in GB.
    pub vram_gb: f32,
    /// Max latency in milliseconds.
    pub max_latency_ms: u32,
    /// Maximum concurrent tasks.
    pub max_concurrency: u32,
}

/// Plugin IO schema definitions (stored but not validated in Rust).
#[derive(Debug, Clone, Deserialize)]
pub struct PluginIoSchema {
    /// Input JSON schema.
    pub input_schema: serde_yaml::Value,
    /// Output JSON schema.
    pub output_schema: serde_yaml::Value,
}

/// Plugin manifest loaded from manifest.yaml.
#[derive(Debug, Clone, Deserialize)]
pub struct PluginManifest {
    pub schema_version: String,
    pub name: String,
    pub version: String,
    pub entrypoint: String,
    pub description: String,
    pub supported_lanes: Vec<String>,
    pub deterministic: bool,
    pub resources: PluginResources,
    pub io: PluginIoSchema,
}

impl PluginManifest {
    /// Validate the manifest has required fields and sane values.
    pub fn validate(&self) -> SidecarResult<()> {
        if self.schema_version.trim().is_empty() {
            return Err(SidecarError::InvalidRequest(
                "plugin manifest schema_version is empty".to_string(),
            ));
        }
        if self.name.trim().is_empty() {
            return Err(SidecarError::InvalidRequest(
                "plugin manifest name is empty".to_string(),
            ));
        }
        if self.version.trim().is_empty() {
            return Err(SidecarError::InvalidRequest(
                "plugin manifest version is empty".to_string(),
            ));
        }
        if self.entrypoint.trim().is_empty() {
            return Err(SidecarError::InvalidRequest(
                "plugin manifest entrypoint is empty".to_string(),
            ));
        }
        if self.supported_lanes.is_empty() {
            return Err(SidecarError::InvalidRequest(
                "plugin manifest supported_lanes is empty".to_string(),
            ));
        }
        if self.resources.vram_gb <= 0.0 {
            return Err(SidecarError::InvalidRequest(
                "plugin manifest vram_gb must be > 0".to_string(),
            ));
        }
        if self.resources.max_latency_ms == 0 {
            return Err(SidecarError::InvalidRequest(
                "plugin manifest max_latency_ms must be > 0".to_string(),
            ));
        }
        if self.resources.max_concurrency == 0 {
            return Err(SidecarError::InvalidRequest(
                "plugin manifest max_concurrency must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Returns true if the manifest supports the requested lane.
    pub fn supports_lane(&self, lane: u32) -> bool {
        let lane_name = match lane {
            0 => "lane0",
            1 => "lane1",
            _ => "unknown",
        };
        self.supported_lanes
            .iter()
            .any(|lane| lane.eq_ignore_ascii_case(lane_name))
    }
}

/// Policy constraints for plugin execution.
#[derive(Debug, Clone)]
pub struct PluginPolicy {
    pub max_vram_gb: f32,
    pub lane0_max_latency_ms: u32,
    pub lane1_max_latency_ms: u32,
    pub allow_untrusted: bool,
    pub allowlist: HashSet<String>,
}

impl Default for PluginPolicy {
    fn default() -> Self {
        Self {
            max_vram_gb: 11.5,
            lane0_max_latency_ms: 15_000,
            lane1_max_latency_ms: 120_000,
            // SECURITY: Untrusted plugins are disabled by default.
            // Only explicitly allowlisted plugins can execute.
            allow_untrusted: false,
            allowlist: HashSet::new(),
        }
    }
}

impl PluginPolicy {
    /// Validate a manifest against policy constraints for a lane.
    pub fn check(&self, manifest: &PluginManifest, lane: u32) -> SidecarResult<()> {
        if manifest.resources.vram_gb > self.max_vram_gb {
            return Err(SidecarError::PluginPolicyViolation(format!(
                "plugin '{}' requires {:.2}GB VRAM, exceeds policy max {:.2}GB",
                manifest.name, manifest.resources.vram_gb, self.max_vram_gb
            )));
        }

        if !self.allow_untrusted && !self.allowlist.contains(&manifest.name) {
            return Err(SidecarError::PluginPolicyViolation(format!(
                "plugin '{}' not in allowlist",
                manifest.name
            )));
        }

        match lane {
            0 => {
                if !manifest.deterministic {
                    return Err(SidecarError::PluginPolicyViolation(format!(
                        "plugin '{}' must be deterministic for lane0",
                        manifest.name
                    )));
                }
                if manifest.resources.max_latency_ms > self.lane0_max_latency_ms {
                    return Err(SidecarError::PluginPolicyViolation(format!(
                        "plugin '{}' latency {}ms exceeds lane0 max {}ms",
                        manifest.name, manifest.resources.max_latency_ms, self.lane0_max_latency_ms
                    )));
                }
            }
            1 => {
                if manifest.resources.max_latency_ms > self.lane1_max_latency_ms {
                    return Err(SidecarError::PluginPolicyViolation(format!(
                        "plugin '{}' latency {}ms exceeds lane1 max {}ms",
                        manifest.name, manifest.resources.max_latency_ms, self.lane1_max_latency_ms
                    )));
                }
            }
            _ => {
                return Err(SidecarError::InvalidRequest(
                    "invalid lane value".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Registry of loaded plugin manifests.
#[derive(Debug, Clone)]
pub struct PluginRegistry {
    plugins: HashMap<String, PluginManifest>,
    policy: PluginPolicy,
}

impl PluginRegistry {
    /// Create an empty registry with a policy.
    pub fn empty(policy: PluginPolicy) -> Self {
        Self {
            plugins: HashMap::new(),
            policy,
        }
    }

    /// Load plugin manifests from a directory.
    pub fn load_from_dir(dir: &Path, policy: PluginPolicy) -> SidecarResult<Self> {
        let mut plugins = HashMap::new();

        if !dir.exists() {
            return Ok(Self { plugins, policy });
        }

        for entry in std::fs::read_dir(dir)
            .map_err(|e| SidecarError::PluginRegistryError(e.to_string()))?
        {
            let entry = entry.map_err(|e| SidecarError::PluginRegistryError(e.to_string()))?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let manifest_path = path.join("manifest.yaml");
            let manifest_path_alt = path.join("manifest.yml");
            let manifest_path = if manifest_path.exists() {
                manifest_path
            } else if manifest_path_alt.exists() {
                manifest_path_alt
            } else {
                continue;
            };

            let manifest_str = std::fs::read_to_string(&manifest_path)
                .map_err(|e| SidecarError::PluginRegistryError(e.to_string()))?;
            let manifest: PluginManifest = serde_yaml::from_str(&manifest_str)
                .map_err(|e| SidecarError::PluginRegistryError(e.to_string()))?;
            manifest.validate()?;

            if plugins.contains_key(&manifest.name) {
                return Err(SidecarError::PluginRegistryError(format!(
                    "duplicate plugin name '{}'",
                    manifest.name
                )));
            }

            plugins.insert(manifest.name.clone(), manifest);
        }

        Ok(Self { plugins, policy })
    }

    /// Access the policy for this registry.
    pub fn policy(&self) -> &PluginPolicy {
        &self.policy
    }

    /// Fetch a manifest by name.
    pub fn get(&self, name: &str) -> Option<&PluginManifest> {
        self.plugins.get(name)
    }

    /// Return manifest list.
    pub fn list(&self) -> Vec<PluginManifest> {
        self.plugins.values().cloned().collect()
    }
}
