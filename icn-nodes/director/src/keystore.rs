//! Keystore management for Director node identity
//!
//! Handles secure Ed25519 keypair loading and PeerId generation.
//!
//! Security considerations:
//! - Keys stored in JSON format compatible with libp2p
//! - File permissions checked (must be 0600 or more restrictive)
//! - No plaintext key logging
//! - Redacted debug output

#![allow(clippy::result_large_err)]

use libp2p::identity::Keypair;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tracing::info;

use crate::error::DirectorError;

/// Ed25519 keypair stored in JSON format
#[derive(Debug, Serialize, Deserialize)]
struct KeypairJson {
    /// Base64-encoded secret key (32 bytes)
    secret_key: String,
}

/// Keystore for managing node identity
pub struct Keystore {
    keypair: Keypair,
}

impl Keystore {
    /// Load keypair from file
    ///
    /// # Security
    /// - Validates file permissions (Unix: checks for world/group read)
    /// - Validates key format
    /// - No key material logged
    ///
    /// # Errors
    /// Returns error if:
    /// - File doesn't exist
    /// - File has insecure permissions (readable by group/others on Unix)
    /// - Invalid JSON format
    /// - Invalid base64 encoding
    /// - Invalid Ed25519 key bytes
    pub fn load(path: impl AsRef<Path>) -> Result<Self, DirectorError> {
        let path = path.as_ref();

        info!("Loading keypair from {:?}", path);

        // Check file exists
        if !path.exists() {
            return Err(DirectorError::Config(format!(
                "Keypair file not found: {:?}",
                path
            )));
        }

        // Validate file permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = fs::metadata(path).map_err(|e| {
                DirectorError::Config(format!("Failed to read keypair file metadata: {}", e))
            })?;
            let mode = metadata.permissions().mode();

            // Check if group or others can read (bits 4-5 or 1-2)
            if mode & 0o077 != 0 {
                return Err(DirectorError::Config(format!(
                    "Insecure keypair file permissions: {:o}. Expected 0600 or stricter",
                    mode & 0o777
                )));
            }
        }

        // Read and parse JSON
        let content = fs::read_to_string(path)
            .map_err(|e| DirectorError::Config(format!("Failed to read keypair file: {}", e)))?;

        let keypair_json: KeypairJson = serde_json::from_str(&content)
            .map_err(|e| DirectorError::Config(format!("Invalid keypair JSON format: {}", e)))?;

        // Decode base64 to get protobuf bytes
        use base64::{engine::general_purpose, Engine as _};
        let protobuf_bytes = general_purpose::STANDARD
            .decode(&keypair_json.secret_key)
            .map_err(|e| {
                DirectorError::Config(format!("Invalid base64 encoding in secret_key: {}", e))
            })?;

        // Decode keypair from protobuf encoding
        let keypair = Keypair::from_protobuf_encoding(&protobuf_bytes)
            .map_err(|e| DirectorError::Config(format!("Failed to decode keypair: {}", e)))?;

        info!(
            "Keypair loaded successfully, PeerId: {}",
            keypair.public().to_peer_id()
        );

        Ok(Self { keypair })
    }

    /// Generate a new random keypair and save to file
    ///
    /// # Security
    /// - Sets file permissions to 0600 (Unix)
    /// - Uses cryptographically secure RNG
    ///
    /// # Errors
    /// Returns error if file write fails
    #[allow(dead_code)] // Used in tests and future CLI tools
    pub fn generate_and_save(path: impl AsRef<Path>) -> Result<Self, DirectorError> {
        let path = path.as_ref();

        // Generate new Ed25519 keypair
        let keypair = Keypair::generate_ed25519();

        // Use protobuf encoding for full keypair (secret + public)
        let protobuf_bytes = keypair
            .to_protobuf_encoding()
            .map_err(|e| DirectorError::Config(format!("Failed to encode keypair: {}", e)))?;

        // Encode to base64
        use base64::{engine::general_purpose, Engine as _};
        let secret_b64 = general_purpose::STANDARD.encode(&protobuf_bytes);

        let keypair_json = KeypairJson {
            secret_key: secret_b64,
        };

        // Serialize to JSON
        let json_content = serde_json::to_string_pretty(&keypair_json)
            .map_err(|e| DirectorError::Config(format!("Failed to serialize keypair: {}", e)))?;

        // Write to file
        fs::write(path, json_content)
            .map_err(|e| DirectorError::Config(format!("Failed to write keypair file: {}", e)))?;

        // Set permissions to 0600 (Unix)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(path)
                .map_err(|e| DirectorError::Config(format!("Failed to read file metadata: {}", e)))?
                .permissions();
            perms.set_mode(0o600);
            fs::set_permissions(path, perms).map_err(|e| {
                DirectorError::Config(format!("Failed to set file permissions: {}", e))
            })?;
        }

        info!(
            "Generated new keypair at {:?}, PeerId: {}",
            path,
            keypair.public().to_peer_id()
        );

        Ok(Self { keypair })
    }

    /// Get reference to the keypair
    #[allow(dead_code)] // Public API for future use
    pub fn keypair(&self) -> &Keypair {
        &self.keypair
    }

    /// Get the PeerId derived from the keypair
    pub fn peer_id(&self) -> libp2p::PeerId {
        self.keypair.public().to_peer_id()
    }
}

/// Redact secret key material in debug output
impl std::fmt::Debug for Keystore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Keystore")
            .field("peer_id", &self.peer_id())
            .field("keypair", &"<redacted>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    /// Test Case 1: Valid keypair loads successfully
    /// Purpose: Verify keystore can load valid Ed25519 keypair
    /// Contract: Returns Keystore with correct PeerId
    #[test]
    fn test_keystore_load_valid() {
        // Generate a temporary keypair
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        // Generate and save
        let keystore1 = Keystore::generate_and_save(path).expect("Failed to generate keypair");
        let peer_id1 = keystore1.peer_id();

        // Load again
        let keystore2 = Keystore::load(path).expect("Failed to load keypair");
        let peer_id2 = keystore2.peer_id();

        // Should be identical
        assert_eq!(peer_id1, peer_id2);
    }

    /// Test Case 2: Missing file returns error
    /// Purpose: Verify error handling for non-existent file
    /// Contract: Returns Config error with descriptive message
    #[test]
    fn test_keystore_load_missing_file() {
        let result = Keystore::load("/nonexistent/path/to/keypair.json");
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, DirectorError::Config(_)));
        assert!(err.to_string().contains("not found"));
    }

    /// Test Case 3: Invalid JSON returns error
    /// Purpose: Verify validation of keypair file format
    /// Contract: Returns Config error for malformed JSON
    #[test]
    fn test_keystore_load_invalid_json() {
        let tmp_file = NamedTempFile::new().unwrap();
        fs::write(tmp_file.path(), "invalid json content").unwrap();

        let result = Keystore::load(tmp_file.path());
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, DirectorError::Config(_)));
        assert!(err.to_string().contains("Invalid keypair JSON"));
    }

    /// Test Case 4: Invalid base64 returns error
    /// Purpose: Verify validation of base64 encoding
    /// Contract: Returns Config error for invalid base64
    #[test]
    fn test_keystore_load_invalid_base64() {
        let tmp_file = NamedTempFile::new().unwrap();
        let invalid_json = r#"{"secret_key": "not-valid-base64!!!"}"#;
        fs::write(tmp_file.path(), invalid_json).unwrap();

        let result = Keystore::load(tmp_file.path());
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, DirectorError::Config(_)));
        assert!(err.to_string().contains("Invalid base64"));
    }

    /// Test Case 5: Wrong key length returns error
    /// Purpose: Verify validation of protobuf-encoded keypair size
    /// Contract: Returns Config error for incorrect key length
    #[test]
    fn test_keystore_load_wrong_key_length() {
        use base64::{engine::general_purpose, Engine as _};

        let tmp_file = NamedTempFile::new().unwrap();
        // Too short for protobuf-encoded Ed25519 keypair (need 68+ bytes)
        let short_key = general_purpose::STANDARD.encode(&[0u8; 16]);
        let invalid_json = format!(r#"{{"secret_key": "{}"}}"#, short_key);
        fs::write(tmp_file.path(), invalid_json).unwrap();

        let result = Keystore::load(tmp_file.path());
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, DirectorError::Config(_)));
        // Error could be "Failed to decode keypair" or protobuf parsing error
        assert!(
            err.to_string().contains("Failed to decode keypair")
                || err.to_string().contains("protobuf")
        );
    }

    /// Test Case 6: Insecure file permissions rejected (Unix only)
    /// Purpose: Verify security check for world/group readable files
    /// Contract: Returns Config error if permissions too permissive
    #[test]
    #[cfg(unix)]
    fn test_keystore_load_insecure_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        // Generate valid keypair
        Keystore::generate_and_save(path).unwrap();

        // Set insecure permissions (world readable)
        let mut perms = fs::metadata(path).unwrap().permissions();
        perms.set_mode(0o644); // rw-r--r--
        fs::set_permissions(path, perms).unwrap();

        // Should fail to load
        let result = Keystore::load(path);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, DirectorError::Config(_)));
        assert!(err
            .to_string()
            .contains("Insecure keypair file permissions"));
    }

    /// Test Case 7: Generate creates valid keypair
    /// Purpose: Verify keypair generation produces valid Ed25519 key
    /// Contract: Generated keypair can be loaded and has valid PeerId
    #[test]
    fn test_keystore_generate_and_save() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        // Generate
        let keystore = Keystore::generate_and_save(path).expect("Failed to generate");

        // Should have valid PeerId
        let peer_id = keystore.peer_id();
        assert!(!peer_id.to_string().is_empty());

        // File should exist
        assert!(path.exists());

        // Should be loadable
        let loaded = Keystore::load(path).expect("Failed to load generated keypair");
        assert_eq!(loaded.peer_id(), peer_id);
    }

    /// Test Case 8: Generated file has secure permissions (Unix only)
    /// Purpose: Verify generated files have 0600 permissions
    /// Contract: File mode is 0600
    #[test]
    #[cfg(unix)]
    fn test_keystore_generate_secure_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        Keystore::generate_and_save(path).unwrap();

        let metadata = fs::metadata(path).unwrap();
        let mode = metadata.permissions().mode() & 0o777;

        assert_eq!(mode, 0o600, "Expected 0600, got {:o}", mode);
    }

    /// Test Case 9: Debug output redacts secret key
    /// Purpose: Verify debug trait doesn't leak sensitive data
    /// Contract: Debug output contains PeerId but not keypair bytes
    #[test]
    fn test_keystore_debug_redaction() {
        let tmp_file = NamedTempFile::new().unwrap();
        let keystore = Keystore::generate_and_save(tmp_file.path()).unwrap();

        let debug_output = format!("{:?}", keystore);

        // Should contain PeerId
        assert!(debug_output.contains("peer_id"));

        // Should redact keypair
        assert!(debug_output.contains("<redacted>"));

        // Should NOT contain "secret_key" or raw bytes
        assert!(!debug_output.contains("secret_key"));
    }

    /// Test Case 10: PeerId consistency
    /// Purpose: Verify PeerId is deterministic from keypair
    /// Contract: Same keypair always produces same PeerId
    #[test]
    fn test_keystore_peer_id_consistency() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path();

        // Generate once
        let keystore1 = Keystore::generate_and_save(path).unwrap();
        let peer_id1 = keystore1.peer_id();

        // Load multiple times
        let keystore2 = Keystore::load(path).unwrap();
        let peer_id2 = keystore2.peer_id();

        let keystore3 = Keystore::load(path).unwrap();
        let peer_id3 = keystore3.peer_id();

        // All should be identical
        assert_eq!(peer_id1, peer_id2);
        assert_eq!(peer_id2, peer_id3);
    }
}
