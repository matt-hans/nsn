//! P2P identity management
//!
//! Provides Ed25519 keypair generation, PeerId derivation, and conversion
//! between libp2p PeerId and Substrate AccountId32 for cross-layer identity.

use libp2p::identity::{Keypair, PublicKey};
use libp2p::PeerId;
use sp_core::crypto::AccountId32;
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IdentityError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Invalid keypair format")]
    InvalidKeypair,

    #[error("PeerId conversion failed: {0}")]
    ConversionError(String),
}

/// Generate a new Ed25519 keypair
///
/// # Returns
/// A new Ed25519 keypair suitable for libp2p identity
pub fn generate_keypair() -> Keypair {
    Keypair::generate_ed25519()
}

/// Convert libp2p PeerId to Substrate AccountId32
///
/// The conversion uses the public key bytes from the PeerId.
/// For Ed25519 keys, this provides a stable 32-byte identifier
/// that can be used as a Substrate AccountId32.
///
/// # Arguments
/// * `peer_id` - The libp2p PeerId to convert
///
/// # Returns
/// AccountId32 derived from the PeerId's public key
pub fn peer_id_to_account_id(peer_id: &PeerId) -> Result<AccountId32, IdentityError> {
    // Extract public key from PeerId
    let public_key = PublicKey::try_decode_protobuf(&peer_id.to_bytes())
        .map_err(|e| IdentityError::ConversionError(format!("Failed to decode PeerId: {}", e)))?;

    // Encode the public key to get bytes
    let encoded = public_key
        .encode_protobuf()
        .into_iter()
        .collect::<Vec<u8>>();

    // For Ed25519, extract the last 32 bytes (the actual public key)
    if encoded.len() >= 32 {
        let key_bytes: [u8; 32] = encoded[encoded.len() - 32..]
            .try_into()
            .map_err(|_| IdentityError::ConversionError("Invalid key length".to_string()))?;
        Ok(AccountId32::from(key_bytes))
    } else {
        Err(IdentityError::ConversionError(
            "Public key too short".to_string(),
        ))
    }
}

/// Save keypair to file
///
/// WARNING: This stores the keypair in plaintext. In production,
/// use encrypted storage or HSM.
///
/// # Arguments
/// * `keypair` - The keypair to save
/// * `path` - File path to save to
pub fn save_keypair(keypair: &Keypair, path: &Path) -> Result<(), IdentityError> {
    let bytes = keypair
        .to_protobuf_encoding()
        .map_err(|_| IdentityError::InvalidKeypair)?;

    let mut file = fs::File::create(path)?;
    file.write_all(&bytes)?;

    // Set restrictive permissions (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = file.metadata()?.permissions();
        perms.set_mode(0o600); // Only owner can read/write
        fs::set_permissions(path, perms)?;
    }

    Ok(())
}

/// Load keypair from file
///
/// # Arguments
/// * `path` - File path to load from
pub fn load_keypair(path: &Path) -> Result<Keypair, IdentityError> {
    let mut file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    Keypair::from_protobuf_encoding(&bytes).map_err(|_| IdentityError::InvalidKeypair)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_keypair_generation() {
        let keypair = generate_keypair();

        // Verify PeerId can be derived
        let peer_id = PeerId::from(keypair.public());
        assert!(!peer_id.to_string().is_empty());
    }

    #[test]
    fn test_peer_id_to_account_id() {
        let keypair = generate_keypair();
        let peer_id = PeerId::from(keypair.public());

        let account_id = peer_id_to_account_id(&peer_id).expect("Failed to convert PeerId");

        // AccountId32 should be 32 bytes
        let bytes: &[u8] = account_id.as_ref();
        assert_eq!(bytes.len(), 32);

        // Conversion should be deterministic
        let account_id2 = peer_id_to_account_id(&peer_id).expect("Failed to convert PeerId");
        assert_eq!(account_id, account_id2);
    }

    #[test]
    fn test_save_and_load_keypair() {
        let keypair = generate_keypair();
        let peer_id_original = PeerId::from(keypair.public());

        // Save to temp file
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path();

        save_keypair(&keypair, path).expect("Failed to save keypair");

        // Load back
        let loaded_keypair = load_keypair(path).expect("Failed to load keypair");
        let peer_id_loaded = PeerId::from(loaded_keypair.public());

        // Should be the same
        assert_eq!(peer_id_original, peer_id_loaded);
    }

    #[test]
    fn test_load_invalid_keypair() {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(b"invalid keypair data")
            .expect("Failed to write");

        let result = load_keypair(temp_file.path());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), IdentityError::InvalidKeypair));
    }

    #[test]
    fn test_account_id_cross_layer_compatibility() {
        // Test that the same keypair can be used for both P2P and on-chain operations
        let keypair = generate_keypair();
        let peer_id = PeerId::from(keypair.public());

        let account_id = peer_id_to_account_id(&peer_id).expect("Failed to convert");

        // Verify we can use this AccountId32 in Substrate context
        let account_str = format!("{:?}", account_id);
        assert!(!account_str.is_empty());

        // Verify account_id is 32 bytes
        let bytes: &[u8] = account_id.as_ref();
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_load_nonexistent_file() {
        use std::path::PathBuf;

        let nonexistent = PathBuf::from("/tmp/nonexistent_keypair_test_12345.key");
        let result = load_keypair(&nonexistent);

        assert!(result.is_err(), "Loading nonexistent file should fail");
        assert!(
            matches!(result.unwrap_err(), IdentityError::Io(_)),
            "Should be IO error"
        );
    }

    #[test]
    fn test_load_empty_file() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        // File is empty by default

        let result = load_keypair(temp_file.path());

        assert!(result.is_err(), "Loading empty file should fail");
        assert!(
            matches!(result.unwrap_err(), IdentityError::InvalidKeypair),
            "Should be InvalidKeypair error"
        );
    }

    #[test]
    fn test_load_corrupted_keypair() {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

        // Write partially valid protobuf (but invalid keypair)
        temp_file
            .write_all(&[0x08, 0x01, 0x12, 0x40]) // Partial protobuf header
            .expect("Failed to write");

        let result = load_keypair(temp_file.path());

        assert!(result.is_err(), "Loading corrupted file should fail");
        assert!(
            matches!(result.unwrap_err(), IdentityError::InvalidKeypair),
            "Should be InvalidKeypair error"
        );
    }

    #[test]
    fn test_identity_error_display() {
        // Test error message formatting
        let err = IdentityError::InvalidKeypair;
        assert_eq!(err.to_string(), "Invalid keypair format");

        let err = IdentityError::ConversionError("test error".to_string());
        assert_eq!(err.to_string(), "PeerId conversion failed: test error");

        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err = IdentityError::Io(io_err);
        assert!(err.to_string().contains("IO error"));
    }

    #[test]
    fn test_keypair_persistence_across_multiple_saves() {
        let keypair = generate_keypair();
        let original_peer_id = PeerId::from(keypair.public());

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path();

        // Save multiple times
        save_keypair(&keypair, path).expect("First save failed");
        save_keypair(&keypair, path).expect("Second save failed (overwrite)");

        // Load back
        let loaded = load_keypair(path).expect("Failed to load");
        let loaded_peer_id = PeerId::from(loaded.public());

        assert_eq!(
            original_peer_id, loaded_peer_id,
            "PeerId should be stable across multiple saves"
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_keypair_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let keypair = generate_keypair();
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path();

        save_keypair(&keypair, path).expect("Failed to save");

        let metadata = fs::metadata(path).expect("Failed to get metadata");
        let permissions = metadata.permissions();
        let mode = permissions.mode();

        // On Unix, should be 0o600 (owner read/write only)
        assert_eq!(
            mode & 0o777,
            0o600,
            "Keypair file should have restrictive permissions (0o600)"
        );
    }

    #[test]
    fn test_multiple_keypairs_unique() {
        let keypair1 = generate_keypair();
        let keypair2 = generate_keypair();

        let peer_id1 = PeerId::from(keypair1.public());
        let peer_id2 = PeerId::from(keypair2.public());

        assert_ne!(
            peer_id1, peer_id2,
            "Different keypairs should produce different PeerIds"
        );

        let account1 = peer_id_to_account_id(&peer_id1).expect("Failed to convert");
        let account2 = peer_id_to_account_id(&peer_id2).expect("Failed to convert");

        assert_ne!(
            account1, account2,
            "Different PeerIds should produce different AccountIds"
        );
    }
}
