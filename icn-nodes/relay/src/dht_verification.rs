//! DHT record signature verification
//!
//! This module implements Ed25519 signature verification for DHT records
//! to prevent DHT poisoning attacks where malicious actors inject fake records.

use ed25519_dalek::{PUBLIC_KEY_LENGTH, SIGNATURE_LENGTH};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tracing::{debug, warn};

use crate::error::Result;

/// Public key of a known publisher (e.g., Super-Node)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublisherPublicKey(pub [u8; PUBLIC_KEY_LENGTH]);

/// Signature on a DHT record
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DhtSignature(pub [u8; SIGNATURE_LENGTH]);

// Implement serde serialization using hex encoding
impl Serialize for PublisherPublicKey {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(self.0))
    }
}

impl<'de> Deserialize<'de> for PublisherPublicKey {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s).map_err(serde::de::Error::custom)?;
        if bytes.len() != PUBLIC_KEY_LENGTH {
            return Err(serde::de::Error::custom(format!(
                "invalid public key length: {}",
                bytes.len()
            )));
        }
        let mut key = [0u8; PUBLIC_KEY_LENGTH];
        key.copy_from_slice(&bytes);
        Ok(PublisherPublicKey(key))
    }
}

impl Serialize for DhtSignature {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(self.0))
    }
}

impl<'de> Deserialize<'de> for DhtSignature {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s).map_err(serde::de::Error::custom)?;
        if bytes.len() != SIGNATURE_LENGTH {
            return Err(serde::de::Error::custom(format!(
                "invalid signature length: {}",
                bytes.len()
            )));
        }
        let mut sig = [0u8; SIGNATURE_LENGTH];
        sig.copy_from_slice(&bytes);
        Ok(DhtSignature(sig))
    }
}

/// Signed DHT record with publisher signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedDhtRecord<T> {
    /// The actual record data
    pub record: T,
    /// Ed25519 public key of the publisher
    pub publisher_public_key: PublisherPublicKey,
    /// Ed25519 signature over serialized record
    pub signature: DhtSignature,
    /// Timestamp when record was created (Unix seconds)
    pub timestamp: u64,
}

/// Shard manifest from DHT (signed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManifest {
    /// Content identifier
    pub cid: String,
    /// List of Super-Nodes hosting this content
    pub super_nodes: Vec<String>,
    /// Content size in bytes
    pub size_bytes: u64,
    /// Number of shards (10 data + 4 parity = 14)
    pub shard_count: usize,
}

/// DHT signature verifier
pub struct DhtVerifier {
    /// Trusted publisher public keys (e.g., known Super-Nodes)
    trusted_publishers: Vec<PublisherPublicKey>,
}

impl DhtVerifier {
    /// Create new DHT verifier with trusted publishers
    pub fn new(trusted_publishers: Vec<PublisherPublicKey>) -> Self {
        Self { trusted_publishers }
    }

    /// Create verifier with no trusted publishers (accepts any signed record)
    pub fn new_permissive() -> Self {
        Self {
            trusted_publishers: Vec::new(),
        }
    }

    /// Verify a signed DHT record
    ///
    /// # Arguments
    /// * `signed_record` - The signed DHT record to verify
    ///
    /// # Returns
    /// Ok(()) if signature is valid, Err otherwise
    pub fn verify_record<T: Serialize>(&self, signed_record: &SignedDhtRecord<T>) -> Result<()> {
        // Step 1: Check timestamp (reject old records to prevent replay attacks)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        const MAX_RECORD_AGE_SECS: u64 = 3600; // 1 hour

        if now.saturating_sub(signed_record.timestamp) > MAX_RECORD_AGE_SECS {
            return Err(crate::error::RelayError::DhtSignatureVerificationFailed(
                format!(
                    "record too old: {} seconds",
                    now.saturating_sub(signed_record.timestamp)
                ),
            ));
        }

        // Step 2: Serialize record for signature verification
        let record_bytes = serde_json::to_vec(&signed_record.record).map_err(|e| {
            crate::error::RelayError::InvalidMerkleProof(format!(
                "failed to serialize record: {}",
                e
            ))
        })?;

        // Step 3: Verify Ed25519 signature using ed25519-dalek
        let signature_bytes = &signed_record.signature.0;
        let public_key_bytes = &signed_record.publisher_public_key.0;

        if signature_bytes.len() != SIGNATURE_LENGTH {
            return Err(crate::error::RelayError::InvalidMerkleProof(
                "invalid signature length".to_string(),
            ));
        }

        if public_key_bytes.len() != PUBLIC_KEY_LENGTH {
            return Err(crate::error::RelayError::InvalidMerkleProof(
                "invalid public key length".to_string(),
            ));
        }

        // Perform actual Ed25519 signature verification
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        // Reconstruct VerifyingKey from bytes (ed25519-dalek 2.x API)
        let mut verifying_key_array = [0u8; PUBLIC_KEY_LENGTH];
        verifying_key_array.copy_from_slice(public_key_bytes);
        let verifying_key = VerifyingKey::from_bytes(&verifying_key_array).map_err(|_| {
            crate::error::RelayError::InvalidMerkleProof("invalid public key format".to_string())
        })?;

        // Reconstruct Signature from bytes (ed25519-dalek 2.x returns Signature directly)
        let mut signature_array = [0u8; SIGNATURE_LENGTH];
        signature_array.copy_from_slice(signature_bytes);
        let signature = Signature::from_bytes(&signature_array);

        // Verify the signature
        verifying_key
            .verify(&record_bytes, &signature)
            .map_err(|_| {
                crate::error::RelayError::DhtSignatureVerificationFailed(
                    "signature verification failed".to_string(),
                )
            })?;

        debug!(
            "DHT signature verified: publisher={}",
            hex::encode(public_key_bytes)
        );

        // Step 4: Check if publisher is trusted (if whitelist is configured)
        if !self.trusted_publishers.is_empty()
            && !self
                .trusted_publishers
                .contains(&signed_record.publisher_public_key)
        {
            warn!(
                "DHT record from untrusted publisher: {}",
                hex::encode(signed_record.publisher_public_key.0)
            );
            return Err(crate::error::RelayError::DhtSignatureVerificationFailed(
                "untrusted publisher".to_string(),
            ));
        }

        debug!(
            "DHT signature format validated: publisher={}",
            hex::encode(signed_record.publisher_public_key.0)
        );

        Ok(())
    }

    /// Add a trusted publisher
    pub fn add_trusted_publisher(&mut self, public_key: PublisherPublicKey) {
        if !self.trusted_publishers.contains(&public_key) {
            self.trusted_publishers.push(public_key);
        }
    }

    /// Remove a trusted publisher
    pub fn remove_trusted_publisher(&mut self, public_key: &PublisherPublicKey) {
        self.trusted_publishers.retain(|pk| pk != public_key);
    }

    /// Get count of trusted publishers
    pub fn trusted_publisher_count(&self) -> usize {
        self.trusted_publishers.len()
    }
}

/// Helper function to sign a DHT record (for testing)
#[cfg(test)]
pub fn sign_record<T: Serialize + for<'de> Deserialize<'de>>(
    record: &T,
    secret_key: &[u8; 32], // Secret key bytes
) -> SignedDhtRecord<T> {
    use ed25519_dalek::{Signer, SigningKey};

    // Create SigningKey from bytes (ed25519-dalek 2.x API)
    let sk = SigningKey::from_bytes(secret_key);
    let public_key = sk.verifying_key();
    let record_bytes = serde_json::to_vec(record).unwrap();
    let signature = sk.sign(&record_bytes);

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Get the raw bytes from verifying key and signature (ed25519-dalek 2.x API)
    let mut pk_bytes = [0u8; PUBLIC_KEY_LENGTH];
    pk_bytes.copy_from_slice(public_key.as_bytes());

    let mut sig_bytes = [0u8; SIGNATURE_LENGTH];
    sig_bytes.copy_from_slice(signature.to_bytes().as_slice());

    SignedDhtRecord {
        record: serde_json::from_str(&serde_json::to_string(record).unwrap()).unwrap(),
        publisher_public_key: PublisherPublicKey(pk_bytes),
        signature: DhtSignature(sig_bytes),
        timestamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn test_verify_signed_record() {
        // Generate keypair
        let mut csprng = rand::rngs::OsRng {};
        let mut secret_key_bytes = [0u8; 32];
        csprng.fill_bytes(&mut secret_key_bytes);

        // Create test manifest
        let manifest = ShardManifest {
            cid: "test_cid".to_string(),
            super_nodes: vec!["super1.example.com:9002".to_string()],
            size_bytes: 1024,
            shard_count: 14,
        };

        // Sign the manifest
        let signed_record = sign_record(&manifest, &secret_key_bytes);

        // Verify with permissive verifier
        let verifier = DhtVerifier::new_permissive();
        let result = verifier.verify_record(&signed_record);
        assert!(result.is_ok(), "Should verify valid signature");
    }

    #[test]
    fn test_verify_with_trusted_publisher() {
        let mut csprng = rand::rngs::OsRng {};
        let mut secret_key_bytes = [0u8; 32];
        csprng.fill_bytes(&mut secret_key_bytes);

        // Get the public key for this secret key
        use ed25519_dalek::SigningKey;
        let sk = SigningKey::from_bytes(&secret_key_bytes);
        let mut pk_bytes = [0u8; PUBLIC_KEY_LENGTH];
        pk_bytes.copy_from_slice(sk.verifying_key().as_bytes());
        let public_key = PublisherPublicKey(pk_bytes);

        let manifest = ShardManifest {
            cid: "test_cid".to_string(),
            super_nodes: vec![],
            size_bytes: 1024,
            shard_count: 14,
        };

        let signed_record = sign_record(&manifest, &secret_key_bytes);

        // Verify with trusted publisher list
        let verifier = DhtVerifier::new(vec![public_key]);
        let result = verifier.verify_record(&signed_record);
        assert!(
            result.is_ok(),
            "Should verify record from trusted publisher"
        );
    }

    #[test]
    fn test_reject_untrusted_publisher() {
        let mut csprng = rand::rngs::OsRng {};
        let mut secret_key_bytes = [0u8; 32];
        csprng.fill_bytes(&mut secret_key_bytes);
        let untrusted_key = PublisherPublicKey([1u8; PUBLIC_KEY_LENGTH]); // Different key

        let manifest = ShardManifest {
            cid: "test_cid".to_string(),
            super_nodes: vec![],
            size_bytes: 1024,
            shard_count: 14,
        };

        let signed_record = sign_record(&manifest, &secret_key_bytes);

        // Verify with trusted publisher list that doesn't include signer
        let verifier = DhtVerifier::new(vec![untrusted_key]);
        let result = verifier.verify_record(&signed_record);
        assert!(
            result.is_err(),
            "Should reject record from untrusted publisher"
        );
    }

    #[test]
    fn test_reject_invalid_signature() {
        let mut csprng = rand::rngs::OsRng {};
        let mut secret_key_bytes = [0u8; 32];
        csprng.fill_bytes(&mut secret_key_bytes);

        let manifest = ShardManifest {
            cid: "test_cid".to_string(),
            super_nodes: vec![],
            size_bytes: 1024,
            shard_count: 14,
        };

        let mut signed_record = sign_record(&manifest, &secret_key_bytes);

        // Corrupt the signature
        signed_record.signature.0[0] = signed_record.signature.0[0].wrapping_add(1);

        let verifier = DhtVerifier::new_permissive();
        let result = verifier.verify_record(&signed_record);
        assert!(result.is_err(), "Should reject invalid signature");
    }

    #[test]
    fn test_trusted_publisher_management() {
        let key1 = PublisherPublicKey([1u8; PUBLIC_KEY_LENGTH]);
        let key2 = PublisherPublicKey([2u8; PUBLIC_KEY_LENGTH]);

        let mut verifier = DhtVerifier::new(Vec::new());
        assert_eq!(verifier.trusted_publisher_count(), 0);

        verifier.add_trusted_publisher(key1);
        assert_eq!(verifier.trusted_publisher_count(), 1);

        verifier.add_trusted_publisher(key2);
        assert_eq!(verifier.trusted_publisher_count(), 2);

        // Adding same key should not duplicate
        verifier.add_trusted_publisher(key1);
        assert_eq!(verifier.trusted_publisher_count(), 2);

        verifier.remove_trusted_publisher(&key1);
        assert_eq!(verifier.trusted_publisher_count(), 1);
    }
}
