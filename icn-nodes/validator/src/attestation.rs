use base64::Engine;
use chrono::Utc;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Result, ValidatorError};

/// Attestation of video chunk validation result
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Attestation {
    /// Slot number being validated
    pub slot: u64,

    /// Validator's PeerId (derived from public key)
    pub validator_id: String,

    /// CLIP ensemble score [0.0, 1.0]
    pub clip_score: f32,

    /// Whether validation passed (score >= threshold)
    pub passed: bool,

    /// Unix timestamp (seconds)
    pub timestamp: u64,

    /// Optional failure reason
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,

    /// Ed25519 signature (base64 encoded)
    pub signature: String,
}

impl Attestation {
    /// Create a new unsigned attestation
    pub fn new(slot: u64, validator_id: String, clip_score: f32, threshold: f32) -> Result<Self> {
        // Validate score range
        if !(0.0..=1.0).contains(&clip_score) {
            return Err(ValidatorError::InvalidScore(clip_score));
        }

        let passed = clip_score >= threshold;
        let reason = if !passed {
            Some("semantic_mismatch".to_string())
        } else {
            None
        };

        Ok(Self {
            slot,
            validator_id,
            clip_score,
            passed,
            timestamp: Utc::now().timestamp() as u64,
            reason,
            signature: String::new(), // Will be filled by sign()
        })
    }

    /// Sign the attestation with Ed25519 keypair
    pub fn sign(mut self, signing_key: &SigningKey) -> Result<Self> {
        let message = self.canonical_message();
        let signature = signing_key.sign(message.as_bytes());
        self.signature = base64::engine::general_purpose::STANDARD.encode(signature.to_bytes());
        Ok(self)
    }

    /// Verify attestation signature
    pub fn verify(&self, verifying_key: &VerifyingKey) -> Result<()> {
        let message = self.canonical_message();
        let sig_bytes = base64::engine::general_purpose::STANDARD
            .decode(&self.signature)
            .map_err(|e| ValidatorError::AttestationVerification(e.to_string()))?;

        let signature = Signature::from_bytes(&sig_bytes.try_into().map_err(|_| {
            ValidatorError::AttestationVerification("Invalid signature length".to_string())
        })?);

        verifying_key
            .verify(message.as_bytes(), &signature)
            .map_err(|e| ValidatorError::AttestationVerification(e.to_string()))?;

        Ok(())
    }

    /// Verify timestamp is within acceptable range (±5 minutes)
    pub fn verify_timestamp(&self, tolerance_secs: u64) -> Result<()> {
        let now = Utc::now().timestamp() as u64;
        let diff = now.abs_diff(self.timestamp);

        if diff > tolerance_secs {
            return Err(ValidatorError::InvalidTimestamp(format!(
                "Timestamp difference {} seconds exceeds tolerance {} seconds",
                diff, tolerance_secs
            )));
        }

        Ok(())
    }

    /// Get canonical message for signing/verification
    fn canonical_message(&self) -> String {
        // Deterministic message format: slot:score:timestamp:passed
        format!(
            "{}:{}:{}:{}",
            self.slot, self.clip_score, self.timestamp, self.passed as u8
        )
    }

    /// Compute attestation hash for BFT comparison
    pub fn compute_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.canonical_message().as_bytes());
        hasher.finalize().into()
    }
}

/// Load Ed25519 keypair from JSON file
pub fn load_keypair(path: &std::path::Path) -> Result<SigningKey> {
    let contents = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&contents)?;

    // Extract secret key bytes (expect "secretKey" field with hex or base64)
    let secret_key_str = json
        .get("secretKey")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            ValidatorError::Config("Missing 'secretKey' field in keypair JSON".to_string())
        })?;

    // Try base64 first, then hex
    let secret_bytes = base64::engine::general_purpose::STANDARD
        .decode(secret_key_str)
        .or_else(|_| hex::decode(secret_key_str))
        .map_err(|e| ValidatorError::Config(format!("Failed to decode secret key: {}", e)))?;

    if secret_bytes.len() != 32 {
        return Err(ValidatorError::Config(format!(
            "Invalid secret key length: expected 32 bytes, got {}",
            secret_bytes.len()
        )));
    }

    let key_bytes: [u8; 32] = secret_bytes
        .try_into()
        .map_err(|_| ValidatorError::Config("Invalid key length".to_string()))?;

    Ok(SigningKey::from_bytes(&key_bytes))
}

/// Derive PeerId string from signing key (for libp2p compatibility)
pub fn derive_peer_id(signing_key: &SigningKey) -> String {
    let public_key = signing_key.verifying_key();
    let public_bytes = public_key.to_bytes();

    // Hash public key to get PeerId (simplified - real libp2p uses multihash)
    let mut hasher = Sha256::new();
    hasher.update(public_bytes);
    let hash = hasher.finalize();

    format!(
        "12D3KooW{}",
        base64::engine::general_purpose::STANDARD.encode(&hash[..16])
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signing_key() -> SigningKey {
        // Deterministic test key
        let secret_bytes = [42u8; 32];
        SigningKey::from_bytes(&secret_bytes)
    }

    #[test]
    fn test_attestation_creation() {
        let attestation = Attestation::new(100, "test_validator".to_string(), 0.85, 0.75).unwrap();

        assert_eq!(attestation.slot, 100);
        assert_eq!(attestation.clip_score, 0.85);
        assert!(attestation.passed);
        assert!(attestation.reason.is_none());
    }

    #[test]
    fn test_attestation_failed_validation() {
        let attestation = Attestation::new(100, "test_validator".to_string(), 0.65, 0.75).unwrap();

        assert!(!attestation.passed);
        assert_eq!(attestation.reason, Some("semantic_mismatch".to_string()));
    }

    #[test]
    fn test_invalid_score_range() {
        let result = Attestation::new(100, "test_validator".to_string(), 1.5, 0.75);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid CLIP score"));
    }

    #[test]
    fn test_signature_generation() {
        let signing_key = test_signing_key();
        let attestation = Attestation::new(100, "test_validator".to_string(), 0.85, 0.75)
            .unwrap()
            .sign(&signing_key)
            .unwrap();

        assert!(!attestation.signature.is_empty());

        // Signature should be base64 encoded
        let decoded = base64::engine::general_purpose::STANDARD.decode(&attestation.signature);
        assert!(decoded.is_ok());
        assert_eq!(decoded.unwrap().len(), 64); // Ed25519 signature is 64 bytes
    }

    #[test]
    fn test_signature_verification_success() {
        let signing_key = test_signing_key();
        let verifying_key = signing_key.verifying_key();

        let attestation = Attestation::new(100, "test_validator".to_string(), 0.85, 0.75)
            .unwrap()
            .sign(&signing_key)
            .unwrap();

        let result = attestation.verify(&verifying_key);
        assert!(result.is_ok());
    }

    #[test]
    fn test_signature_verification_failure() {
        let signing_key = test_signing_key();
        let wrong_key = SigningKey::from_bytes(&[99u8; 32]);
        let wrong_verifying_key = wrong_key.verifying_key();

        let attestation = Attestation::new(100, "test_validator".to_string(), 0.85, 0.75)
            .unwrap()
            .sign(&signing_key)
            .unwrap();

        let result = attestation.verify(&wrong_verifying_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_signature_deterministic() {
        let signing_key = test_signing_key();

        // Create two attestations with same data
        let att1 = Attestation {
            slot: 100,
            validator_id: "test".to_string(),
            clip_score: 0.85,
            passed: true,
            timestamp: 1234567890,
            reason: None,
            signature: String::new(),
        }
        .sign(&signing_key)
        .unwrap();

        let att2 = Attestation {
            slot: 100,
            validator_id: "test".to_string(),
            clip_score: 0.85,
            passed: true,
            timestamp: 1234567890,
            reason: None,
            signature: String::new(),
        }
        .sign(&signing_key)
        .unwrap();

        // Signatures should be identical for same input
        assert_eq!(att1.signature, att2.signature);
    }

    #[test]
    fn test_timestamp_validation() {
        let attestation = Attestation::new(100, "test_validator".to_string(), 0.85, 0.75).unwrap();

        // Should pass with 5 minute tolerance
        let result = attestation.verify_timestamp(300);
        assert!(result.is_ok());
    }

    #[test]
    fn test_canonical_message_format() {
        let attestation = Attestation {
            slot: 100,
            validator_id: "test".to_string(),
            clip_score: 0.85,
            passed: true,
            timestamp: 1234567890,
            reason: None,
            signature: String::new(),
        };

        let message = attestation.canonical_message();
        assert_eq!(message, "100:0.85:1234567890:1");
    }

    #[test]
    fn test_attestation_hash() {
        let attestation = Attestation::new(100, "test_validator".to_string(), 0.85, 0.75).unwrap();
        let hash = attestation.compute_hash();

        assert_eq!(hash.len(), 32);

        // Same attestation should produce same hash
        let hash2 = attestation.compute_hash();
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_derive_peer_id() {
        let signing_key = test_signing_key();
        let peer_id = derive_peer_id(&signing_key);

        assert!(peer_id.starts_with("12D3KooW"));
        assert!(!peer_id.is_empty());
    }
}
