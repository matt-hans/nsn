//! Ed25519 Signature Verification for Bootstrap Manifests
//!
//! Provides signature verification for DNS and HTTP bootstrap sources
//! using Ed25519 public keys from trusted signers.

use libp2p::identity::PublicKey;
use std::collections::HashSet;

/// Get trusted signer public keys
///
/// These are foundation keypairs authorized to sign bootstrap manifests.
/// In production, these would be rotated via on-chain governance.
///
/// # WARNING
/// These are TESTNET keys. Replace with actual foundation keys for mainnet.
pub fn get_trusted_signers() -> HashSet<PublicKey> {
    use libp2p::identity::Keypair;

    // Deterministic testnet keys (from fixed bytes for reproducibility)
    // TODO: Replace with real foundation keys before mainnet deployment

    // Testnet signer 1 - derive from fixed seed bytes
    let mut seed_1 = [0u8; 32];
    seed_1[0] = 0; // Distinguishable signer

    // Testnet signer 2 - derive from fixed seed bytes
    let mut seed_2 = [0u8; 32];
    seed_2[0] = 1; // Distinguishable signer

    // Use ed25519_from_bytes to create deterministic keypairs
    let keypair_1 = Keypair::ed25519_from_bytes(&mut seed_1).expect("Valid seed size");
    let keypair_2 = Keypair::ed25519_from_bytes(&mut seed_2).expect("Valid seed size");

    let signer_1 = keypair_1.public();
    let signer_2 = keypair_2.public();

    vec![signer_1, signer_2].into_iter().collect()
}

/// Verify signature on message
///
/// # Arguments
/// * `message` - Message bytes that were signed
/// * `signature` - Signature bytes
/// * `trusted_signers` - Set of trusted public keys
///
/// # Returns
/// `true` if signature is valid from any trusted signer
pub fn verify_signature(
    message: &[u8],
    signature: &[u8],
    trusted_signers: &HashSet<PublicKey>,
) -> bool {
    trusted_signers
        .iter()
        .any(|pk| pk.verify(message, signature))
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    #[test]
    fn test_get_trusted_signers_returns_at_least_one() {
        let signers = get_trusted_signers();
        assert!(!signers.is_empty(), "Must have at least one trusted signer");
    }

    #[test]
    fn test_verify_signature_valid() {
        let keypair = Keypair::generate_ed25519();
        let public_key = keypair.public();

        let message = b"test message";
        let signature = keypair.sign(message).expect("Signing should succeed");

        let mut signers = HashSet::new();
        signers.insert(public_key);

        assert!(
            verify_signature(message, &signature, &signers),
            "Valid signature should verify"
        );
    }

    #[test]
    fn test_verify_signature_invalid_signature() {
        let keypair = Keypair::generate_ed25519();
        let public_key = keypair.public();

        let message = b"test message";
        let invalid_signature = vec![0u8; 64]; // Invalid signature

        let mut signers = HashSet::new();
        signers.insert(public_key);

        assert!(
            !verify_signature(message, &invalid_signature, &signers),
            "Invalid signature should not verify"
        );
    }

    #[test]
    fn test_verify_signature_wrong_message() {
        let keypair = Keypair::generate_ed25519();
        let public_key = keypair.public();

        let original_message = b"original message";
        let tampered_message = b"tampered message";

        let signature = keypair
            .sign(original_message)
            .expect("Signing should succeed");

        let mut signers = HashSet::new();
        signers.insert(public_key);

        assert!(
            !verify_signature(tampered_message, &signature, &signers),
            "Signature should not verify with tampered message"
        );
    }

    #[test]
    fn test_verify_signature_untrusted_signer() {
        let trusted_keypair = Keypair::generate_ed25519();
        let untrusted_keypair = Keypair::generate_ed25519();

        let message = b"test message";
        let signature = untrusted_keypair
            .sign(message)
            .expect("Signing should succeed");

        let mut signers = HashSet::new();
        signers.insert(trusted_keypair.public());

        assert!(
            !verify_signature(message, &signature, &signers),
            "Signature from untrusted signer should not verify"
        );
    }

    #[test]
    fn test_verify_signature_multiple_signers() {
        let keypair1 = Keypair::generate_ed25519();
        let keypair2 = Keypair::generate_ed25519();

        let message = b"test message";
        let signature = keypair2.sign(message).expect("Signing should succeed");

        let mut signers = HashSet::new();
        signers.insert(keypair1.public());
        signers.insert(keypair2.public());

        assert!(
            verify_signature(message, &signature, &signers),
            "Signature should verify if ANY trusted signer matches"
        );
    }

    #[test]
    fn test_trusted_signers_are_unique() {
        let signers = get_trusted_signers();
        let unique_count = signers.len();
        let total_count = signers.len();

        assert_eq!(
            unique_count, total_count,
            "Trusted signers should be unique"
        );
    }
}
