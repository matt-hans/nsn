//! WebRTC certificate management
//!
//! Provides certificate generation, persistence, and loading for WebRTC transport.
//! Certificates must persist across restarts to maintain stable certhash in multiaddrs.

use libp2p_webrtc::tokio::Certificate;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::info;

#[derive(Debug, Error)]
pub enum CertError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Certificate generation failed: {0}")]
    Generation(String),

    #[error("Certificate parse error: {0}")]
    Parse(String),
}

/// Manages WebRTC certificate persistence
///
/// WebRTC transport requires a certificate for DTLS encryption. The certificate
/// fingerprint (certhash) is included in the multiaddr and must remain stable
/// across restarts for browsers to connect.
pub struct CertificateManager {
    cert_path: PathBuf,
}

impl CertificateManager {
    /// Create a new certificate manager
    ///
    /// # Arguments
    /// * `data_dir` - Directory where certificate will be stored
    pub fn new(data_dir: &Path) -> Self {
        Self {
            cert_path: data_dir.join("webrtc_cert.pem"),
        }
    }

    /// Load existing certificate or generate a new one
    ///
    /// If a certificate exists at the configured path, it is loaded.
    /// Otherwise, a new certificate is generated and saved.
    ///
    /// # Returns
    /// The WebRTC certificate
    pub fn load_or_generate(&self) -> Result<Certificate, CertError> {
        if self.cert_path.exists() {
            info!("Loading WebRTC certificate from {:?}", self.cert_path);
            self.load()
        } else {
            info!("Generating new WebRTC certificate at {:?}", self.cert_path);
            self.generate_and_save()
        }
    }

    /// Load certificate from disk
    fn load(&self) -> Result<Certificate, CertError> {
        let pem = fs::read_to_string(&self.cert_path)?;
        Certificate::from_pem(&pem).map_err(|e| CertError::Parse(format!("{:?}", e)))
    }

    /// Generate a new certificate and save to disk
    fn generate_and_save(&self) -> Result<Certificate, CertError> {
        let cert = Certificate::generate(&mut rand::thread_rng())
            .map_err(|e| CertError::Generation(format!("{:?}", e)))?;

        let pem = cert.serialize_pem();

        // Ensure parent directory exists
        if let Some(parent) = self.cert_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&self.cert_path, &pem)?;

        // Set restrictive permissions (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&self.cert_path)?.permissions();
            perms.set_mode(0o600); // Only owner can read/write
            fs::set_permissions(&self.cert_path, perms)?;
        }

        Ok(cert)
    }

    /// Get the path where the certificate is stored
    pub fn cert_path(&self) -> &Path {
        &self.cert_path
    }

    /// Check if a certificate already exists
    pub fn exists(&self) -> bool {
        self.cert_path.exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_and_load_certificate() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let manager = CertificateManager::new(temp_dir.path());

        // First call should generate
        let cert1 = manager.load_or_generate().expect("Failed to generate cert");
        assert!(manager.exists(), "Certificate file should exist");

        // Second call should load existing
        let cert2 = manager.load_or_generate().expect("Failed to load cert");

        // Fingerprints should match (same certificate)
        assert_eq!(
            cert1.fingerprint(),
            cert2.fingerprint(),
            "Loaded certificate should have same fingerprint"
        );
    }

    #[test]
    fn test_certificate_persistence_across_instances() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // First manager generates
        let manager1 = CertificateManager::new(temp_dir.path());
        let cert1 = manager1.load_or_generate().expect("Failed to generate");
        let fingerprint1 = cert1.fingerprint();

        // Second manager loads same certificate
        let manager2 = CertificateManager::new(temp_dir.path());
        let cert2 = manager2.load_or_generate().expect("Failed to load");
        let fingerprint2 = cert2.fingerprint();

        assert_eq!(
            fingerprint1, fingerprint2,
            "Certificate fingerprint should persist across manager instances"
        );
    }

    #[test]
    fn test_load_nonexistent_generates_new() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let manager = CertificateManager::new(temp_dir.path());

        assert!(!manager.exists(), "No certificate should exist initially");

        let cert = manager.load_or_generate().expect("Should generate new cert");
        assert!(manager.exists(), "Certificate should exist after generation");

        // Certificate should be valid (has fingerprint - can debug print)
        let fingerprint = cert.fingerprint();
        // Fingerprint exists and has content (use Debug format since Display not impl)
        let debug_str = format!("{:?}", fingerprint);
        assert!(debug_str.contains("Fingerprint"), "Should have valid fingerprint");
    }

    #[cfg(unix)]
    #[test]
    fn test_certificate_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let manager = CertificateManager::new(temp_dir.path());

        manager.load_or_generate().expect("Failed to generate");

        let metadata = fs::metadata(manager.cert_path()).expect("Failed to get metadata");
        let mode = metadata.permissions().mode();

        assert_eq!(
            mode & 0o777,
            0o600,
            "Certificate file should have restrictive permissions (0o600)"
        );
    }

    #[test]
    fn test_cert_error_display() {
        let err = CertError::Generation("test error".to_string());
        assert!(err.to_string().contains("Certificate generation failed"));

        let err = CertError::Parse("parse error".to_string());
        assert!(err.to_string().contains("Certificate parse error"));
    }
}
