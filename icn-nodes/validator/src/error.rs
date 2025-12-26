use thiserror::Error;

/// Errors that can occur in the validator node
#[derive(Error, Debug)]
pub enum ValidatorError {
    #[error("CLIP engine error: {0}")]
    ClipEngine(String),

    #[error("ONNX model loading failed: {0}")]
    ModelLoad(String),

    #[error("ONNX inference failed: {0}")]
    Inference(String),

    #[error("Video decoding error: {0}")]
    VideoDecode(String),

    #[error("Frame extraction failed: {0}")]
    FrameExtraction(String),

    #[error("Attestation signing failed: {0}")]
    AttestationSigning(String),

    #[error("Attestation verification failed: {0}")]
    AttestationVerification(String),

    #[error("P2P service error: {0}")]
    P2PService(String),

    #[error("Chain client error: {0}")]
    ChainClient(String),

    #[error("Challenge monitor error: {0}")]
    ChallengeMonitor(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Metrics error: {0}")]
    Metrics(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("TOML parsing error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("Timeout error: operation exceeded {0}s")]
    Timeout(u64),

    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),

    #[error("Invalid CLIP score: {0} (must be in range [0.0, 1.0])")]
    InvalidScore(f32),
}

pub type Result<T> = std::result::Result<T, ValidatorError>;
