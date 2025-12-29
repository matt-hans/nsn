# ICN Validator Node

CLIP-based semantic verification node for the Interdimensional Cable Network.

## Overview

The Validator Node performs semantic verification of director-generated content using CLIP (Contrastive Language-Image Pretraining) models. Validators ensure content quality without trusting directors, providing an independent verification layer.

### Core Functions

1. **Content Verification**: Run CLIP-ViT-B-32 + CLIP-ViT-L-14 dual ensemble on video frames
2. **Attestation Generation**: Sign verification results with Ed25519 keypair
3. **Challenge Participation**: Provide attestations during BFT dispute resolution

## Requirements

- **Hardware**: CPU-only, no GPU required
- **Stake**: Minimum 10 ICN tokens
- **Network**: Stable internet connection for P2P networking

## Installation

```bash
# Build from source
cargo build --release -p icn-validator

# Binary location
./target/release/icn-validator
```

## Configuration

Create `config/validator.toml`:

```toml
chain_endpoint = "ws://localhost:9944"
keypair_path = "keys/validator.json"
models_dir = "models/clip"

[clip]
model_b32_path = "clip-vit-b-32.onnx"
model_l14_path = "clip-vit-l-14.onnx"
b32_weight = 0.4
l14_weight = 0.6
threshold = 0.75
keyframe_count = 5
inference_timeout_secs = 5

[p2p]
listen_addresses = ["/ip4/0.0.0.0/tcp/0"]
bootstrap_peers = []
max_peers = 50

[metrics]
listen_address = "0.0.0.0"
port = 9101

[challenge]
enabled = true
response_buffer_blocks = 40
poll_interval_secs = 6
```

## CLIP Models

Download ONNX models:

```bash
mkdir -p models/clip
# TODO: Add model download links when available
# These models will be ~500MB each
```

## Usage

```bash
# Start validator with config file
./icn-validator --config config/validator.toml

# Override chain endpoint
./icn-validator --chain-endpoint ws://testnet.icn.network:9944

# Override models directory
./icn-validator --models-dir /path/to/models

# Enable verbose logging
./icn-validator --verbose
```

## Metrics

Prometheus metrics available at `http://localhost:9101/metrics`:

- `icn_validator_validations_total` - Total validations performed
- `icn_validator_attestations_total` - Total attestations broadcast
- `icn_validator_challenges_total` - Total challenges participated in
- `icn_validator_clip_score` - Distribution of CLIP scores
- `icn_validator_validation_duration_seconds` - Validation latency
- `icn_validator_clip_inference_duration_seconds` - CLIP inference latency
- `icn_validator_connected_peers` - Current P2P peer count

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              Validator Node (Tokio Runtime)           │
├──────────────────────────────────────────────────────┤
│                                                        │
│  ┌─────────────┐    ┌─────────────┐   ┌───────────┐ │
│  │   Config    │───▶│    Main     │───▶│  Metrics  │ │
│  │   Loader    │    │   Runtime   │    │  Server   │ │
│  └─────────────┘    └─────────────┘   └───────────┘ │
│                            │                          │
│         ┌──────────────────┼──────────────────┐      │
│         ▼                  ▼                  ▼      │
│  ┌─────────────┐    ┌─────────────┐   ┌────────────┐│
│  │    Chain    │    │    CLIP     │   │    P2P     ││
│  │   Client    │    │   Engine    │   │  Service   ││
│  │  (subxt)    │    │  (ONNX RT)  │   │ (libp2p)   ││
│  └─────────────┘    └─────────────┘   └────────────┘│
│         │                  │                  │      │
│         ▼                  ▼                  ▼      │
│  ┌─────────────┐    ┌─────────────┐   ┌────────────┐│
│  │  Challenge  │    │    Video    │   │Attestation ││
│  │   Monitor   │    │   Decoder   │   │  Signer    ││
│  └─────────────┘    └─────────────┘   └────────────┘│
│                                                        │
└──────────────────────────────────────────────────────┘
```

## Development

```bash
# Run unit tests
cargo test -p icn-validator --lib

# Run with test coverage
cargo tarpaulin -p icn-validator

# Run clippy
cargo clippy -p icn-validator -- -D warnings

# Format code
cargo fmt -p icn-validator
```

## Testing

33 unit tests covering:
- Attestation signing and verification
- CLIP score computation and ensemble
- Video frame extraction
- Configuration validation
- Challenge monitoring
- P2P networking

## Deployment Status

**Current Status**: STUB IMPLEMENTATION

The validator node compiles and has a complete API, but requires:
- **ONNX models**: CLIP-ViT-B-32 and CLIP-ViT-L-14 in ONNX format
- **libp2p integration**: Full GossipSub implementation
- **subxt integration**: ICN Chain metadata and event subscriptions
- **ffmpeg integration**: Real video decoding (currently stubbed)

These will be completed when:
1. CLIP models are converted to ONNX format
2. ICN Chain is deployed to testnet
3. Full P2P mesh is operational

## License

MIT

## See Also

- [ICN Architecture](../../.claude/rules/architecture.md)
- [ICN PRD](../../.claude/rules/prd.md)
- [Task Specification](../../.tasks/tasks/T010-validator-node.md)
