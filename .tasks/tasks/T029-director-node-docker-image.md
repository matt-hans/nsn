---
id: T029
title: Production Director Node Docker Image
status: pending
priority: 2
agent: infrastructure
dependencies: [T009, T028]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [devops, docker, infrastructure, director, production, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - Architecture Section 6.1 (Deployment Architecture)
  - PRD Section 16 (DevOps & Tooling)

est_tokens: 11000
actual_tokens: null
---

## Description

Build a production-ready Docker image for NSN Director nodes that includes the Rust binary (core runtime), Python sidecar (Vortex engine), pre-loaded AI model weights, and all necessary dependencies. The image must support GPU passthrough for CUDA operations and expose required ports for P2P networking, metrics, and gRPC communication.

This image is the deployable artifact for all Director nodes on NSN Testnet/NSN Chain and must be optimized for size, startup time, and runtime performance.

**Technical Approach:**
- Multi-stage Docker build to minimize final image size
- Ubuntu 22.04 base with NVIDIA Driver 535+ support
- Rust binary built in release mode with link-time optimization
- Python 3.11 with PyTorch CUDA support
- Model weights baked into image or mounted as volume
- Health check endpoint for Kubernetes/orchestration
- Non-root user for security

**Integration Points:**
- Deployed by Kubernetes (T030) or bare metal operators
- Connects to NSN Chain RPC endpoints
- Publishes to P2P mesh (libp2p)
- Exposes metrics to Prometheus (T033)

## Business Context

**User Story:** As a node operator, I want a single Docker image that runs a complete Director node, so that I can deploy NSN infrastructure without complex multi-step setup.

**Why This Matters:**
- Simplifies deployment for community node operators
- Ensures consistent runtime environment across all Directors
- Enables automated scaling via Kubernetes
- Reduces support burden (single artifact vs. multi-component setup)

**What It Unblocks:**
- NSN Testnet deployment (end of Phase 1)
- Community node operator onboarding
- Kubernetes orchestration (T030)
- Production mainnet launch (Phase 2)

**Priority Justification:** P2 - Critical for deployment but depends on Director node implementation (T009). Must be ready before NSN Testnet deployment in Week 8.

## Acceptance Criteria

- [ ] Multi-stage Dockerfile builds successfully with `docker build -t nsn-director:latest .`
- [ ] Final image size <10GB (excluding model weights if mounted separately)
- [ ] Image startup time <60s from cold start to healthy P2P connections
- [ ] GPU visible inside container (`nvidia-smi` works)
- [ ] All 5 AI models load successfully (Flux, LivePortrait, Kokoro, CLIP-B, CLIP-L)
- [ ] VRAM usage <11.8GB after full model loading
- [ ] Rust binary runs as non-root user (UID 1000)
- [ ] Health check endpoint responds on `/health` (port 9100)
- [ ] P2P port (9000) accepts connections
- [ ] gRPC server (50051) responds to BFT coordination requests
- [ ] Prometheus metrics exposed on port 9100 with all NSN-specific metrics
- [ ] Image published to GitHub Container Registry (`ghcr.io/nsn/director:latest`)
- [ ] Multi-architecture support (amd64, arm64 for future Apple Silicon support)
- [ ] Security scanning passes (no critical CVEs)

## Test Scenarios

**Test Case 1: Clean Build and Startup**
- Given: Clean Docker environment, Dockerfile in repository root
- When: Developer runs `docker build -t nsn-director:latest . && docker run --gpus all -p 9000:9000 nsn-director:latest`
- Then: Container starts, models load, P2P service binds to port 9000, health check returns 200 within 60s

**Test Case 2: GPU Passthrough Verification**
- Given: Container running with `--gpus all` flag
- When: Exec into container: `docker exec <container_id> nvidia-smi`
- Then: GPU device visible, driver version ≥535, CUDA runtime version matches PyTorch requirements

**Test Case 3: Model Loading**
- Given: Container starting for first time
- When: Monitor logs: `docker logs -f <container_id>`
- Then: See log entries: "Loading Flux-Schnell...", "Loading LivePortrait...", "Loading Kokoro-82M...", "Loading CLIP-ViT-B-32...", "Loading CLIP-ViT-L-14...", "All models loaded. VRAM: 11.7 GB"

**Test Case 4: P2P Connectivity**
- Given: Two Director containers running on same network
- When: Second container starts and attempts peer discovery
- Then: Containers exchange PeerInfo via Kademlia DHT, establish QUIC connection, subscribe to GossipSub topics

**Test Case 5: Metrics Endpoint**
- Given: Container running
- When: `curl http://localhost:9100/metrics`
- Then: Response includes Prometheus-format metrics: `nsn_vortex_vram_usage_bytes`, `nsn_p2p_connected_peers`, `nsn_bft_round_duration_seconds`

**Test Case 6: Graceful Shutdown**
- Given: Container running with active P2P connections
- When: `docker stop <container_id>` (SIGTERM)
- Then: Node announces departure on P2P network, closes gRPC server, flushes metrics, exits cleanly within 30s

**Test Case 7: Resource Limits**
- Given: Container running with `--memory=16g --cpus=4` limits
- When: Vortex generates 5 consecutive slots
- Then: No OOM kills, CPU stays <100% per core average, VRAM stable at ~11.7GB

**Test Case 8: Image Security Scan**
- Given: Built image `nsn-director:latest`
- When: Run `docker scan nsn-director:latest` or Trivy scan
- Then: No critical CVEs reported, all base packages up-to-date

## Technical Implementation

**Required Components:**

### 1. Multi-Stage Dockerfile
**File:** `Dockerfile.director`

```dockerfile
# ============================================================================
# Stage 1: Rust Builder
# ============================================================================
FROM rust:1.75-slim as rust-builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    clang \
    libssl-dev \
    pkg-config \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifests
COPY Cargo.toml Cargo.lock ./
COPY pallets ./pallets
COPY off-chain ./off-chain

# Build in release mode with LTO
ENV CARGO_PROFILE_RELEASE_LTO=true
ENV CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1

RUN cargo build --release -p nsn-director

# ============================================================================
# Stage 2: Python Model Downloader
# ============================================================================
FROM python:3.11-slim as model-downloader

WORKDIR /models

# Install download script dependencies
RUN pip install --no-cache-dir requests tqdm huggingface_hub

COPY vortex/scripts/download_models.py .

# Download all models (~15GB)
RUN python download_models.py --output /models --verify-checksums

# ============================================================================
# Stage 3: Final Runtime Image
# ============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash nsn

WORKDIR /app

# Copy Rust binary from builder
COPY --from=rust-builder --chown=nsn:nsn /build/target/release/nsn-director /app/nsn-director

# Copy Vortex Python code
COPY --chown=nsn:nsn vortex ./vortex

# Install Python dependencies
RUN pip3 install --no-cache-dir -r vortex/requirements.txt

# Copy pre-downloaded models (or mount as volume in production)
COPY --from=model-downloader --chown=nsn:nsn /models /models

# Expose ports
EXPOSE 9000   # P2P (libp2p)
EXPOSE 9100   # Prometheus metrics + health check
EXPOSE 50051  # gRPC (BFT coordination)

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:9100/health || exit 1

# Switch to non-root user
USER nsn

# Environment variables (override via docker run -e)
ENV RUST_LOG=info
ENV SUBSTRATE_WS_URL=ws://localhost:9944
ENV MODELS_PATH=/models
ENV P2P_PORT=9000
ENV METRICS_PORT=9100
ENV GRPC_PORT=50051

# Entrypoint
ENTRYPOINT ["/app/nsn-director"]
CMD ["--p2p-port", "9000", "--metrics-port", "9100", "--grpc-port", "50051"]
```

### 2. Model Download Script
**File:** `vortex/scripts/download_models.py`

```python
#!/usr/bin/env python3
"""
Download NSN Vortex AI models from Hugging Face Hub with checksum verification.
"""
import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict

import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Model definitions with expected checksums (SHA256)
MODELS = {
    "flux-schnell-nf4": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "files": [
            "flux1-schnell-nf4.safetensors",
            "model_index.json",
        ],
        "checksums": {
            "flux1-schnell-nf4.safetensors": "abc123...",  # Replace with actual
        },
    },
    "live-portrait-fp16": {
        "repo": "KwaiVGI/LivePortrait",
        "files": [
            "liveportrait_fp16.onnx",
            "config.yaml",
        ],
        "checksums": {
            "liveportrait_fp16.onnx": "def456...",
        },
    },
    "kokoro-82m": {
        "repo": "hexgrad/Kokoro-82M",
        "files": [
            "kokoro-v0_19.pth",
            "voices.json",
        ],
        "checksums": {
            "kokoro-v0_19.pth": "ghi789...",
        },
    },
    "clip-vit-b-32": {
        "repo": "openai/clip-vit-base-patch32",
        "files": [
            "model.safetensors",
        ],
        "checksums": {
            "model.safetensors": "jkl012...",
        },
    },
    "clip-vit-l-14": {
        "repo": "openai/clip-vit-large-patch14",
        "files": [
            "model.safetensors",
        ],
        "checksums": {
            "model.safetensors": "mno345...",
        },
    },
}


def verify_checksum(file_path: Path, expected_hash: str) -> bool:
    """Verify file SHA256 checksum."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest()
    return actual_hash == expected_hash


def download_models(output_dir: Path, verify: bool = True):
    """Download all models to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, config in MODELS.items():
        print(f"\n📦 Downloading {model_name}...")
        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)

        for file_name in config["files"]:
            file_path = model_dir / file_name

            if file_path.exists() and verify:
                expected_hash = config["checksums"].get(file_name)
                if expected_hash and verify_checksum(file_path, expected_hash):
                    print(f"  ✓ {file_name} (cached, checksum valid)")
                    continue

            print(f"  ⬇ {file_name}...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=config["repo"],
                    filename=file_name,
                    cache_dir=str(model_dir),
                    force_download=False,
                )

                # Verify checksum
                if verify and file_name in config["checksums"]:
                    expected_hash = config["checksums"][file_name]
                    if not verify_checksum(Path(downloaded_path), expected_hash):
                        raise ValueError(f"Checksum mismatch for {file_name}")
                    print(f"  ✓ {file_name} (checksum verified)")
                else:
                    print(f"  ✓ {file_name} (downloaded)")

            except Exception as e:
                print(f"  ✗ Failed to download {file_name}: {e}")
                raise

    print("\n✅ All models downloaded successfully!")
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"   Total size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NSN Vortex models")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--verify-checksums", action="store_true", help="Verify checksums")
    args = parser.parse_args()

    download_models(args.output, verify=args.verify_checksums)
```

### 3. Health Check Endpoint
**File:** `nsn-nodes/director/src/health.rs`

```rust
use axum::{response::Json, routing::get, Router};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct HealthState {
    pub p2p_connected: Arc<RwLock<bool>>,
    pub models_loaded: Arc<RwLock<bool>>,
    pub chain_synced: Arc<RwLock<bool>>,
}

pub async fn health_handler(state: axum::extract::State<HealthState>) -> Json<Value> {
    let p2p_ok = *state.p2p_connected.read().await;
    let models_ok = *state.models_loaded.read().await;
    let chain_ok = *state.chain_synced.read().await;

    let healthy = p2p_ok && models_ok && chain_ok;

    Json(json!({
        "status": if healthy { "healthy" } else { "unhealthy" },
        "checks": {
            "p2p_connected": p2p_ok,
            "models_loaded": models_ok,
            "chain_synced": chain_ok,
        },
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

pub fn health_router(state: HealthState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .with_state(state)
}
```

### 4. Docker Build Script
**File:** `scripts/build-director-image.sh`

```bash
#!/bin/bash
set -euo pipefail

VERSION=${1:-latest}
REGISTRY=${REGISTRY:-ghcr.io/nsn}

echo "Building Director image version: $VERSION"

# Multi-architecture build
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file Dockerfile.director \
  --tag $REGISTRY/director:$VERSION \
  --tag $REGISTRY/director:latest \
  --push \
  .

echo "✅ Image published to $REGISTRY/director:$VERSION"
```

### 5. GitHub Actions Build Workflow
**File:** `.github/workflows/docker-director.yml`

```yaml
name: Build Director Docker Image

on:
  push:
    branches: [main, develop]
    tags: ['v*']
  pull_request:
    paths:
      - 'Dockerfile.director'
      - 'nsn-nodes/director/**'
      - 'vortex/**'

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/director

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.director
          platforms: linux/amd64,linux/arm64
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Scan image for vulnerabilities
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ steps.meta.outputs.version }}
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

### Validation Commands

```bash
# Build image locally
docker build -t nsn-director:test -f Dockerfile.director .

# Run with GPU passthrough
docker run --rm --gpus all \
  -p 9000:9000 \
  -p 9100:9100 \
  -p 50051:50051 \
  -e SUBSTRATE_WS_URL=ws://localhost:9944 \
  nsn-director:test

# Test health check
curl http://localhost:9100/health | jq .

# Verify GPU access
docker exec -it <container_id> nvidia-smi

# Check VRAM usage
docker exec <container_id> python3 -c "import torch; print(f'VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB')"

# Scan for vulnerabilities
docker scan nsn-director:test

# Test multi-architecture build
docker buildx build --platform linux/amd64,linux/arm64 -t nsn-director:test -f Dockerfile.director .
```

## Dependencies

**Hard Dependencies** (must be complete first):
- [T009] Director Node Implementation - provides Rust binary to package
- [T028] Local Dev Environment - provides Docker Compose testing framework

**Soft Dependencies** (nice to have):
- [T010] Vortex Engine - Python sidecar code
- [T033] Observability Stack - metrics endpoint specification

**External Dependencies:**
- Docker 24.0+ with BuildKit
- Docker Buildx for multi-architecture builds
- NVIDIA Container Toolkit
- GitHub Container Registry access
- Hugging Face Hub access for model downloads

## Design Decisions

**Decision 1: Multi-Stage Build vs. Single Monolithic Image**
- **Rationale:** Multi-stage build reduces final image size by 60-70% (only runtime deps in final layer), improves build caching, separates build-time from runtime security posture
- **Alternatives:** Single Dockerfile with all build tools, separate build/runtime images
- **Trade-offs:** (+) Smaller image, faster pulls, better security. (-) More complex Dockerfile, longer build times

**Decision 2: Bake Models into Image vs. Volume Mount**
- **Rationale:** Baking models ensures reproducibility, eliminates download failures at runtime, simplifies deployment. Volume mount allows model updates without image rebuilds.
- **Alternatives:** Always download at runtime, hybrid (base models baked, updates mounted)
- **Trade-offs:** (+) Fast startup, offline deployment. (-) Large image (10GB+), model updates require rebuild

**Decision 3: Ubuntu 22.04 vs. Alpine Linux**
- **Rationale:** Ubuntu has better CUDA support, more compatible with NVIDIA runtime, easier debugging (standard GNU tools)
- **Alternatives:** Alpine (smaller), Debian Slim, CentOS
- **Trade-offs:** (+) CUDA compatibility, package availability. (-) Larger base image (~200MB vs 5MB Alpine)

**Decision 4: Non-Root User (UID 1000)**
- **Rationale:** Security best practice, required by many Kubernetes security policies, limits blast radius of container escape
- **Alternatives:** Root user, dynamically assigned UID
- **Trade-offs:** (+) Security, compliance. (-) Slightly more complex file permissions

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Model download fails during build | High | Medium | Pre-cache models in CI/CD, provide fallback CDN, implement retry logic |
| Image size exceeds 10GB | Medium | Medium | Aggressive layer caching, squash final image, consider model volume mount option |
| CUDA version mismatch | High | Low | Pin NVIDIA base image version, test on target GPU hardware, document supported CUDA versions |
| Build time >30 minutes | Medium | High | Use BuildKit caching, parallelize model downloads, build only on code changes |
| Security CVE in base image | High | Medium | Automated Trivy scans in CI/CD, subscribe to NVIDIA security advisories, monthly rebuild cadence |

**Specific Mitigations:**

**Build Time Optimization:**
```dockerfile
# Optimize layer caching - copy only manifests first
COPY Cargo.toml Cargo.lock ./
RUN cargo fetch  # Pre-download dependencies (cached)

# Then copy source and build
COPY off-chain ./off-chain
RUN cargo build --release
```

**Model Download Resilience:**
```python
# In download_models.py
def download_with_retry(url, output_path, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            # ... download logic ...
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            backoff = 2 ** attempt
            print(f"Retry {attempt+1}/{max_retries} in {backoff}s...")
            time.sleep(backoff)
```

## Progress Log

### [2025-12-24] - Task Created

**Created By:** task-creator agent
**Reason:** Provide production-ready deployment artifact for Director nodes
**Dependencies:** T009 (Director Node Implementation), T028 (Local Dev Environment)
**Estimated Complexity:** Standard (multi-stage Docker build with GPU support, model packaging, CI/CD integration)

## Completion Checklist

### Code Complete
- [ ] `Dockerfile.director` with multi-stage build
- [ ] Model download script with checksum verification
- [ ] Health check endpoint in Rust binary
- [ ] Docker build script (`scripts/build-director-image.sh`)
- [ ] GitHub Actions workflow (`.github/workflows/docker-director.yml`)
- [ ] `.dockerignore` to exclude unnecessary files
- [ ] Documentation in `docs/deployment/director-node.md`

### Testing
- [ ] Local build succeeds: `docker build -t nsn-director:test .`
- [ ] Container starts and becomes healthy within 60s
- [ ] GPU visible: `nvidia-smi` works inside container
- [ ] All models load, VRAM <11.8GB
- [ ] Health check returns 200
- [ ] P2P service binds to port 9000
- [ ] Metrics endpoint exposes all NSN metrics
- [ ] Graceful shutdown on SIGTERM
- [ ] Multi-architecture build succeeds (amd64, arm64)
- [ ] Trivy scan shows no critical CVEs

### Documentation
- [ ] Deployment guide includes image pull commands
- [ ] Environment variables documented
- [ ] GPU requirements specified
- [ ] Troubleshooting section for common issues
- [ ] Volume mount options explained

### DevOps
- [ ] Image published to GitHub Container Registry
- [ ] CI/CD builds on every push to main/develop
- [ ] Automated security scanning enabled
- [ ] Image tagged with version, commit SHA, branch
- [ ] Build cache optimization configured

**Definition of Done:**
Task is complete when a node operator can run `docker pull ghcr.io/nsn/director:latest && docker run --gpus all -p 9000:9000 ghcr.io/nsn/director:latest`, and within 60 seconds have a fully functional Director node connected to NSN Chain mainnet, with all AI models loaded and P2P networking active.
