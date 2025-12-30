---
id: T028
title: Local Development Environment with Docker Compose
status: pending
priority: 1
agent: infrastructure
dependencies: []
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [devops, infrastructure, docker, testing, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - Architecture Section 6.1 (Deployment Model)
  - PRD Section 16 (DevOps & Tooling)

est_tokens: 10000
actual_tokens: null
---

## Description

Create a comprehensive local development environment using Docker Compose that enables developers to test the entire NSN stack locally. This environment includes a local NSN Chain node, mock STUN/TURN servers, GPU passthrough for Vortex testing, and pre-configured model weights volume.

This is the foundation for all developer workflows and enables rapid iteration on pallets, off-chain nodes, and AI generation logic without cloud infrastructure costs.

**Technical Approach:**
- Multi-service Docker Compose configuration with GPU support
- Local Substrate node with NSN custom pallets pre-deployed
- Mock STUN/TURN servers for NAT traversal testing
- Shared volume for AI model weights (~15GB)
- Prometheus + Grafana for local observability
- Pre-seeded test accounts with staked tokens

**Integration Points:**
- Used by all developers for local pallet testing
- Required for CI/CD integration test runners
- Foundation for E2E testing (T037)

## Business Context

**User Story:** As a developer, I want to run the entire NSN stack locally, so that I can test changes without deploying to NSN Testnet or requiring expensive cloud infrastructure.

**Why This Matters:**
- Reduces development cycle time from hours (deploy to testnet) to minutes (local testing)
- Enables offline development
- Lowers infrastructure costs during development phase
- Provides consistent development environment across team

**What It Unblocks:**
- All pallet development (T002-T007)
- Off-chain node development (T009-T012)
- Integration testing (T035)
- Developer onboarding

**Priority Justification:** P1 - Critical path blocker for all development work. Without local environment, developers must deploy to ICN Testnet for every code change, slowing iteration by 10-20×.

## Acceptance Criteria

- [ ] `docker-compose.yml` successfully brings up all services with single `docker-compose up` command
- [ ] Local Substrate node starts with NSN pallets pre-deployed and accessible on port 9944
- [ ] Mock STUN/TURN servers respond to ICE negotiation requests
- [ ] GPU passthrough works for Vortex container (verified via `nvidia-smi` inside container)
- [ ] Model weights volume auto-downloads on first startup (or provides clear instructions)
- [ ] Prometheus scrapes metrics from all services (verified in http://localhost:9090)
- [ ] Grafana dashboards load with NSN-specific panels (http://localhost:3000)
- [ ] Test accounts pre-funded with 1000 NSN tokens each
- [ ] All services start within 120 seconds on developer laptop (16GB RAM, RTX 3060)
- [ ] `docker-compose down` cleanly shuts down all services without orphaned processes
- [ ] README.md includes setup, troubleshooting, and common workflows
- [ ] Environment variables documented in `.env.example`

## Test Scenarios

**Test Case 1: Clean First Startup**
- Given: Fresh clone of repository, Docker and Docker Compose installed, NVIDIA drivers 535+
- When: Developer runs `docker-compose up`
- Then: All 7 services start successfully, Substrate node produces blocks, Prometheus shows "Up" for all targets

**Test Case 2: GPU Passthrough Verification**
- Given: Docker Compose environment running
- When: Developer executes `docker-compose exec vortex nvidia-smi`
- Then: GPU is visible with correct VRAM (12GB for RTX 3060), CUDA version matches host

**Test Case 3: Pallet Interaction**
- Given: Local Substrate node running
- When: Developer submits `deposit_stake` extrinsic via Polkadot.js Apps
- Then: Extrinsic succeeds, `StakeDeposited` event emitted, storage updated (verified via RPC)

**Test Case 4: STUN Server Functionality**
- Given: Mock STUN server running on port 3478
- When: Director node attempts NAT discovery
- Then: External IP/port mapping returned successfully

**Test Case 5: Model Weights Volume Mount**
- Given: Model weights downloaded to `./volumes/models/` (or auto-downloaded)
- When: Vortex container starts
- Then: All 5 models load successfully (Flux, LivePortrait, Kokoro, CLIP-B, CLIP-L), VRAM usage <11.8GB

**Test Case 6: Service Restart Persistence**
- Given: Services running, developer stakes tokens and triggers director election
- When: `docker-compose restart substrate-node`
- Then: Chain state persists, staked tokens remain, elected directors unchanged

**Test Case 7: Observability Stack**
- Given: All services running
- When: Developer opens Grafana (http://localhost:3000)
- Then: Pre-configured dashboards show live metrics (block production, P2P peers, VRAM usage)

## Technical Implementation

**Required Components:**

### 1. Docker Compose Configuration
**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  # Local Substrate node with NSN pallets
  substrate-node:
    build:
      context: .
      dockerfile: docker/Dockerfile.substrate-local
    ports:
      - "9944:9944"   # WebSocket RPC
      - "9933:9933"   # HTTP RPC
      - "30333:30333" # P2P
    volumes:
      - substrate-data:/data
    command: >
      --dev
      --ws-external
      --rpc-external
      --rpc-cors all
      --tmp
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9933/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Vortex engine (GPU required)
  vortex:
    build:
      context: .
      dockerfile: docker/Dockerfile.vortex
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - model-weights:/models:ro
      - vortex-output:/output
    ports:
      - "50051:50051"  # gRPC
      - "9101:9100"    # Prometheus metrics
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Mock STUN server
  stun-server:
    image: coturn/coturn:latest
    ports:
      - "3478:3478/udp"
      - "3478:3478/tcp"
    command: >
      -n
      --listening-port=3478
      --fingerprint
      --lt-cred-mech
      --user=nsn:password
      --realm=nsn.local

  # Mock TURN server (fallback)
  turn-server:
    image: coturn/coturn:latest
    ports:
      - "3479:3479/udp"
      - "3479:3479/tcp"
      - "49152-49200:49152-49200/udp"
    command: >
      -n
      --listening-port=3479
      --fingerprint
      --lt-cred-mech
      --user=nsn:password
      --realm=nsn.local
      --relay-ip=127.0.0.1

  # Prometheus
  prometheus:
    image: prom/prometheus:v2.47.0
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=7d'

  # Grafana
  grafana:
    image: grafana/grafana:10.0.0
    ports:
      - "3000:3000"
    volumes:
      - ./docker/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - ./docker/grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer

  # Jaeger (distributed tracing)
  jaeger:
    image: jaegertracing/all-in-one:1.50
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP
    environment:
      - COLLECTOR_OTLP_ENABLED=true

volumes:
  substrate-data:
  model-weights:
  vortex-output:
  prometheus-data:
  grafana-data:
```

### 2. Local Substrate Node Dockerfile
**File:** `docker/Dockerfile.substrate-local`

```dockerfile
FROM paritytech/substrate-relay:latest

WORKDIR /app

# Copy built runtime (assumes cargo build completed)
COPY --from=builder /app/target/release/nsn_runtime.wasm /app/runtime.wasm

# Pre-fund development accounts
COPY docker/chain-spec-dev.json /app/chain-spec.json

EXPOSE 9944 9933 30333

ENTRYPOINT ["/usr/local/bin/substrate"]
```

### 3. Vortex Dockerfile
**File:** `docker/Dockerfile.vortex`

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY vortex/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY vortex/ ./vortex/

# Download models on build (or provide init script)
RUN python3 vortex/scripts/download_models.py --output /models

EXPOSE 50051 9100

CMD ["python3", "vortex/main.py", "--models-path", "/models"]
```

### 4. Prometheus Configuration
**File:** `docker/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'substrate-node'
    static_configs:
      - targets: ['substrate-node:9615']

  - job_name: 'vortex'
    static_configs:
      - targets: ['vortex:9100']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

### 5. Grafana Dashboard Provisioning
**File:** `docker/grafana/provisioning/dashboards/icn-local.yaml`

```yaml
apiVersion: 1

providers:
  - name: 'NSN Local Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /var/lib/grafana/dashboards
```

### 6. Environment Configuration
**File:** `.env.example`

```bash
# Substrate Node
SUBSTRATE_WS_URL=ws://localhost:9944
SUBSTRATE_LOG_LEVEL=info

# Vortex Engine
VORTEX_GRPC_PORT=50051
VORTEX_MODELS_PATH=/models
VORTEX_MAX_VRAM_GB=11.8

# STUN/TURN
STUN_SERVER=stun://localhost:3478
TURN_SERVER=turn://localhost:3479
TURN_USER=nsn
TURN_PASSWORD=password

# Observability
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
JAEGER_URL=http://localhost:16686

# Development Accounts (pre-funded)
ALICE_SEED=//Alice
BOB_SEED=//Bob
CHARLIE_SEED=//Charlie
```

### 7. Developer README
**File:** `docs/local-development.md`

```markdown
# Local Development Environment

## Prerequisites

- Docker 24.0+ with Docker Compose
- NVIDIA GPU with 12GB+ VRAM
- NVIDIA Container Toolkit
- 50GB free disk space (for model weights)

## Quick Start

1. Clone repository:
   ```bash
   git clone <repo-url>
   cd interdim-cable
   ```

2. Copy environment template:
   ```bash
   cp .env.example .env
   ```

3. Start services:
   ```bash
   docker-compose up
   ```

4. Wait for all services to be healthy (~120s)

5. Access services:
   - Substrate RPC: ws://localhost:9944
   - Polkadot.js Apps: https://polkadot.js.org/apps/?rpc=ws://localhost:9944
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)
   - Jaeger: http://localhost:16686

## Model Weights Setup

Option 1: Auto-download (first startup)
- Vortex container will download ~15GB of models on first run
- Takes 10-30 minutes depending on connection

Option 2: Manual download
```bash
mkdir -p volumes/models
python3 vortex/scripts/download_models.py --output volumes/models
```

## Common Workflows

### Test a Pallet Change
```bash
# 1. Edit pallet code
vim pallets/nsn-stake/src/lib.rs

# 2. Rebuild and restart Substrate node
docker-compose build substrate-node
docker-compose restart substrate-node

# 3. Verify via Polkadot.js Apps
```

### Test Vortex Generation
```bash
# 1. Exec into Vortex container
docker-compose exec vortex bash

# 2. Run test generation
python3 -c "from vortex.pipeline import VortexPipeline; p = VortexPipeline(); print('VRAM:', torch.cuda.memory_allocated() / 1e9, 'GB')"
```

### Reset Chain State
```bash
docker-compose down -v  # Warning: deletes all volumes
docker-compose up
```

## Troubleshooting

### GPU Not Visible
```bash
# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi

# Check Docker Compose GPU config
docker-compose config | grep -A5 nvidia
```

### Out of VRAM
- Ensure no other GPU processes running
- Check VRAM usage: `nvidia-smi`
- Reduce model precision in vortex config

### Substrate Node Not Producing Blocks
- Check logs: `docker-compose logs substrate-node`
- Verify ports not in use: `lsof -i :9944`
- Try clean restart: `docker-compose down && docker-compose up`
```

### Validation Commands

```bash
# Verify all services running
docker-compose ps | grep "Up"

# Check Substrate node health
curl http://localhost:9933/health | jq .

# Verify GPU passthrough
docker-compose exec vortex nvidia-smi

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health == "up")'

# Test STUN server
docker run --rm --network host alpine nc -u localhost 3478

# Verify model weights loaded
docker-compose exec vortex ls -lh /models
```

## Dependencies

**Hard Dependencies** (must be complete first):
- None (this is foundation task)

**Soft Dependencies** (nice to have):
- T001 (ICN Chain Bootstrap) - provides pallet code to deploy

**External Dependencies:**
- Docker 24.0+ with Compose plugin
- NVIDIA Container Toolkit 1.14+
- NVIDIA drivers 535+
- 50GB free disk space
- Internet connection for model downloads

## Design Decisions

**Decision 1: Docker Compose over Kubernetes**
- **Rationale:** Simpler for local development, lower resource overhead, easier developer onboarding
- **Alternatives:** Minikube, kind, Docker Swarm
- **Trade-offs:** (+) Easy setup, fast iteration. (-) Doesn't match production K8s exactly, limited scaling

**Decision 2: GPU Passthrough via nvidia-docker2**
- **Rationale:** Industry standard for GPU containers, well-documented, supports all CUDA features
- **Alternatives:** Singularity, Podman with NVIDIA CDI
- **Trade-offs:** (+) Mature, widely supported. (-) Requires NVIDIA Container Toolkit installation

**Decision 3: Mock STUN/TURN Servers**
- **Rationale:** Enables NAT traversal testing without internet dependency or external services
- **Alternatives:** Use public STUN servers (stun.l.google.com), run actual coturn with credentials
- **Trade-offs:** (+) Offline testing, predictable behavior. (-) Doesn't test real-world NAT scenarios

**Decision 4: Substrate --dev Mode**
- **Rationale:** Instant block finality, pre-funded accounts, no consensus needed for local testing
- **Alternatives:** --local mode (with BABE/GRANDPA), full parachain setup
- **Trade-offs:** (+) Fast iteration, simple. (-) Doesn't test consensus edge cases

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Model download fails (15GB) | High | Medium | Provide torrent/CDN fallback, checksum verification |
| GPU drivers incompatible | High | Low | Document tested driver versions (535, 545, 550) |
| VRAM OOM on startup | Medium | Medium | Provide lower-precision config option (INT8 for all) |
| Docker Compose version mismatch | Medium | Medium | Pin to specific version in docs, add version check script |
| Services fail to start in 120s | Low | Medium | Add startup probes, increase timeout in docs |

**Specific Mitigations:**

**Model Download Reliability:**
```python
# vortex/scripts/download_models.py
def download_with_retry(url, output_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            # Verify checksum
            if verify_checksum(output_path):
                return True
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

**GPU Driver Compatibility Check:**
```bash
# scripts/check-gpu.sh
#!/bin/bash
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
MAJOR_VERSION=${DRIVER_VERSION%%.*}

if [ "$MAJOR_VERSION" -lt 535 ]; then
    echo "ERROR: NVIDIA driver $DRIVER_VERSION too old. Minimum: 535.x"
    exit 1
fi
```

## Progress Log

### [2025-12-24] - Task Created

**Created By:** task-creator agent
**Reason:** Enable local development environment for all developers
**Dependencies:** None (foundation task)
**Estimated Complexity:** Standard (multi-service Docker setup, GPU config, observability stack)

## Completion Checklist

### Code Complete
- [ ] `docker-compose.yml` with all 7 services defined
- [ ] Substrate node Dockerfile with dev chain spec
- [ ] Vortex Dockerfile with GPU support and model downloads
- [ ] Prometheus config scraping all services
- [ ] Grafana dashboard provisioning with NSN panels
- [ ] Mock STUN/TURN server configuration
- [ ] `.env.example` with all required variables
- [ ] `docs/local-development.md` with setup and troubleshooting

### Testing
- [ ] Clean startup on fresh Ubuntu 22.04 system
- [ ] All services healthy within 120s
- [ ] GPU visible in Vortex container
- [ ] Substrate node produces blocks
- [ ] Prometheus shows all targets "Up"
- [ ] Grafana dashboards load with live data
- [ ] Model weights download successfully (or manual setup works)
- [ ] Test accounts pre-funded with 1000 NSN

### Documentation
- [ ] README includes prerequisites
- [ ] Common workflows documented
- [ ] Troubleshooting section complete
- [ ] GPU setup instructions clear
- [ ] Model download options explained

### DevOps
- [ ] All Dockerfiles use multi-stage builds where appropriate
- [ ] Health checks defined for all services
- [ ] Volumes persist data across restarts
- [ ] Graceful shutdown tested (`docker-compose down`)
- [ ] Resource limits documented

**Definition of Done:**
Task is complete when a new developer can clone the repository, run `docker-compose up`, and have a fully functional NSN development environment within 5 minutes (excluding model download time), with all services healthy and test accounts ready for pallet interaction.
