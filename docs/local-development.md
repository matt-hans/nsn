# NSN Local Development Environment

Complete guide for setting up and using the NSN local development environment with Docker Compose.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Service Overview](#service-overview)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)
- [Performance Tuning](#performance-tuning)

## Prerequisites

### Required Software

- **Docker 24.0+** with Docker Compose plugin
- **NVIDIA GPU** with 12GB+ VRAM (RTX 3060 or better)
- **NVIDIA drivers** 535+
- **NVIDIA Container Toolkit** 1.14+
- **50GB+ free disk space** (for model weights and containers)
- **16GB+ system RAM** (recommended)

### System Requirements by Component

| Component | CPU | RAM | Disk | GPU |
|-----------|-----|-----|------|-----|
| Substrate Node | 2 cores | 4GB | 10GB | - |
| Vortex Engine | 4 cores | 8GB | 20GB | RTX 3060+ (12GB VRAM) |
| Observability Stack | 2 cores | 4GB | 5GB | - |
| STUN/TURN Servers | 1 core | 512MB | 100MB | - |

### Supported Operating Systems

- **Ubuntu 22.04 LTS** (recommended)
- **Debian 12**
- **Fedora 38+**
- **macOS** (Intel/Apple Silicon with Docker Desktop, no GPU support)

## Quick Start

### 1. Verify GPU Compatibility

Run the GPU check script to ensure your system is ready:

```bash
./scripts/check-gpu.sh
```

Expected output:
```
==========================================
NSN GPU Compatibility Check
==========================================

1. Checking for NVIDIA GPU...
NVIDIA GeForce RTX 3060, 12288 MiB, 535.146.02
✓ NVIDIA GPU detected

2. Checking NVIDIA driver version...
   Detected driver version: 535.146.02
✓ Driver version is compatible (>= 535)

3. Checking GPU VRAM...
   Total VRAM: 12 GB
✓ VRAM meets requirements (>= 12 GB)

4. Checking Docker installation...
   Docker version 24.0.7, build afdd53b
✓ Docker is installed

5. Checking Docker Compose...
   Docker Compose version v2.23.3
✓ Docker Compose is available

6. Checking NVIDIA Container Toolkit...
✓ NVIDIA Container Toolkit is properly configured
   GPU passthrough to containers is working.

7. Checking available disk space...
   Available space: 120 GB
✓ Sufficient disk space available

==========================================
Summary
==========================================
✓ All checks passed!
```

### 2. Configure Environment

Copy the environment template and customize if needed:

```bash
cp .env.example .env
```

Default configuration is suitable for most development setups. Key variables:

```bash
# Substrate node ports
SUBSTRATE_WS_URL=ws://localhost:9944
SUBSTRATE_HTTP_URL=http://localhost:9933

# Vortex configuration
VORTEX_MAX_VRAM_GB=11.8

# Observability
GRAFANA_ADMIN_PASSWORD=admin  # Change for production!
```

### 3. Start All Services

Start the entire NSN stack with a single command:

```bash
docker compose up
```

Or run in detached mode:

```bash
docker compose up -d
```

**First-time startup:**
- Model download: ~10-30 minutes (15GB)
- Container builds: ~5-10 minutes
- Total time: ~40 minutes

**Subsequent startups:**
- All services ready: ~60-120 seconds

### 4. Verify Services

Check that all services are healthy:

```bash
docker compose ps
```

Expected output:
```
NAME                    STATUS              PORTS
nsn-substrate-node      running (healthy)   0.0.0.0:9933->9933/tcp, 0.0.0.0:9944->9944/tcp
nsn-vortex              running (healthy)   0.0.0.0:50051->50051/tcp
nsn-stun                running             0.0.0.0:3478->3478/tcp
nsn-turn                running             0.0.0.0:3479->3479/tcp
nsn-prometheus          running (healthy)   0.0.0.0:9090->9090/tcp
nsn-grafana             running (healthy)   0.0.0.0:3000->3000/tcp
nsn-jaeger              running (healthy)   0.0.0.0:16686->16686/tcp
```

### 5. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **Substrate RPC (WebSocket)** | ws://localhost:9944 | - |
| **Substrate RPC (HTTP)** | http://localhost:9933 | - |
| **Polkadot.js Apps** | https://polkadot.js.org/apps/?rpc=ws://localhost:9944 | - |
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | admin/admin |
| **Jaeger UI** | http://localhost:16686 | - |
| **Vortex gRPC** | localhost:50051 | - |

## Service Overview

### Substrate Node (NSN Chain)

Local NSN blockchain node running in `--dev` mode with:

- **Instant block finality** (no consensus delay)
- **Pre-funded development accounts** (Alice, Bob, Charlie, Dave, Eve)
- **All NSN pallets deployed** (stake, reputation, director, bft, storage, treasury, task-market, model-registry)
- **RPC methods exposed** for testing

**Logs:**
```bash
docker compose logs -f substrate-node
```

**Restart:**
```bash
docker compose restart substrate-node
```

**Reset chain state:**
```bash
docker compose down -v  # WARNING: Deletes all data!
docker compose up
```

### Vortex AI Engine

GPU-accelerated AI generation pipeline with:

- **Flux-Schnell** (NF4, ~6GB VRAM) - Actor image generation
- **LivePortrait** (FP16, ~3.5GB VRAM) - Video warping
- **Kokoro-82M** (FP32, ~400MB VRAM) - Text-to-speech
- **CLIP-ViT-B-32** (INT8, ~300MB VRAM) - Semantic verification
- **CLIP-ViT-L-14** (INT8, ~600MB VRAM) - Secondary verification

**Check GPU usage:**
```bash
docker compose exec vortex nvidia-smi
```

**Test generation:**
```bash
docker compose exec vortex python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
print(f'VRAM allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')
"
```

### STUN/TURN Servers

Mock NAT traversal servers for P2P testing:

- **STUN**: Port 3478 (UDP/TCP)
- **TURN**: Port 3479 (UDP/TCP) + relay ports 49152-49200 (UDP)

**Credentials:**
- Username: `nsn`
- Password: `password`
- Realm: `nsn.local`

**Test STUN:**
```bash
docker run --rm --network host alpine/socat - UDP:localhost:3478
```

### Prometheus

Metrics collection and time-series database.

**Targets:**
- Substrate node metrics (port 9615)
- Vortex metrics (port 9100)
- Prometheus self-monitoring

**Check targets:**
```bash
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'
```

### Grafana

Visualization and dashboards.

**Pre-configured dashboards:**
- NSN Local Development Overview
- Substrate Chain Metrics
- Vortex GPU Performance

**Access:** http://localhost:3000 (admin/admin)

**Import additional dashboards:**
1. Navigate to Dashboards → Import
2. Use dashboard ID or JSON file
3. Select Prometheus datasource

### Jaeger

Distributed tracing for debugging.

**Access:** http://localhost:16686

**Configure traces:**
Export OTLP traces to `localhost:4317` (gRPC) or `localhost:4318` (HTTP)

## Common Workflows

### Workflow 1: Test Pallet Changes

Modify, rebuild, and test custom pallets:

```bash
# 1. Edit pallet code
vim nsn-chain/pallets/nsn-stake/src/lib.rs

# 2. Rebuild Substrate node container
docker compose build substrate-node

# 3. Restart node with new runtime
docker compose up -d substrate-node

# 4. Wait for node to be healthy
docker compose ps substrate-node

# 5. Test via Polkadot.js Apps
# Open https://polkadot.js.org/apps/?rpc=ws://localhost:9944
# Submit extrinsic: nsnStake.depositStake(100 NSN, 1000 blocks, "us-east-1")
```

### Workflow 2: Test Vortex Generation

Generate AI content locally:

```bash
# 1. Exec into Vortex container
docker compose exec vortex bash

# 2. Check VRAM availability
nvidia-smi

# 3. Run test generation (Python REPL)
python3
>>> from vortex.pipeline import VortexPipeline
>>> pipeline = VortexPipeline(models_path="/models")
>>> print(f"VRAM used: {pipeline.get_vram_usage():.2f} GB")
>>> result = pipeline.generate(recipe={
...     "audio_track": {"script": "Hello world", "voice_id": "default"},
...     "visual_track": {"prompt": "A scientist speaking"}
... })
```

### Workflow 3: Monitor System Health

Use Grafana and Prometheus:

```bash
# 1. Open Grafana
open http://localhost:3000

# 2. View NSN Overview dashboard
# Navigate to: Dashboards → NSN Local Development Overview

# 3. Check key metrics:
# - Block production rate (should be ~1 block/6s in dev mode)
# - P2P peers (should be 0 in local single-node setup)
# - Vortex VRAM usage (should be <11.8 GB)
# - GPU temperature (should be <85°C)

# 4. Query Prometheus directly
curl 'http://localhost:9090/api/v1/query?query=substrate_block_height' | jq .
```

### Workflow 4: Debug with Jaeger

Trace requests across services:

```bash
# 1. Open Jaeger UI
open http://localhost:16686

# 2. Select service: "nsn-substrate-node" or "vortex"

# 3. Search for traces with specific tags
# - operation: deposit_stake
# - error: true

# 4. Inspect trace timeline and spans
```

### Workflow 5: Reset Development Environment

Clean slate for fresh testing:

```bash
# Option 1: Restart services (keeps volumes)
docker compose restart

# Option 2: Stop and start (keeps volumes)
docker compose down
docker compose up

# Option 3: Full reset (deletes all data!)
docker compose down -v
docker compose up

# Option 4: Rebuild all containers
docker compose down
docker compose build --no-cache
docker compose up
```

### Workflow 6: Download Model Weights Manually

If auto-download fails or you want to pre-download:

```bash
# 1. Create models directory
mkdir -p volumes/models

# 2. Run download script
docker run --rm \
  -v $(pwd)/volumes/models:/models \
  python:3.11-slim \
  bash -c "
    pip install requests tqdm &&
    python /app/scripts/download-models.py --output /models
  "

# 3. Verify downloads
ls -lh volumes/models/
# Should show:
# - flux-schnell/
# - liveportrait/
# - kokoro/
# - clip-vit-b32/
# - clip-vit-l14/
```

## Troubleshooting

### Issue: GPU Not Visible in Vortex Container

**Symptoms:**
```bash
docker compose exec vortex nvidia-smi
# Error: Failed to initialize NVML: Driver/library version mismatch
```

**Solutions:**

1. **Check NVIDIA Container Toolkit installation:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
   If this fails, install nvidia-docker2:
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Verify Docker daemon configuration:**
   ```bash
   cat /etc/docker/daemon.json
   ```
   Should contain:
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

3. **Check GPU visibility:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

### Issue: Out of VRAM

**Symptoms:**
```
CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Close other GPU applications:**
   ```bash
   nvidia-smi
   # Kill processes using GPU
   kill <PID>
   ```

2. **Reduce Vortex VRAM budget:**
   Edit `.env`:
   ```bash
   VORTEX_MAX_VRAM_GB=10.0  # Reduce from 11.8
   ```

3. **Use lower precision models** (requires code changes):
   - Flux-Schnell: NF4 → INT8
   - LivePortrait: FP16 → INT8
   - CLIP: Already INT8

### Issue: Substrate Node Not Producing Blocks

**Symptoms:**
```bash
curl http://localhost:9933/health
# Error: Connection refused
```

**Solutions:**

1. **Check container status:**
   ```bash
   docker compose ps substrate-node
   ```

2. **View logs:**
   ```bash
   docker compose logs substrate-node | tail -50
   ```

3. **Check port conflicts:**
   ```bash
   lsof -i :9944
   lsof -i :9933
   ```
   If port is in use, change mapping in `docker-compose.yml`

4. **Restart with clean state:**
   ```bash
   docker compose down -v
   docker compose up substrate-node
   ```

### Issue: Prometheus Targets Down

**Symptoms:**
Prometheus UI shows targets as "Down" or "Unhealthy"

**Solutions:**

1. **Check target health:**
   ```bash
   curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health == "down")'
   ```

2. **Verify target endpoints:**
   ```bash
   # Substrate metrics
   curl http://localhost:9615/metrics

   # Vortex metrics
   curl http://localhost:9100/metrics
   ```

3. **Check Docker networking:**
   ```bash
   docker compose exec prometheus ping substrate-node
   docker compose exec prometheus ping vortex
   ```

### Issue: Model Download Fails

**Symptoms:**
```
Failed to download https://huggingface.co/... after 3 attempts
```

**Solutions:**

1. **Check internet connectivity:**
   ```bash
   docker compose exec vortex curl -I https://huggingface.co
   ```

2. **Use mirror or manual download:**
   ```bash
   # Download on host, then copy to volume
   wget https://huggingface.co/.../model.bin -O model.bin
   docker cp model.bin nsn-vortex:/models/flux-schnell/model.bin
   ```

3. **Increase retry attempts:**
   Edit `docker/scripts/download-models.py`:
   ```python
   max_retries=5  # Increase from 3
   ```

### Issue: Services Slow to Start

**Symptoms:**
Services take >5 minutes to become healthy

**Solutions:**

1. **Check system resources:**
   ```bash
   docker stats
   ```
   Ensure sufficient CPU, RAM available

2. **Increase health check intervals:**
   Edit `docker-compose.yml`:
   ```yaml
   healthcheck:
     start_period: 120s  # Increase from 60s
   ```

3. **Start services incrementally:**
   ```bash
   docker compose up -d substrate-node prometheus grafana
   # Wait for healthy
   docker compose up -d vortex
   ```

## Advanced Configuration

### Custom Chain Spec

To use a custom genesis configuration:

1. Generate chain spec:
   ```bash
   docker compose run --rm substrate-node build-spec --chain local > custom-spec.json
   ```

2. Edit genesis config in `custom-spec.json`

3. Convert to raw format:
   ```bash
   docker compose run --rm substrate-node build-spec --chain custom-spec.json --raw > custom-spec-raw.json
   ```

4. Update `docker-compose.yml`:
   ```yaml
   substrate-node:
     command: >
       --chain=/app/custom-spec-raw.json
       --rpc-external
       --rpc-cors=all
   ```

### Multi-Node Local Testnet

Run multiple validators locally:

```yaml
# docker-compose.yml additions
substrate-node-alice:
  build:
    context: .
    dockerfile: docker/Dockerfile.substrate-local
  command: >
    --chain=local
    --alice
    --port 30333
    --rpc-port 9944
    --node-key=0000000000000000000000000000000000000000000000000000000000000001

substrate-node-bob:
  build:
    context: .
    dockerfile: docker/Dockerfile.substrate-local
  command: >
    --chain=local
    --bob
    --port 30334
    --rpc-port 9945
    --bootnodes /dns/substrate-node-alice/tcp/30333/p2p/12D3KooWEyoppNCUx8Yx66oV9fJnriXwCcXwDDUA2kj6vnc6iDEp
  depends_on:
    - substrate-node-alice
```

### Volume Persistence

Use host directories instead of Docker volumes:

```yaml
# docker-compose.yml
volumes:
  - ./volumes/substrate-data:/data
  - ./volumes/models:/models
  - ./volumes/prometheus-data:/prometheus
  - ./volumes/grafana-data:/var/lib/grafana
```

Create directories:
```bash
mkdir -p volumes/{substrate-data,models,prometheus-data,grafana-data}
chmod 777 volumes/*  # Or set proper ownership
```

### Resource Limits

Constrain resource usage:

```yaml
# docker-compose.yml
vortex:
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 8G
      reservations:
        cpus: '2'
        memory: 4G
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## Performance Tuning

### Optimize for Development Speed

Prioritize fast iteration over resource efficiency:

```yaml
# .env
SUBSTRATE_LOG_LEVEL=warn  # Reduce log verbosity
VORTEX_LOG_LEVEL=ERROR
RUST_LOG=warn

# docker-compose.yml
substrate-node:
  command: >
    --dev
    --execution=Native  # Faster than WASM
    --wasm-execution=Compiled
    --pruning=archive  # Keep all blocks
```

### Optimize for Resource Constraints

Reduce resource usage on constrained systems:

```bash
# .env
VORTEX_MAX_VRAM_GB=10.0  # Lower VRAM budget

# Disable non-essential services
ENABLE_JAEGER=false
ENABLE_STUN_TURN=false

# docker-compose.yml - reduce Prometheus retention
prometheus:
  command:
    - '--storage.tsdb.retention.time=1d'  # From 7d
```

### Optimize Model Loading

Pre-warm models on container start:

```python
# vortex/entrypoint.py
def preload_models():
    from vortex.models import load_all_models
    load_all_models(models_path="/models")
    print("All models loaded and resident in VRAM")

if __name__ == "__main__":
    preload_models()
    start_grpc_server()
```

---

## Quick Reference

### Essential Commands

```bash
# Start environment
docker compose up -d

# View logs
docker compose logs -f [service-name]

# Restart service
docker compose restart [service-name]

# Stop environment
docker compose down

# Full reset
docker compose down -v && docker compose up

# Check service health
docker compose ps

# Execute command in container
docker compose exec [service-name] [command]
```

### Key Ports

| Port | Service |
|------|---------|
| 9944 | Substrate WebSocket RPC |
| 9933 | Substrate HTTP RPC |
| 30333 | Substrate P2P |
| 9615 | Substrate metrics |
| 50051 | Vortex gRPC |
| 9100 | Vortex metrics |
| 3478 | STUN server |
| 3479 | TURN server |
| 9090 | Prometheus |
| 3000 | Grafana |
| 16686 | Jaeger UI |
| 4317 | OTLP gRPC |

### Development Accounts

| Account | Address | Seed |
|---------|---------|------|
| Alice | 5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY | //Alice |
| Bob | 5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty | //Bob |
| Charlie | 5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y | //Charlie |

All accounts pre-funded with 1,000,000 NSN tokens in dev mode.

---

**Need help?** Open an issue at https://github.com/neon-stream/nsn-chain/issues
