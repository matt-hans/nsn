# NSN Docker Configuration

This directory contains Docker configuration files for the NSN local development environment.

## Directory Structure

```
docker/
├── Dockerfile.substrate-local  # NSN Chain node container
├── Dockerfile.vortex            # Vortex AI engine with GPU support
├── chain-spec-dev.json          # Development chain specification
├── prometheus.yml               # Prometheus scrape configuration
├── grafana/
│   ├── dashboards/              # Pre-configured Grafana dashboards
│   │   └── nsn-overview.json
│   └── provisioning/            # Grafana auto-provisioning
│       ├── datasources/
│       │   └── prometheus.yml
│       └── dashboards/
│           └── nsn-local.yml
└── scripts/
    └── download-models.py       # AI model downloader with retry logic
```

## Container Images

### substrate-local

Multi-stage build for NSN Substrate node:

**Base:** `rust:1.75-bookworm` (builder), `debian:bookworm-slim` (runtime)
**Purpose:** Runs NSN Chain with custom pallets in development mode
**Build time:** ~8-12 minutes (first build)
**Image size:** ~300MB (runtime)

**Exposed ports:**
- 9944: WebSocket RPC
- 9933: HTTP RPC
- 30333: P2P networking
- 9615: Prometheus metrics

**Volumes:**
- `/data`: Blockchain database (persistent)

**Environment variables:**
- `SUBSTRATE_LOG_LEVEL`: Log level (default: info)
- `RUST_LOG`: Rust-specific logging

### vortex

GPU-accelerated AI engine:

**Base:** `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
**Purpose:** Runs Vortex generation pipeline with all models resident in VRAM
**Build time:** ~15-20 minutes (includes model downloads)
**Image size:** ~20GB (with models)

**Exposed ports:**
- 50051: gRPC server
- 9100: Prometheus metrics

**Volumes:**
- `/models`: AI model weights (persistent, ~15GB)
- `/output`: Generated content cache

**Environment variables:**
- `VORTEX_MAX_VRAM_GB`: VRAM budget limit (default: 11.8)
- `VORTEX_MODELS_PATH`: Model weights directory
- `CUDA_VISIBLE_DEVICES`: GPU device index

**GPU requirements:**
- NVIDIA GPU with 12GB+ VRAM
- NVIDIA drivers 535+
- NVIDIA Container Toolkit

## Configuration Files

### prometheus.yml

Prometheus scrape targets:

- `substrate-node:9615` - Chain metrics
- `vortex:9100` - GPU/generation metrics
- `prometheus:9090` - Self-monitoring
- `grafana:3000` - Grafana metrics
- `jaeger:14269` - Jaeger metrics

Scrape interval: 15s
Retention: 7 days

### chain-spec-dev.json

Development chain specification:

- Chain ID: `nsn_dev`
- Chain type: Development
- Block time: ~6 seconds (instant finality)
- Pre-funded accounts: Alice, Bob, Charlie, Dave, Eve (1M NSN each)
- All NSN pallets enabled

### grafana/dashboards/nsn-overview.json

Pre-configured Grafana dashboard panels:

1. **Block Height** - Current chain height (gauge)
2. **P2P Peers** - Connected peer count (gauge)
3. **Block Production Rate** - Blocks per minute (time series)
4. **VRAM Usage** - GPU memory usage (gauge)
5. **GPU Temperature** - GPU thermal state (gauge)

Refresh interval: 5 seconds

## Build Arguments

### substrate-local

```bash
docker build \
  --build-arg RUST_VERSION=1.75 \
  --build-arg POLKADOT_SDK_VERSION=polkadot-stable2409 \
  -f docker/Dockerfile.substrate-local \
  -t nsn-substrate-local:latest \
  .
```

### vortex

```bash
docker build \
  --build-arg CUDA_VERSION=12.1.0 \
  --build-arg PYTHON_VERSION=3.11 \
  -f docker/Dockerfile.vortex \
  -t nsn-vortex:latest \
  .
```

## Development vs Production

**Development (current):**
- `--dev` mode (instant finality, pre-funded accounts)
- Unsafe RPC methods enabled
- CORS all origins
- Single node
- Temporary chain state

**Production (future):**
- Multi-validator setup
- Safe RPC methods only
- Restricted CORS
- Persistent chain state
- TLS/HTTPS
- Secrets via Docker secrets or vault
- Resource limits enforced
- Health checks with automatic recovery

## Troubleshooting

### Container fails to start

**Check logs:**
```bash
docker compose logs [service-name]
```

**Common issues:**
- Port conflicts: Change port mappings in `docker-compose.yml`
- Insufficient resources: Increase Docker resource limits
- Volume permissions: `chmod 777 volumes/*`

### GPU not visible in vortex

**Verify NVIDIA Container Toolkit:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Check Docker daemon config:**
```bash
cat /etc/docker/daemon.json
```

Must include:
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

### Model download fails

**Manual download:**
```bash
docker compose run --rm vortex python3 /app/scripts/download-models.py --output /models --force
```

**Use host-mounted volume:**
```bash
mkdir -p ./volumes/models
# Download models to ./volumes/models manually
# Then update docker-compose.yml to use bind mount
```

## Security Notes

**WARNING:** This configuration is for local development only!

**Insecure defaults:**
- All RPC methods exposed
- CORS accepts all origins
- Default credentials (admin/admin)
- No TLS/HTTPS
- Pre-funded development accounts with known seeds

**Never use these containers in production!**

For production deployment, see:
- `docs/deployment-production.md` (when available)
- NSN Mainnet deployment guide

## Performance Optimization

### Fast iteration (development)

```yaml
# docker-compose.yml
substrate-node:
  command: >
    --dev
    --execution=Native  # Skip WASM compilation
    --tmp  # No database persistence (faster)
```

### Resource-constrained systems

```yaml
vortex:
  deploy:
    resources:
      limits:
        memory: 8G
        cpus: '4'
```

```bash
# .env
VORTEX_MAX_VRAM_GB=10.0  # Reduce VRAM budget
```

## Maintenance

### Update base images

```bash
docker compose pull
docker compose build --pull --no-cache
```

### Clean up dangling images

```bash
docker image prune -a
```

### Backup volumes

```bash
docker run --rm -v nsn-local_substrate-data:/data -v $(pwd):/backup ubuntu tar czf /backup/substrate-data-backup.tar.gz /data
```

### Restore volumes

```bash
docker run --rm -v nsn-local_substrate-data:/data -v $(pwd):/backup ubuntu tar xzf /backup/substrate-data-backup.tar.gz -C /
```

---

**For usage instructions, see:** `docs/local-development.md`
