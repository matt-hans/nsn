# Phase 6: Testnet Deployment Config

## Plan 01: Production-Ready Docker Configuration

**Created:** 2026-01-11
**Estimated Scope:** Medium (8 tasks)
**Prerequisites:** Phases 1-5 complete (simulation harness validates system behavior)

---

## Objective

Create production-ready Docker Compose configuration and supporting artifacts for NSN testnet deployment. This phase transforms the existing development-only Docker setup into a multi-node testnet configuration with proper security, environment configuration, and bootstrap infrastructure.

---

## Execution Context

```
@docker-compose.yml                              # Existing dev config (reference)
@docker/Dockerfile.substrate-local               # Substrate node build
@docker/Dockerfile.vortex                        # Vortex GPU container
@docker/README.md                                # Current documentation
@nsn-chain/node/src/chain_spec.rs               # Chain spec definitions
@nsn-chain/runtime/src/genesis_config_presets.rs # Genesis presets
@node-core/bin/nsn-node/src/main.rs             # Off-chain node CLI
@viewer/scripts/signaling-server.js             # WebRTC signaling
```

---

## Context

### Current Infrastructure

**Existing Development Config:**
- `docker-compose.yml` - Single-node dev environment with insecure defaults
- `docker/Dockerfile.substrate-local` - Multi-stage Substrate node build
- `docker/Dockerfile.vortex` - GPU-enabled AI engine container
- Prometheus/Grafana/Jaeger observability stack
- STUN/TURN servers for NAT traversal

**Chain Spec Support:**
- `development_chain_spec()` - Single node, dev mode
- `local_chain_spec()` - Multi-node local testnet
- `nsn_testnet_chain_spec()` - Public testnet (3 validators: Alice, Bob, Charlie)
- `nsn_mainnet_chain_spec()` - Template with token allocations

**Off-Chain Node Modes:**
- SuperNode: Director + Validator + Storage
- DirectorOnly: Lane 0 generation
- ValidatorOnly: CLIP verification
- StorageOnly: Pinning and distribution

### Production Requirements

**Security:**
- Restricted RPC methods (Safe mode)
- Limited CORS origins
- Non-root container execution
- Secret management via Docker secrets
- TLS termination for external endpoints

**Scalability:**
- Multi-validator configuration (3 initial validators)
- Horizontal scaling for off-chain nodes
- Load balancing for RPC endpoints

**Reliability:**
- Health checks with automatic restart
- Volume persistence for chain data
- Graceful shutdown handling
- Log aggregation

---

## Tasks

### Task 1: Create testnet docker-compose configuration

**Goal:** Production-ready multi-node Docker Compose for testnet

**Actions:**
1. Create `docker/testnet/docker-compose.yml` with services:
   - 3 validator nodes (alice, bob, charlie)
   - 2 off-chain director nodes
   - 1 vortex GPU node (for Lane 0)
   - 1 signaling server (WebRTC)
   - Prometheus + Grafana stack
   - Jaeger tracing

2. Configure validator services:
   ```yaml
   validator-alice:
     image: nsn-node:${NSN_VERSION:-latest}
     command: >
       --chain=/chain-spec/testnet.json
       --alice
       --validator
       --rpc-cors=https://polkadot.js.org,https://nsn.network
       --rpc-methods=Safe
       --prometheus-external
       --telemetry-url "wss://telemetry.nsn.network/submit 0"
     volumes:
       - alice-data:/data
       - ./chain-spec:/chain-spec:ro
     networks:
       - nsn-testnet
     deploy:
       resources:
         limits:
           memory: 4G
           cpus: '2'
   ```

3. Configure off-chain director services:
   ```yaml
   director-1:
     image: nsn-offchain:${NSN_VERSION:-latest}
     command: >
       director-only
       --config=/config/director.toml
       --rpc-url=ws://validator-alice:9944
       --p2p-listen-port=9000
       --attestation-submit-mode=dual
     volumes:
       - director1-data:/data
       - ./config:/config:ro
     depends_on:
       - validator-alice
     networks:
       - nsn-testnet
   ```

4. Add Docker secrets for sensitive configuration:
   ```yaml
   secrets:
     attestation_key_alice:
       file: ./secrets/alice.key
     attestation_key_bob:
       file: ./secrets/bob.key
   ```

5. Configure resource limits and health checks for all services

**Verification:**
```bash
cd docker/testnet && docker compose config --quiet
```

**Checkpoint:** Docker Compose validates successfully

---

### Task 2: Create testnet chain spec

**Goal:** Export and customize testnet chain specification

**Actions:**
1. Export testnet chain spec from NSN node:
   ```bash
   ./target/release/nsn-node build-spec --chain nsn-testnet --raw > docker/testnet/chain-spec/testnet.json
   ```

2. Create `docker/testnet/chain-spec/` directory structure

3. Customize testnet.json:
   - Set bootnode addresses (placeholder multiaddrs)
   - Configure telemetry endpoint
   - Set appropriate validator bond requirements
   - Enable all NSN pallets

4. Create chain spec generation script:
   ```bash
   #!/bin/bash
   # scripts/generate-testnet-chainspec.sh
   # Generates testnet chain spec with current validator keys
   ```

**Verification:**
```bash
./target/release/nsn-node build-spec --chain docker/testnet/chain-spec/testnet.json --raw >/dev/null
```

**Checkpoint:** Chain spec is valid and loadable

---

### Task 3: Create environment configuration templates

**Goal:** Document and template all required environment variables

**Actions:**
1. Create `docker/testnet/.env.example`:
   ```bash
   # NSN Testnet Environment Configuration

   # Version
   NSN_VERSION=0.1.0

   # Network
   NSN_CHAIN=nsn-testnet
   NSN_P2P_PORT=30333
   NSN_RPC_PORT=9944
   NSN_WS_PORT=9944
   NSN_PROMETHEUS_PORT=9615

   # Validators (set key files in ./secrets/)
   VALIDATOR_ALICE_ENABLED=true
   VALIDATOR_BOB_ENABLED=true
   VALIDATOR_CHARLIE_ENABLED=true

   # Off-chain nodes
   DIRECTOR_COUNT=2
   DIRECTOR_GPU_DEVICE=0

   # Vortex
   VORTEX_MAX_VRAM_GB=11.8
   CUDA_VISIBLE_DEVICES=0

   # Observability
   GRAFANA_ADMIN_PASSWORD=changeme
   PROMETHEUS_RETENTION_DAYS=15
   JAEGER_ENABLED=true

   # External URLs
   NSN_TELEMETRY_URL=wss://telemetry.nsn.network/submit
   NSN_BOOTNODES=
   ```

2. Create `docker/testnet/config/` directory with:
   - `director.toml` - Director node configuration
   - `validator.toml` - Validator node configuration
   - `prometheus.yml` - Prometheus scrape targets
   - `alerting-rules.yml` - Prometheus alerting rules

3. Create secrets template structure:
   ```
   docker/testnet/secrets/
   ├── .gitkeep
   └── README.md  (instructions for generating keys)
   ```

**Verification:**
```bash
cd docker/testnet && cp .env.example .env && docker compose config --quiet
```

**Checkpoint:** Environment loads without errors

---

### Task 4: Create bootstrap node configuration

**Goal:** Configure and document bootstrap node setup

**Actions:**
1. Create `docker/testnet/bootstrap/docker-compose.yml`:
   - Dedicated bootstrap node service
   - Public P2P port exposure
   - Persistent peer ID

2. Create bootstrap node documentation:
   - How to generate persistent peer ID
   - DNS configuration requirements
   - Firewall rules
   - Health monitoring

3. Add bootnode multiaddr generation script:
   ```bash
   #!/bin/bash
   # scripts/generate-bootnode-addr.sh
   # Outputs multiaddr for use in chain spec
   ```

4. Create bootstrap peer ID persistence:
   - Volume mount for libp2p identity
   - Instructions for backup/restore

**Reference:** Polkadot SDK bootnode patterns

**Verification:**
```bash
# Verify bootstrap config validates
cd docker/testnet/bootstrap && docker compose config --quiet
```

**Checkpoint:** Bootstrap configuration complete

---

### Task 5: Create Dockerfile for off-chain node

**Goal:** Build optimized container image for node-core binary

**Actions:**
1. Create `docker/Dockerfile.nsn-offchain`:
   ```dockerfile
   # Multi-stage build for NSN off-chain node
   FROM rust:1.75-bookworm as builder

   RUN apt-get update && apt-get install -y \
       clang libclang-dev cmake git protobuf-compiler

   WORKDIR /app
   COPY node-core/ ./node-core/
   COPY nsn-chain/runtime/Cargo.toml ./nsn-chain/runtime/

   RUN cd node-core && cargo build --release --bin nsn-node

   # Runtime stage
   FROM debian:bookworm-slim

   RUN apt-get update && apt-get install -y ca-certificates curl && \
       useradd -m -u 1000 -U -s /bin/sh nsn

   COPY --from=builder /app/node-core/target/release/nsn-node /usr/local/bin/

   USER nsn
   EXPOSE 9000 9100 50051
   ENTRYPOINT ["/usr/local/bin/nsn-node"]
   ```

2. Add build arguments for feature flags:
   - `--build-arg FEATURES=lane0,lane1,storage`

3. Configure health check:
   ```dockerfile
   HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
       CMD curl -f http://localhost:9100/metrics || exit 1
   ```

4. Optimize image size:
   - Strip debug symbols
   - Remove build dependencies
   - Use minimal base image

**Verification:**
```bash
docker build -f docker/Dockerfile.nsn-offchain -t nsn-offchain:test .
docker run --rm nsn-offchain:test --help
```

**Checkpoint:** Off-chain node container builds and runs

---

### Task 6: Create signaling server container

**Goal:** Containerize WebRTC signaling server for production

**Actions:**
1. Create `docker/Dockerfile.signaling`:
   ```dockerfile
   FROM node:20-alpine

   WORKDIR /app

   COPY viewer/scripts/signaling-server.js ./
   COPY viewer/package.json ./

   RUN npm install --production

   RUN adduser -D -u 1000 signaling
   USER signaling

   EXPOSE 8080

   HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
       CMD wget -q --spider http://localhost:8080/health || exit 1

   CMD ["node", "signaling-server.js", "8080"]
   ```

2. Add signaling server to testnet compose:
   ```yaml
   signaling:
     build:
       context: ../..
       dockerfile: docker/Dockerfile.signaling
     ports:
       - "8080:8080"
     networks:
       - nsn-testnet
   ```

3. Create Kubernetes-ready configuration (optional):
   - Service definition
   - Deployment manifest
   - Ingress for TLS termination

**Verification:**
```bash
docker build -f docker/Dockerfile.signaling -t nsn-signaling:test .
docker run --rm -d -p 8080:8080 nsn-signaling:test
curl http://localhost:8080/health
```

**Checkpoint:** Signaling server container operational

---

### Task 7: Update Prometheus and Grafana configuration

**Goal:** Production-ready observability for testnet

**Actions:**
1. Update `docker/testnet/prometheus.yml`:
   - Add all validator scrape targets
   - Add off-chain node metrics
   - Add signaling server metrics
   - Configure alerting rules

2. Create alerting rules (`docker/testnet/alerts/nsn-alerts.yml`):
   ```yaml
   groups:
     - name: nsn-testnet
       rules:
         - alert: ValidatorDown
           expr: up{job="validators"} == 0
           for: 1m
           annotations:
             summary: "Validator {{ $labels.instance }} is down"

         - alert: HighBlockTime
           expr: rate(substrate_block_height[5m]) < 0.15
           for: 5m
           annotations:
             summary: "Block production rate below expected"

         - alert: ConsensusStalled
           expr: increase(substrate_finality_grandpa_round[10m]) == 0
           for: 10m
           annotations:
             summary: "GRANDPA consensus has stalled"
   ```

3. Update Grafana dashboards (`docker/testnet/grafana/dashboards/`):
   - Multi-validator overview
   - Off-chain node performance
   - Network health dashboard
   - Lane 0/Lane 1 metrics

4. Configure Grafana provisioning for testnet:
   - Datasource auto-discovery
   - Dashboard auto-import
   - Alert notification channels

**Verification:**
```bash
promtool check config docker/testnet/prometheus.yml
promtool check rules docker/testnet/alerts/nsn-alerts.yml
```

**Checkpoint:** Prometheus config validates, dashboards ready

---

### Task 8: Create deployment documentation and scripts

**Goal:** Complete deployment guide with operational scripts

**Actions:**
1. Create `docker/testnet/README.md`:
   - Prerequisites (Docker, Docker Compose, NVIDIA Container Toolkit)
   - Quick start guide
   - Configuration reference
   - Troubleshooting guide

2. Create deployment scripts:
   ```bash
   # scripts/testnet-up.sh - Start testnet
   # scripts/testnet-down.sh - Stop testnet
   # scripts/testnet-logs.sh - View aggregated logs
   # scripts/testnet-status.sh - Health check all services
   # scripts/backup-chain-data.sh - Backup validator data
   ```

3. Create validator key generation guide:
   - Session key rotation
   - Key backup procedures
   - Key injection via Docker secrets

4. Document upgrade procedure:
   - Rolling validator upgrades
   - Runtime upgrade process
   - Rollback procedures

5. Create health check script:
   ```bash
   #!/bin/bash
   # scripts/testnet-status.sh
   echo "=== NSN Testnet Status ==="
   docker compose -f docker/testnet/docker-compose.yml ps
   echo ""
   echo "=== Block Height ==="
   curl -s http://localhost:9933 -H "Content-Type: application/json" \
     -d '{"id":1,"jsonrpc":"2.0","method":"chain_getHeader"}' | jq .result.number
   ```

**Verification:**
```bash
# Documentation renders correctly
cat docker/testnet/README.md
# Scripts are executable
chmod +x docker/testnet/scripts/*.sh
```

**Checkpoint:** Documentation complete, scripts operational

---

## Verification

### Per-Task Verification

Each task includes specific verification commands in its section.

### Phase-Level Verification

After all tasks complete:

```bash
# 1. All Docker configs validate
cd docker/testnet && docker compose config --quiet
cd docker/testnet/bootstrap && docker compose config --quiet

# 2. Build all images
docker compose -f docker/testnet/docker-compose.yml build

# 3. Start testnet (dry run)
docker compose -f docker/testnet/docker-compose.yml up -d
sleep 30

# 4. Verify all services healthy
docker compose -f docker/testnet/docker-compose.yml ps --format json | jq '.[] | select(.Health != "healthy")'

# 5. Verify chain producing blocks
curl -s http://localhost:9933 -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"chain_getHeader"}' | jq .result.number

# 6. Clean up
docker compose -f docker/testnet/docker-compose.yml down -v
```

### Service Health Matrix

| Service | Health Check | Expected Response |
|---------|--------------|-------------------|
| validator-alice | HTTP :9933/health | 200 OK |
| validator-bob | HTTP :9943/health | 200 OK |
| validator-charlie | HTTP :9953/health | 200 OK |
| director-1 | HTTP :9100/metrics | 200 OK |
| director-2 | HTTP :9101/metrics | 200 OK |
| vortex | HTTP :9101/metrics | 200 OK |
| signaling | HTTP :8080/health | {"status":"ok"} |
| prometheus | HTTP :9090/-/healthy | 200 OK |
| grafana | HTTP :3000/api/health | 200 OK |

---

## Success Criteria

1. **Docker Compose validates** for testnet configuration
2. **3 validator nodes** can start and produce blocks
3. **Off-chain nodes** connect to validators successfully
4. **Chain spec** loads with testnet preset
5. **Environment templates** document all configuration
6. **Bootstrap node** configuration documented
7. **Observability stack** collects metrics from all services
8. **Documentation** covers deployment and operations

---

## Output

Upon completion:
- `docker/testnet/docker-compose.yml` - Multi-node testnet configuration
- `docker/testnet/chain-spec/testnet.json` - Testnet chain specification
- `docker/testnet/.env.example` - Environment template
- `docker/testnet/config/` - Service configuration files
- `docker/testnet/secrets/README.md` - Key generation instructions
- `docker/testnet/bootstrap/` - Bootstrap node configuration
- `docker/Dockerfile.nsn-offchain` - Off-chain node container
- `docker/Dockerfile.signaling` - Signaling server container
- `docker/testnet/prometheus.yml` - Updated Prometheus config
- `docker/testnet/alerts/` - Alerting rules
- `docker/testnet/grafana/dashboards/` - Production dashboards
- `docker/testnet/README.md` - Deployment documentation
- `docker/testnet/scripts/` - Operational scripts

---

## Notes

### Design Decisions

1. **3 validators for testnet**: Matches nsn_testnet_chain_spec() preset, provides 2/3+1 BFT threshold

2. **Docker secrets for keys**: Avoids hardcoding credentials, follows Docker security best practices

3. **Separate bootstrap compose**: Allows independent bootstrap node management and public exposure

4. **Environment-based configuration**: Enables easy transition between testnet and mainnet

### Security Considerations

**Production Changes from Dev:**
- `--rpc-methods=Safe` instead of `Unsafe`
- Limited `--rpc-cors` to known origins
- Non-root container execution
- Resource limits enforced
- Health checks with restart policies
- Secrets via Docker secrets (not environment variables)

### Future Extensions

- Kubernetes manifests for cloud deployment
- Terraform modules for infrastructure provisioning
- CI/CD pipeline for automated deployment
- Multi-region validator distribution
- Automated backup and disaster recovery
