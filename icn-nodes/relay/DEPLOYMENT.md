# Regional Relay Node Deployment Guide

**ICN Regional Relay Node** - Tier 2 content distribution layer for the Interdimensional Cable Network.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running](#running)
- [Monitoring](#monitoring)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Storage | 1 TB SSD | 2 TB NVMe SSD |
| Bandwidth | 100 Mbps symmetric | 1 Gbps symmetric |
| OS | Ubuntu 22.04+ | Ubuntu 22.04 LTS |

### Network Requirements

- Low latency to nearest ICN Super-Node (<50ms)
- Open ports:
  - **9003** - QUIC viewer connections (incoming)
  - **30333** - libp2p P2P (incoming)
  - **9103** - Prometheus metrics (outgoing)

---

## Installation

### Option 1: Binary Deployment

#### 1. Build from Source

```bash
# Clone the repository
git clone https://github.com/your-org/icn-nodes.git
cd icn-nodes/relay

# Build the release binary
cargo build --release

# The binary will be at: ../target/release/icn-relay
```

#### 2. Install System-Wide

```bash
# Create dedicated user
sudo useradd -r -s /bin/false icn-relay
sudo mkdir -p /var/lib/icn-relay /etc/icn
sudo chown -R icn-relay:icn-relay /var/lib/icn-relay

# Install binary
sudo cp target/release/icn-relay /usr/local/bin/
sudo chmod +x /usr/local/bin/icn-relay
```

#### 3. Create Configuration

```bash
# Copy example configuration
sudo cp config/relay.toml.example /etc/icn/relay.toml

# Edit configuration
sudo nano /etc/icn/relay.toml
```

#### 4. Create systemd Service

```bash
sudo cat > /etc/systemd/system/icn-relay.service << 'EOF'
[Unit]
Description=ICN Regional Relay Node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=icn-relay
Group=icn-relay

# Working directory
WorkingDirectory=/var/lib/icn-relay

# Executable
ExecStart=/usr/local/bin/icn-relay --config /etc/icn/relay.toml --cache-path /var/lib/icn-relay/cache

# Restart policy
Restart=always
RestartSec=10
StartLimitInterval=60
StartLimitBurst=3

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/icn-relay

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=icn-relay

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable icn-relay
sudo systemctl start icn-relay
```

#### 5. Check Status

```bash
# Check service status
sudo systemctl status icn-relay

# View logs
sudo journalctl -u icn-relay -f

# Check metrics
curl http://localhost:9103/metrics | grep icn_relay_
```

---

### Option 2: Docker Deployment

#### 1. Build Docker Image

```dockerfile
# Dockerfile
FROM debian:bookworm-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy source
WORKDIR /build
COPY . .

# Build binary
RUN cargo build --release

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -r -s /bin/false icn-relay

# Copy binary from builder
COPY --from=builder /build/target/release/icn-relay /usr/local/bin/

# Create directories
RUN mkdir -p /var/lib/icn-relay/cache && \
    chown -R icn-relay:icn-relay /var/lib/icn-relay

# Switch to non-root user
USER icn-relay

# Expose ports
EXPOSE 9003 30333 9103

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:9103/health || exit 1

# Set entrypoint
ENTRYPOINT ["icn-relay"]
CMD ["--config", "/config/relay.toml", "--cache-path", "/var/lib/icn-relay/cache"]
```

#### 2. Build and Run

```bash
# Build image
docker build -t icn-relay:latest .

# Create volume for cache
docker volume create icn-relay-cache

# Run container
docker run -d \
  --name icn-relay \
  --restart unless-stopped \
  -p 9003:9003 \
  -p 30333:30333 \
  -p 9103:9103 \
  -v icn-relay-cache:/var/lib/icn-relay/cache \
  -v $(pwd)/config/relay.toml:/config/relay.toml:ro \
  icn-relay:latest

# View logs
docker logs -f icn-relay

# Check status
docker exec icn-relay curl http://localhost:9103/metrics
```

#### 3. Docker Compose

```yaml
version: '3.8'

services:
  icn-relay:
    build: .
    image: icn-relay:latest
    container_name: icn-relay
    restart: unless-stopped
    ports:
      - "9003:9003"  # QUIC
      - "30333:30333" # P2P
      - "9103:9103"  # Metrics
    volumes:
      - icn-relay-cache:/var/lib/icn-relay/cache
      - ./config/relay.toml:/config/relay.toml:ro
    environment:
      - RUST_LOG=info,icn_relay=debug
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9103/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
    resource_limits:
      cpus: '2'
      memory: 4G
    networks:
      - icn-network

volumes:
  icn-relay-cache:
    driver: local

networks:
  icn-network:
    driver: bridge
```

Run with:
```bash
docker-compose up -d
```

---

### Option 3: Kubernetes Deployment

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: icn-relay-config
  namespace: icn
data:
  relay.toml: |
    # Relay configuration here
    cache_max_size_bytes = 1099511627776  # 1TB
    quic_port = 9003
    p2p_port = 30333
    metrics_port = 9103
    # ... rest of config
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: icn-relay
  namespace: icn
spec:
  replicas: 1
  selector:
    matchLabels:
      app: icn-relay
  template:
    metadata:
      labels:
        app: icn-relay
    spec:
      containers:
      - name: relay
        image: icn-relay:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 9003
          name: quic
          protocol: UDP
        - containerPort: 30333
          name: p2p
          protocol: TCP
        - containerPort: 9103
          name: metrics
          protocol: TCP
        env:
        - name: RUST_LOG
          value: "info,icn_relay=debug"
        volumeMounts:
        - name: config
          mountPath: /config
          readOnly: true
        - name: cache
          mountPath: /var/lib/icn-relay/cache
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 9103
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 9103
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
      volumes:
      - name: config
        configMap:
          name: icn-relay-config
      - name: cache
        persistentVolumeClaim:
          claimName: icn-relay-cache-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: icn-relay-cache-pvc
  namespace: icn
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 1Ti
  storageClassName: fast-ssd
---
apiVersion: v1
kind: Service
metadata:
  name: icn-relay
  namespace: icn
spec:
  selector:
    app: icn-relay
  ports:
  - name: quic
    port: 9003
    targetPort: 9003
    protocol: UDP
  - name: p2p
    port: 30333
    targetPort: 30333
    protocol: TCP
  - name: metrics
    port: 9103
    targetPort: 9103
    protocol: TCP
  type: LoadBalancer
```

Deploy with:
```bash
kubectl apply -f k8s/
```

---

## Configuration

### Configuration File (`/etc/icn/relay.toml`)

```toml
# Network Configuration
quic_port = 9003
p2p_port = 30333
metrics_port = 9103

# Cache Configuration
cache_max_size_bytes = 1099511627776  # 1 TB
cache_path = "/var/lib/icn-relay/cache"

# Region Configuration (leave empty for auto-detection)
region = ""  # Options: "NA-WEST", "NA-EAST", "EU-WEST", "EU-CENTRAL", "APAC", "SA-EAST", "AF-SOUTH"

# Upstream Super-Nodes
super_nodes = [
  "super-node-1.icn.network:9002",
  "super-node-2.icn.network:9002",
  "super-node-3.icn.network:9002",
]

# Rate Limiting
global_connection_rate = 100  # connections per second
per_ip_connection_rate = 10   # connections per second per IP

# Authentication (optional)
require_auth = false
auth_tokens = []  # List of valid auth tokens if require_auth = true

# TLS Configuration
tls_cert_path = ""  # Path to TLS certificate (optional, auto-generated if empty)
tls_key_path = ""   # Path to TLS private key (optional, auto-generated if empty)

# Chain Connection
chain_ws_url = "wss://mainnet.icn.network"

# Logging
log_level = "info"  # Options: "error", "warn", "info", "debug", "trace"
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Log level | `info` |
| `ICN_RELAY_CONFIG` | Path to config file | `/etc/icn/relay.toml` |
| `ICN_RELAY_CACHE_PATH` | Path to cache directory | `/var/lib/icn-relay/cache` |
| `ICN_RELAY_REGION` | Override region | (auto-detect) |

---

## Running

### Start the Service

```bash
# systemd
sudo systemctl start icn-relay

# Docker
docker start icn-relay

# Manual
/usr/local/bin/icn-relay --config /etc/icn/relay.toml --cache-path /var/lib/icn-relay/cache
```

### Stop the Service

```bash
# systemd
sudo systemctl stop icn-relay

# Docker
docker stop icn-relay

# Manual (SIGTERM for graceful shutdown)
kill $(pidof icn-relay)
```

### Graceful Shutdown

The relay node supports graceful shutdown with cache persistence:

1. **SIGTERM** - Flushes cache to disk and exits
2. **SIGINT** (Ctrl+C) - Same as SIGTERM
3. **SIGKILL** - Immediate termination (cache may be corrupted)

**Note:** Always use `systemctl stop` or `docker stop` for graceful shutdown.

---

## Monitoring

### Metrics Endpoint

The relay exposes Prometheus metrics on port 9103:

```bash
# View all metrics
curl http://localhost:9103/metrics

# View relay-specific metrics
curl http://localhost:9103/metrics | grep icn_relay_
```

#### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `icn_relay_cache_hits_total` | Counter | Total cache hits |
| `icn_relay_cache_misses_total` | Counter | Total cache misses |
| `icn_relay_cache_evictions_total` | Counter | Total cache evictions |
| `icn_relay_bytes_served_total` | Counter | Total bytes served to viewers |
| `icn_relay_upstream_fetches_total` | Counter | Total upstream fetches |
| `icn_relay_viewer_connections` | Gauge | Current viewer connections |
| `icn_relay_shard_serve_latency_seconds` | Histogram | Shard serve latency |
| `icn_relay_upstream_fetch_latency_seconds` | Histogram | Upstream fetch latency |

### Health Check

```bash
# Health endpoint
curl http://localhost:9103/health

# Response
{"status":"healthy","cache_size_bytes":536870912,"viewer_connections":12}
```

### Logging

Logs are written to journald (systemd) or stdout (Docker):

```bash
# View real-time logs
sudo journalctl -u icn-relay -f

# View logs from last hour
sudo journalctl -u icn-relay --since "1 hour ago"

# Docker logs
docker logs -f icn-relay

# Filter by log level
sudo journalctl -u icn-relay | grep "ERROR"
```

### Prometheus Integration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'icn-relay'
    static_configs:
      - targets: ['localhost:9103']
    scrape_interval: 15s
```

### Grafana Dashboard

Import the provided dashboard JSON: `icn-nodes/relay/grafana-dashboard.json`

---

## Security

### TLS/SSL

- **Production mode**: Uses WebPKI root certificates (default)
- **Dev mode**: Skips certificate verification (requires `--features dev-mode`)

**Never use dev mode in production!**

### Rate Limiting

Default rate limits:
- **Global**: 100 connections/second
- **Per-IP**: 10 connections/second

Adjust in configuration if needed.

### Authentication (Optional)

For private relays, enable token-based authentication:

```toml
require_auth = true
auth_tokens = ["token1", "token2", "token3"]
```

Viewers must include auth token in requests:

```
AUTH token1
GET /shards/QmXxx.../shard_0.bin
```

### Firewall

Configure firewall to allow only necessary ports:

```bash
# UFW
sudo ufw allow 9003/udp comment 'ICN Relay QUIC'
sudo ufw allow 30333/tcp comment 'ICN Relay P2P'
sudo ufw allow from 127.0.0.1 to any port 9103 proto tcp comment 'ICN Relay Metrics'

# iptables
sudo iptables -A INPUT -p udp --dport 9003 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 30333 -j ACCEPT
sudo iptables -A INPUT -s 127.0.0.1 -p tcp --dport 9103 -j ACCEPT
```

### Best Practices

1. **Run as non-root user** (create `icn-relay` user)
2. **Use firewall** to restrict access
3. **Enable authentication** for private deployments
4. **Monitor logs** for suspicious activity
5. **Keep software updated**
6. **Use TLS** in production (never dev mode)
7. **Backup cache directory** regularly
8. **Monitor disk usage** (cache can grow to 1TB)

---

## Troubleshooting

### Relay Won't Start

**Symptom:** Service fails to start

**Solutions:**

1. Check logs:
```bash
sudo journalctl -u icn-relay -n 50
```

2. Verify configuration:
```bash
/usr/local/bin/icn-relay --config /etc/icn/relay.toml --validate
```

3. Check port availability:
```bash
sudo netstat -tulpn | grep -E '(9003|30333|9103)'
```

4. Verify file permissions:
```bash
ls -la /var/lib/icn-relay
ls -la /etc/icn/relay.toml
```

### High Cache Miss Rate

**Symptom:** Cache hit rate <70%

**Solutions:**

1. Check cache size:
```bash
du -sh /var/lib/icn-relay/cache
```

2. Monitor cache metrics:
```bash
curl http://localhost:9103/metrics | grep cache
```

3. Increase cache size if needed

4. Check if content is popular (unpopular content will have low hit rate)

### Cannot Connect to Upstream

**Symptom:** Errors connecting to Super-Nodes

**Solutions:**

1. Check network connectivity:
```bash
ping super-node.icn.network
telnet super-node.icn.network 9002
```

2. Verify Super-Node addresses in config

3. Check if Super-Nodes are operational

4. Verify TLS certificates (production mode only)

### High Memory Usage

**Symptom:** Relay using >4GB RAM

**Solutions:**

1. Check connection count:
```bash
curl http://localhost:9103/metrics | grep viewer_connections
```

2. Reduce max concurrent streams in config

3. Check for memory leaks (restart if needed)

### Cache Corruption

**Symptom:** Errors reading from cache

**Solutions:**

1. Stop relay gracefully:
```bash
sudo systemctl stop icn-relay
```

2. Clear cache:
```bash
rm -rf /var/lib/icn-relay/cache/*
```

3. Restart relay:
```bash
sudo systemctl start icn-relay
```

---

## Performance Tuning

### Cache Size

Adjust based on content popularity:

```toml
cache_max_size_bytes = 2199023255552  # 2 TB for very popular relays
cache_max_size_bytes = 549755813888   # 512 GB for less popular relays
```

### Connection Limits

Adjust based on available bandwidth:

```toml
global_connection_rate = 200  # For 1 Gbps networks
per_ip_connection_rate = 20
```

### Region Selection

For best performance, manually set region:

```toml
region = "NA-WEST"  # Force specific region
```

---

## Upgrading

### Upgrade Procedure

1. **Stop service gracefully:**
```bash
sudo systemctl stop icn-relay
```

2. **Backup configuration and cache:**
```bash
sudo cp /etc/icn/relay.toml /etc/icn/relay.toml.bak
sudo tar -czf /tmp/icn-relay-cache-backup.tar.gz /var/lib/icn-relay/cache
```

3. **Install new binary:**
```bash
cargo build --release
sudo cp target/release/icn-relay /usr/local/bin/
```

4. **Start service:**
```bash
sudo systemctl start icn-relay
```

5. **Verify upgrade:**
```bash
sudo journalctl -u icn-relay -n 50
curl http://localhost:9103/health
```

### Rollback

If upgrade fails:

1. Stop service
2. Restore old binary:
```bash
sudo cp /usr/local/bin/icn-relay.old /usr/local/bin/icn-relay
```
3. Start service

---

## Support

For issues and questions:

- **GitHub Issues**: https://github.com/your-org/icn-nodes/issues
- **Documentation**: https://docs.icn.network
- **Community Discord**: https://discord.gg/icn-network

---

**Version:** 0.1.0
**Last Updated:** 2025-12-28
