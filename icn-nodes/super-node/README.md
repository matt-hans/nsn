# ICN Super-Node

Tier 1 storage and relay infrastructure for the Interdimensional Cable Network (ICN).

## Features

- **Reed-Solomon Erasure Coding (10+4)**: Encodes video chunks into 14 shards (10 data + 4 parity) for fault tolerance
- **CID-Based Storage**: Content-addressed shard persistence with IPFS-compatible CIDs
- **Kademlia DHT**: Decentralized shard manifest publishing and discovery
- **QUIC Transport**: High-performance shard transfers to Regional Relays
- **On-Chain Audit Response**: Automated pinning audit proof generation and submission
- **Geographic Replication**: Distributed across 7 regions (NA-WEST, NA-EAST, EU-WEST, EU-EAST, APAC, LATAM, MENA)

## Architecture

```
Directors → GossipSub → Erasure Encoder → Storage Layer → DHT Manifest
                                               ↓
                                          QUIC Server ← Regional Relays
                                               ↓
                                       Audit Monitor → Chain (submit_audit_proof)
```

## Requirements

- **Storage**: 10TB+ capacity (SSD for latency, HDD for cost)
- **Bandwidth**: 500 Mbps symmetric
- **Stake**: Minimum 50 ICN tokens
- **ICN Chain**: Access to running ICN Chain RPC endpoint

## Configuration

Create `config/super-node.toml`:

```toml
chain_endpoint = "ws://127.0.0.1:9944"
storage_path = "/mnt/icn-storage"
quic_port = 9002
metrics_port = 9102
p2p_listen_addr = "/ip4/0.0.0.0/tcp/30333"
bootstrap_peers = ["/ip4/127.0.0.1/tcp/30334/p2p/12D3KooWA"]
region = "NA-WEST"
max_storage_gb = 10000
audit_poll_secs = 30
cleanup_interval_blocks = 1000
```

## Running

### Build

```bash
cargo build --release -p icn-super-node
```

### Run

```bash
./target/release/icn-super-node \
  --config config/super-node.toml \
  --storage-path /mnt/icn-storage \
  --region NA-WEST \
  --chain-endpoint ws://localhost:9944
```

### CLI Options

- `--config <path>`: Path to configuration file (default: `config/super-node.toml`)
- `--storage-path <path>`: Override storage root directory
- `--region <region>`: Override geographic region
- `--chain-endpoint <url>`: Override ICN Chain RPC endpoint

## Monitoring

Prometheus metrics available at `http://localhost:9102/metrics`:

- `icn_super_node_shard_count`: Total number of shards stored
- `icn_super_node_bytes_stored`: Total bytes stored
- `icn_super_node_audit_success_total`: Successful audit responses
- `icn_super_node_audit_failure_total`: Failed audit responses

## Storage Layout

```
<storage_path>/
  <CID_1>/
    shard_00.bin
    shard_01.bin
    ...
    shard_13.bin
  <CID_2>/
    ...
```

## Testing

### Unit Tests

```bash
cargo test -p icn-super-node --lib
```

### Integration Tests

```bash
cargo test -p icn-super-node --test '*'
```

## Erasure Coding

Super-Nodes use Reed-Solomon (10+4) encoding:

- **10 data shards**: Original video chunk split into 10 equal parts
- **4 parity shards**: Redundancy for fault tolerance
- **Reconstruction**: Any 10 of 14 shards can reconstruct the original
- **Overhead**: 1.4× (14/10) vs 3× for simple replication

**Example**:
- 50MB video chunk → 14 shards × ~7MB each
- Withstand loss of up to 4 shards
- 53% storage cost reduction vs 3× replication

## Audit Response

When pallet-icn-pinning initiates an audit:

1. **Detection**: Audit monitor polls `PendingAudits` storage every 30s
2. **Challenge**: Read specified bytes from shard at given offset
3. **Proof**: Compute `SHA256(challenged_bytes || nonce)`
4. **Submission**: Submit proof via `submit_audit_proof` extrinsic
5. **Deadline**: 100 blocks (~10 minutes)

**Success**: +10 reputation, continue earning rewards
**Failure**: -50 reputation, 10 ICN slashed

## Deployment

### Systemd Service

Create `/etc/systemd/system/icn-super-node.service`:

```ini
[Unit]
Description=ICN Super-Node
After=network.target

[Service]
Type=simple
User=icn
WorkingDirectory=/opt/icn-super-node
ExecStart=/opt/icn-super-node/icn-super-node --config /etc/icn/super-node.toml
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable icn-super-node
sudo systemctl start icn-super-node
sudo systemctl status icn-super-node
```

### Docker

```dockerfile
FROM rust:1.75 AS builder
WORKDIR /build
COPY . .
RUN cargo build --release -p icn-super-node

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/icn-super-node /usr/local/bin/
VOLUME ["/mnt/icn-storage"]
EXPOSE 9002 9102
ENTRYPOINT ["icn-super-node"]
CMD ["--config", "/etc/icn/super-node.toml"]
```

Run:

```bash
docker build -t icn-super-node .
docker run -d \
  -v /mnt/icn-storage:/mnt/icn-storage \
  -v /etc/icn:/etc/icn \
  -p 9002:9002 \
  -p 9102:9102 \
  --name icn-super-node \
  icn-super-node
```

## Disaster Recovery

### Restore from Erasure Shards

If a shard is corrupted or lost:

1. Retrieve remaining shards from storage
2. Run reconstruction:

```rust
let coder = ErasureCoder::new()?;
let shards: Vec<Option<Vec<u8>>> = /* load available shards */;
let original = coder.decode(shards, original_size)?;
```

3. Re-encode to regenerate missing shards
4. Store repaired shards

### Regional Failure

If entire region fails:
- 5× geographic replication ensures availability
- Automatic failover to other Super-Nodes in different regions
- Regional Relays fetch from next-closest Super-Node

## Troubleshooting

### High Disk Usage

```bash
# Check storage usage
du -sh /mnt/icn-storage/*

# Manual cleanup (beyond automatic interval)
curl -X POST http://localhost:9102/admin/cleanup
```

### Audit Failures

Check logs for:
- Shard file not found → Storage corruption, restore from other Super-Nodes
- I/O timeout → Disk performance issue, check SMART metrics
- Hash mismatch → Data corruption, re-encode from original

### QUIC Connection Issues

- Verify firewall allows UDP port 9002
- Check NAT traversal with: `netstat -an | grep 9002`
- Review QUIC handshake logs

## License

MIT OR Apache-2.0
