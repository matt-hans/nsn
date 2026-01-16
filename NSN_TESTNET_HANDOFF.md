# NSN Testnet Handoff Document

**Generated:** 2026-01-15T09:02:34Z
**Author:** Claude Code Agent
**Purpose:** Network state handoff for continued testing and penetration testing

---

## 1. Executive Summary

The NSN (Neural Sovereign Network) testnet has been partially deployed. The on-chain layer (blockchain) is operational, but the off-chain layer (validators, directors, Vortex AI) requires additional configuration before full operation.

### Current Deployment Status

| Component | Status | Details |
|-----------|--------|---------|
| NSN Chain (Blockchain) | ✅ Running | Dev mode, Alice validator |
| Off-Chain Validators | ❌ Blocked | Missing trusted signers |
| Off-Chain Directors | ❌ Blocked | Missing trusted signers |
| Vortex AI Server | ❌ Not Started | Missing Python dependencies |
| P2P Mesh Network | ❌ Not Started | Depends on validators |

---

## 2. Network Configuration

### 2.1 Chain Details

| Property | Value |
|----------|-------|
| Chain Name | NSN Development |
| Node Name | NSN Node |
| Version | 0.1.0-unknown |
| Runtime | nsn-runtime |
| Spec Version | 1 |
| Implementation Version | 0 |
| Available APIs | 14 |
| Local Peer ID | `12D3KooWKfEMomtd9qnWiHP1zi85hsoZmTViCajZz4XF8wRbJ7CZ` |

### 2.2 Endpoints

| Service | Endpoint | Status |
|---------|----------|--------|
| Chain RPC (HTTP) | http://127.0.0.1:9944 | ✅ Active |
| Chain RPC (WebSocket) | ws://127.0.0.1:9944 | ✅ Active |
| Chain P2P | /ip4/127.0.0.1/tcp/30333 | ✅ Active |
| Prometheus Metrics | http://127.0.0.1:9615 | ✅ Active |

### 2.3 Running Processes

```bash
# Chain process (PID varies)
./target/release/nsn-node --dev --alice --validator --rpc-port 9944 --rpc-cors all --unsafe-rpc-external
```

### 2.4 Log Locations

| Component | Log File |
|-----------|----------|
| NSN Chain | `/tmp/claude/-home-matt-nsn/tasks/b3c26a5.output` |
| Database | `/tmp/substrate3O7HUx/chains/nsn-dev/db/full` |

---

## 3. Built Artifacts

### 3.1 Binary Locations

| Binary | Path | Size | Purpose |
|--------|------|------|---------|
| nsn-node (chain) | `/home/matt/nsn/nsn-chain/target/release/nsn-node` | 160MB | Blockchain node |
| nsn-node (off-chain) | `/home/matt/nsn/node-core/target/release/nsn-node` | 32MB | Validator/Director/Storage |

### 3.2 Build Environment

| Requirement | Installed Version | Min Required |
|-------------|-------------------|--------------|
| Rust | 1.92.0 | 1.75+ |
| WASM Target | wasm32-unknown-unknown | Required |
| Python | 3.12.3 | 3.10+ |
| CUDA | 13.0 | Required for Vortex |
| GPU | RTX 3090 24GB | 12GB+ VRAM |

### 3.3 Build Workarounds Applied

1. **LIBCLANG_PATH**: Created symlink at `/tmp/libclang-workaround/libclang.so` → `/usr/lib/x86_64-linux-gnu/libclang-18.so.18`
2. **rust-src**: Installed via `rustup component add rust-src`
3. **C_INCLUDE_PATH**: Set to `/usr/lib/gcc/x86_64-linux-gnu/13/include`

---

## 4. Blocking Issues

### 4.1 Critical: No Trusted Signers Configured

**Impact:** All off-chain nodes (validators, directors) fail to bootstrap

**Error Message:**
```
Error: Bootstrap error: Chain signer fetch failed: No trusted signers configured
```

**Root Cause:** The `pallet-nsn-bootstrap` requires trusted signers to be configured before off-chain nodes can verify chain state.

**Resolution Options:**

#### Option A: Genesis Configuration (Requires Chain Restart)

Edit `/home/matt/nsn/nsn-chain/runtime/src/genesis_config_presets.rs`:

```rust
// Add to genesis config
nsn_bootstrap: NsnBootstrapConfig {
    trusted_signers: vec![
        // Add public keys of trusted signers
        hex!("d43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d").to_vec(), // Alice
        hex!("8eaf04151687736326c9fea17e25fc5287613693c912909cb226aa4794f26a48").to_vec(), // Bob
    ],
    ..Default::default()
},
```

#### Option B: Runtime Extrinsic (No Restart Required)

1. Connect to chain via Polkadot.js Apps: https://polkadot.js.org/apps/?rpc=ws://127.0.0.1:9944
2. Navigate to: Developer → Extrinsics
3. Select: `sudo` → `sudo(call)`
4. Inner call: `nsnBootstrap` → `setTrustedSigners`
5. Add signer public keys and submit

#### Option C: Programmatic via subxt

```rust
use subxt::{OnlineClient, PolkadotConfig};

let api = OnlineClient::<PolkadotConfig>::from_url("ws://127.0.0.1:9944").await?;
let tx = api.tx().nsn_bootstrap().set_trusted_signers(signers, quorum);
api.sign_and_submit_then_watch_default(&tx, &signer).await?;
```

### 4.2 Medium: Python Environment Missing

**Impact:** Cannot start Vortex AI server

**Required Packages:**
```bash
sudo apt install python3-venv python3-pip
```

**Setup Commands:**
```bash
cd /home/matt/nsn/vortex
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 4.3 Low: Port Conflicts

**Conflicting Ports:**
- Port 9101: Unknown process (may be stale from previous session)
- Port 50051: Unknown process (expected for Vortex gRPC)

**Resolution:**
```bash
sudo lsof -i :9101 -i :50051
# Kill stale processes if necessary
```

---

## 5. Architecture Overview

### 5.1 Dual-Lane Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NSN Network                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ON-CHAIN LAYER                         │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │   │
│  │  │ nsn-stake   │ │ nsn-reputa- │ │ nsn-bootstrap       │ │   │
│  │  │             │ │ tion        │ │ (NEEDS CONFIG)      │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │   │
│  │  │ nsn-epochs  │ │ nsn-bft     │ │ nsn-task-market     │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   OFF-CHAIN LAYER                         │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │   │
│  │  │ Validators  │ │ Directors   │ │ Storage Nodes       │ │   │
│  │  │ (BLOCKED)   │ │ (BLOCKED)   │ │ (BLOCKED)           │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      AI LAYER                             │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │ Vortex (Lane 0: Video Generation)                   │ │   │
│  │  │ Status: NOT STARTED (missing Python deps)           │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Component Dependencies

```
pallet-nsn-stake (foundation)
    │
    ├── pallet-nsn-reputation
    │       │
    │       └── pallet-nsn-epochs
    │               │
    │               └── pallet-nsn-bft
    │
    ├── pallet-nsn-bootstrap  ◄── BLOCKING ISSUE
    │
    ├── pallet-nsn-treasury
    │
    └── pallet-nsn-task-market
            │
            └── pallet-nsn-model-registry
```

---

## 6. Testing Procedures

### 6.1 Pre-Bootstrap Testing (Current State)

These tests can be run with the current configuration:

#### Chain Health Check
```bash
curl -s -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"system_health"}' \
  http://127.0.0.1:9944 | jq
```

#### Runtime Metadata
```bash
curl -s -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"state_getMetadata"}' \
  http://127.0.0.1:9944 | jq -r '.result' | head -c 200
```

#### Block Production (after transactions)
```bash
curl -s -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"chain_getBlock"}' \
  http://127.0.0.1:9944 | jq '.result.block.header.number'
```

### 6.2 Post-Bootstrap Testing

After configuring trusted signers, run these tests:

#### Start Validator Nodes
```bash
for i in {0..4}; do
  PORT=$((9000 + i))
  METRICS=$((9100 + i))
  /home/matt/nsn/node-core/target/release/nsn-node \
    --rpc-url ws://127.0.0.1:9944 \
    --p2p-listen-port $PORT \
    --p2p-metrics-port $METRICS \
    validator-only &
done
```

#### Verify P2P Connectivity
```bash
# Check each validator's metrics endpoint
for i in {0..4}; do
  curl -s http://127.0.0.1:$((9100 + i))/metrics | grep libp2p
done
```

#### Start Director Node
```bash
/home/matt/nsn/node-core/target/release/nsn-node \
  --rpc-url ws://127.0.0.1:9944 \
  --p2p-listen-port 9010 \
  --p2p-metrics-port 9110 \
  director-only &
```

### 6.3 Full Stack Testing

After all components are running:

1. **Epoch Election Test**: Verify directors are elected each epoch (100 blocks)
2. **BFT Consensus Test**: Submit video generation request, verify 3-of-5 consensus
3. **Reputation Oracle Test**: Verify on-chain reputation syncs to off-chain nodes
4. **Task Market Test**: Submit Lane 1 task, verify execution and payment

---

## 7. Penetration Testing Guide

### 7.1 Attack Surface Overview

| Component | Exposure | Risk Level | Notes |
|-----------|----------|------------|-------|
| Chain RPC | Network | High | Unsafe RPC enabled for testing |
| P2P Network | Network | Medium | GossipSub protocol |
| Metrics | Localhost | Low | Prometheus endpoints |
| Vortex gRPC | Network | Medium | AI inference endpoint |

### 7.2 Recommended Pentest Scenarios

#### 7.2.1 Chain Layer Tests

**RPC Injection Tests:**
```bash
# Test for method enumeration
curl -s -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"rpc_methods"}' \
  http://127.0.0.1:9944 | jq '.result.methods[]'

# Test for unauthorized sudo access
curl -s -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"author_submitExtrinsic","params":["0x..."]}' \
  http://127.0.0.1:9944
```

**State Manipulation:**
- Attempt to modify storage without proper keys
- Test runtime upgrade authorization
- Verify sudo is properly restricted

#### 7.2.2 P2P Layer Tests

**Eclipse Attack Simulation:**
- Flood peer discovery with malicious nodes
- Test Sybil resistance in reputation system

**Message Tampering:**
- Inject malformed GossipSub messages
- Test signature verification on video chunks

**DoS Vectors:**
- Connection exhaustion
- Message flood
- Large message attacks

#### 7.2.3 Consensus Tests

**BFT Byzantine Tests:**
- Simulate 2 of 5 validators colluding
- Test equivocation detection
- Verify slashing for double-signing

**Epoch Manipulation:**
- Attempt to front-run epoch elections
- Test stake manipulation timing

#### 7.2.4 AI Layer Tests (When Vortex Running)

**Prompt Injection:**
- Test for model manipulation via crafted prompts
- Verify input sanitization

**Resource Exhaustion:**
- VRAM exhaustion attacks
- Inference queue flooding

### 7.3 Security Tools Recommendations

| Tool | Purpose | Installation |
|------|---------|--------------|
| subxt | Substrate interaction | `cargo install subxt-cli` |
| polkadot-js | Web-based chain interaction | Browser extension |
| nmap | Port scanning | `apt install nmap` |
| grpcurl | gRPC testing | `go install github.com/fullstorydev/grpcurl` |
| libp2p-lookup | P2P inspection | Build from source |

### 7.4 Logging and Monitoring for Pentests

```bash
# Enable debug logging for chain
RUST_LOG=debug ./target/release/nsn-node --dev ...

# Monitor metrics
watch -n 1 'curl -s http://127.0.0.1:9615/metrics | grep -E "block|peer|transaction"'

# Capture network traffic
tcpdump -i any port 9944 or port 30333 -w nsn_capture.pcap
```

---

## 8. Next Steps Checklist

### Immediate (Required for Full Testnet)

- [ ] Configure trusted signers in bootstrap pallet
- [ ] Restart or update chain with signers
- [ ] Start 5 validator nodes
- [ ] Start 1+ director nodes
- [ ] Verify P2P mesh connectivity

### Short-Term (Enhanced Testing)

- [ ] Install Python dependencies (`python3-venv`, `pip`)
- [ ] Setup Vortex virtual environment
- [ ] Start Vortex AI server (placeholder)
- [ ] Test Lane 0 video generation flow (when implemented)
- [ ] Test Lane 1 task marketplace

### Medium-Term (Penetration Testing)

- [ ] Document all RPC methods and test authorization
- [ ] Perform P2P eclipse attack simulation
- [ ] Test BFT consensus with Byzantine validators
- [ ] Fuzz test all extrinsic inputs
- [ ] Load test with concurrent transactions

### Long-Term (Production Readiness)

- [ ] Replace `--unsafe-rpc-external` with proper auth
- [ ] Implement rate limiting on RPC
- [ ] Add TLS to all endpoints
- [ ] Set up monitoring and alerting
- [ ] Document operational runbooks

---

## 9. Quick Reference Commands

### Start Chain
```bash
cd /home/matt/nsn/nsn-chain
./target/release/nsn-node --dev --alice --validator \
  --rpc-port 9944 --rpc-cors all --unsafe-rpc-external
```

### Check Chain Status
```bash
curl -s http://127.0.0.1:9944 -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"system_health"}' | jq
```

### Start Validators (After Bootstrap Config)
```bash
for i in {0..4}; do
  /home/matt/nsn/node-core/target/release/nsn-node \
    --rpc-url ws://127.0.0.1:9944 \
    --p2p-listen-port $((9000+i)) \
    --p2p-metrics-port $((9100+i)) \
    validator-only &
done
```

### View Logs
```bash
tail -f /tmp/claude/-home-matt-nsn/tasks/b3c26a5.output
```

### Kill All NSN Processes
```bash
pkill -f "nsn-node"
```

---

## 10. Contact and Resources

| Resource | Location |
|----------|----------|
| Architecture Document | `.claude/rules/architecture.md` |
| Product Requirements | `.claude/rules/prd.md` |
| Task Manifest | `.tasks/manifest.json` |
| Project Root | `/home/matt/nsn` |
| NSN Chain | `/home/matt/nsn/nsn-chain` |
| Node Core | `/home/matt/nsn/node-core` |
| Vortex | `/home/matt/nsn/vortex` |

---

*Document generated by Claude Code Agent. For questions or updates, reference this handoff in subsequent sessions.*
