# Phase 1: Rust Node Core Upgrade - Research

**Researched:** 2026-01-18
**Domain:** libp2p WebRTC transport, certificate persistence, hybrid transport
**Confidence:** HIGH

## Summary

This research investigates adding WebRTC transport to the NSN node-core p2p crate to enable browser connections via WebRTC-Direct. The current codebase uses libp2p 0.53 with TCP, QUIC, noise, and yamux transports. WebRTC support requires adding the `libp2p-webrtc` crate as a direct dependency (NOT a feature of the libp2p umbrella crate) with the `tokio` and `pem` features.

Key findings:
1. **libp2p-webrtc 0.9.0-alpha.2** provides the WebRTC transport with certificate persistence via the `pem` feature
2. The transport is added using `SwarmBuilder::with_other_transport()` to create a hybrid TCP+QUIC+WebRTC setup
3. Certificate persistence uses `Certificate::serialize_pem()` and `Certificate::from_pem()` methods
4. The certhash is automatically included in the multiaddr when listening on WebRTC

**Primary recommendation:** Add libp2p-webrtc as a direct dependency with `tokio` and `pem` features. Create a `cert.rs` module for certificate persistence. Use `with_other_transport()` to add WebRTC alongside existing TCP/QUIC transports.

## Current State Analysis

### Existing P2P Architecture

**File:** `/home/matt/nsn/node-core/crates/p2p/Cargo.toml`

Current libp2p configuration:
```toml
libp2p = { workspace = true, features = ["macros", "relay", "dcutr", "autonat", "kad", "mdns", "tokio", "tcp", "noise", "yamux"] }
```

**Workspace version:** libp2p 0.53 (from `/home/matt/nsn/node-core/Cargo.toml`)

### Transport Configuration

**File:** `/home/matt/nsn/node-core/crates/p2p/src/service.rs` (lines 328-342)

Current SwarmBuilder pattern:
```rust
let mut swarm = SwarmBuilder::with_existing_identity(keypair)
    .with_tokio()
    .with_tcp(
        libp2p::tcp::Config::default(),
        libp2p::noise::Config::new,
        libp2p::yamux::Config::default,
    )
    .map_err(|e| ServiceError::Swarm(format!("TCP transport error: {}", e)))?
    .with_quic()
    .with_behaviour(|_| behaviour)
    .map_err(|e| ServiceError::Swarm(format!("Failed to create behaviour: {}", e)))?
    .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(config.connection_timeout))
    .build();
```

### Listening Addresses

**File:** `/home/matt/nsn/node-core/crates/p2p/src/service.rs` (lines 440-458)

Current setup:
```rust
let quic_addr: Multiaddr = format!("/ip4/0.0.0.0/udp/{}/quic-v1", self.config.listen_port).parse()?;
let tcp_addr: Multiaddr = format!("/ip4/0.0.0.0/tcp/{}", self.config.listen_port).parse()?;

self.swarm.listen_on(quic_addr.clone())?;
self.swarm.listen_on(tcp_addr.clone())?;
```

### Identity Management

**File:** `/home/matt/nsn/node-core/crates/p2p/src/identity.rs`

Existing keypair persistence pattern:
```rust
pub fn save_keypair(keypair: &Keypair, path: &Path) -> Result<(), IdentityError>
pub fn load_keypair(path: &Path) -> Result<Keypair, IdentityError>
```

### CLI Arguments

**File:** `/home/matt/nsn/node-core/bin/nsn-node/src/main.rs` (lines 33-108)

Existing P2P-related CLI flags:
- `--p2p-listen-port` (default: 9000)
- `--p2p-metrics-port` (default: 9100)
- `--p2p-keypair-path` (optional)

### P2pConfig

**File:** `/home/matt/nsn/node-core/crates/p2p/src/config.rs`

Current config structure (extensible via serde):
```rust
pub struct P2pConfig {
    pub listen_port: u16,
    pub max_connections: usize,
    pub keypair_path: Option<PathBuf>,
    pub enable_upnp: bool,
    pub enable_relay: bool,
    pub stun_servers: Vec<String>,
    // ... security and bootstrap configs
}
```

## Standard Stack

### Core Dependencies

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| libp2p-webrtc | 0.9.0-alpha.2 | WebRTC transport | Official rust-libp2p implementation |
| pem | 3.0 | Certificate serialization (optional) | Standard PEM format handling |

### Feature Configuration

| Feature | Required | Purpose |
|---------|----------|---------|
| tokio | YES | Async runtime for transport |
| pem | YES | Certificate persistence via PEM format |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| libp2p-webrtc pem feature | Manual DER serialization | pem feature simpler, standard format |
| Ephemeral certificates | None | Would break certhash stability |

**Installation:**
```toml
# In node-core/crates/p2p/Cargo.toml
[dependencies]
libp2p-webrtc = { version = "0.9.0-alpha.2", features = ["tokio", "pem"] }
```

## Architecture Patterns

### Recommended Project Structure

```
node-core/crates/p2p/src/
├── cert.rs              # NEW: Certificate persistence
├── config.rs            # MODIFY: Add WebRTC config fields
├── identity.rs          # EXISTING: Keypair persistence (reference pattern)
├── service.rs           # MODIFY: Add WebRTC transport
└── lib.rs               # MODIFY: Export cert module
```

### Pattern 1: Certificate Persistence Module

**What:** Module to load or generate WebRTC certificates with disk persistence
**When to use:** Always - certificate must persist for stable certhash

```rust
// Source: libp2p-webrtc certificate.rs (verified)
use libp2p_webrtc::tokio::Certificate;
use std::fs;
use std::path::Path;

pub struct CertificateManager {
    cert_path: PathBuf,
}

impl CertificateManager {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            cert_path: data_dir.join("webrtc_cert.pem"),
        }
    }

    pub fn load_or_generate(&self) -> Result<Certificate, CertError> {
        if self.cert_path.exists() {
            let pem = fs::read_to_string(&self.cert_path)?;
            Certificate::from_pem(&pem)
                .map_err(|e| CertError::Parse(e.to_string()))
        } else {
            let cert = Certificate::generate(&mut rand::thread_rng())?;
            let pem = cert.serialize_pem();
            fs::write(&self.cert_path, pem)?;
            Ok(cert)
        }
    }
}
```

### Pattern 2: Hybrid Transport with SwarmBuilder

**What:** Add WebRTC transport alongside existing TCP/QUIC
**When to use:** During swarm construction in P2pService::new()

```rust
// Source: rust-libp2p browser-webrtc example (verified)
use libp2p_webrtc as webrtc;
use libp2p::core::muxing::StreamMuxerBox;

let webrtc_cert = cert_manager.load_or_generate()?;

let mut swarm = SwarmBuilder::with_existing_identity(keypair)
    .with_tokio()
    .with_tcp(
        libp2p::tcp::Config::default(),
        libp2p::noise::Config::new,
        libp2p::yamux::Config::default,
    )?
    .with_quic()
    .with_other_transport(|id_keys| {
        Ok(webrtc::tokio::Transport::new(
            id_keys.clone(),
            webrtc_cert.clone(),
        )
        .map(|(peer_id, conn), _| (peer_id, StreamMuxerBox::new(conn))))
    })?
    .with_behaviour(|_| behaviour)?
    .build();
```

### Pattern 3: WebRTC Listen Address

**What:** Add WebRTC listener with automatic certhash
**When to use:** In P2pService::start()

```rust
// Source: libp2p WebRTC documentation (verified)
use libp2p::multiaddr::Protocol;

// WebRTC uses UDP, default port 9003 per requirements
let webrtc_port = config.webrtc_port.unwrap_or(9003);
let webrtc_addr: Multiaddr = format!("/ip4/0.0.0.0/udp/{}/webrtc-direct", webrtc_port)
    .parse()?;

self.swarm.listen_on(webrtc_addr)?;

// The certhash is automatically included in NewListenAddr events
```

### Anti-Patterns to Avoid

- **Generating new certificate on each startup:** Would break certhash in advertised multiaddr
- **Using `webrtc` feature on libp2p umbrella crate:** Does not exist for native targets, only `webrtc-websys` for WASM
- **Hardcoding external IP in multiaddr:** Use CLI flag and advertise via swarm.add_external_address()

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Certificate generation | Custom X.509 code | `Certificate::generate()` | Handles DTLS requirements correctly |
| PEM serialization | Manual base64/headers | `serialize_pem()` / `from_pem()` | Matches OpenSSL format, handles key export |
| Certhash in multiaddr | Manual hash computation | libp2p automatic | `NewListenAddr` event includes certhash |
| WebRTC negotiation | Custom SDP/ICE | libp2p-webrtc Transport | Handles DTLS, SCTP, muxing |

**Key insight:** The libp2p-webrtc crate handles all WebRTC complexity including ICE, DTLS, SCTP muxing, and certhash advertisement. Focus on certificate persistence and transport configuration only.

## Common Pitfalls

### Pitfall 1: Missing `pem` Feature Flag

**What goes wrong:** Cannot persist certificates; `serialize_pem()` and `from_pem()` methods unavailable
**Why it happens:** `pem` feature is optional on libp2p-webrtc
**How to avoid:** Always enable both `tokio` and `pem` features:
```toml
libp2p-webrtc = { version = "0.9.0-alpha.2", features = ["tokio", "pem"] }
```
**Warning signs:** Compile error mentioning missing methods on Certificate type

### Pitfall 2: Alpha Crate Incompatibility

**What goes wrong:** libp2p-webrtc 0.9.0-alpha.2 may not match libp2p 0.53 internal crate versions
**Why it happens:** Alpha versioning, workspace version mismatches
**How to avoid:** Pin exact versions; test compilation early in implementation
**Warning signs:** Cargo version conflict errors during `cargo build`

### Pitfall 3: UDP Port Conflicts

**What goes wrong:** WebRTC and QUIC both use UDP; potential port confusion
**Why it happens:** Default QUIC uses same port as listen_port
**How to avoid:** Use separate port for WebRTC (9003 vs 9000 for QUIC)
**Warning signs:** "Address already in use" errors

### Pitfall 4: Certificate File Permissions

**What goes wrong:** Certificate file readable by other users leaks private key
**Why it happens:** Default file permissions may be too permissive
**How to avoid:** Follow existing keypair pattern - set 0o600 permissions:
```rust
#[cfg(unix)]
{
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(&self.cert_path)?.permissions();
    perms.set_mode(0o600);
    fs::set_permissions(&self.cert_path, perms)?;
}
```
**Warning signs:** Security audit flagging certificate file permissions

### Pitfall 5: External Address Not Advertised

**What goes wrong:** Browser cannot connect because internal IP advertised instead of external
**Why it happens:** Docker/NAT environment, no explicit external address
**How to avoid:** Support `--p2p-external-address` CLI flag:
```rust
if let Some(external) = config.external_address {
    self.swarm.add_external_address(external);
}
```
**Warning signs:** Browsers failing to connect with "unreachable" errors

## Code Examples

### Example 1: Complete Certificate Manager

```rust
// Source: Pattern derived from identity.rs + libp2p-webrtc certificate.rs
use libp2p_webrtc::tokio::Certificate;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CertError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Certificate generation failed: {0}")]
    Generation(String),

    #[error("Certificate parse error: {0}")]
    Parse(String),
}

pub struct CertificateManager {
    cert_path: PathBuf,
}

impl CertificateManager {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            cert_path: data_dir.join("webrtc_cert.pem"),
        }
    }

    pub fn load_or_generate(&self) -> Result<Certificate, CertError> {
        if self.cert_path.exists() {
            tracing::info!("Loading WebRTC certificate from {:?}", self.cert_path);
            let pem = fs::read_to_string(&self.cert_path)?;
            Certificate::from_pem(&pem)
                .map_err(|e| CertError::Parse(format!("{:?}", e)))
        } else {
            tracing::info!("Generating new WebRTC certificate at {:?}", self.cert_path);
            let cert = Certificate::generate(&mut rand::thread_rng())
                .map_err(|e| CertError::Generation(format!("{:?}", e)))?;

            let pem = cert.serialize_pem();
            fs::write(&self.cert_path, &pem)?;

            // Set restrictive permissions
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(&self.cert_path)?.permissions();
                perms.set_mode(0o600);
                fs::set_permissions(&self.cert_path, perms)?;
            }

            Ok(cert)
        }
    }

    pub fn fingerprint(&self) -> Result<String, CertError> {
        let cert = self.load_or_generate()?;
        Ok(cert.fingerprint().to_string())
    }
}
```

### Example 2: Config Extension

```rust
// Source: Extension to existing config.rs pattern
/// P2P network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2pConfig {
    // ... existing fields ...

    /// Enable WebRTC transport for browser connections
    #[serde(default)]
    pub enable_webrtc: bool,

    /// UDP port for WebRTC connections (default: 9003)
    #[serde(default = "default_webrtc_port")]
    pub webrtc_port: u16,

    /// Path to data directory for certificate persistence
    pub data_dir: Option<PathBuf>,

    /// External address to advertise (for NAT/Docker)
    pub external_address: Option<Multiaddr>,
}

fn default_webrtc_port() -> u16 { 9003 }

impl Default for P2pConfig {
    fn default() -> Self {
        Self {
            // ... existing defaults ...
            enable_webrtc: false,
            webrtc_port: 9003,
            data_dir: None,
            external_address: None,
        }
    }
}
```

### Example 3: CLI Flags

```rust
// Source: Extension to existing main.rs CLI pattern
#[derive(Parser)]
struct Cli {
    // ... existing fields ...

    /// Enable WebRTC transport for browser connections
    #[arg(long)]
    p2p_enable_webrtc: bool,

    /// UDP port for WebRTC connections
    #[arg(long, default_value = "9003")]
    p2p_webrtc_port: u16,

    /// External address to advertise (e.g., /ip4/1.2.3.4/udp/9003/webrtc-direct)
    #[arg(long)]
    p2p_external_address: Option<String>,

    /// Data directory for certificate persistence
    #[arg(long, default_value = "/var/lib/nsn")]
    data_dir: PathBuf,
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| webrtc-rs direct | libp2p-webrtc crate | 2023 | Integrated with libp2p transport system |
| Ephemeral certs only | `pem` feature for persistence | 2024 | Enables stable certhash |
| Manual SDP munging | Automatic WebRTC-Direct | 2023 | No signaling server needed |

**Deprecated/outdated:**
- wngr/libp2p-webrtc (third-party): Replaced by official libp2p-webrtc
- Ring certificate generation: Being replaced by rcgen in newer versions

## Version Compatibility Analysis

### Current Workspace Versions (from `/home/matt/nsn/node-core/Cargo.toml`)

| Crate | Workspace Version |
|-------|-------------------|
| libp2p | 0.53 |
| tokio | 1.35 |
| sp-core | 28.0 |

### libp2p-webrtc Compatibility

libp2p-webrtc 0.9.0-alpha.2 depends on:
- libp2p-core (workspace)
- libp2p-noise (workspace)
- libp2p-identity (workspace)
- webrtc 0.12.0

**Risk Assessment:** The alpha status of libp2p-webrtc and potential version mismatches with libp2p 0.53 internals is a MEDIUM risk. Recommend early compilation test.

## Open Questions

1. **Version Compatibility**
   - What we know: libp2p-webrtc 0.9.0-alpha.2 exists, libp2p 0.53 is current workspace version
   - What's unclear: Whether alpha crate versions align with stable libp2p 0.53 internal crates
   - Recommendation: Add dependency early and verify compilation before detailed implementation

2. **HTTP Discovery Endpoint Location**
   - What we know: Requirements specify `/p2p/info` endpoint for certhash discovery
   - What's unclear: Whether to add to existing metrics HTTP server or create new endpoint
   - Recommendation: Research in Phase 2 (Discovery Bridge)

## Sources

### Primary (HIGH confidence)
- [rust-libp2p GitHub](https://github.com/libp2p/rust-libp2p) - Official repository
- [libp2p-webrtc certificate.rs](https://github.com/libp2p/rust-libp2p/blob/master/transports/webrtc/src/tokio/certificate.rs) - Certificate API
- [browser-webrtc example](https://github.com/libp2p/rust-libp2p/tree/master/examples/browser-webrtc) - SwarmBuilder pattern
- [libp2p-webrtc Cargo.toml](https://raw.githubusercontent.com/libp2p/rust-libp2p/master/transports/webrtc/Cargo.toml) - Version and features

### Secondary (MEDIUM confidence)
- [libp2p WebRTC docs](https://docs.libp2p.io/concepts/transports/webrtc/) - Protocol overview
- [crates.io libp2p-webrtc](https://crates.io/crates/libp2p-webrtc) - Package registry

### Local Codebase (HIGH confidence)
- `/home/matt/nsn/node-core/crates/p2p/Cargo.toml` - Current dependencies
- `/home/matt/nsn/node-core/crates/p2p/src/service.rs` - SwarmBuilder usage
- `/home/matt/nsn/node-core/crates/p2p/src/identity.rs` - Keypair persistence pattern
- `/home/matt/nsn/node-core/crates/p2p/src/config.rs` - Config structure
- `/home/matt/nsn/node-core/bin/nsn-node/src/main.rs` - CLI argument pattern

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Direct dependency approach verified via official examples
- Architecture: HIGH - SwarmBuilder pattern confirmed in codebase and examples
- Pitfalls: MEDIUM - Alpha crate compatibility needs runtime verification

**Research date:** 2026-01-18
**Valid until:** 2026-02-18 (30 days - alpha crate may have updates)

---

## Key Files to Modify

| File | Change Type | Purpose |
|------|-------------|---------|
| `node-core/crates/p2p/Cargo.toml` | ADD | libp2p-webrtc dependency |
| `node-core/crates/p2p/src/cert.rs` | CREATE | Certificate persistence module |
| `node-core/crates/p2p/src/config.rs` | MODIFY | Add WebRTC config fields |
| `node-core/crates/p2p/src/service.rs` | MODIFY | Add WebRTC transport and listener |
| `node-core/crates/p2p/src/lib.rs` | MODIFY | Export cert module |
| `node-core/bin/nsn-node/src/main.rs` | MODIFY | Add CLI flags |
| `node-core/Cargo.toml` | MODIFY | Add libp2p-webrtc to workspace deps |

## Implementation Order

1. Add dependency to workspace Cargo.toml and p2p/Cargo.toml
2. Verify compilation succeeds (check alpha compatibility)
3. Create cert.rs module with CertificateManager
4. Extend P2pConfig with WebRTC fields
5. Modify SwarmBuilder to add WebRTC transport
6. Add WebRTC listener in start() method
7. Add CLI flags to nsn-node
8. Write unit tests for certificate persistence
