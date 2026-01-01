# T046: Implement Bootstrap Diversification

## Priority: P1 (Critical Path)
## Complexity: 3 days
## Status: Pending
## Depends On: None

---

## Objective

Implement 5+ independent bootstrap sources across different registrars, TLDs, and jurisdictions to ensure censorship resistance and prevent single points of failure.

## Background

Current configuration resolves ALL bootstrap paths to a single domain:

```yaml
bootstrap:
  dns_seeds: ["_dnsaddr.nsn.network"]
  http_seeds: ["https://seeds.nsn.network/peers.json"]
  hardcoded_peers:
    - "/dns4/boot1.nsn.network/tcp/30333/..."
```

This is a critical centralization failure - if `nsn.network` is seized or blocked, the entire network becomes unreachable.

## Implementation

### Step 1: Multi-Registrar DNS Seeds

Configure DNS seeds across different registrars and TLDs:

```yaml
bootstrap:
  dns_seeds:
    # Different registrars, different TLDs
    - "_dnsaddr.nsn.network"      # Primary (.network TLD)
    - "_dnsaddr.nsn.systems"      # Backup (.systems TLD)
    - "_dnsaddr.interdim.io"      # Different brand (.io TLD)
    - "_dnsaddr.nsn-peers.xyz"    # Alternative (.xyz TLD)
    - "_dnsaddr.nsn.lat"          # Geographic diversity (.lat TLD)
```

### Step 2: Distributed HTTP Seeds

Host peer lists on independent infrastructure:

```yaml
http_seeds:
  # Different hosting providers, different jurisdictions
  - "https://seeds.nsn.network/peers.json"       # Primary
  - "https://nsn-seeds.netlify.app/peers.json"   # Netlify CDN
  - "https://nsn-bootstrap.vercel.app/peers.json" # Vercel CDN
  - "https://seeds.interdim.io/peers.json"       # Alternative domain
  - "ipns://k51qzi5uqu5dg4lz..."                  # IPNS (truly decentralized)
```

### Step 3: Geographically Distributed Hardcoded Peers

```yaml
hardcoded_peers:
  # At least 5 peers in different jurisdictions
  - "/dns4/boot-us.nsn.network/tcp/30333/p2p/12D3KooW..."    # US
  - "/dns4/boot-eu.interdim.io/tcp/30333/p2p/12D3KooW..."    # EU
  - "/dns4/boot-asia.nsn.systems/tcp/30333/p2p/12D3KooW..."  # Asia
  - "/ip4/185.xx.xx.xx/tcp/30333/p2p/12D3KooW..."            # Direct IP (Iceland)
  - "/ip4/45.xx.xx.xx/tcp/30333/p2p/12D3KooW..."             # Direct IP (Switzerland)
```

### Step 4: Bootstrap Source Resolver

```rust
pub struct BootstrapResolver {
    dns_seeds: Vec<String>,
    http_seeds: Vec<String>,
    hardcoded_peers: Vec<Multiaddr>,
    timeout: Duration,
}

impl BootstrapResolver {
    pub async fn resolve_all(&self) -> Vec<Multiaddr> {
        let mut peers = Vec::new();
        let mut successful_sources = 0;

        // Try all sources in parallel
        let (dns_peers, http_peers) = tokio::join!(
            self.resolve_dns_seeds(),
            self.resolve_http_seeds()
        );

        // Aggregate results
        if !dns_peers.is_empty() {
            peers.extend(dns_peers);
            successful_sources += 1;
        }

        if !http_peers.is_empty() {
            peers.extend(http_peers);
            successful_sources += 1;
        }

        // Always include hardcoded as fallback
        peers.extend(self.hardcoded_peers.clone());
        successful_sources += 1;

        // Log diversity metrics
        tracing::info!(
            "Bootstrap resolved {} peers from {} sources",
            peers.len(),
            successful_sources
        );

        // Deduplicate by peer ID
        peers.dedup_by(|a, b| {
            extract_peer_id(a) == extract_peer_id(b)
        });

        peers
    }

    async fn resolve_dns_seeds(&self) -> Vec<Multiaddr> {
        let mut results = Vec::new();

        for seed in &self.dns_seeds {
            match self.resolve_dns_seed(seed).await {
                Ok(addrs) => {
                    tracing::debug!("DNS seed {} returned {} peers", seed, addrs.len());
                    results.extend(addrs);
                }
                Err(e) => {
                    tracing::warn!("DNS seed {} failed: {}", seed, e);
                }
            }
        }

        results
    }
}
```

### Step 5: Source Health Monitoring

```rust
pub struct BootstrapHealthMonitor {
    sources: Vec<BootstrapSource>,
}

impl BootstrapHealthMonitor {
    pub async fn check_diversity(&self) -> DiversityReport {
        let mut report = DiversityReport::default();

        for source in &self.sources {
            let health = self.check_source(source).await;
            report.add_source(source.clone(), health);
        }

        // Warn if fewer than 3 sources are healthy
        if report.healthy_count < 3 {
            tracing::warn!(
                "Bootstrap diversity degraded: only {} healthy sources",
                report.healthy_count
            );
        }

        report
    }
}
```

## Acceptance Criteria

- [ ] At least 5 DNS seeds across 3+ different TLDs
- [ ] At least 3 HTTP seeds on different hosting providers
- [ ] At least 5 hardcoded peers in 3+ different jurisdictions
- [ ] At least 2 direct IP addresses (no DNS dependency)
- [ ] Bootstrap succeeds even if primary domain is blocked
- [ ] Health monitoring alerts on diversity degradation
- [ ] Documentation of all bootstrap sources and their jurisdictions

## Testing

```rust
#[tokio::test]
async fn test_bootstrap_resilience() {
    let resolver = BootstrapResolver::new_from_config();

    // Should resolve peers even with primary domain failures
    let peers = resolver.resolve_all().await;
    assert!(peers.len() >= 5, "Should have at least 5 bootstrap peers");
}

#[tokio::test]
async fn test_bootstrap_diversity() {
    let monitor = BootstrapHealthMonitor::new();
    let report = monitor.check_diversity().await;

    assert!(report.unique_tlds >= 3, "Should use at least 3 TLDs");
    assert!(report.unique_registrars >= 2, "Should use at least 2 registrars");
    assert!(report.unique_jurisdictions >= 3, "Should span at least 3 jurisdictions");
}
```

## Deliverables

1. Updated `config.yaml` with diversified bootstrap sources
2. `node-core/crates/p2p/src/bootstrap.rs` - Multi-source resolver
3. Health monitoring integration
4. Documentation of censorship resistance design

---

**This task is critical for censorship resistance.**
