# T045: Implement DHT Walk with Hybrid Lookup

## Priority: P1 (Critical Path)
## Complexity: 1 week
## Status: Pending
## Depends On: T024 (Kademlia DHT)

---

## Objective

Replace the placeholder DHT walk implementation with a real hybrid lookup system that combines random walk for general discovery with content-addressed lookups for finding specific node types.

## Background

Current implementation in `node-core/crates/p2p/src/discovery.rs` returns empty results:

```rust
pub async fn dht_walk(&self) -> Vec<PeerId> {
    // TODO: Implement DHT walk
    vec![]
}
```

This is a critical decentralization failure - nodes cannot discover peers through the DHT.

## Implementation

### Step 1: Implement Random Walk Discovery

```rust
pub async fn random_walk(&self, count: usize) -> Vec<PeerId> {
    let mut discovered = Vec::new();

    // Generate random key for walk
    let random_key = self.generate_random_key();

    // Query DHT for closest peers to random key
    let closest = self.kademlia.get_closest_peers(random_key).await;

    // Continue walking from discovered peers
    for peer in closest.iter().take(count) {
        discovered.push(peer.clone());
    }

    discovered
}
```

### Step 2: Implement Content-Addressed Lookup

```rust
pub async fn find_providers(&self, role: NodeRole) -> Vec<PeerId> {
    // Use role-based DHT keys for discovery
    let role_key = self.role_to_dht_key(role);

    // Find providers for this role
    let providers = self.kademlia.get_providers(role_key).await;

    providers.into_iter().map(|p| p.peer_id).collect()
}

fn role_to_dht_key(&self, role: NodeRole) -> Key {
    match role {
        NodeRole::Director => Key::new(b"/nsn/role/director"),
        NodeRole::Validator => Key::new(b"/nsn/role/validator"),
        NodeRole::SuperNode => Key::new(b"/nsn/role/supernode"),
        NodeRole::Relay => Key::new(b"/nsn/role/relay"),
    }
}
```

### Step 3: Hybrid Lookup Strategy

```rust
pub async fn dht_walk(&self, strategy: DiscoveryStrategy) -> Vec<PeerId> {
    match strategy {
        DiscoveryStrategy::Random { count } => {
            self.random_walk(count).await
        }
        DiscoveryStrategy::ByRole { role, count } => {
            self.find_providers(role).await.into_iter().take(count).collect()
        }
        DiscoveryStrategy::Hybrid { random_count, role } => {
            let mut peers = self.random_walk(random_count).await;
            peers.extend(self.find_providers(role).await);
            peers.dedup();
            peers
        }
    }
}
```

### Step 4: Provider Announcement

Nodes must announce themselves as providers for their role:

```rust
pub async fn announce_role(&self, role: NodeRole) -> Result<(), P2pError> {
    let role_key = self.role_to_dht_key(role);
    self.kademlia.start_providing(role_key).await?;
    Ok(())
}
```

## Acceptance Criteria

- [ ] Random walk returns actual peers from DHT
- [ ] Role-based discovery finds nodes by type (Director, Validator, etc.)
- [ ] Hybrid lookup combines both strategies
- [ ] Nodes announce their roles on startup
- [ ] Integration tests verify discovery works with 10+ nodes
- [ ] Performance: discovery completes in <5 seconds

## Testing

```rust
#[tokio::test]
async fn test_random_walk_discovers_peers() {
    let network = TestNetwork::new(10).await;
    let node = network.get_node(0);

    // Wait for DHT to stabilize
    tokio::time::sleep(Duration::from_secs(2)).await;

    let peers = node.discovery.random_walk(5).await;
    assert!(peers.len() >= 3, "Should discover at least 3 peers");
}

#[tokio::test]
async fn test_role_based_discovery() {
    let network = TestNetwork::new(10).await;

    // Announce roles
    network.get_node(0).announce_role(NodeRole::Director).await.unwrap();
    network.get_node(1).announce_role(NodeRole::Director).await.unwrap();

    // Wait for announcement propagation
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Discover directors from another node
    let directors = network.get_node(5).find_providers(NodeRole::Director).await;
    assert!(directors.len() >= 2, "Should find at least 2 directors");
}
```

## Deliverables

1. `node-core/crates/p2p/src/discovery.rs` - Updated with real DHT walk
2. `node-core/crates/p2p/src/dht/provider.rs` - Provider announcement logic
3. Integration tests
4. Documentation updates

---

**This task is critical for network decentralization.**
