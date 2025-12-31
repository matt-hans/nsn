# T024 Kademlia DHT Implementation Notes

## Status: 80% Complete

### Completed Components

1. ✅ Integration tests (`tests/integration_kademlia.rs`)
2. ✅ Kademlia module (`src/kademlia.rs`)
3. ✅ Kademlia helpers (`src/kademlia_helpers.rs`)
4. ✅ NsnBehaviour updated with Kademlia
5. ✅ Cargo.toml dependencies
6. ✅ lib.rs exports

### Remaining Work

**service.rs integration** - Need to:

1. Update constructor to use `build_kademlia` helper:

```rust
// In P2pService::new(), replace Kademlia creation with:
use super::kademlia_helpers::build_kademlia;

let kademlia = build_kademlia(local_peer_id);
let behaviour = NsnBehaviour::new(gossipsub, kademlia);
```

2. Initialize query tracking in constructor:

```rust
// In P2pService::new(), after connection_manager:
pending_get_closest_peers: HashMap::new(),
pending_get_providers: HashMap::new(),
pending_start_providing: HashMap::new(),
local_provided_shards: Vec::new(),
```

3. Add DHT command handlers in `handle_command()`:

```rust
ServiceCommand::GetClosestPeers(target, result_tx) => {
    let query_id = self.swarm.behaviour_mut().kademlia.get_closest_peers(target);
    self.pending_get_closest_peers.insert(query_id, result_tx);
    debug!("get_closest_peers query initiated: {:?}", query_id);
}

ServiceCommand::PublishProvider(shard_hash, result_tx) => {
    let key = RecordKey::new(&shard_hash);
    match self.swarm.behaviour_mut().kademlia.start_providing(key) {
        Ok(query_id) => {
            self.pending_start_providing.insert(query_id, result_tx);
            if !self.local_provided_shards.contains(&shard_hash) {
                self.local_provided_shards.push(shard_hash);
            }
            info!("start_providing: shard={}", hex::encode(shard_hash));
        }
        Err(e) => {
            let _ = result_tx.send(Err(KademliaError::ProviderPublishFailed(format!("{:?}", e))));
        }
    }
}

ServiceCommand::GetProviders(shard_hash, result_tx) => {
    let key = RecordKey::new(&shard_hash);
    let query_id = self.swarm.behaviour_mut().kademlia.get_providers(key);
    self.pending_get_providers.insert(query_id, result_tx);
    debug!("get_providers query: shard={}", hex::encode(shard_hash));
}

ServiceCommand::GetRoutingTableSize(result_tx) => {
    let size = self.swarm.behaviour_mut().kademlia.iter_peers().count();
    let _ = result_tx.send(Ok(size));
}

ServiceCommand::TriggerRoutingTableRefresh(result_tx) => {
    let random_peer = PeerId::random();
    self.swarm.behaviour_mut().kademlia.get_closest_peers(random_peer);
    let _ = result_tx.send(Ok(()));
}
```

4. Handle KademliaEvent in event loop:

```rust
// In start() event loop, add handling for Kademlia events:
SwarmEvent::Behaviour(NsnBehaviourEvent::Kademlia(kad_event)) => {
    self.handle_kademlia_event(kad_event)?;
}
```

5. Add helper method to handle Kademlia events:

```rust
fn handle_kademlia_event(&mut self, event: kad::Event) -> Result<(), ServiceError> {
    use libp2p::kad::{QueryResult, GetProvidersOk, GetClosestPeersError, GetProvidersError};

    match event {
        kad::Event::OutboundQueryProgressed { id, result, .. } => {
            match result {
                QueryResult::GetClosestPeers(Ok(ok)) => {
                    if let Some(tx) = self.pending_get_closest_peers.remove(&id) {
                        let _ = tx.send(Ok(ok.peers));
                    }
                }
                QueryResult::GetClosestPeers(Err(GetClosestPeersError::Timeout { .. })) => {
                    if let Some(tx) = self.pending_get_closest_peers.remove(&id) {
                        let _ = tx.send(Err(KademliaError::Timeout));
                    }
                }
                QueryResult::GetProviders(Ok(GetProvidersOk { providers, .. })) => {
                    if let Some(tx) = self.pending_get_providers.remove(&id) {
                        let _ = tx.send(Ok(providers));
                    }
                }
                QueryResult::GetProviders(Err(GetProvidersError::Timeout { .. })) => {
                    if let Some(tx) = self.pending_get_providers.remove(&id) {
                        let _ = tx.send(Err(KademliaError::Timeout));
                    }
                }
                QueryResult::StartProviding(Ok(_)) => {
                    if let Some(tx) = self.pending_start_providing.remove(&id) {
                        let _ = tx.send(Ok(true));
                    }
                }
                QueryResult::StartProviding(Err(e)) => {
                    if let Some(tx) = self.pending_start_providing.remove(&id) {
                        let _ = tx.send(Err(KademliaError::ProviderPublishFailed(format!("{:?}", e))));
                    }
                }
                _ => {}
            }
        }
        kad::Event::RoutingUpdated { peer, .. } => {
            debug!("Routing table updated: added {}", peer);
        }
        _ => {}
    }
    Ok(())
}
```

### metrics.rs Updates

Add to P2pMetrics struct:

```rust
// DHT metrics
pub dht_routing_table_size: prometheus::Gauge,
pub dht_query_latency: prometheus::Histogram,
pub dht_provider_records_published: prometheus::Counter,
pub dht_routing_table_refreshes: prometheus::Counter,
```

Initialize in `new()`:

```rust
let dht_routing_table_size = prometheus::Gauge::new(
    "nsn_dht_routing_table_size",
    "Number of peers in DHT routing table"
)?;
registry.register(Box::new(dht_routing_table_size.clone()))?;

let dht_query_latency = prometheus::Histogram::new(
    "nsn_dht_query_latency_seconds",
    "DHT query latency in seconds"
)?;
registry.register(Box::new(dht_query_latency.clone()))?;

let dht_provider_records_published = prometheus::Counter::new(
    "nsn_dht_provider_records_published_total",
    "Total provider records published"
)?;
registry.register(Box::new(dht_provider_records_published.clone()))?;

let dht_routing_table_refreshes = prometheus::Counter::new(
    "nsn_dht_routing_table_refreshes_total",
    "Total routing table refresh operations"
)?;
registry.register(Box::new(dht_routing_table_refreshes.clone()))?;
```

### Build & Test Commands

```bash
# Build
cd node-core/crates/p2p
cargo build

# Run unit tests
cargo test --lib kademlia

# Run integration tests
cargo test --test integration_kademlia -- --test-threads=1

# Check metrics
curl http://localhost:9100/metrics | grep dht
```

### Acceptance Criteria Checklist

- [ ] DHT Initialization with `/nsn/kad/1.0.0` protocol ID
- [ ] Peer Discovery via `get_closest_peers`
- [ ] Provider Records: publish/query
- [ ] Content Addressing with shard hash
- [ ] Bootstrap from known peers
- [ ] k-bucket routing table (k=20)
- [ ] Periodic refresh every 5 minutes
- [ ] Provider records republished every 12 hours
- [ ] Query timeout 10 seconds
- [ ] Metrics exposed

## Next Steps

1. Apply the code snippets above to complete service.rs
2. Update metrics.rs with DHT metrics
3. Run `cargo build` and fix any compilation errors
4. Run tests
5. Use `/task-complete` to validate
