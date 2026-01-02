# T064: Implement Storage Crate with Pluggable Backends

## Priority: P2
## Complexity: 3-4 weeks
## Status: Pending
## Depends On: T005 (Pinning Pallet), T011 (Super-Node)

---

## Objective

Implement the `crates/storage` backend layer with at least two adapters (local filesystem + IPFS) and integrate with Super-Node for real persistence.

## Background

`crates/storage` is currently a placeholder. On-chain pinning commitments exist, but there is no real storage backend for content availability.

## Implementation

### Step 1: Define Storage Trait

```rust
pub trait StorageBackend {
    fn put(&self, cid: &Cid, data: &[u8]) -> Result<(), StorageError>;
    fn get(&self, cid: &Cid) -> Result<Vec<u8>, StorageError>;
    fn pin(&self, cid: &Cid) -> Result<(), StorageError>;
    fn unpin(&self, cid: &Cid) -> Result<(), StorageError>;
}
```

### Step 2: Local Filesystem Backend

- CID-based path layout (`storage/<CID>/chunk.bin`)
- Atomic writes + checksum validation
- Background cleanup for expired deals

### Step 3: IPFS Backend

- Use HTTP API (`/api/v0/add`, `/api/v0/cat`, `/api/v0/pin/add`)
- Configurable IPFS endpoint
- Retry + exponential backoff

### Step 4: Integration with Super-Node

- Replace ad-hoc filesystem logic with `StorageBackend`
- Hook audit proof generation into backend reads

### Step 5: Configuration + Metrics

- Configurable backend selection
- Metrics for put/get latency and pin success

## Acceptance Criteria

- [ ] Storage trait defined with test coverage
- [ ] Local backend works for put/get/pin/unpin
- [ ] IPFS backend works against local IPFS daemon
- [ ] Super-Node uses storage crate instead of direct FS calls
- [ ] Audit paths use storage backend reads
- [ ] Metrics emitted for storage operations

## Testing

- Unit tests for local backend
- Integration test with local IPFS daemon (docker)
- Super-Node end-to-end test storing and retrieving shard

## Deliverables

1. `crates/storage/src/lib.rs` (trait + errors)
2. `crates/storage/src/local.rs` (filesystem backend)
3. `crates/storage/src/ipfs.rs` (IPFS backend)
4. Super-Node integration changes

---

**This task enables real storage persistence and availability.**
