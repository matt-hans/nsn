# Pitfalls Research

Research into common mistakes when building libp2p bridges and @polkadot/api integrations for NSN v1.1 viewer networking.

**Reliability:** Research based on libp2p documentation, polkadot-js issues, and community discussions.

---

## libp2p Interoperability

### Topic Hashing Mismatch

Rust libp2p hashes GossipSub topic names by default, while Go and JS implementations use the topic string verbatim. A topic named `libp2p-demo-chat` becomes `RDEpsjSPrAZF9JCK5REt3tao` in Rust but stays as `libp2p-demo-chat` in JS.

- **Warning signs:**
  - Messages published from JS never arrive at Rust nodes
  - Rust nodes show subscription to different topic hash
  - `peer:subscribe` events show mismatched topic identifiers

- **Prevention:**
  - Configure Rust libp2p to use `IdentityHash` for topic names instead of SHA256
  - Verify topic strings match exactly across implementations before integration
  - Log subscribed topics on both sides during development

- **Phase:** Phase 1 (Video Bridge) — configure before first GossipSub connection

**Source:** [rust-libp2p#473](https://github.com/libp2p/rust-libp2p/issues/473)

---

### Noise Protocol Handshake Pattern

Only the `NoiseConfig::xx` handshake pattern is guaranteed interoperable across libp2p implementations. Other patterns may work within a single implementation but fail cross-language.

- **Warning signs:**
  - Connection established but immediately closed
  - Handshake timeout errors in logs
  - Security protocol negotiation failures

- **Prevention:**
  - Use `XX` handshake pattern explicitly in both Rust and JS configurations
  - Do not rely on default handshake patterns
  - Test handshake with minimal config before adding other protocols

- **Phase:** Phase 1 (Video Bridge) — verify during initial connection setup

**Source:** [libp2p Noise documentation](https://docs.libp2p.io/concepts/secure-comm/noise/)

---

### Transport Protocol Selection for Browsers

Browsers cannot access raw TCP/QUIC sockets. WebSocket requires CA-signed TLS certificates and domain names in secure contexts. WebRTC works without CA certificates but has browser API limitations.

- **Warning signs:**
  - Mixed content errors in browser console
  - Certificate validation failures
  - Connection timeouts on localhost with HTTPS pages

- **Prevention:**
  - For local testnet: Use `ws://` (insecure WebSocket) with HTTP-served viewer, not HTTPS
  - For production: Use WebRTC-direct or WebTransport (when stable)
  - Note: WebSocket results in double encryption (TLS + Noise) — acceptable overhead for testnet

- **Phase:** Phase 1 (Video Bridge) — architecture decision during bridge design

**Source:** [libp2p WebTransport blog](https://blog.libp2p.io/2022-12-19-libp2p-webtransport/)

---

### Stream Reset and Half-Close Limitations

Browser WebRTC implementation does not support stream resets or half-closing streams. js-libp2p implements message framing on data channels to work around this.

- **Warning signs:**
  - Streams hang when one side finishes sending
  - Resource leaks from unclosed streams
  - "Stream already closed" errors on partial sends

- **Prevention:**
  - Use explicit application-level end-of-stream markers
  - Implement timeout-based stream cleanup
  - Test bidirectional communication patterns early

- **Phase:** Phase 2 (Chunk Reception) — important for video chunk streaming

**Source:** [libp2p WebRTC documentation](https://docs.libp2p.io/concepts/transports/webrtc/)

---

## GossipSub

### Sequence Number Implementation Differences

Rust libp2p sends sequence numbers as 64-bit big-endian unsigned integers. When signed, they're monotonically increasing from a random start. Go implementation uses sequential integers. This can cause message deduplication issues.

- **Warning signs:**
  - Duplicate messages received intermittently
  - Messages dropped as "already seen" when they're new
  - Inconsistent message delivery between peers

- **Prevention:**
  - Use consistent message ID function across implementations
  - Configure `message_id_fn` to use content-based hashing, not sequence numbers
  - Test with high message rates to expose deduplication bugs early

- **Phase:** Phase 2 (Chunk Reception) — verify during chunk delivery testing

**Source:** [libp2p-gossipsub Rust docs](https://docs.rs/libp2p-gossipsub)

---

### Insufficient Peers for Publishing

GossipSub requires a minimum number of mesh peers (typically 6) to publish messages. New nodes may fail to publish immediately after connecting.

- **Warning signs:**
  - "InsufficientPeers" or "NotEnoughPeers" errors on publish
  - Messages sent but never received
  - Publishing works from CLI but fails programmatically

- **Prevention:**
  - Wait for mesh formation after connecting (implement ready check)
  - Subscribe to topic before attempting to publish
  - For bridge: implement connection health check before marking service ready

- **Phase:** Phase 1 (Video Bridge) — handle during bridge startup sequence

**Source:** [Rust forum discussion](https://users.rust-lang.org/t/help-with-libp2p-gossipsub-message-publishing/73337)

---

### JS-to-Rust Stream Closure

Documented issue where JS side closes outbound peer immediately after publishing to Rust node. Rust-to-Rust and Rust-to-JS work fine.

- **Warning signs:**
  - Connection established, then immediately dropped
  - "stream suddenly closed" errors on JS side
  - One-way communication works, bidirectional fails

- **Prevention:**
  - Ensure proper async handling in JS publish flow
  - Add connection keepalive mechanisms
  - Test bidirectional messaging explicitly during integration

- **Phase:** Phase 1 (Video Bridge) — critical for bridge reliability

**Source:** [libp2p forum](https://discuss.libp2p.io/t/js-to-rust-via-gossipsub-stream-suddenly-closed-on-js-side/389)

---

### Peer Discovery Not Included

GossipSub does not provide peer discovery. Nodes must discover each other through other mechanisms (DHT, bootstrap nodes, manual configuration).

- **Warning signs:**
  - Nodes subscribe but never see each other's messages
  - Empty peer lists despite running multiple nodes
  - Works with manual `dial()` but not automatically

- **Prevention:**
  - Configure bootstrap nodes with known mesh peers
  - For testnet: hardcode mesh node addresses in bridge config
  - Implement connection status monitoring in bridge

- **Phase:** Phase 1 (Video Bridge) — configure during bridge initialization

**Source:** [libp2p GossipSub documentation](https://docs.libp2p.io/concepts/pubsub/overview/)

---

## SCALE Decoding

### Type Definition Mismatch

SCALE encoding is binary with no field names. Decoder must know exact types and field order. Any mismatch causes complete decode failure or silent data corruption.

- **Warning signs:**
  - "Could not convert parameter" errors
  - Decoded values are nonsense (wrong numbers, corrupted strings)
  - Works on one chain version, fails after upgrade

- **Prevention:**
  - Generate TypeScript types from chain metadata (use `@polkadot/typegen`)
  - Pin type definitions to specific runtime version
  - Validate decoded data with sanity checks (bounds, enums)

- **Phase:** Phase 2 (Chunk Reception) — critical for video chunk decoding

**Source:** [polkadot-js FAQ](https://polkadot.js.org/docs/api/FAQ/)

---

### Field Order Sensitivity

SCALE serialization contains no field names, only encoded values. Decoding depends entirely on size and order of fields matching the definition.

- **Warning signs:**
  - Decoded struct has values in wrong fields
  - Integer overflow errors on small numbers
  - Partial decode success followed by garbage

- **Prevention:**
  - Define types in exact order they appear in Rust structs
  - Use code generation from metadata, not manual definitions
  - Add integration tests that round-trip encode/decode

- **Phase:** Phase 2 (Chunk Reception) — enforce via automated type generation

**Source:** [polkadot-js docs](https://polkadot.js.org/docs/api/start/types.extend/)

---

### Option/Vec Encoding Edge Cases

`Option<Vec<u8>>` and similar nested types have encoding quirks. The issue is with hex representation in JSON RPC, not the binary SCALE format itself.

- **Warning signs:**
  - `None` values decode as empty arrays instead of null
  - Nested options produce unexpected byte patterns
  - RPC responses differ from direct storage reads

- **Prevention:**
  - Test edge cases explicitly: `None`, `Some([])`, `Some([0])`
  - Use typed queries rather than raw storage reads when possible
  - Validate option handling in both directions

- **Phase:** Phase 3 (Chain RPC) — test during director query implementation

**Source:** [polkadot-js/api#4208](https://github.com/polkadot-js/api/issues/4208)

---

### Type Clashes Across Pallets

A chain can define `Balance` as `u128` in one pallet and `u64` in another. polkadot-js uses global type definitions and will apply the wrong one.

- **Warning signs:**
  - Correct encoding for one pallet, wrong for another
  - "Unexpected number of bytes" errors
  - Values are powers of 2 off (u64 vs u128)

- **Prevention:**
  - Use explicit type aliases: `TreasuryBalance`, `AssetsBalance`
  - Configure type aliases in API initialization
  - Check for type name collisions when adding new pallets

- **Phase:** Phase 3 (Chain RPC) — review pallet types during integration

**Source:** [polkadot-js FAQ](https://polkadot.js.org/docs/api/FAQ/)

---

### Runtime Upgrade Type Breakage

Runtime upgrades can change type definitions. API with old types will fail to decode new data. The `RefCount` type changed from `u8` to `u32` in Substrate 2.0.

- **Warning signs:**
  - Queries work, then suddenly fail after block N
  - "Unable to resolve type" errors mentioning new type names
  - AccountInfo or balance queries return wrong values

- **Prevention:**
  - For testnet: pin to specific runtime version
  - Monitor for runtime upgrade events
  - Plan for API type updates when upgrading chain

- **Phase:** Out of scope for v1.1 — document for future mainnet considerations

**Source:** [polkadot-js/api#4518](https://github.com/polkadot-js/api/issues/4518)

---

## WebSocket Management

### Silent Connection Failures

Browsers don't enable WebSocket keepalive by default. Broken connections may not fire `close` events for extended periods, until TCP timeout (minutes).

- **Warning signs:**
  - No messages received but no error events
  - Reconnection logic never triggers
  - "Working" connections that aren't delivering data

- **Prevention:**
  - Implement application-level heartbeat (ping/pong every 15-20 seconds)
  - Set explicit timeout for expected responses
  - Monitor last-message-received timestamp

- **Phase:** Phase 1 (Video Bridge) — implement in bridge-to-viewer connection

**Source:** [websockets documentation](https://websockets.readthedocs.io/en/stable/topics/keepalive.html)

---

### Background Tab Throttling

Browsers throttle WebSocket activity when tabs are in background. Connections may drop and reconnect every 3 minutes after 5 minutes of background.

- **Warning signs:**
  - Video stops when user switches tabs
  - Periodic reconnection events in inactive tabs
  - Works fine when tab is focused

- **Prevention:**
  - Accept this limitation for testnet — document expected behavior
  - Consider Web Worker for WebSocket handling (partial mitigation)
  - Buffer recent chunks for quick resume on tab focus

- **Phase:** Phase 2 (Chunk Reception) — document as known limitation

**Source:** [supabase/realtime-js#121](https://github.com/supabase/realtime-js/issues/121)

---

### Missing Event Handlers

Without handlers for `open`, `error`, and `close` events, connection failures become invisible. Debugging is significantly harder.

- **Warning signs:**
  - "It just stopped working" with no error logs
  - Inconsistent connection behavior across browsers
  - Works in dev tools but fails in production

- **Prevention:**
  - Always register all four handlers: `open`, `message`, `error`, `close`
  - Log all connection state transitions
  - Include reconnection state in UI during development

- **Phase:** Phase 1 (Video Bridge) — enforce in code review

**Source:** [Postman WebSocket guide](https://blog.postman.com/websocket-connection-failed/)

---

### Thundering Herd on Reconnection

If many viewers disconnect simultaneously (server restart, network blip), they all reconnect at once, potentially overwhelming the bridge.

- **Warning signs:**
  - Bridge crashes or becomes unresponsive during reconnection spike
  - Memory/CPU spike followed by timeout cascade
  - Works with few viewers, fails with many

- **Prevention:**
  - Implement exponential backoff with jitter for reconnection
  - Add connection rate limiting on bridge side
  - For testnet: acceptable risk with limited viewers

- **Phase:** Phase 1 (Video Bridge) — implement backoff in viewer client

**Source:** [Making.Close engineering blog](https://making.close.com/posts/reliable-websockets/)

---

### Proxy Idle Timeout

HTTP/1.1 infrastructure (proxies, load balancers) typically closes idle connections after 30-120 seconds. WebSocket connections without traffic will be terminated.

- **Warning signs:**
  - Connections drop after consistent idle period
  - Works locally but fails through reverse proxy
  - Docker networking introduces unexpected timeouts

- **Prevention:**
  - Implement heartbeat below proxy timeout threshold (every 20 seconds)
  - Configure proxy timeouts explicitly in Docker Compose
  - Test through full network stack, not just localhost

- **Phase:** Phase 1 (Video Bridge) — configure in Docker Compose deployment

**Source:** [websockets keepalive docs](https://websockets.readthedocs.io/en/stable/topics/keepalive.html)

---

## Chain RPC

### API Instance Memory Leak

Every `ApiPromise.create()` opens a WebSocket that never auto-disconnects. Creating new instances per call causes connection and memory leaks.

- **Warning signs:**
  - Memory usage grows over time
  - "Too many open files" errors
  - Increasing WebSocket connections in network tab

- **Prevention:**
  - Create single API instance at startup, reuse everywhere
  - Implement singleton pattern or dependency injection
  - Call `api.disconnect()` explicitly when shutting down

- **Phase:** Phase 3 (Chain RPC) — architecture decision for viewer

**Source:** [polkadot-js/docs#11](https://github.com/polkadot-js/docs/issues/11)

---

### Subscription Cleanup

Subscriptions that aren't unsubscribed prevent garbage collection. In React, forgetting to return cleanup function from `useEffect` causes leaks.

- **Warning signs:**
  - Memory grows with navigation/component remounts
  - Stale data callbacks fire after component unmount
  - React warnings about setState on unmounted component

- **Prevention:**
  - Always store and call unsubscribe function
  - Use `useEffect` cleanup pattern in React
  - Consider custom hook that handles subscription lifecycle

- **Phase:** Phase 3 (Chain RPC) — enforce in React component patterns

**Source:** [polkadot-js documentation](https://polkadot.js.org/docs/api/start/api.query.subs/)

**Example pattern:**
```typescript
useEffect(() => {
  let unsubscribe: () => void;

  api.query.nsnDirector.electedDirectors((directors) => {
    setDirectors(directors.toJSON());
  }).then((unsub) => {
    unsubscribe = unsub;
  });

  return () => {
    unsubscribe?.();
  };
}, [api]);
```

---

### WebSocket Disconnect During Subscription

Removing API disconnect while subscriptions are active causes "normal closure" errors. But leaving connection open when done wastes resources.

- **Warning signs:**
  - Subscription callbacks stop firing unexpectedly
  - Connection closed errors in console
  - Inconsistent behavior between query and subscription

- **Prevention:**
  - Track active subscriptions count
  - Only disconnect when all subscriptions cleaned up
  - Implement connection health monitoring

- **Phase:** Phase 3 (Chain RPC) — implement connection lifecycle management

**Source:** [Tanssi documentation](https://docs.tanssi.network/builders/toolkit/substrate-api/libraries/polkadot-js-api/)

---

### Bootstrap Tag Expiration

Bootstrap peers are tagged with 2-minute TTL by default. Connections may be pruned when connection limit is reached, breaking connectivity.

- **Warning signs:**
  - Initial connection works, then fails after ~2 minutes
  - Reconnection attempts fail to find peers
  - Works with few connections, fails under load

- **Prevention:**
  - For browser clients needing persistent bootstrap: set TTL to `Infinity`
  - Monitor peer count and connection health
  - Configure appropriate `maxConnections` limit

- **Phase:** Phase 1 (Video Bridge) — configure in js-libp2p setup

**Source:** [js-libp2p-bootstrap](https://github.com/libp2p/js-libp2p-bootstrap)

---

## Summary: Phase Mapping

| Phase | Critical Pitfalls to Address |
|-------|------------------------------|
| **Phase 1: Video Bridge** | Topic hashing mismatch, Noise XX pattern, transport selection, insufficient peers, JS stream closure, WebSocket handlers, backoff/jitter, heartbeat |
| **Phase 2: Chunk Reception** | Stream half-close, sequence numbers, SCALE type mismatch, field order, background throttling |
| **Phase 3: Chain RPC** | API instance reuse, subscription cleanup, connection lifecycle, Option/Vec edge cases, type clashes |
| **Future (v1.2+)** | Runtime upgrade handling, production scaling |

---

*Research completed: 2026-01-18*
*Sources: libp2p documentation, polkadot-js GitHub issues, community forums*
