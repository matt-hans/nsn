# Task T041: Snowbridge/Bridge Hub Integration Planning

## Metadata
```yaml
id: T041
title: Snowbridge/Bridge Hub Integration Planning
status: pending
priority: P3
tags: [bridge, ethereum, snowbridge, on-chain, phase-d]
estimated_tokens: 10000
actual_tokens: 0
dependencies: [T039]
created_at: 2025-12-24
updated_at: 2025-12-24
```

## Description

**PHASE D TASK** - Plan and implement Snowbridge integration for trustless Ethereum ↔ NSN bridging once NSN is a Polkadot parachain. This provides Ethereum mainnet access without adding Frontier EVM directly to NSN Chain.

**Alternative to T008**: Instead of adding EVM to NSN Chain, use Snowbridge for Ethereum ecosystem access.

This task is **NOT required for MVP** and only relevant after T039 (Cumulus integration) is complete.

## Business Context

**Why Snowbridge**:
- **Ethereum Access**: Bridge NSN tokens to Ethereum for DeFi, CEX listings
- **Trustless**: Decentralized bridge, not custodial
- **No EVM Overhead**: Don't need Frontier on NSN Chain
- **Existing Infrastructure**: Uses Polkadot's Bridge Hub system chain

**When to do this**:
- After NSN is parachain (T039)
- When Ethereum ecosystem access is needed
- When token bridging demand exists

## Acceptance Criteria

1. Snowbridge architecture understood and documented
2. Bridge Hub XCM integration designed
3. ICN token registration on Ethereum Gateway planned
4. Bridge message flow designed (ICN → Ethereum and back)
5. Security considerations documented
6. Bridge fee structure analyzed
7. Testing plan on testnet (Goerli/Sepolia ↔ Rococo)

## Snowbridge Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ETHEREUM MAINNET                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 GATEWAY CONTRACT                         │    │
│  │  - Receives messages from Polkadot                       │    │
│  │  - Sends messages to Polkadot                           │    │
│  │  - Manages token locking/minting                        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │ Bridge Messages
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POLKADOT BRIDGE HUB                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 SNOWBRIDGE PALLETS                       │    │
│  │  - Ethereum Light Client                                │    │
│  │  - Message Verification                                 │    │
│  │  - XCM Router to Parachains                            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │ XCM Messages
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NSN PARACHAIN                               │
│  - Receives bridged assets                                      │
│  - Sends assets to Bridge Hub for Ethereum                      │
└─────────────────────────────────────────────────────────────────┘
```

## Bridge Flow: NSN → Ethereum

1. User initiates transfer on NSN Chain
2. NSN locked on NSN Chain
3. XCM message sent to Bridge Hub
4. Bridge Hub sends message to Ethereum Gateway
5. Gateway mints wrapped NSN (wNSN) on Ethereum
6. User receives wNSN in their Ethereum wallet

## Bridge Flow: Ethereum → NSN

1. User deposits wNSN to Gateway contract
2. Gateway sends message to Bridge Hub
3. Bridge Hub routes XCM to NSN Chain
4. NSN unlocked on NSN Chain
5. User receives native NSN

## Technical Implementation

### XCM Configuration for Bridge Hub

```rust
// runtime/src/xcm_config.rs
pub struct XcmConfig;

impl xcm_executor::Config for XcmConfig {
    // ... existing config ...
    
    // Route messages to Bridge Hub
    type XcmSender = XcmRouter;
}

// Define route to Bridge Hub
parameter_types! {
    pub BridgeHubLocation: Location = (Parent, Parachain(BRIDGE_HUB_PARA_ID)).into();
}
```

### ICN Token Registration

```solidity
// On Ethereum - interact with Gateway
interface IGateway {
    function registerForeignToken(
        bytes32 assetId,
        string memory name,
        string memory symbol,
        uint8 decimals
    ) external;
    
    function sendToken(
        bytes32 destinationChain,
        bytes32 destinationAddress,
        bytes32 assetId,
        uint128 amount
    ) external payable;
}
```

## Fee Structure

| Direction | Fee Components |
|-----------|---------------|
| NSN → ETH | XCM fee + Bridge fee + Ethereum gas |
| ETH → NSN | Ethereum gas + Bridge fee + XCM fee |

Estimated total: ~$5-20 depending on Ethereum gas prices

## Security Considerations

1. **Light Client Security**: Snowbridge uses Ethereum light client on Bridge Hub
2. **Message Verification**: All messages cryptographically verified
3. **Rate Limiting**: Bridge may have per-period limits
4. **Finality**: Wait for Ethereum finality (~15 min) before NSN unlock

## Testing Plan

1. Deploy NSN test token on testnet
2. Register with Snowbridge on Rococo
3. Test transfers in both directions
4. Verify token accounting
5. Test failure scenarios (insufficient fees, invalid addresses)
6. Load test for expected volumes

## Dependencies

- **T039**: Cumulus integration (must be parachain first)
- **Snowbridge**: Operational on Polkadot
- **Bridge Hub**: Polkadot Bridge Hub system chain

## Alternatives

### Alternative: T008 (Frontier EVM)
- Add EVM to NSN Chain directly
- Simpler for EVM dApp deployment
- More runtime complexity
- **Recommendation**: Use if EVM tooling critical for developers

### Comparison

| Aspect | T008 (Frontier) | T041 (Snowbridge) |
|--------|-----------------|-------------------|
| EVM Contracts on NSN | Yes | No |
| Ethereum Mainnet Access | Indirect | Direct |
| Runtime Complexity | Higher | Lower |
| Token on Ethereum | Via bridge from NSN | Native on ETH |
| Use Case | EVM dev convenience | ETH ecosystem access |

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Bridge vulnerability | Critical | Low | Use well-audited Snowbridge |
| Long finality delays | Medium | High | Set user expectations, async UX |
| High bridge fees | Medium | Medium | Batch transactions, timing |
| Snowbridge changes | Medium | Low | Pin versions, monitor updates |

## Completion Checklist

- [ ] Snowbridge architecture documented
- [ ] XCM integration designed
- [ ] Token registration planned
- [ ] Fee analysis completed
- [ ] Security review done
- [ ] Testnet testing completed
- [ ] Operations runbook written

