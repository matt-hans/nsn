# Task T038: ICN Chain Specification and Genesis Configuration

## Metadata
```yaml
id: T038
title: ICN Chain Specification and Genesis Configuration
status: pending
priority: P1
tags: [chain, genesis, configuration, on-chain, phase-a]
estimated_tokens: 8000
actual_tokens: 0
dependencies: [T001]
created_at: 2025-12-24
updated_at: 2025-12-24
```

## Description

Define the ICN Chain specification and genesis configuration for both testnet and mainnet deployments. This includes network identity (SS58 prefix, token decimals), initial balances, sudo account configuration, validator session keys, bootnode multiaddrs, and chain properties.

## Business Context

**Why this matters**: The chain spec defines ICN's identity in the Polkadot ecosystem. Proper genesis configuration ensures:
- Correct token economics from block 0
- Trusted validator set for initial security
- Discoverable bootnodes for network formation
- Unique network identity (SS58 prefix, chain ID)

**Value delivered**: Enables ICN Testnet and Mainnet deployment with proper configuration.

## Acceptance Criteria

1. ICN SS58 prefix registered (or reserved range selected)
2. Chain properties defined (token symbol: ICN, decimals: 18)
3. Genesis configuration for ICN Testnet created
4. Genesis configuration for ICN Mainnet template created
5. Initial balances configured for treasury, team, and test accounts
6. Sudo account configured for bootstrap governance
7. Validator session keys generation documented
8. Bootnode multiaddrs defined for testnet
9. Chain spec JSON files generated (`icn-testnet.json`, `icn-mainnet.json`)
10. Raw chain spec for production deployment
11. Documentation for validator onboarding

## Test Scenarios

### Scenario 1: Generate Chain Spec
```gherkin
GIVEN ICN Chain runtime compiled
WHEN developer runs: ./target/release/icn-node build-spec --chain=icn-testnet --raw > icn-testnet-raw.json
THEN chain spec JSON generated
  AND contains correct token properties
  AND contains genesis balances
  AND contains bootnode addresses
```

### Scenario 2: Start Node with Chain Spec
```gherkin
GIVEN icn-testnet.json chain spec
WHEN validator runs: ./target/release/icn-node --chain=icn-testnet.json --validator
THEN node starts successfully
  AND connects to bootnodes
  AND syncs from genesis
```

### Scenario 3: Validator Session Key Generation
```gherkin
GIVEN new validator node
WHEN operator runs: curl -X POST http://localhost:9944 -d '{"jsonrpc":"2.0","method":"author_rotateKeys"}'
THEN new session keys generated
  AND BABE, GRANDPA, ImOnline keys created
  AND keys can be submitted to chain
```

### Scenario 4: Genesis Balance Verification
```gherkin
GIVEN ICN Chain started from genesis
WHEN query balances at block 0
THEN treasury has allocated balance (40% of supply)
  AND team accounts have vested allocations
  AND sudo account has operational balance
```

## Technical Implementation

### Chain Properties

```rust
// runtime/src/lib.rs
pub mod properties {
    /// ICN SS58 prefix (use 42 for generic, or register custom)
    pub const SS58_PREFIX: u16 = 42; // TODO: Register ICN-specific prefix
    
    /// Token symbol
    pub const TOKEN_SYMBOL: &str = "ICN";
    
    /// Token decimals (same as DOT/ETH)
    pub const TOKEN_DECIMALS: u8 = 18;
    
    /// Existential deposit (minimum balance)
    pub const EXISTENTIAL_DEPOSIT: u128 = 1_000_000_000_000; // 0.001 ICN
    
    /// Block time in milliseconds
    pub const BLOCK_TIME_MS: u64 = 6000; // 6 seconds
}
```

### Genesis Configuration Template

```rust
// node/src/chain_spec.rs
use icn_runtime::{
    AccountId, BalancesConfig, RuntimeGenesisConfig, SessionConfig, 
    SudoConfig, WASM_BINARY,
};
use sp_core::{sr25519, Pair, Public};
use sp_runtime::traits::IdentifyAccount;

/// ICN Testnet chain spec
pub fn icn_testnet_config() -> Result<ChainSpec, String> {
    let wasm_binary = WASM_BINARY.ok_or("WASM not available")?;
    
    Ok(ChainSpec::builder(wasm_binary, Extensions::default())
        .with_name("ICN Testnet")
        .with_id("icn_testnet")
        .with_chain_type(ChainType::Live)
        .with_protocol_id("icn")
        .with_properties(chain_properties())
        .with_genesis_config(icn_testnet_genesis())
        .with_boot_nodes(icn_testnet_bootnodes())
        .build())
}

fn chain_properties() -> sc_chain_spec::Properties {
    let mut properties = sc_chain_spec::Properties::new();
    properties.insert("tokenSymbol".into(), "ICN".into());
    properties.insert("tokenDecimals".into(), 18.into());
    properties.insert("ss58Format".into(), 42.into());
    properties
}

fn icn_testnet_genesis() -> RuntimeGenesisConfig {
    // Test accounts
    let sudo_key = get_account_id_from_seed::<sr25519::Public>("Alice");
    let validators = vec![
        authority_keys_from_seed("Alice"),
        authority_keys_from_seed("Bob"),
        authority_keys_from_seed("Charlie"),
    ];
    
    // Initial balances (testnet - generous for testing)
    let endowed_accounts = vec![
        (get_account_id_from_seed::<sr25519::Public>("Alice"), 1_000_000 * ICN),
        (get_account_id_from_seed::<sr25519::Public>("Bob"), 1_000_000 * ICN),
        (get_account_id_from_seed::<sr25519::Public>("Charlie"), 1_000_000 * ICN),
        // Faucet account
        (get_account_id_from_seed::<sr25519::Public>("Faucet"), 100_000_000 * ICN),
    ];
    
    RuntimeGenesisConfig {
        system: Default::default(),
        balances: BalancesConfig {
            balances: endowed_accounts,
        },
        session: SessionConfig {
            keys: validators.iter().map(|x| {
                (x.0.clone(), x.0.clone(), session_keys(x.1.clone(), x.2.clone()))
            }).collect(),
        },
        aura: Default::default(),
        grandpa: Default::default(),
        sudo: SudoConfig {
            key: Some(sudo_key),
        },
        // ICN pallets initial config
        icn_stake: Default::default(),
        icn_reputation: Default::default(),
        icn_director: Default::default(),
        icn_bft: Default::default(),
        icn_pinning: Default::default(),
        icn_treasury: Default::default(),
    }
}

fn icn_testnet_bootnodes() -> Vec<MultiaddrWithPeerId> {
    vec![
        // Primary bootnode
        "/dns/boot1.icn-testnet.example.com/tcp/30333/p2p/12D3KooW...".parse().unwrap(),
        // Secondary bootnode
        "/dns/boot2.icn-testnet.example.com/tcp/30333/p2p/12D3KooW...".parse().unwrap(),
    ]
}

/// ICN Mainnet chain spec (template - finalize before launch)
pub fn icn_mainnet_config() -> Result<ChainSpec, String> {
    let wasm_binary = WASM_BINARY.ok_or("WASM not available")?;
    
    Ok(ChainSpec::builder(wasm_binary, Extensions::default())
        .with_name("ICN Mainnet")
        .with_id("icn_mainnet")
        .with_chain_type(ChainType::Live)
        .with_protocol_id("icn")
        .with_properties(chain_properties())
        .with_genesis_config(icn_mainnet_genesis())
        .with_boot_nodes(icn_mainnet_bootnodes())
        .build())
}

fn icn_mainnet_genesis() -> RuntimeGenesisConfig {
    // Mainnet configuration
    // TODO: Replace with actual keys before launch
    
    // Sudo key (multisig in production)
    let sudo_key = AccountId::from_ss58check("5GrwvaEF...").unwrap();
    
    // Initial validators (known operators)
    let validators = vec![
        // Validator 1: ICN Foundation
        authority_keys_from_ss58(
            "5GrwvaEF...", // Stash
            "5GNJqTPy...", // Aura
            "5FA9nQDV...", // GRANDPA
        ),
        // Validator 2: Operator 2
        // ...
    ];
    
    // Mainnet allocations (1B total supply)
    let total_supply = 1_000_000_000 * ICN;
    let endowed_accounts = vec![
        // Treasury (40%)
        (treasury_account(), total_supply * 40 / 100),
        // Development Fund (20%)
        (dev_fund_account(), total_supply * 20 / 100),
        // Ecosystem Growth (15%)
        (ecosystem_account(), total_supply * 15 / 100),
        // Team & Advisors (15%) - with vesting
        (team_account(), total_supply * 15 / 100),
        // Initial Liquidity (10%)
        (liquidity_account(), total_supply * 10 / 100),
    ];
    
    RuntimeGenesisConfig {
        system: Default::default(),
        balances: BalancesConfig {
            balances: endowed_accounts,
        },
        session: SessionConfig {
            keys: validators.iter().map(|x| {
                (x.0.clone(), x.0.clone(), session_keys(x.1.clone(), x.2.clone()))
            }).collect(),
        },
        aura: Default::default(),
        grandpa: Default::default(),
        sudo: SudoConfig {
            key: Some(sudo_key),
        },
        icn_stake: Default::default(),
        icn_reputation: Default::default(),
        icn_director: Default::default(),
        icn_bft: Default::default(),
        icn_pinning: Default::default(),
        icn_treasury: Default::default(),
    }
}

// Constants
const ICN: u128 = 1_000_000_000_000_000_000; // 10^18
```

### Commands

```bash
# Build chain spec (human-readable)
./target/release/icn-node build-spec --chain=icn-testnet > chain-specs/icn-testnet.json

# Build raw chain spec (for distribution)
./target/release/icn-node build-spec --chain=icn-testnet --raw > chain-specs/icn-testnet-raw.json

# Generate validator session keys
curl -H "Content-Type: application/json" \
  -d '{"id":1, "jsonrpc":"2.0", "method": "author_rotateKeys", "params":[]}' \
  http://localhost:9944

# Start validator node with chain spec
./target/release/icn-node \
  --chain=chain-specs/icn-testnet-raw.json \
  --validator \
  --name "ICN-Validator-1" \
  --rpc-port 9944 \
  --port 30333
```

## Dependencies

- **T001**: ICN Chain bootstrap (runtime must exist)

## Design Decisions

1. **SS58 Prefix**: Use generic prefix (42) initially, register ICN-specific prefix later
2. **18 Decimals**: Match Ethereum/DOT convention for tooling compatibility
3. **Sudo for Bootstrap**: Enables fast iteration during testnet, replaced with multisig/council for mainnet
4. **3-5 Initial Validators**: Small trusted set for MVP, expand for mainnet

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Genesis configuration error | High | Medium | Test thoroughly on local network first |
| Validator key compromise | Critical | Low | Use HSM, geographic distribution |
| Bootnode unavailability | Medium | Medium | Multiple bootnodes in different regions |

## Completion Checklist

- [ ] Chain properties defined and documented
- [ ] Testnet genesis configuration complete
- [ ] Mainnet genesis template prepared
- [ ] Session key generation documented
- [ ] Bootnode infrastructure planned
- [ ] Chain spec files generated
- [ ] Validator onboarding guide written

