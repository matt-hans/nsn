# Task T008: Optional Frontier EVM Integration (Token + Precompiles)

## Metadata
```yaml
id: T008
title: Optional Frontier EVM Integration (Token + Precompiles)
status: pending
priority: P3
tags: [evm, solidity, precompiles, token, on-chain, phase-d, optional, frontier]
estimated_tokens: 15000
actual_tokens: 0
dependencies: [T001, T002, T003]
created_at: 2025-12-24
updated_at: 2025-12-24
```

## Description

**OPTIONAL TASK** - Add Frontier EVM compatibility to ICN Chain for Ethereum developer convenience. This includes integrating `pallet-evm` and `pallet-ethereum` into the ICN runtime, creating Substrate precompiles for staking and reputation queries from EVM, and enabling dual-interface access (MetaMask + Polkadot.js).

This is Phase D work and is **NOT required for MVP**. ICN token is native to ICN Chain - EVM compatibility is only needed if:
- dApp developers want familiar Ethereum tooling
- Integration with EVM-based DeFi is desired
- MetaMask UX is preferred over Polkadot.js

**Alternative**: For Ethereum mainnet access without adding EVM to ICN Chain, use Snowbridge (T041).

## Business Context

**Why this matters (when needed)**: EVM compatibility can lower barrier to entry:
- **Familiar tooling**: Web3 developers can use ethers.js, Hardhat, Remix
- **MetaMask support**: Users interact with ICN using existing Ethereum wallets
- **DeFi integration**: ICN token tradeable on EVM DEXs
- **Developer reach**: Access to Ethereum ecosystem (~400k developers)

**Value delivered**: When enabled, provides EVM compatibility layer on ICN Chain itself.

**Priority justification**: P3 (optional) because:
- ICN token is NATIVE, not ERC-20 by default
- Core pallet functionality works without EVM
- Substrate API via subxt is the primary integration path
- Snowbridge (T041) provides alternative Ethereum access

## Acceptance Criteria

1. Frontier crates (`pallet-evm`, `pallet-ethereum`) integrated into ICN runtime
2. EVM execution enabled and functional on ICN Chain
3. ICN token ERC-20 wrapper contract deployed (optional)
4. `IcnStakePrecompile` at designated address for EVM → Substrate staking
5. Precompile correctly converts EVM address (20 bytes) to Substrate AccountId (32 bytes)
6. `IcnReputationPrecompile` for reputation queries from EVM
7. MetaMask successfully connects to ICN Chain RPC
8. Hardhat test suite covers basic EVM operations (90%+ coverage)
9. Gas costs optimized: staking precompile <100k gas, reputation query <50k gas
10. Documentation for EVM developers (connection, contracts, precompiles)

## Test Scenarios

### Scenario 1: Enable Frontier EVM in Runtime
```gherkin
GIVEN ICN Chain runtime without EVM
WHEN Frontier pallets added:
  - pallet-evm
  - pallet-ethereum
  - pallet-base-fee
THEN runtime compiles successfully
  AND EVM execution is available
  AND eth_* RPC methods respond
```

### Scenario 2: MetaMask Connection
```gherkin
GIVEN ICN Chain running with Frontier enabled
  AND MetaMask installed
WHEN user adds custom network:
  - Network Name: ICN Chain
  - RPC URL: http://localhost:9944
  - Chain ID: <icn-chain-id>
  - Currency Symbol: ICN
THEN MetaMask connects successfully
  AND shows ICN balance (converted from native)
  AND can send transactions
```

### Scenario 3: Deploy ERC-20 Wrapper Contract
```gherkin
GIVEN ICN Chain with Frontier enabled
  AND deployer has ICN for gas
WHEN ICN token wrapper contract deployed
THEN contract deployed successfully
  AND allows wrap()/unwrap() between native and ERC-20
  AND ERC-20 interface works for DeFi integration
```

### Scenario 4: Stake via EVM Precompile
```gherkin
GIVEN Alice has ICN in EVM balance
  AND IcnStakePrecompile registered
WHEN Alice calls precompile to stake for Director role
THEN ICN deducted from EVM balance
  AND pallet_icn_stake::deposit_stake() called
  AND stake visible in Substrate storage
```

### Scenario 5: Query Reputation via EVM
```gherkin
GIVEN Dave has staked and has reputation scores
WHEN EVM contract calls IcnReputationPrecompile
THEN returns reputation scores from pallet-icn-reputation
  AND gas used <50k
```

## Technical Implementation

### Step 1: Add Frontier Dependencies
```toml
# runtime/Cargo.toml
[dependencies]
# Frontier
pallet-evm = { git = "https://github.com/polkadot-evm/frontier", branch = "polkadot-stable2409", default-features = false }
pallet-ethereum = { git = "https://github.com/polkadot-evm/frontier", branch = "polkadot-stable2409", default-features = false }
pallet-base-fee = { git = "https://github.com/polkadot-evm/frontier", branch = "polkadot-stable2409", default-features = false }
fp-evm = { git = "https://github.com/polkadot-evm/frontier", branch = "polkadot-stable2409", default-features = false }
fp-rpc = { git = "https://github.com/polkadot-evm/frontier", branch = "polkadot-stable2409", default-features = false }
```

### Step 2: Configure Runtime for EVM
```rust
// runtime/src/lib.rs
parameter_types! {
    pub BlockGasLimit: U256 = U256::from(NORMAL_DISPATCH_RATIO * MAXIMUM_BLOCK_WEIGHT.ref_time() / WEIGHT_PER_GAS);
    pub PrecompilesValue: Precompiles = ICNPrecompiles::<Runtime>::new();
    pub WeightPerGas: Weight = Weight::from_parts(WEIGHT_PER_GAS, 0);
}

impl pallet_evm::Config for Runtime {
    type FeeCalculator = BaseFee;
    type GasWeightMapping = pallet_evm::FixedGasWeightMapping<Self>;
    type WeightPerGas = WeightPerGas;
    type BlockHashMapping = pallet_ethereum::EthereumBlockHashMapping<Self>;
    type CallOrigin = EnsureAddressRoot<AccountId>;
    type WithdrawOrigin = EnsureAddressNever<AccountId>;
    type AddressMapping = HashedAddressMapping<BlakeTwo256>;
    type Currency = Balances;
    type RuntimeEvent = RuntimeEvent;
    type PrecompilesType = ICNPrecompiles<Self>;
    type PrecompilesValue = PrecompilesValue;
    type ChainId = ConstU64<ICN_CHAIN_ID>;
    type BlockGasLimit = BlockGasLimit;
    type Runner = pallet_evm::runner::stack::Runner<Self>;
    type OnChargeTransaction = ();
    type OnCreate = ();
    type FindAuthor = ();
    type GasLimitPovSizeRatio = ConstU64<GAS_LIMIT_POV_SIZE_RATIO>;
    type Timestamp = Timestamp;
    type WeightInfo = ();
}
```

### Step 3: ICN Precompiles
```rust
// precompiles/src/lib.rs
pub struct ICNPrecompiles<R>(PhantomData<R>);

impl<R> PrecompileSet for ICNPrecompiles<R>
where
    R: pallet_evm::Config + pallet_icn_stake::Config + pallet_icn_reputation::Config,
{
    fn execute(&self, handle: &mut impl PrecompileHandle) -> Option<PrecompileResult> {
        match handle.code_address() {
            // Standard precompiles (0x01-0x09)
            a if a == H160::from_low_u64_be(1) => Some(ECRecover::execute(handle)),
            // ... other standard precompiles ...
            
            // ICN custom precompiles
            a if a == H160::from_low_u64_be(0x0900) => Some(IcnStakePrecompile::execute(handle)),
            a if a == H160::from_low_u64_be(0x0901) => Some(IcnReputationPrecompile::execute(handle)),
            
            _ => None,
        }
    }
}
```

### Step 4: ICN Token Wrapper Contract (Optional)
```solidity
// contracts/ICNTokenWrapper.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

/// @title ICN Token Wrapper
/// @notice Wraps native ICN to ERC-20 for DeFi compatibility
contract ICNTokenWrapper is ERC20 {
    event Wrapped(address indexed user, uint256 amount);
    event Unwrapped(address indexed user, uint256 amount);

    constructor() ERC20("Wrapped ICN", "wICN") {}

    /// @notice Wrap native ICN to ERC-20
    function wrap() external payable {
        require(msg.value > 0, "No ICN sent");
        _mint(msg.sender, msg.value);
        emit Wrapped(msg.sender, msg.value);
    }

    /// @notice Unwrap ERC-20 to native ICN
    function unwrap(uint256 amount) external {
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        _burn(msg.sender, amount);
        payable(msg.sender).transfer(amount);
        emit Unwrapped(msg.sender, amount);
    }

    receive() external payable {
        _mint(msg.sender, msg.value);
        emit Wrapped(msg.sender, msg.value);
    }
}
```

## Dependencies

- **T001**: ICN Chain bootstrap (runtime must exist)
- **T002**: pallet-icn-stake for staking precompile
- **T003**: pallet-icn-reputation for reputation precompile
- **Frontier**: polkadot-evm/frontier repository
- **OpenZeppelin Contracts 5.x**: ERC-20 base (for wrapper contract)

## Design Decisions

1. **Native token + optional EVM**: ICN is native Substrate token. EVM is an optional layer, not the primary interface.

2. **Wrapper pattern**: Native ICN can be wrapped to ERC-20 for DeFi compatibility, rather than making ICN ERC-20 by default.

3. **Precompile addresses**: 0x0900+ range for ICN-specific precompiles, following common conventions.

4. **Address conversion**: EVM address (20 bytes) → Substrate AccountId (32 bytes) via deterministic padding.

5. **Optional integration**: This entire task is optional - ICN works fully without Frontier.

## Alternatives

### Alternative 1: Skip Frontier, Use Snowbridge (T041)
- Don't add EVM to ICN Chain
- Use Snowbridge for Ethereum mainnet bridging
- Simpler runtime, less attack surface
- **Recommendation**: Consider this if EVM tooling not critical

### Alternative 2: Minimal Frontier (view-only)
- Add pallet-evm but no custom precompiles
- Allow basic EVM contract deployment
- No staking/reputation precompiles
- **Recommendation**: Good middle ground if unsure

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Precompile security vulnerability | Critical | Medium | External audit, extensive testing |
| Frontier compatibility issues | Medium | Medium | Pin exact Frontier version, test thoroughly |
| Increased runtime complexity | Medium | High | Comprehensive testing, monitor runtime size |
| Address conversion edge cases | Low | Low | Use standard AddressMapping implementations |

## Progress Log

- 2025-12-24: Task rewritten as optional Frontier integration for ICN Chain (P3)
- 2025-12-24: Removed Moonbeam-specific references

## Completion Checklist

- [ ] Decision made: Enable Frontier or skip (use Snowbridge instead)
- [ ] If enabling: Frontier crates integrated into runtime
- [ ] If enabling: EVM execution functional
- [ ] If enabling: Precompiles registered
- [ ] If enabling: MetaMask connection tested
- [ ] If enabling: Documentation for EVM developers
- [ ] If skipping: Document decision and rationale
