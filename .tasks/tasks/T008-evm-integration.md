# Task T008: EVM Integration (ICN Token ERC-20 & Substrate Precompiles)

## Metadata
```yaml
id: T008
title: EVM Integration (ICN Token ERC-20 & Substrate Precompiles)
status: pending
priority: P2
tags: [evm, solidity, precompiles, token, on-chain, phase1, integration]
estimated_tokens: 15000
actual_tokens: 0
dependencies: [T001, T002, T003]
created_at: 2025-12-24
updated_at: 2025-12-24
```

## Description

Implement the EVM layer integration that bridges Moonbeam's Frontier EVM with ICN's custom Substrate pallets. This includes deploying the ICN ERC-20 token contract (1B supply), creating Substrate precompiles for staking and reputation queries from EVM, and enabling dual-interface access (MetaMask + Polkadot.js) for maximum developer compatibility.

## Business Context

**Why this matters**: EVM compatibility is critical for ICN's adoption. It enables:
- **Familiar tooling**: Web3 developers can use ethers.js, Hardhat, Remix instead of learning Substrate
- **MetaMask support**: Users stake and interact with ICN using their existing Ethereum wallets
- **DeFi integration**: ICN token can be traded on DEXs like StellaSwap, Beamswap (Moonbeam's Uniswap forks)
- **Broader ecosystem**: dApps built on EVM can integrate ICN staking/reputation without Rust knowledge

**Value delivered**: Lowers barrier to entry for developers and users, expanding ICN's potential user base from Substrate-only developers to the entire Ethereum ecosystem (~400k developers).

**Priority justification**: P2 because core pallet functionality can be tested without EVM. Required for mainnet user onboarding but not blocking basic pallet development.

## Acceptance Criteria

1. ICN ERC-20 token contract deployed with 1B total supply (1,000,000,000 × 10^18 wei)
2. OpenZeppelin 5.x ERC-20 base contract used for security-audited foundation
3. `stakeForRole()` function in ICN token burns ERC-20 and calls Substrate pallet via precompile
4. `IcnStakePrecompile` at address 0x0000000000000000000000000000000000000900
5. Precompile correctly converts EVM address (20 bytes) to Substrate AccountId (32 bytes)
6. `getReputation(address)` function queries pallet-icn-reputation via precompile
7. `IcnReputationPrecompile` at address 0x0000000000000000000000000000000000000901
8. MetaMask successfully sends transactions to ICN token contract
9. Hardhat test suite covers token transfer, staking, and reputation queries (90%+ coverage)
10. Gas costs optimized: staking precompile <100k gas, reputation query <50k gas
11. Integration tests verify EVM → Substrate → EVM round-trip (stake via EVM, query via RPC)
12. Security audit completed for precompiles (external auditor or internal review)

## Test Scenarios

### Scenario 1: Deploy ICN ERC-20 Token
```gherkin
GIVEN Moonbeam runtime with Frontier EVM enabled
  AND deployer account has GLMR for gas
WHEN ICN token contract deployed with:
  - name: "Interdimensional Cable Network"
  - symbol: "ICN"
  - totalSupply: 1,000,000,000 × 10^18
THEN contract deployed at address 0x...
  AND balanceOf(deployer) == 1B × 10^18
  AND totalSupply() returns 1B × 10^18
  AND name() returns "Interdimensional Cable Network"
  AND symbol() returns "ICN"
```

### Scenario 2: ERC-20 Transfer via MetaMask
```gherkin
GIVEN Alice has 1000 ICN (1000 × 10^18 wei)
  AND Bob's address is 0xBOB
WHEN Alice calls transfer(0xBOB, 500 × 10^18) via MetaMask
THEN transaction succeeds
  AND balanceOf(Alice) == 500 × 10^18
  AND balanceOf(Bob) == 500 × 10^18
  AND Transfer event emitted
  AND MetaMask shows updated balance
```

### Scenario 3: Stake via EVM (stakeForRole)
```gherkin
GIVEN Charlie has 200 ICN in EVM balance
  AND wants to become a Director (role=0, 100 ICN minimum)
  AND region NA-WEST encoded as bytes32
WHEN Charlie calls ICN.stakeForRole(
  amount=150 × 10^18,
  role=0,  // Director
  region=0x4E415F57455354...  // "NA_WEST" as bytes32
)
THEN 150 ICN burned from Charlie's ERC-20 balance
  AND IcnStakePrecompile called with:
    - EVM address: 0xCHARLIE (20 bytes)
    - amount: 150 ICN
    - role: Director
  AND EVM address converted to Substrate AccountId (32 bytes)
  AND pallet_icn_stake::deposit_stake() called
  AND Stakes[SubstrateAccountId(Charlie)].amount == 150 ICN
  AND Stakes[SubstrateAccountId(Charlie)].role == Director
```

### Scenario 4: Precompile Address Conversion
```gherkin
GIVEN EVM address: 0x1234567890abcdef1234567890abcdef12345678
WHEN IcnStakePrecompile converts to Substrate AccountId
THEN AccountId = 0x1234567890abcdef1234567890abcdef12345678000000000000000000000000
  (EVM address left-padded with zeros to 32 bytes)
AND conversion is deterministic and reversible
AND same EVM address always maps to same AccountId
```

### Scenario 5: Query Reputation via EVM
```gherkin
GIVEN Dave has staked 100 ICN via Substrate
  AND Dave's reputation: director=500, validator=300, seeder=100
  AND Dave's EVM address is 0xDAVE
WHEN EVM contract calls IcnReputationPrecompile.getReputation(0xDAVE)
THEN precompile queries pallet_icn_reputation::ReputationScores
  AND returns ReputationScore struct:
    - director_score: 500
    - validator_score: 300
    - seeder_score: 100
    - total: (500×50 + 300×30 + 100×20)/100 = 344
AND gas used <50k
```

### Scenario 6: Gas Cost Optimization
```gherkin
GIVEN IcnStakePrecompile implementation
WHEN benchmark stakeForRole() with 100 ICN stake
THEN total gas cost breakdown:
  - ERC-20 burn: ~20k gas
  - Precompile call overhead: ~5k gas
  - Substrate pallet call: ~60k gas
  - Storage writes: ~10k gas
  Total: ~95k gas (<100k target)
```

### Scenario 7: Failed Stake (Insufficient Balance)
```gherkin
GIVEN Eve has 50 ICN ERC-20 balance
WHEN Eve calls stakeForRole(amount=100 ICN, role=0, region=...)
THEN transaction reverts with "Insufficient balance"
  AND no ICN burned
  AND no Substrate state changed
  AND MetaMask shows error message
```

### Scenario 8: Hardhat Test Suite
```solidity
describe("ICN Token", function() {
  it("Should deploy with correct supply", async function() {
    const ICN = await ethers.deployContract("ICNToken");
    expect(await ICN.totalSupply()).to.equal(ethers.parseEther("1000000000"));
  });

  it("Should stake via precompile", async function() {
    const [owner] = await ethers.getSigners();
    await ICN.stakeForRole(
      ethers.parseEther("100"),
      0,  // Director
      ethers.encodeBytes32String("NA_WEST")
    );
    // Verify Substrate state via RPC
    const stakes = await api.query.icnStake.stakes(substrateAccount);
    expect(stakes.amount.toString()).to.equal("100000000000000000000");
  });
});
```

### Scenario 9: Cross-Chain Token Bridge (Future)
```gherkin
GIVEN ICN token deployed on Moonbeam
  AND XCM cross-chain messaging enabled
WHEN user wants to bridge ICN to Polkadot Asset Hub
THEN ICN locked in Moonbeam vault contract
  AND XCM message sent to Asset Hub
  AND wrapped ICN minted on Asset Hub
  (Future enhancement, not MVP)
```

### Scenario 10: Precompile Security - Reentrancy Protection
```gherkin
GIVEN malicious contract attempts reentrancy attack on stakeForRole()
WHEN malicious contract calls stakeForRole() and tries to re-enter during execution
THEN precompile has nonReentrant guard
  AND second call fails with "ReentrancyGuard: reentrant call"
  AND attacker cannot exploit to double-stake
```

### Scenario 11: Event Emission from Precompile
```gherkin
GIVEN Frank stakes 200 ICN via EVM
WHEN stakeForRole() succeeds
THEN EVM event emitted: StakedViaPrecompile(address indexed user, uint256 amount, uint8 role)
  AND Substrate event emitted: pallet_icn_stake::Event::StakeDeposited
  AND both events indexed by block explorers (Moonscan)
```

### Scenario 12: Governance Upgrade of Precompile
```gherkin
GIVEN IcnStakePrecompile v1.0 deployed at 0x...0900
  AND bug discovered requiring fix
WHEN Moonbeam governance approves precompile upgrade
THEN new precompile code deployed at same address
  AND existing EVM contracts calling precompile use new logic automatically
  AND no storage migration required (stateless precompile)
```

## Technical Implementation

### ICN ERC-20 Token Contract
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/// @title ICN Token - Interdimensional Cable Network
/// @notice ERC-20 token with Substrate pallet integration via precompiles
contract ICNToken is ERC20, Ownable {
    // Precompile addresses
    address constant STAKE_PRECOMPILE = 0x0000000000000000000000000000000000000900;
    address constant REPUTATION_PRECOMPILE = 0x0000000000000000000000000000000000000901;

    // Total supply: 1 billion ICN
    uint256 public constant INITIAL_SUPPLY = 1_000_000_000 * 10**18;

    // Node roles (must match Substrate NodeRole enum)
    enum NodeRole { Director, SuperNode, Validator, Relay }

    // Events
    event StakedViaPrecompile(address indexed user, uint256 amount, NodeRole role);

    constructor() ERC20("Interdimensional Cable Network", "ICN") Ownable(msg.sender) {
        _mint(msg.sender, INITIAL_SUPPLY);
    }

    /// @notice Stake ICN tokens for a node role
    /// @dev Burns ERC-20 tokens and calls Substrate staking pallet via precompile
    /// @param amount Amount of ICN to stake (in wei, 18 decimals)
    /// @param role Node role (0=Director, 1=SuperNode, 2=Validator, 3=Relay)
    /// @param region Geographic region as bytes32 (e.g., "NA_WEST")
    function stakeForRole(uint256 amount, NodeRole role, bytes32 region) external {
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");

        // Burn ERC-20 tokens
        _burn(msg.sender, amount);

        // Call Substrate staking precompile
        (bool success, ) = STAKE_PRECOMPILE.call(
            abi.encodeWithSignature(
                "deposit_stake(uint256,uint8,bytes32)",
                amount,
                uint8(role),
                region
            )
        );
        require(success, "Staking failed");

        emit StakedViaPrecompile(msg.sender, amount, role);
    }

    /// @notice Get reputation score for an account
    /// @param account EVM address to query
    /// @return directorScore Director reputation component
    /// @return validatorScore Validator reputation component
    /// @return seederScore Seeder reputation component
    /// @return total Weighted total reputation
    function getReputation(address account) external view returns (
        uint256 directorScore,
        uint256 validatorScore,
        uint256 seederScore,
        uint256 total
    ) {
        (bool success, bytes memory data) = REPUTATION_PRECOMPILE.staticcall(
            abi.encodeWithSignature("reputation_scores(address)", account)
        );
        require(success, "Reputation query failed");

        // Decode Substrate ReputationScore struct
        (directorScore, validatorScore, seederScore) = abi.decode(data, (uint256, uint256, uint256));
        total = (directorScore * 50 + validatorScore * 30 + seederScore * 20) / 100;
    }
}
```

### IcnStakePrecompile (Rust)
```rust
use pallet_evm::{
    ExitError, ExitSucceed, Precompile, PrecompileFailure, PrecompileHandle, PrecompileOutput, PrecompileResult,
};
use sp_core::{H160, H256, U256};
use sp_std::vec::Vec;

pub struct IcnStakePrecompile;

impl IcnStakePrecompile {
    const ADDRESS: H160 = H160([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0]);

    /// Convert EVM address (20 bytes) to Substrate AccountId (32 bytes)
    fn evm_to_substrate(evm_address: H160) -> AccountId32 {
        let mut account_id = [0u8; 32];
        account_id[0..20].copy_from_slice(&evm_address.0);
        AccountId32::from(account_id)
    }
}

impl Precompile for IcnStakePrecompile {
    fn execute(handle: &mut impl PrecompileHandle) -> PrecompileResult {
        let input = handle.input();
        let selector = &input[0..4];

        match selector {
            // deposit_stake(uint256,uint8,bytes32)
            [0x12, 0x34, 0x56, 0x78] => {  // Function selector (computed from keccak256)
                let amount = U256::from_big_endian(&input[4..36]);
                let role = input[36];
                let region = &input[37..69];

                // Convert EVM caller to Substrate account
                let caller = handle.context().caller;
                let substrate_account = Self::evm_to_substrate(caller);

                // Call Substrate pallet
                pallet_icn_stake::Pallet::<Runtime>::deposit_stake(
                    frame_system::RawOrigin::Signed(substrate_account).into(),
                    amount.as_u128().into(),
                    1000u32.into(),  // lock_blocks (configurable)
                    Self::decode_region(region)?,
                )
                .map_err(|_| PrecompileFailure::Error {
                    exit_status: ExitError::Other("Stake deposit failed".into()),
                })?;

                Ok(PrecompileOutput {
                    exit_status: ExitSucceed::Returned,
                    output: Vec::new(),
                })
            }
            _ => Err(PrecompileFailure::Error {
                exit_status: ExitError::Other("Unknown selector".into()),
            }),
        }
    }
}

impl IcnStakePrecompile {
    fn decode_region(region_bytes: &[u8]) -> Result<pallet_icn_stake::Region, PrecompileFailure> {
        // Parse bytes32 region identifier to Substrate Region enum
        match &region_bytes[0..7] {
            b"NA_WEST" => Ok(pallet_icn_stake::Region::NA_WEST),
            b"NA_EAST" => Ok(pallet_icn_stake::Region::NA_EAST),
            b"EU_WEST" => Ok(pallet_icn_stake::Region::EU_WEST),
            b"EU_EAST" => Ok(pallet_icn_stake::Region::EU_EAST),
            b"APAC" => Ok(pallet_icn_stake::Region::APAC),
            b"LATAM" => Ok(pallet_icn_stake::Region::LATAM),
            b"MENA" => Ok(pallet_icn_stake::Region::MENA),
            _ => Err(PrecompileFailure::Error {
                exit_status: ExitError::Other("Invalid region".into()),
            }),
        }
    }
}
```

### IcnReputationPrecompile (Rust)
```rust
pub struct IcnReputationPrecompile;

impl IcnReputationPrecompile {
    const ADDRESS: H160 = H160([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 1]);
}

impl Precompile for IcnReputationPrecompile {
    fn execute(handle: &mut impl PrecompileHandle) -> PrecompileResult {
        let input = handle.input();
        let selector = &input[0..4];

        match selector {
            // reputation_scores(address)
            [0xAB, 0xCD, 0xEF, 0x12] => {
                let evm_address = H160::from_slice(&input[16..36]);  // Skip 12 byte padding
                let substrate_account = IcnStakePrecompile::evm_to_substrate(evm_address);

                // Query Substrate pallet
                let reputation = pallet_icn_reputation::Pallet::<Runtime>::reputation_scores(&substrate_account);

                // Encode as (uint256, uint256, uint256)
                let mut output = Vec::new();
                output.extend_from_slice(&U256::from(reputation.director_score).as_bytes());
                output.extend_from_slice(&U256::from(reputation.validator_score).as_bytes());
                output.extend_from_slice(&U256::from(reputation.seeder_score).as_bytes());

                Ok(PrecompileOutput {
                    exit_status: ExitSucceed::Returned,
                    output,
                })
            }
            _ => Err(PrecompileFailure::Error {
                exit_status: ExitError::Other("Unknown selector".into()),
            }),
        }
    }
}
```

### Hardhat Test Suite
```javascript
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ICN Token Integration", function() {
  let icnToken;
  let owner, alice, bob;

  beforeEach(async function() {
    [owner, alice, bob] = await ethers.getSigners();
    const ICNToken = await ethers.getContractFactory("ICNToken");
    icnToken = await ICNToken.deploy();
  });

  it("Should have correct initial supply", async function() {
    const totalSupply = await icnToken.totalSupply();
    expect(totalSupply).to.equal(ethers.parseEther("1000000000"));
  });

  it("Should transfer tokens correctly", async function() {
    await icnToken.transfer(alice.address, ethers.parseEther("1000"));
    expect(await icnToken.balanceOf(alice.address)).to.equal(ethers.parseEther("1000"));
  });

  it("Should stake via precompile", async function() {
    // Transfer tokens to alice
    await icnToken.transfer(alice.address, ethers.parseEther("200"));

    // Alice stakes 150 ICN as Director
    await icnToken.connect(alice).stakeForRole(
      ethers.parseEther("150"),
      0,  // Director
      ethers.encodeBytes32String("NA_WEST")
    );

    // Verify ERC-20 burned
    expect(await icnToken.balanceOf(alice.address)).to.equal(ethers.parseEther("50"));

    // Verify Substrate state (requires RPC access)
    // const api = await ApiPromise.create({ provider: wsProvider });
    // const stakes = await api.query.icnStake.stakes(substrateAccount(alice.address));
    // expect(stakes.amount.toString()).to.equal("150000000000000000000");
  });

  it("Should query reputation via precompile", async function() {
    // Setup: Stake first
    await icnToken.transfer(alice.address, ethers.parseEther("200"));
    await icnToken.connect(alice).stakeForRole(
      ethers.parseEther("100"),
      0,
      ethers.encodeBytes32String("EU_WEST")
    );

    // Query reputation (after some activity)
    const [directorScore, validatorScore, seederScore, total] =
      await icnToken.getReputation(alice.address);

    expect(directorScore).to.be.gte(0);  // May have reputation events
    expect(total).to.equal((directorScore * 50n + validatorScore * 30n + seederScore * 20n) / 100n);
  });
});
```

## Dependencies

- **T001**: Moonbeam fork (Frontier EVM must be enabled)
- **T002**: pallet-icn-stake for staking integration
- **T003**: pallet-icn-reputation for reputation queries
- **OpenZeppelin Contracts 5.x**: ERC-20 base implementation
- **Hardhat**: Testing and deployment framework

## Design Decisions

1. **OpenZeppelin base**: Industry-standard, audited ERC-20 implementation reduces security risks vs custom implementation.

2. **Burn-then-stake**: ERC-20 tokens burned and re-minted as Substrate native balance prevents double-spending across EVM/Substrate boundary.

3. **Precompile addresses**: 0x...0900, 0x...0901 follow Moonbeam's precompile address convention (system precompiles at 0x...0800+).

4. **Left-padding address conversion**: EVM address (20 bytes) → Substrate AccountId (32 bytes) via left-padding ensures deterministic bidirectional mapping.

5. **Function selectors**: Use keccak256 hash of function signature for EVM compatibility (standard Solidity ABI encoding).

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Precompile security vulnerability | Critical | Medium | External audit (Oak Security), extensive testing |
| Address conversion collision | High | Very Low | Left-padding is deterministic, no known collisions |
| Gas cost too high for users | Medium | Medium | Optimize precompile logic, benchmark on testnet |
| OpenZeppelin contract bug | Medium | Very Low | Use latest stable version, monitor security advisories |

## Progress Log

- 2025-12-24: Task created from PRD §5 and Architecture §4.2

## Completion Checklist

- [ ] All 12 acceptance criteria met
- [ ] All 12 test scenarios implemented and passing
- [ ] ICN token contract deployed to Moonriver testnet
- [ ] Hardhat test coverage ≥90%
- [ ] Precompiles registered in Moonbeam runtime
- [ ] Gas benchmarks meet targets (<100k stake, <50k query)
- [ ] MetaMask integration tested end-to-end
- [ ] Security audit completed (external or internal)
- [ ] Documentation for EVM developers (README, examples)
- [ ] No regression in existing tests
