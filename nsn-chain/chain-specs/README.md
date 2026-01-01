# NSN Chain Specifications

This directory contains chain specification files for NSN Chain deployments.

## Files (Generated After Build)

- `nsn-dev.json` - Development chain spec (human-readable)
- `nsn-local.json` - Local testnet chain spec (human-readable)
- `nsn-testnet.json` - NSN Testnet chain spec (human-readable)
- `nsn-testnet-raw.json` - NSN Testnet raw spec (for distribution)
- `nsn-mainnet.json` - NSN Mainnet chain spec (human-readable)
- `nsn-mainnet-raw.json` - NSN Mainnet raw spec (for distribution)

## Generation Commands

See `docs/chain-spec-guide.md` for detailed instructions.

### Quick Start

```bash
# Build the node first
cargo build --release

# Generate testnet spec
./target/release/nsn-node build-spec --chain=nsn-testnet > chain-specs/nsn-testnet.json
./target/release/nsn-node build-spec --chain=nsn-testnet --raw > chain-specs/nsn-testnet-raw.json

# Generate mainnet spec (TEMPLATE - DO NOT USE IN PRODUCTION)
./target/release/nsn-node build-spec --chain=nsn-mainnet > chain-specs/nsn-mainnet.json
./target/release/nsn-node build-spec --chain=nsn-mainnet --raw > chain-specs/nsn-mainnet-raw.json
```

## Usage

```bash
# Start node with chain spec
./target/release/nsn-node --chain=./chain-specs/nsn-testnet-raw.json --validator
```

## Security Warning

**NEVER** use the mainnet chain spec template in production without:
1. Replacing all test accounts with production accounts
2. Generating unique production validator keys
3. Updating bootnode addresses
4. Completing security audit
5. Testing on private network first

