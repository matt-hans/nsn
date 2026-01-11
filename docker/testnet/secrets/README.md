# NSN Testnet Secrets Directory

This directory contains sensitive key material for NSN testnet nodes.

**IMPORTANT:** Never commit actual keys to version control!

## Required Files

For each validator, you need to generate a persistent node key:

```bash
# Generate Alice's node key
subkey generate-node-key --file alice-node-key

# Generate Bob's node key
subkey generate-node-key --file bob-node-key

# Generate Charlie's node key
subkey generate-node-key --file charlie-node-key
```

## Directory Structure

After setup, this directory should contain:

```
secrets/
├── README.md           # This file
├── .gitkeep           # Keeps directory in git
├── alice-node-key     # Alice's persistent node key
├── bob-node-key       # Bob's persistent node key
└── charlie-node-key   # Charlie's persistent node key
```

## Getting Peer IDs

After generating node keys, get the peer IDs for bootnode configuration:

```bash
# Get Alice's peer ID
subkey inspect-node-key --file alice-node-key

# Output example:
# 12D3KooWAliceHashHere
```

Update `.env` with the peer IDs:

```bash
ALICE_PEER_ID=12D3KooW...
BOB_PEER_ID=12D3KooW...
CHARLIE_PEER_ID=12D3KooW...
```

## Session Keys

Validator session keys are managed separately from node keys. For testnet using
`--alice`, `--bob`, `--charlie` flags, session keys are derived automatically.

For custom validators, rotate session keys via RPC:

```bash
# Connect to node RPC
curl -H "Content-Type: application/json" \
  -d '{"id":1,"jsonrpc":"2.0","method":"author_rotateKeys","params":[]}' \
  http://localhost:9944

# Submit session keys via extrinsic (using polkadot.js.org/apps)
session.setKeys(keys, proof)
```

## Security Recommendations

1. **File Permissions:** Ensure keys are only readable by owner
   ```bash
   chmod 600 *-node-key
   ```

2. **Backup:** Keep secure backups of all keys
   - Loss of keys = loss of validator identity

3. **Rotation:** Periodically rotate session keys
   - Node keys should remain stable for consistent peer IDs

4. **Production:** Use Docker secrets or Kubernetes secrets
   - Never store in plain files on shared systems

## Quick Setup Script

```bash
#!/bin/bash
# Generate all required keys for testnet

set -e

echo "Generating validator node keys..."

for name in alice bob charlie; do
    if [ ! -f "${name}-node-key" ]; then
        subkey generate-node-key --file "${name}-node-key"
        chmod 600 "${name}-node-key"
        echo "Generated ${name}-node-key"

        # Print peer ID
        echo "Peer ID for ${name}:"
        subkey inspect-node-key --file "${name}-node-key"
        echo ""
    else
        echo "${name}-node-key already exists, skipping"
    fi
done

echo "Done! Update .env with the peer IDs above."
```

Save as `setup-keys.sh` and run:

```bash
chmod +x setup-keys.sh
./setup-keys.sh
```
