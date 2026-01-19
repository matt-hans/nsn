#!/bin/bash
cd /home/matt/nsn/node-core

# Skip bootstrap for local testing
export NSN_LOCAL_TESTNET=1

./target/release/nsn-node \
  --p2p-enable-webrtc \
  --p2p-webrtc-port 9003 \
  --p2p-metrics-port 9100 \
  --p2p-listen-port 9000 \
  --storage-backend local \
  --storage-path /home/matt/nsn/.data/nsn-sandbox \
  --data-dir /home/matt/nsn/.data/nsn-sandbox \
  --rpc-url ws://127.0.0.1:9944 \
  --log-level info \
  --attestation-submit-mode none \
  super-node
