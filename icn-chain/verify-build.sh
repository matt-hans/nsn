#!/bin/bash
# ICN Chain Build Verification Script
# This script verifies that the ICN Chain can be built successfully

set -e

echo "========================================="
echo "ICN Chain Build Verification"
echo "========================================="
echo ""

# Check Rust toolchain
echo "1. Checking Rust toolchain..."
if ! command -v cargo &> /dev/null; then
    echo "❌ ERROR: Cargo not found. Please install Rust:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi
echo "✅ Cargo found: $(cargo --version)"

# Check Rust version
echo ""
echo "2. Checking Rust version..."
RUST_VERSION=$(rustc --version | awk '{print $2}')
echo "✅ Rust version: $RUST_VERSION"
echo "   (rust-toolchain.toml will override to stable-2024-09-05)"

# Check wasm32 target
echo ""
echo "3. Checking wasm32-unknown-unknown target..."
if rustup target list --installed | grep -q wasm32-unknown-unknown; then
    echo "✅ wasm32-unknown-unknown target installed"
else
    echo "⚠️  wasm32-unknown-unknown target not found, installing..."
    rustup target add wasm32-unknown-unknown
    echo "✅ wasm32-unknown-unknown target installed"
fi

# Check dependencies
echo ""
echo "4. Checking for build dependencies..."
cargo --version > /dev/null 2>&1 && echo "✅ Cargo available"
rustc --version > /dev/null 2>&1 && echo "✅ Rustc available"

# Run cargo check
echo ""
echo "5. Running cargo check (verifies project compiles without building)..."
if cargo check --release --locked 2>&1 | tee /tmp/icn-check.log; then
    echo "✅ cargo check passed"
else
    echo "❌ cargo check failed. See /tmp/icn-check.log for details"
    exit 1
fi

# Run cargo build
echo ""
echo "6. Running cargo build --release (this may take 30-60 minutes on first build)..."
echo "   Compiling runtime to WASM and native..."
if cargo build --release --locked 2>&1 | tee /tmp/icn-build.log; then
    echo "✅ cargo build --release passed"
else
    echo "❌ cargo build failed. See /tmp/icn-build.log for details"
    exit 1
fi

# Verify binary exists
echo ""
echo "7. Verifying icn-node binary..."
if [ -f "./target/release/icn-node" ]; then
    echo "✅ icn-node binary exists"
    ls -lh ./target/release/icn-node
else
    echo "❌ icn-node binary not found"
    exit 1
fi

# Run binary --version
echo ""
echo "8. Testing binary execution..."
if ./target/release/icn-node --version; then
    echo "✅ icn-node binary executes successfully"
else
    echo "❌ icn-node binary failed to execute"
    exit 1
fi

# Summary
echo ""
echo "========================================="
echo "✅ ALL CHECKS PASSED"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Run dev node:    ./target/release/icn-node --dev"
echo "  2. Run tests:       cargo test --all"
echo "  3. Run clippy:      cargo clippy --all-targets --all-features"
echo ""
echo "Build artifacts:"
echo "  Binary:     ./target/release/icn-node"
echo "  WASM:       ./target/release/wbuild/icn-runtime/icn_runtime.wasm"
echo ""
