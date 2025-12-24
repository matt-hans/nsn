---
id: T031
title: CI/CD Pipeline for Substrate Pallets
status: pending
priority: 1
agent: infrastructure
dependencies: [T001, T002]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [devops, cicd, pallets, testing, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - PRD Section 16 (CI/CD - GitHub Actions)
  - Architecture Section 6.6 (DevOps & Tooling)

est_tokens: 9000
actual_tokens: null
---

## Description

Implement comprehensive CI/CD pipeline for ICN Chain runtime and pallets using GitHub Actions. Pipeline includes Rust compilation, unit/integration tests, Clippy linting, security auditing (cargo-audit, cargo-deny), runtime WASM build, and automated ICN Testnet deployment on merge to develop branch.

**Technical Approach:**
- GitHub Actions workflow with matrix strategy for pallet testing
- Rust nightly toolchain pinned to nightly-2024-01-01
- wasm32-unknown-unknown target for runtime builds
- Parallel test execution across pallets
- Cached dependencies for faster builds (<5 min)
- Automated runtime upgrade submission to ICN Testnet

**Integration Points:**
- Triggered on PR, push to main/develop
- Artifacts: Runtime WASM, benchmarks, coverage reports
- Deploys to ICN Testnet on develop branch merge

## Business Context

**User Story:** As a developer, I want automated testing and deployment for pallets, so that I catch bugs before production and reduce manual deployment overhead.

**Why This Matters:**
- Reduces time-to-deployment from hours to minutes
- Catches bugs before code review
- Ensures runtime WASM always builds successfully
- Automates ICN Testnet deployments

**What It Unblocks:**
- Rapid pallet iteration (Phase 1)
- Confidence in mainnet deployments (Phase 2)
- Third-party contributor submissions

**Priority Justification:** P1 - Needed from day 1 of development to prevent broken builds and enable continuous testing.

## Acceptance Criteria

- [ ] GitHub Actions workflow file `.github/workflows/pallets.yml` exists
- [ ] Workflow triggers on: push to main/develop, pull requests, manual dispatch
- [ ] All 6 pallets build successfully in parallel matrix
- [ ] Unit tests run with coverage report generated (>85% target)
- [ ] Integration tests execute with local Substrate node
- [ ] Clippy linting passes with `-D warnings` (no warnings allowed)
- [ ] cargo-audit reports no vulnerabilities in dependencies
- [ ] cargo-deny checks pass (licenses, advisories, bans)
- [ ] Runtime WASM builds successfully for wasm32-unknown-unknown target
- [ ] Benchmark weights checked for excessive growth (>10% increase fails)
- [ ] Automated ICN Testnet runtime upgrade on develop branch merge
- [ ] Build time <10 minutes (with caching)
- [ ] Artifacts uploaded: WASM, coverage, benchmarks

## Test Scenarios

**Test Case 1: PR Build**
- Given: Developer opens PR modifying pallet-icn-stake
- When: GitHub Actions workflow triggers
- Then: Only icn-stake pallet builds/tests (optimized), Clippy passes, unit tests pass

**Test Case 2: Security Vulnerability Detection**
- Given: Dependency has known CVE
- When: cargo-audit runs
- Then: Build fails with clear error message, issue linked to CVE advisory

**Test Case 3: Benchmark Weight Regression**
- Given: PR modifies extrinsic logic
- When: Benchmarks run and compare to baseline
- Then: If weights increase >10%, build fails with "Weight regression detected" error

**Test Case 4: Automated ICN Testnet Deployment**
- Given: PR merged to develop branch
- When: CI/CD pipeline completes all checks
- Then: Runtime WASM submitted to ICN Testnet via `submit-runtime-upgrade.sh`, upgrade proposed on-chain

**Test Case 5: Caching Effectiveness**
- Given: Two sequential builds with no dependency changes
- When: Second build runs
- Then: Cargo dependencies restored from cache, build time <3 minutes (vs 10 min cold)

## Technical Implementation

**Required Components:**

### 1. GitHub Actions Workflow
**File:** `.github/workflows/pallets.yml`

```yaml
name: Substrate Pallets CI

on:
  push:
    branches: [main, develop]
    paths:
      - 'pallets/**'
      - 'Cargo.toml'
      - 'Cargo.lock'
  pull_request:
    paths:
      - 'pallets/**'
      - 'Cargo.toml'
  workflow_dispatch:

env:
  RUST_TOOLCHAIN: nightly-2024-01-01
  CARGO_TERM_COLOR: always

jobs:
  check:
    name: Check & Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-action@master
        with:
          toolchain: ${{ env.RUST_TOOLCHAIN }}
          components: rustfmt, clippy
          targets: wasm32-unknown-unknown

      - name: Cache Cargo
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: cargo-${{ runner.os }}-${{ hashFiles('**/Cargo.lock') }}

      - name: Check Formatting
        run: cargo fmt --all -- --check

      - name: Clippy
        run: cargo clippy --all-features --all-targets -- -D warnings

  test:
    name: Test Pallets
    runs-on: ubuntu-latest
    strategy:
      matrix:
        pallet:
          - icn-stake
          - icn-reputation
          - icn-director
          - icn-bft
          - icn-pinning
          - icn-treasury
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-action@master
        with:
          toolchain: ${{ env.RUST_TOOLCHAIN }}
          targets: wasm32-unknown-unknown

      - name: Cache
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: cargo-test-${{ runner.os }}-${{ matrix.pallet }}-${{ hashFiles('**/Cargo.lock') }}

      - name: Unit Tests
        run: cargo test --package pallet-${{ matrix.pallet }} --all-features -- --nocapture

      - name: Coverage
        uses: actions-rs/tarpaulin@v0.1
        with:
          version: '0.22.0'
          args: '--package pallet-${{ matrix.pallet }} --all-features --out Xml'

      - name: Upload Coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./cobertura.xml
          flags: pallet-${{ matrix.pallet }}

  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-action@master
        with:
          toolchain: ${{ env.RUST_TOOLCHAIN }}

      - name: Cache
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: cargo-integration-${{ runner.os }}-${{ hashFiles('**/Cargo.lock') }}

      - name: Integration Tests
        run: cargo test --features integration-tests --test '*' -- --nocapture

  security:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Cargo Audit
        uses: actions-rs/audit-check@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Cargo Deny
        uses: EmbarkStudios/cargo-deny-action@v1
        with:
          log-level: warn
          command: check
          arguments: --all-features

  build-wasm:
    name: Build Runtime WASM
    runs-on: ubuntu-latest
    needs: [check, test]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-action@master
        with:
          toolchain: ${{ env.RUST_TOOLCHAIN }}
          targets: wasm32-unknown-unknown

      - name: Cache
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: cargo-wasm-${{ runner.os }}-${{ hashFiles('**/Cargo.lock') }}

      - name: Build WASM
        run: |
          cargo build --release --target wasm32-unknown-unknown -p icn-runtime

      - name: Upload WASM Artifact
        uses: actions/upload-artifact@v3
        with:
          name: icn-runtime-wasm
          path: target/wasm32-unknown-unknown/release/icn_runtime.wasm

      - name: Check Weight Benchmarks
        run: cargo run --release -p icn-weights-check

  deploy-icn-testnet:
    name: Deploy to ICN Testnet
    runs-on: ubuntu-latest
    needs: [build-wasm, security]
    if: github.ref == 'refs/heads/develop'
    steps:
      - uses: actions/checkout@v4

      - name: Download WASM
        uses: actions/download-artifact@v3
        with:
          name: icn-runtime-wasm
          path: ./wasm

      - name: Setup Substrate Tools
        run: |
          curl -L https://github.com/paritytech/substrate/releases/download/latest/subxt-cli -o subxt
          chmod +x subxt
          sudo mv subxt /usr/local/bin/

      - name: Submit Runtime Upgrade
        env:
          MOONRIVER_SUDO_KEY: ${{ secrets.MOONRIVER_SUDO_KEY }}
        run: |
          ./scripts/submit-runtime-upgrade.sh \
            --network icn-testnet \
            --wasm ./wasm/icn_runtime.wasm \
            --sudo-seed "$MOONRIVER_SUDO_KEY"
```

### 2. Runtime Upgrade Script
**File:** `scripts/submit-runtime-upgrade.sh`

```bash
#!/bin/bash
set -euo pipefail

NETWORK=""
WASM_PATH=""
SUDO_SEED=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --network) NETWORK="$2"; shift 2 ;;
    --wasm) WASM_PATH="$2"; shift 2 ;;
    --sudo-seed) SUDO_SEED="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ "$NETWORK" == "icn-testnet" ]]; then
  WS_URL="wss://testnet.icn.example.com"
elif [[ "$NETWORK" == "icn-mainnet" ]]; then
  WS_URL="wss://mainnet.icn.example.com"
else
  echo "Invalid network: $NETWORK"
  exit 1
fi

echo "Submitting runtime upgrade to $NETWORK..."

subxt tx \
  --url "$WS_URL" \
  --suri "$SUDO_SEED" \
  sudo sudo_unchecked_weight \
  --call system set_code \
  --code "$(cat $WASM_PATH | xxd -p | tr -d '\n')" \
  --weight 1000000000

echo "✅ Runtime upgrade proposed on $NETWORK"
```

### 3. Weight Check Tool
**File:** `crates/icn-weights-check/src/main.rs`

```rust
use std::fs;
use std::process;

fn main() {
    let baseline = fs::read_to_string("benchmarks/baseline.json")
        .expect("Baseline weights not found");
    let current = fs::read_to_string("target/benchmarks/current.json")
        .expect("Current weights not found");

    let baseline_weights: serde_json::Value = serde_json::from_str(&baseline).unwrap();
    let current_weights: serde_json::Value = serde_json::from_str(&current).unwrap();

    let mut regressions = Vec::new();

    for (pallet, extrinsics) in current_weights.as_object().unwrap() {
        if let Some(baseline_extrinsics) = baseline_weights.get(pallet) {
            for (extrinsic, weight) in extrinsics.as_object().unwrap() {
                if let Some(baseline_weight) = baseline_extrinsics.get(extrinsic) {
                    let current_val = weight.as_u64().unwrap();
                    let baseline_val = baseline_weight.as_u64().unwrap();

                    let increase_pct = ((current_val as f64 - baseline_val as f64) / baseline_val as f64) * 100.0;

                    if increase_pct > 10.0 {
                        regressions.push(format!("{}.{}: {:.1}% increase", pallet, extrinsic, increase_pct));
                    }
                }
            }
        }
    }

    if !regressions.is_empty() {
        eprintln!("❌ Weight regressions detected:");
        for regression in regressions {
            eprintln!("  - {}", regression);
        }
        process::exit(1);
    } else {
        println!("✅ No weight regressions detected");
    }
}
```

### Validation Commands

```bash
# Run workflow locally (act)
act -j test

# Check workflow syntax
actionlint .github/workflows/pallets.yml

# Simulate PR build
gh pr checks <pr_number>

# View recent workflow runs
gh run list --workflow=pallets.yml

# Download artifacts
gh run download <run_id>
```

## Dependencies

**Hard Dependencies:**
- [T001] ICN Chain Bootstrap - provides pallet code
- [T002] pallet-icn-stake - first pallet to test

**External Dependencies:**
- GitHub Actions runner with 4 CPU cores
- ICN Testnet access (our own chain)
- MOONRIVER_SUDO_KEY secret in GitHub repo

## Design Decisions

**Decision 1: Nightly Rust vs. Stable**
- **Rationale:** Substrate FRAME requires nightly for certain features (const generics, unstable APIs)
- **Trade-offs:** (+) Feature access. (-) Potential breakage on nightly updates

**Decision 2: Matrix Strategy for Pallets**
- **Rationale:** Parallel testing reduces total CI time from 30 min (sequential) to <10 min
- **Trade-offs:** (+) Fast feedback. (-) More complex workflow YAML

**Decision 3: Cargo-deny for License Compliance**
- **Rationale:** Prevents GPL-3.0 dependencies in ERC-20 token code (license conflict)
- **Trade-offs:** (+) License safety. (-) May block useful crates

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Nightly toolchain breaks | High | Low | Pin to specific nightly (nightly-2024-01-01), test upgrades in separate PR |
| CI queue time >30 min | Medium | High | Use GitHub Actions matrix parallelism, aggressive caching |
| False positive security alerts | Low | Medium | Maintain cargo-deny allowlist for false positives |
| ICN Testnet upgrade fails | High | Low | Test on local node first, require manual approval step for mainnet |

## Progress Log

### [2025-12-24] - Task Created

**Created By:** task-creator agent
**Reason:** Automate pallet testing and deployment
**Dependencies:** T001, T002
**Estimated Complexity:** Standard (GitHub Actions YAML, Rust tooling)

## Completion Checklist

### Code Complete
- [ ] `.github/workflows/pallets.yml` workflow file
- [ ] `scripts/submit-runtime-upgrade.sh` deployment script
- [ ] `crates/icn-weights-check/` weight regression tool
- [ ] `benchmarks/baseline.json` baseline weights

### Testing
- [ ] Workflow runs successfully on sample PR
- [ ] All 6 pallets build in <10 minutes
- [ ] Coverage reports upload to Codecov
- [ ] Security scans detect intentional vulnerable dependency
- [ ] Weight check fails on >10% regression

### Documentation
- [ ] README section on running CI locally
- [ ] Troubleshooting common CI failures

**Definition of Done:**
Task is complete when GitHub Actions workflow automatically builds, tests, and deploys all 6 pallets on every PR/merge, with full test coverage reporting, security scanning, and automated ICN Testnet runtime upgrades on develop branch.
