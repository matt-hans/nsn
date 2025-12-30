---
id: T044
title: Deprecate and Remove legacy-nodes After P2P Migration
status: pending
priority: 1
agent: backend
dependencies: [T042, T043]
blocked_by: []
created: 2025-12-30T08:00:00Z
updated: 2025-12-30T08:00:00Z

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - .claude/rules/architecture.md
  - .claude/rules/prd.md

est_tokens: 10000
actual_tokens: null
---

## Description

Deprecate and remove the entire `legacy-nodes/` directory after successfully migrating all P2P networking code to `node-core/crates/p2p/`. This completes the architectural consolidation and eliminates technical debt from the legacy implementation.

**Context**: The `legacy-nodes/` directory was created early in the project to prototype off-chain node implementations. T022 completed GossipSub implementation in `legacy-nodes/common/src/p2p/`, but the architecture requires all P2P code to live in `node-core/`. T042 and T043 migrate this code, making `legacy-nodes/` obsolete.

**Technical Approach**:
- Verify T042 and T043 migrations are complete (all tests passing)
- Update all references from `icn-common` (legacy) to `nsn-p2p` (node-core)
- Remove `legacy-nodes/` workspace members from root `Cargo.toml`
- Delete `legacy-nodes/` directory entirely
- Update CI/CD pipelines to remove legacy-nodes build steps
- Update documentation to reflect node-core as the canonical implementation
- Add deprecation notice to CHANGELOG.md

**Integration Points**:
- **node-core/crates/p2p/**: Canonical P2P implementation (from T042+T043)
- **Root Cargo.toml**: Remove legacy-nodes from workspace.members
- **CI/CD**: Update GitHub Actions workflows to skip legacy-nodes
- **Documentation**: Update CLAUDE.md, README.md, architecture docs

## Business Context

**User Story**: As a developer, I need a clean, single-source architecture for off-chain nodes so that I can confidently build on node-core without confusion about which codebase to use.

**Why This Matters**:
- **Eliminates Technical Debt**: Removes deprecated code that could confuse developers
- **Reduces Maintenance Burden**: No need to keep legacy-nodes in sync with changes
- **Clarifies Architecture**: Single source of truth for off-chain node implementations
- **Improves Onboarding**: New developers see clear node-core structure, not legacy confusion
- **Enables Future Development**: All future off-chain work builds on node-core foundation

**What It Unblocks**:
- Clean architecture for future Director, Validator, Super-Node implementations
- Simplified CI/CD pipelines (fewer builds, faster iteration)
- Updated documentation for external contributors
- Production deployment clarity (no ambiguity about which nodes to run)

**Priority Justification**: Priority 1 (Critical Path) because:
- Completes P2P migration epic (T042 → T043 → T044)
- Removes architectural confusion and technical debt
- Required for clean production deployment
- Affects all future off-chain node development

## Acceptance Criteria

- [ ] **T042 Complete**: P2P core migration verified complete (all tests passing)
- [ ] **T043 Complete**: GossipSub migration verified complete (all tests passing)
- [ ] **Migration Verification**: `cargo test -p nsn-p2p` passes with 100% of legacy-nodes P2P tests migrated
- [ ] **Workspace Updated**: `legacy-nodes/` removed from root `Cargo.toml` workspace.members
- [ ] **Directory Deleted**: `legacy-nodes/` directory completely removed from repository
- [ ] **Imports Updated**: No remaining imports of `icn-common` anywhere in codebase
- [ ] **CI/CD Updated**: GitHub Actions workflows no longer reference legacy-nodes
- [ ] **CLAUDE.md Updated**: Repository structure section reflects removal of legacy-nodes
- [ ] **README Updated**: Architecture documentation references node-core only
- [ ] **CHANGELOG Updated**: Deprecation notice added explaining removal and migration path
- [ ] **Compilation**: `cargo build --release --all-features` succeeds without legacy-nodes
- [ ] **Tests**: `cargo test --all-features` passes without legacy-nodes
- [ ] **Git History**: Deprecation commit clearly documents removal rationale

## Test Scenarios

**Test Case 1: Workspace Compilation**
- **Given**: `legacy-nodes/` directory deleted, workspace.members updated
- **When**: `cargo build --release --all-features` is run
- **Then**: Build succeeds, no references to legacy-nodes crates

**Test Case 2: Test Suite**
- **Given**: All P2P tests migrated to `nsn-p2p` crate
- **When**: `cargo test --all-features` is run
- **Then**: All tests pass, no legacy-nodes tests executed

**Test Case 3: Import Verification**
- **Given**: Codebase search for `icn-common` imports
- **When**: `grep -r "use icn_common" .` is run
- **Then**: No matches found (all imports use `nsn-p2p` or other node-core crates)

**Test Case 4: CI/CD Pipeline**
- **Given**: Updated GitHub Actions workflows
- **When**: CI pipeline runs on main branch
- **Then**: No legacy-nodes build steps execute, all checks pass

**Test Case 5: Documentation Links**
- **Given**: Updated CLAUDE.md and README.md
- **When**: Documentation is reviewed
- **Then**: All references point to node-core, no legacy-nodes mentions

**Test Case 6: Dependency Graph**
- **Given**: `cargo tree` output
- **When**: Dependency tree is inspected
- **Then**: No legacy-nodes crates appear anywhere

**Test Case 7: Fresh Clone Build**
- **Given**: Fresh clone of repository (post-removal)
- **When**: New developer runs `cargo build --release`
- **Then**: Build succeeds without any legacy-nodes references or errors

## Technical Implementation

**Required Actions**:

1. **Verification Phase**
   - Run `cargo test -p nsn-p2p` to verify T042+T043 completeness
   - Run `cargo test --workspace` to ensure no legacy-nodes dependencies
   - Search codebase for `icn-common` imports: `grep -r "use icn_common" .`
   - Search codebase for `icn_common` in Cargo.toml: `grep -r "icn-common" . --include="Cargo.toml"`

2. **Update Root Cargo.toml**
   - Remove `legacy-nodes/` workspace members:
     ```toml
     # BEFORE
     members = [
         "nsn-chain",
         "legacy-nodes/common",
         "legacy-nodes/director",
         "legacy-nodes/validator",
         "legacy-nodes/super-node",
         "node-core",
     ]

     # AFTER
     members = [
         "nsn-chain",
         "node-core",
     ]
     ```
   - Remove `icn-common` from workspace.dependencies if present

3. **Update CI/CD Pipelines**
   - `.github/workflows/rust.yml`: Remove legacy-nodes build steps
   - `.github/workflows/test.yml`: Remove legacy-nodes test steps
   - Verify no references in workflow files: `grep -r "legacy-nodes" .github/workflows/`

4. **Update Documentation**
   - **CLAUDE.md**: Update repository structure section to remove legacy-nodes
     ```markdown
     # REMOVE
     legacy-nodes/              # Off-chain node implementations
       ├── common/             # Shared P2P, chain client, types
       ├── director/           # Lane 0: GPU video generation + BFT coordination
       ├── validator/          # Lane 0: CLIP semantic verification
       └── super-node/         # Tier 1 erasure-coded storage

     # KEEP (already present)
     node-core/                # Universal node implementation (Rust)
       ├── crates/p2p/         # P2P networking (migrated from legacy-nodes)
       ├── crates/scheduler/   # Task scheduler with On-Deck protocol
       └── crates/sidecar/     # Compute execution runtime
     ```
   - **README.md**: Update build commands to remove legacy-nodes references
   - **.claude/rules/architecture.md**: Ensure no legacy-nodes references remain

5. **Add Deprecation Notice**
   - **CHANGELOG.md**: Add entry explaining removal
     ```markdown
     ## [Unreleased]

     ### Removed
     - **legacy-nodes/**: Deprecated and removed after P2P migration to node-core
       - Reason: Architectural consolidation - all P2P networking now in `node-core/crates/p2p/`
       - Migration: T042 (P2P core), T043 (GossipSub, reputation, metrics)
       - Impact: Future off-chain nodes build on `node-core` foundation
       - If using legacy-nodes: Migrate to `node-core/crates/p2p/` (see T042, T043 for migration guide)
     ```

6. **Delete Directory**
   - Execute: `git rm -r legacy-nodes/`
   - Commit with descriptive message:
     ```
     refactor: Remove legacy-nodes after P2P migration to node-core

     All P2P networking code has been migrated to node-core/crates/p2p/
     via T042 (core) and T043 (GossipSub). legacy-nodes/ is no longer needed.

     - Removed: legacy-nodes/ directory (common, director, validator, super-node)
     - Updated: Root Cargo.toml workspace.members
     - Updated: CI/CD workflows
     - Updated: Documentation (CLAUDE.md, README.md, CHANGELOG.md)

     Resolves: T044
     Related: T042, T043
     ```

7. **Post-Deletion Verification**
   - Run `cargo build --release --all-features`
   - Run `cargo test --all-features`
   - Run `cargo clippy --all-features -- -D warnings`
   - Verify CI/CD pipeline passes on branch
   - Review git diff to confirm clean removal

**Validation Commands**:

```bash
# 1. Verify T042+T043 complete
cargo test -p nsn-p2p

# 2. Search for remaining icn-common references
grep -r "icn.common" . --exclude-dir=target --exclude-dir=.git
grep -r "icn-common" . --include="Cargo.toml"

# 3. Update workspace
vim Cargo.toml  # Remove legacy-nodes members

# 4. Delete directory
git rm -r legacy-nodes/

# 5. Update documentation
vim CLAUDE.md README.md CHANGELOG.md
vim .claude/rules/architecture.md

# 6. Update CI/CD
vim .github/workflows/rust.yml
vim .github/workflows/test.yml

# 7. Verify build
cargo build --release --all-features
cargo test --all-features
cargo clippy --all-features -- -D warnings

# 8. Commit
git commit -m "refactor: Remove legacy-nodes after P2P migration"
```

## Dependencies

**Hard Dependencies** (must be complete first):
- [T042] Migrate P2P Core Implementation - **PENDING** 🟡70 [REPORTED] - Must be complete and tested before removing legacy-nodes
- [T043] Migrate GossipSub to node-core - **PENDING** 🟡70 [REPORTED] - Must be complete and tested before removing legacy-nodes

**Soft Dependencies** (nice to have):
- None - this task is terminal (end of migration epic)

**External Dependencies**:
- Git for version control (directory removal, commit)
- GitHub for CI/CD pipeline updates

## Design Decisions

**Decision 1: Complete Deletion vs. Archive**
- **Rationale**: Complete deletion (git rm -r) is cleanest approach for deprecated code; git history preserves legacy-nodes if needed
- **Alternatives**: Move to `archive/legacy-nodes/` (clutters repo), keep as reference (confusing)
- **Trade-offs**:
  - (+) Clean repository structure
  - (+) No confusion about which code to use
  - (+) Git history preserves legacy-nodes for reference
  - (-) Requires git checkout to access old code
  - (-) Loses "side-by-side comparison" capability

**Decision 2: Deprecation Notice in CHANGELOG**
- **Rationale**: CHANGELOG.md is canonical source for breaking changes; provides migration path for anyone using legacy-nodes
- **Alternatives**: No notice (breaks users silently), separate MIGRATION.md (not standard)
- **Trade-offs**:
  - (+) Standard location for breaking changes
  - (+) Clear migration path documented
  - (+) Future reference for "why was this removed?"
  - (-) Assumes users read CHANGELOG
  - (-) One-time notice (removed in future versions)

**Decision 3: Update CI/CD in Same Commit**
- **Rationale**: Atomic commit with code removal + CI/CD update prevents CI failures on intermediate commits
- **Alternatives**: Separate commits (causes CI failure), disable CI (risky)
- **Trade-offs**:
  - (+) CI passes on every commit
  - (+) Single logical change (remove legacy-nodes)
  - (+) Easy to revert if needed
  - (-) Larger commit (harder to review)

**Decision 4: Verify T042+T043 Before Removal**
- **Rationale**: Ensure migration is 100% complete before deleting source code to prevent loss of functionality
- **Alternatives**: Delete immediately (risky), keep both (technical debt)
- **Trade-offs**:
  - (+) Safety: Prevents accidental loss of functionality
  - (+) Confidence: Tests prove migration completeness
  - (+) Reversibility: Can fix migration issues before deletion
  - (-) Slower: Requires waiting for T042+T043 completion
  - (-) Dependency: Blocks on T042+T043 quality

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| T042/T043 incomplete, functionality lost | Critical | Low | Verify 100% test coverage before deletion, run comprehensive test suite |
| Hidden dependencies on legacy-nodes | High | Low | Search entire codebase for `icn-common` imports before deletion |
| CI/CD breaks after removal | Medium | Low | Update workflows atomically with code removal, test CI on branch |
| Documentation out of sync | Low | Medium | Update CLAUDE.md, README.md, architecture docs in same commit |
| New developers confused by removal | Low | Low | CHANGELOG.md explains removal, git history preserves old code |
| Accidental reintroduction of legacy-nodes | Low | Low | Remove from workspace.members, prevent compilation if re-added |
| Migration path unclear for external users | Low | Very Low | Document migration in CHANGELOG.md, reference T042/T043 tasks |

## Progress Log

### [2025-12-30T08:00:00Z] - Task Created

**Created By**: task-creator agent
**Reason**: Deprecate and remove legacy-nodes after successful P2P migration to node-core
**Dependencies**: T042 (P2P Core Migration), T043 (GossipSub Migration)
**Estimated Complexity**: Standard (10,000 tokens)

## Completion Checklist

**Pre-Deletion Verification**:
- [ ] T042 status = completed
- [ ] T043 status = completed
- [ ] `cargo test -p nsn-p2p` passes (100% P2P tests migrated)
- [ ] No `icn-common` imports found in codebase
- [ ] No `icn-common` in any Cargo.toml files

**Workspace & Build**:
- [ ] Root `Cargo.toml` workspace.members updated (legacy-nodes removed)
- [ ] `cargo build --release --all-features` succeeds
- [ ] `cargo test --all-features` passes
- [ ] `cargo clippy --all-features -- -D warnings` passes

**Documentation**:
- [ ] CLAUDE.md repository structure updated
- [ ] README.md build commands updated
- [ ] CHANGELOG.md deprecation notice added
- [ ] architecture.md no longer references legacy-nodes

**CI/CD**:
- [ ] .github/workflows/rust.yml updated (no legacy-nodes builds)
- [ ] .github/workflows/test.yml updated (no legacy-nodes tests)
- [ ] CI pipeline passes on branch

**Deletion & Commit**:
- [ ] `legacy-nodes/` directory deleted (`git rm -r legacy-nodes/`)
- [ ] Commit message describes removal rationale
- [ ] Commit references T042, T043, T044
- [ ] Changes pushed to branch

**Post-Deletion Verification**:
- [ ] Fresh clone builds successfully
- [ ] All tests pass in fresh environment
- [ ] No broken documentation links
- [ ] Ready for merge to main

**Definition of Done**:
Task is complete when `legacy-nodes/` is completely removed, ALL references updated, ALL tests pass, CI/CD works, documentation is accurate, and CHANGELOG.md explains the deprecation clearly.
