# ICN Task Management System

## Overview

This directory contains the task management system for the Interdimensional Cable Network (ICN) project. Tasks are organized into phases matching the PRD roadmap, with detailed specifications for ICN Chain pallet development, off-chain infrastructure, and AI/ML pipeline implementation.

## Directory Structure

```
.tasks/
├── manifest.json              # Task index and dependency graph
├── metrics.json               # Performance tracking and token usage
├── tasks/                     # Individual task files
│   ├── T001-icn-chain-bootstrap.md
│   ├── T002-pallet-icn-stake.md
│   └── ... (additional tasks)
├── context/                   # Session-loaded context (<1000 tokens total)
│   ├── project.md            # Vision, goals, constraints (~300 tokens)
│   ├── architecture.md       # Tech stack, system design (~300 tokens)
│   ├── acceptance-templates.md  # Standard patterns (~200 tokens)
│   └── test-scenarios/       # Reusable test cases
├── completed/                 # Archived completed tasks
├── updates/                   # Atomic task updates
└── README.md                  # This file
```

## Task Phases

### Phase A: Documentation Refactor (Weeks 1-2)
**Focus**: Strategic pivot documentation and task alignment

**Tasks**:
- Update PRD, TAD, and ADRs for ICN Chain strategy
- Rewrite task dependencies and critical path

### Phase B: ICN Solochain MVP (Weeks 3-10)
**Focus**: Core pallet development and local testing

**Tasks**:
- T001: ICN Chain bootstrap and development environment setup
- T002: pallet-icn-stake implementation
- T003: pallet-icn-reputation with Merkle proofs
- T004: pallet-icn-director with VRF election
- T005: pallet-icn-pinning with erasure coding
- T006: pallet-icn-treasury for reward distribution
- T007: pallet-icn-bft with challenge mechanism
- T038: Chain spec and genesis configuration

**Exit Criteria**:
- All pallets compile and pass unit tests
- ICN Solochain running with minimal validator set
- Staking → Election → Reputation → BFT flow demonstrated
- 10+ test nodes participating

### Phase C: Parachain Readiness (Weeks 11-18)
**Focus**: Cumulus integration and shared security

**Tasks**:
- T039: Cumulus parachain integration
- T040: Coretime planning and acquisition
- Security audit (Oak Security/SRLabs)
- Parachain slot registration

**Exit Criteria**:
- Security audit passed
- Runtime compatible with Cumulus
- Coretime acquisition plan documented
- 50+ nodes, 500+ community members

### Phase D: Ethereum Integration (Optional)
**Focus**: EVM and bridge connectivity

**Tasks**:
- T008: Optional Frontier EVM integration
- T041: Snowbridge integration for Ethereum bridging

**Exit Criteria**:
- Frontier EVM enabled OR Snowbridge gateway integration
- Ethereum access story documented

### Phase E: Scale & Iterate (Ongoing)
**Focus**: Feature expansion and optimization

**Future Work**:
- EVM precompiles for staking/reputation
- Mobile viewer apps (iOS/Android)
- Cross-chain messaging (XCM)
- Director hardware acceleration (TensorRT)

## Using Tasks

### Start a Task
```bash
# View all pending tasks
/task-status

# Get next recommended task
/task-next

# Start specific task
/task-start T001
```

### Task Development Workflow
1. Read task file completely (Description, Acceptance Criteria, Test Scenarios)
2. Review context files in `.tasks/context/`
3. Implement following Technical Implementation section
4. Run validation commands from `acceptance-templates.md`
5. Update Progress Log with findings
6. Complete Completion Checklist
7. Mark task as completed via `/task-complete T001`

### Token Budgets

Each task includes an estimated token count to prevent context overflow:
- **Simple tasks**: 5-8k tokens (single pallet implementation)
- **Standard tasks**: 8-12k tokens (pallet + integration tests)
- **Complex tasks**: 12-20k tokens (multi-pallet coordination)

If a task exceeds 20k tokens, it should be split into subtasks.

## Context Loading Strategy

Context files are designed to stay under 1000 tokens total when loaded together:
- `project.md`: ~300 tokens (vision, goals, constraints)
- `architecture.md`: ~300 tokens (tech stack, system design)
- `acceptance-templates.md`: ~200 tokens (validation patterns)

This allows every task session to load full context without exceeding budget.

## Task File Format

Each task follows a consistent structure:

```markdown
# Task TXXX: Title

## Metadata (YAML frontmatter)
id, title, status, priority, dependencies, tags, estimated_tokens

## Description
Brief overview of what needs to be done

## Business Context
Why this matters, value delivered, impact

## Acceptance Criteria (minimum 8)
Specific, measurable, testable requirements

## Test Scenarios (minimum 6)
Given/When/Then format, covering success and edge cases

## Technical Implementation
Code snippets, commands, step-by-step guide

## Dependencies
Other tasks that must complete first

## Design Decisions
Rationale for key technical choices

## Risks & Mitigations (minimum 4)
Identified risks with probability and mitigation strategies

## Progress Log
Timestamped updates during development

## Completion Checklist
Final verification before marking complete
```

## Validation Tools

### Substrate Pallets
```bash
cargo build --release --all-features
cargo test --all-features -- --nocapture
cargo clippy --all-features -- -D warnings
cargo fmt -- --check
```

### Runtime WASM
```bash
cargo build --release --target wasm32-unknown-unknown -p icn-chain-runtime
```

### Python (Vortex Engine)
```bash
pytest vortex/tests/unit --cov=vortex
python vortex/benchmarks/slot_generation.py --slots 5 --max-time 15
```

### Integration
```bash
./scripts/submit-runtime-upgrade.sh --network icn-testnet --wasm <path>
```

## Metrics Tracking

The system tracks:
- **Token Efficiency**: Estimated vs actual tokens used per task
- **Phase Progress**: Percentage complete for each phase
- **Token Savings**: Modular approach vs monolithic (93%+ savings expected)
- **Task Velocity**: Average tasks completed per week

View metrics: `cat .tasks/metrics.json`

## Best Practices

1. **Read context first**: Always load `.tasks/context/` files before starting a task
2. **Follow acceptance criteria**: They define "done" - don't skip any
3. **Write comprehensive tests**: Minimum 6 test scenarios in Given/When/Then format
4. **Document decisions**: Update Design Decisions section with rationale
5. **Log progress**: Update Progress Log with timestamps and findings
6. **Complete checklist**: Verify all items before marking task complete

## Getting Help

- **PRD Reference**: See `.claude/rules/prd.md` for full product requirements
- **Architecture**: See `.claude/rules/architecture.md` for system design
- **Task Issues**: Create update in `.tasks/updates/` directory
- **Clarifications**: Use Conditional Interview per Minion Engine v3.0

## Quick Reference

| Command | Purpose |
|---------|---------|
| `/task-status` | View all task statuses |
| `/task-next` | Get recommended next task |
| `/task-start T001` | Begin specific task |
| `/task-complete T001` | Mark task done |
| `/task-health` | Check system health |

---

**Initialized**: 2025-12-24
**Project**: Interdimensional Cable Network (ICN)
**Architecture**: ICN Polkadot SDK Chain (v9.0)
**Timeline**: 3-6 months to MVP
