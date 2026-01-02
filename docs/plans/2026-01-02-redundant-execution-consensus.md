# Redundant Execution + Consensus Policy

## Purpose
Ensure Lane 1 tasks are executed by multiple independent executors, with
consensus enforced before any attestation is submitted on-chain.

## Default Policy
- Replicas: 3
- Quorum: 2
- Deterministic tasks: majority hash match (output CID hash)
- Non-deterministic tasks: semantic score threshold with epsilon consensus
- Consensus timeout: 120 seconds

## Deterministic Consensus
- Executors submit `output_cid`.
- Scheduler hashes the CID to `output_hash`.
- Quorum is reached when >= `quorum` outputs share the same hash.

## Semantic Consensus
- Executors submit `output_cid` plus `semantic_score` in [0, 1].
- Scores must be >= `min_score`.
- Consensus achieved when max(score) - min(score) <= `epsilon`.

## Failure Handling
- If quorum becomes impossible (failures exceed remaining capacity), reject.
- If all results arrive without consensus, reject.
- If timeout exceeds consensus window, reject.

## Attestation Submission
- Once consensus is achieved, an attestation bundle is generated:
  - task id
  - output CID
  - score (0-100)
  - optional attestation CID
  - executor set
- Default submission path publishes to the P2P `Attestations` GossipSub topic.
- Optional chain submitter posts directly to `NsnTaskMarket::submit_attestation`.

## Config Example
```toml
[attestation]
# p2p | chain | dual | none
submit_mode = "dual"

# Required for chain/dual submissions. Example dev key only.
suri = "//Alice"
```

## Observability
- Metrics track success rate, failure rate, and average latency per task.

## Executor Selection
- Executors are selected from the director/super-node registry (validator-only nodes excluded).
- Selection prefers higher reputation and rotates deterministically per task.
