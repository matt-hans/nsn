# Task T040: Coretime Acquisition Planning and Implementation

## Metadata
```yaml
id: T040
title: Coretime Acquisition Planning and Implementation
status: pending
priority: P3
tags: [coretime, scaling, polkadot, on-chain, phase-c]
estimated_tokens: 8000
actual_tokens: 0
dependencies: [T039]
created_at: 2025-12-24
updated_at: 2025-12-24
```

## Description

**PHASE C TASK** - Plan and implement coretime acquisition strategy for NSN Chain once it becomes a Polkadot parachain. Coretime is Polkadot's execution allocation model managed by the Broker pallet on the Coretime system chain.

This task is **NOT required for MVP** and only relevant after T039 (Cumulus integration) is complete.

## Business Context

**Why Coretime**:
- **Elastic Scaling**: Acquire more coretime as NSN adoption grows
- **Cost Efficiency**: Pay only for execution time needed
- **On-Demand Option**: Start with on-demand coretime, move to bulk as usage increases
- **Multi-Core Future**: JAM architecture will expand coretime capabilities

## Acceptance Criteria

1. Coretime acquisition strategy documented
2. On-demand vs bulk coretime analysis completed
3. Integration with Broker pallet planned
4. Coretime region management implemented
5. Cost projections for different usage scenarios
6. Monitoring for coretime utilization
7. Documentation for operations team

## Coretime Model Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CORETIME SYSTEM CHAIN                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    BROKER PALLET                         │    │
│  │  - Bulk Coretime Sales (28-day regions)                 │    │
│  │  - On-Demand Coretime (per-block)                       │    │
│  │  - Region NFTs (transferable)                           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NSN PARACHAIN                                 │
│  Uses coretime allocation for block production/validation       │
└─────────────────────────────────────────────────────────────────┘
```

## Coretime Strategies

### Strategy A: On-Demand (Early Phase)
- Pay per block as needed
- Lower commitment
- Good for uncertain/growing usage
- Higher per-block cost

### Strategy B: Bulk Coretime (Established Phase)
- Purchase 28-day regions
- Lower per-block cost
- Requires usage prediction
- Can resell unused regions

### Hybrid Approach
- Bulk coretime for baseline load
- On-demand for peak periods
- Optimize cost vs availability

## Cost Projections

| Scenario | Blocks/Day | Coretime Type | Est. Monthly Cost |
|----------|------------|---------------|-------------------|
| Low (MVP) | 1,000 | On-Demand | ~$500-1,000 |
| Medium | 5,000 | Bulk + On-Demand | ~$2,000-4,000 |
| High | 14,400 (full) | Bulk | ~$5,000-10,000 |

*Costs are estimates and will vary with DOT price and demand*

## Technical Implementation

### Monitor Coretime Usage

```rust
// Track block production rate
fn estimate_coretime_needs() -> CoretimeEstimate {
    let blocks_per_day = get_average_blocks_per_day();
    let peak_multiplier = 1.5; // Buffer for peak times
    
    CoretimeEstimate {
        baseline_blocks: blocks_per_day,
        peak_blocks: (blocks_per_day as f64 * peak_multiplier) as u32,
        recommended_strategy: if blocks_per_day < 2000 {
            CoretimeStrategy::OnDemand
        } else {
            CoretimeStrategy::Bulk
        },
    }
}
```

### Acquire Bulk Coretime

```rust
// Via Broker pallet on Coretime chain
fn acquire_bulk_coretime(
    region_id: RegionId,
    duration: BlockNumber,
) -> DispatchResult {
    // XCM message to Coretime chain
    // Purchase region via Broker pallet
    // Region becomes available next cycle
}
```

## Dependencies

- **T039**: Cumulus integration (must be parachain first)
- **Polkadot Coretime Chain**: Operational coretime system

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Coretime price volatility | Medium | High | Hold DOT reserves, flexible strategy |
| Underestimate needs | High | Medium | Monitor closely, on-demand backup |
| Overestimate needs | Low | Medium | Resell unused regions |

## Completion Checklist

- [ ] Coretime strategy document written
- [ ] Cost projections completed
- [ ] Monitoring infrastructure planned
- [ ] Broker pallet integration designed
- [ ] Operations runbook drafted
- [ ] DOT reserves allocated

