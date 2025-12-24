---
id: T033
title: Production Observability Stack (Prometheus, Grafana, Jaeger, AlertManager)
status: pending
priority: 2
agent: infrastructure
dependencies: [T028, T029]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [devops, observability, monitoring, prometheus, grafana, phase1]

context_refs:
  - context/project.md
  - context/architecture.md

docs_refs:
  - PRD Section 15 (Observability)
  - Architecture Section 6.5 (Observability)

est_tokens: 10000
actual_tokens: null
---

## Description

Deploy production-grade observability stack including Prometheus (metrics), Grafana (dashboards), Jaeger (distributed tracing), and AlertManager (alerting). Configure scraping of all ICN components (Substrate node, Director/Super-Nodes, Vortex engine) with ICN-specific dashboards and critical alerts.

**Technical Approach:**
- Prometheus with 15s scrape interval, 30d retention
- Grafana with pre-configured ICN dashboards (block production, P2P mesh, VRAM usage, BFT latency)
- Jaeger for OpenTelemetry traces (10% sampling)
- AlertManager with critical/warning alert rules
- Vector + Loki for structured log aggregation (7d retention)

**Integration Points:**
- Scrapes metrics from all node types
- Dashboards visualize system health
- Alerts route to PagerDuty/Slack

## Acceptance Criteria

- [ ] Prometheus deployed with persistent storage (30d retention)
- [ ] Grafana accessible with ICN dashboards provisioned
- [ ] Jaeger UI shows traces from Director nodes
- [ ] AlertManager routes alerts to configured channels
- [ ] Vector aggregates logs from all pods/containers
- [ ] Loki stores logs for 7 days
- [ ] Key metrics defined and scraped:
  - `icn_vortex_generation_time_seconds` (P99 <15s)
  - `icn_bft_round_duration_seconds` (P99 <10s)
  - `icn_p2p_connected_peers` (>10)
  - `icn_total_staked_tokens`
  - `icn_slashing_events_total`
- [ ] Critical alerts configured:
  - DirectorSlotMissed
  - VortexOOM (VRAM >11.5GB)
  - ChainDisconnected
  - StakeConcentration (region >25%)
- [ ] ServiceMonitors auto-discover ICN pods

## Test Scenarios

**Test Case 1: Metrics Scraping**
- When: Director node starts
- Then: Prometheus shows icn-director target as "Up", metrics appear within 30s

**Test Case 2: Dashboard Visualization**
- When: Open Grafana → ICN Overview dashboard
- Then: See live block production, P2P peer count, VRAM usage graphs

**Test Case 3: Alert Triggering**
- When: Manually set VRAM to 11.6GB in test
- Then: VortexOOM alert fires within 1 minute, routed to Slack/PagerDuty

**Test Case 4: Distributed Tracing**
- When: Director generates slot
- Then: Jaeger shows trace spanning: recipe received → Vortex generation → BFT coordination → P2P publish

**Test Case 5: Log Aggregation**
- When: Query Loki for "BFT consensus failed" logs
- Then: Return logs from last 7 days with structured fields (slot, directors, error)

## Technical Implementation

**File:** `charts/observability/values.yaml`

```yaml
prometheus:
  server:
    retention: 30d
    resources:
      requests:
        cpu: 500m
        memory: 2Gi
    persistentVolume:
      size: 100Gi

  serverFiles:
    prometheus.yml:
      scrape_configs:
        - job_name: 'substrate-nodes'
          kubernetes_sd_configs:
            - role: pod
          relabel_configs:
            - source_labels: [__meta_kubernetes_pod_label_app]
              regex: icn-.*
              action: keep
            - source_labels: [__meta_kubernetes_pod_container_port_name]
              regex: metrics
              action: keep

        - job_name: 'director-nodes'
          static_configs:
            - targets: ['icn-director:9100']

        - job_name: 'super-nodes'
          static_configs:
            - targets:
                - 'icn-super-node-na-west:9100'
                - 'icn-super-node-eu-west:9100'
                # ... all regions

grafana:
  adminPassword: admin
  persistence:
    enabled: true
    size: 10Gi

  datasources:
    datasources.yaml:
      apiVersion: 1
      datasources:
        - name: Prometheus
          type: prometheus
          url: http://prometheus-server
          isDefault: true
        - name: Loki
          type: loki
          url: http://loki:3100
        - name: Jaeger
          type: jaeger
          url: http://jaeger-query:16686

  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
        - name: 'icn'
          folder: 'ICN'
          type: file
          options:
            path: /var/lib/grafana/dashboards/icn

jaeger:
  allInOne:
    enabled: true
  collector:
    otlp:
      grpc:
        enabled: true

alertmanager:
  config:
    route:
      receiver: 'slack'
      routes:
        - match:
            severity: critical
          receiver: 'pagerduty'

    receivers:
      - name: 'slack'
        slack_configs:
          - api_url: '$SLACK_WEBHOOK_URL'
            channel: '#icn-alerts'

      - name: 'pagerduty'
        pagerduty_configs:
          - service_key: '$PAGERDUTY_KEY'

loki:
  persistence:
    enabled: true
    size: 50Gi
  config:
    limits_config:
      retention_period: 168h  # 7 days
```

**File:** `charts/observability/dashboards/icn-overview.json`

```json
{
  "dashboard": {
    "title": "ICN System Overview",
    "panels": [
      {
        "title": "Block Production Rate",
        "targets": [
          {
            "expr": "rate(substrate_block_height[5m])"
          }
        ]
      },
      {
        "title": "P2P Connected Peers",
        "targets": [
          {
            "expr": "icn_p2p_connected_peers"
          }
        ]
      },
      {
        "title": "Vortex Generation Time (P99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, icn_vortex_generation_time_seconds_bucket)"
          }
        ],
        "alert": {
          "name": "VortexSlowGeneration",
          "condition": "value > 15"
        }
      },
      {
        "title": "BFT Round Duration (P99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, icn_bft_round_duration_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Total Staked Tokens by Region",
        "targets": [
          {
            "expr": "sum(icn_staked_amount) by (region)"
          }
        ]
      },
      {
        "title": "Slashing Events (Last 24h)",
        "targets": [
          {
            "expr": "increase(icn_slashing_events_total[24h])"
          }
        ]
      }
    ]
  }
}
```

**File:** `charts/observability/alerts/icn-rules.yaml`

```yaml
groups:
  - name: icn-critical
    interval: 30s
    rules:
      - alert: DirectorSlotMissed
        expr: increase(icn_bft_failures_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Director slot missed - BFT consensus failed"

      - alert: VortexOOM
        expr: icn_vortex_vram_usage_bytes > 11.5e9
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Vortex VRAM near capacity (>11.5GB)"

      - alert: ChainDisconnected
        expr: time() - substrate_block_timestamp > 60
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "No new blocks in 60 seconds"

  - name: icn-warning
    interval: 1m
    rules:
      - alert: StakeConcentration
        expr: (sum by (region) (icn_staked_amount)) / sum(icn_staked_amount) > 0.25
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Region has >25% of total stake"

      - alert: BftLatencyHigh
        expr: histogram_quantile(0.99, icn_bft_round_duration_seconds_bucket) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "BFT rounds taking >10 seconds"
```

**File:** `scripts/deploy-observability.sh`

```bash
#!/bin/bash
set -euo pipefail

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/prometheus \
  -f charts/observability/values.yaml \
  --namespace monitoring --create-namespace

helm install grafana grafana/grafana \
  -f charts/observability/values.yaml \
  --namespace monitoring

helm install jaeger jaegertracing/jaeger \
  -f charts/observability/values.yaml \
  --namespace monitoring

helm install loki grafana/loki-stack \
  -f charts/observability/values.yaml \
  --namespace monitoring

echo "✅ Observability stack deployed"
echo "Access Grafana: kubectl port-forward -n monitoring svc/grafana 3000:80"
```

### Validation Commands

```bash
# Deploy stack
./scripts/deploy-observability.sh

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health == "up")'

# Access Grafana
kubectl port-forward -n monitoring svc/grafana 3000:80

# Query metrics
curl -G http://localhost:9090/api/v1/query --data-urlencode 'query=icn_vortex_generation_time_seconds' | jq .

# Trigger test alert
kubectl exec -it <director-pod> -- python -c "import torch; torch.cuda.empty_cache(); torch.zeros(12e9, device='cuda')"

# View logs in Loki
logcli query '{app="icn-director"}' --limit=100 --since=1h
```

## Dependencies

**Hard Dependencies:**
- [T028] Local Dev Environment - provides docker-compose observability setup
- [T029] Director Docker Image - exposes /metrics endpoint

**External Dependencies:**
- Kubernetes cluster with 10GB+ storage for Prometheus
- Slack/PagerDuty for alert routing

## Design Decisions

**Decision 1: Prometheus vs. VictoriaMetrics**
- **Rationale:** Prometheus is industry standard, better Grafana integration, simpler setup
- **Trade-offs:** (+) Standard. (-) Higher resource usage than VictoriaMetrics

**Decision 2: 30d Retention vs. 90d**
- **Rationale:** 30 days balances cost and debugging needs (most issues caught within weeks)
- **Trade-offs:** (+) Lower storage costs. (-) Can't debug older issues

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Prometheus storage fills disk | High | Medium | Set retention limits, alert on disk >80% |
| Alert fatigue from false positives | Medium | High | Tune thresholds, use for/silence rules |

## Progress Log

### [2025-12-24] - Task Created
**Created By:** task-creator agent
**Dependencies:** T028, T029
**Estimated Complexity:** Standard

## Completion Checklist

- [ ] Helm charts for Prometheus, Grafana, Jaeger, Loki
- [ ] ICN dashboards provisioned
- [ ] Alert rules configured
- [ ] ServiceMonitors for auto-discovery
- [ ] Deployment script tested

**Definition of Done:**
Observability stack deployed to Kubernetes, scraping all ICN components, dashboards show live metrics, critical alerts configured and tested.
