---
id: T030
title: Super-Node Kubernetes Deployment (7 Regions)
status: pending
priority: 2
agent: infrastructure
dependencies: [T011, T029]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [devops, kubernetes, infrastructure, super-node, production, phase2]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - Architecture Section 6.1 (Deployment Model - Super-Node K8s)
  - PRD Section 17.2 (Hierarchical Swarm - Tier 1 Super-Nodes)

est_tokens: 13000
actual_tokens: null
---

## Description

Create Kubernetes manifests and Helm charts to deploy Super-Node infrastructure across 7 geographic regions (NA-WEST, NA-EAST, EU-WEST, EU-EAST, APAC, LATAM, MENA) with 2 replicas per region for high availability. Super-Nodes provide erasure-coded shard storage (10TB+ per node), CLIP validation, and regional P2P relay services.

This deployment must handle PersistentVolumeClaims for storage, horizontal pod autoscaling based on bandwidth utilization, and service mesh configuration for cross-region routing.

**Technical Approach:**
- Helm chart with region-specific value overrides
- StatefulSets for persistent node identity
- Regional PersistentVolumeClaims (10TB SSD)
- HorizontalPodAutoscaler based on network I/O
- Service mesh (Istio/Linkerd) for regional routing
- Node affinity rules to ensure geographic distribution

**Integration Points:**
- Receives video chunks from Directors via GossipSub
- Stores erasure-coded shards (Reed-Solomon 10+4)
- Relays content to Regional Relays (Tier 2)
- Runs CLIP validation for BFT challenges

## Business Context

**User Story:** As an ICN operator, I want Super-Nodes deployed across 7 regions with automatic failover, so that video content remains available even during regional outages.

**Why This Matters:**
- Ensures 99.5% availability through geographic redundancy
- Reduces latency for viewers in different continents
- Meets architecture requirement of 7-region minimum
- Enables horizontal scaling as network grows

**What It Unblocks:**
- Production mainnet launch (Phase 2)
- Geographic diversity for censorship resistance
- Regional compliance (data sovereignty)
- Community node operator participation

**Priority Justification:** P2 - Critical for mainnet but depends on Super-Node implementation (T011). Required before public launch but can be tested on ICN Testnet with 3 regions initially.

## Acceptance Criteria

- [ ] Helm chart successfully installs with `helm install icn-super-nodes ./charts/super-node`
- [ ] 14 total pods running (7 regions × 2 replicas)
- [ ] Each pod has 10TB PersistentVolumeClaim attached and mounted
- [ ] All pods show "Ready" status within 5 minutes of deployment
- [ ] Pod affinity rules ensure no two replicas of same region on same node
- [ ] HorizontalPodAutoscaler scales up when bandwidth >80% utilization
- [ ] Service mesh routes traffic to nearest healthy Super-Node based on latency
- [ ] Pod disruption budgets prevent simultaneous failure of both regional replicas
- [ ] StatefulSet maintains stable network identities (icn-super-node-na-west-0, icn-super-node-na-west-1, ...)
- [ ] Prometheus ServiceMonitor configured for all Super-Node pods
- [ ] ConfigMap includes all required P2P bootstrap nodes
- [ ] Secrets contain Super-Node staking keys (securely managed)
- [ ] Rolling updates preserve at least 1 replica per region during deployments

## Test Scenarios

**Test Case 1: Clean Deployment**
- Given: Kubernetes cluster with 7+ nodes across regions, Helm installed
- When: `helm install icn-super-nodes ./charts/super-node --set global.regions=all`
- Then: 14 pods deploy, all reach "Running" status, PVCs bound, services created

**Test Case 2: Regional Pod Affinity**
- Given: Helm chart deployed with all regions
- When: Inspect pod placement: `kubectl get pods -o wide | grep icn-super-node`
- Then: Pods with same region prefix never on same Kubernetes node, spread across 7+ nodes

**Test Case 3: Storage Persistence**
- Given: Super-Node pod running with 10TB PVC
- When: Write test data: `kubectl exec icn-super-node-na-west-0 -- dd if=/dev/zero of=/data/test.dat bs=1M count=1000`
- Then: Data persists after pod restart, `kubectl exec icn-super-node-na-west-0 -- ls -lh /data/test.dat` shows 1GB file

**Test Case 4: Autoscaling**
- Given: HPA configured with network I/O target (80% bandwidth utilization)
- When: Simulate high traffic with load generator sending video chunks
- Then: HPA scales replicas from 2 to 4 per region when bandwidth >80%, scales down when traffic drops

**Test Case 5: Service Mesh Routing**
- Given: Istio VirtualService configured for latency-based routing
- When: Client in EU makes request to `/chunks/<shard_hash>`
- Then: Request routed to EU-WEST or EU-EAST Super-Node (latency <50ms), not NA-WEST (latency >100ms)

**Test Case 6: Disruption Budget**
- Given: PodDisruptionBudget set to `minAvailable: 1` per region
- When: Initiate node drain: `kubectl drain <node> --ignore-daemonsets`
- Then: Drain pauses if it would violate PDB, waits for rescheduled pod to become Ready

**Test Case 7: Rolling Update**
- Given: 14 pods running version v1.0.0
- When: `helm upgrade icn-super-nodes ./charts/super-node --set image.tag=v1.1.0`
- Then: Pods update one region at a time, always 1 replica available per region during rollout

## Technical Implementation

**Required Components:**

### 1. Helm Chart Structure
```
charts/super-node/
├── Chart.yaml
├── values.yaml
├── values-prod.yaml
├── templates/
│   ├── statefulset.yaml
│   ├── service.yaml
│   ├── pvc.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── servicemonitor.yaml
│   └── istio/
│       ├── virtualservice.yaml
│       └── destinationrule.yaml
└── README.md
```

### 2. Helm Chart Metadata
**File:** `charts/super-node/Chart.yaml`

```yaml
apiVersion: v2
name: icn-super-node
description: Helm chart for ICN Super-Node tier (7 regions)
type: application
version: 0.1.0
appVersion: "1.0.0"

dependencies:
  - name: prometheus-operator
    version: "^45.0.0"
    repository: https://prometheus-community.github.io/helm-charts
    condition: prometheus.enabled
```

### 3. Default Values
**File:** `charts/super-node/values.yaml`

```yaml
# Global configuration
global:
  registry: ghcr.io/icn
  imagePullSecrets: []
  regions:
    - na-west
    - na-east
    - eu-west
    - eu-east
    - apac
    - latam
    - mena

# Super-Node image
image:
  repository: super-node
  tag: "latest"
  pullPolicy: IfNotPresent

# Replicas per region
replicasPerRegion: 2

# Resource requests/limits
resources:
  requests:
    cpu: "2000m"
    memory: "8Gi"
    storage: "10Ti"
  limits:
    cpu: "4000m"
    memory: "16Gi"

# Storage class for PVCs
storage:
  className: "fast-ssd"  # Cloud provider specific
  accessMode: ReadWriteOnce

# P2P networking
p2p:
  port: 9000
  bootstrapNodes:
    - "/dns4/boot1.icn.network/tcp/9000/p2p/12D3KooW..."
    - "/dns4/boot2.icn.network/tcp/9000/p2p/12D3KooW..."

# Horizontal Pod Autoscaler
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 80
    - type: Pods
      pods:
        metric:
          name: network_bandwidth_utilization
        target:
          type: AverageValue
          averageValue: "800M"  # 80% of 1Gbps

# Pod Disruption Budget
podDisruptionBudget:
  enabled: true
  minAvailable: 1  # Per region

# Service Mesh (Istio)
serviceMesh:
  enabled: true
  istio:
    enabled: true
    gateway: icn-gateway

# Monitoring
prometheus:
  enabled: true
  serviceMonitor:
    interval: 15s

# Secrets (managed externally)
secrets:
  stakingKeys: icn-staking-keys  # K8s Secret name
```

### 4. StatefulSet Template
**File:** `charts/super-node/templates/statefulset.yaml`

```yaml
{{- range .Values.global.regions }}
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: icn-super-node-{{ . }}
  labels:
    app: icn-super-node
    region: {{ . }}
spec:
  serviceName: icn-super-node-{{ . }}
  replicas: {{ $.Values.replicasPerRegion }}
  selector:
    matchLabels:
      app: icn-super-node
      region: {{ . }}

  # Pod Management
  podManagementPolicy: Parallel
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 0

  template:
    metadata:
      labels:
        app: icn-super-node
        region: {{ . }}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9100"

    spec:
      # Anti-affinity: don't schedule replicas on same node
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchExpressions:
                  - key: app
                    operator: In
                    values:
                      - icn-super-node
                  - key: region
                    operator: In
                    values:
                      - {{ . }}
              topologyKey: kubernetes.io/hostname

        # Node affinity: prefer nodes in matching region
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              preference:
                matchExpressions:
                  - key: topology.kubernetes.io/region
                    operator: In
                    values:
                      - {{ . }}

      containers:
        - name: super-node
          image: "{{ $.Values.global.registry }}/{{ $.Values.image.repository }}:{{ $.Values.image.tag }}"
          imagePullPolicy: {{ $.Values.image.pullPolicy }}

          ports:
            - name: p2p
              containerPort: {{ $.Values.p2p.port }}
              protocol: TCP
            - name: metrics
              containerPort: 9100
              protocol: TCP

          env:
            - name: REGION
              value: {{ . | quote }}
            - name: P2P_PORT
              value: {{ $.Values.p2p.port | quote }}
            - name: BOOTSTRAP_NODES
              value: {{ join "," $.Values.p2p.bootstrapNodes | quote }}
            - name: STAKING_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ $.Values.secrets.stakingKeys }}
                  key: {{ . }}-staking-key

          volumeMounts:
            - name: data
              mountPath: /data

          resources:
            {{- toYaml $.Values.resources | nindent 12 }}

          livenessProbe:
            httpGet:
              path: /health
              port: metrics
            initialDelaySeconds: 60
            periodSeconds: 30

          readinessProbe:
            httpGet:
              path: /ready
              port: metrics
            initialDelaySeconds: 30
            periodSeconds: 10

  # PVC template
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes:
          - {{ $.Values.storage.accessMode }}
        storageClassName: {{ $.Values.storage.className }}
        resources:
          requests:
            storage: {{ $.Values.resources.requests.storage }}
{{- end }}
```

### 5. HorizontalPodAutoscaler
**File:** `charts/super-node/templates/hpa.yaml`

```yaml
{{- if .Values.autoscaling.enabled }}
{{- range .Values.global.regions }}
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: icn-super-node-{{ . }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: icn-super-node-{{ . }}
  minReplicas: {{ $.Values.autoscaling.minReplicas }}
  maxReplicas: {{ $.Values.autoscaling.maxReplicas }}
  metrics:
    {{- toYaml $.Values.autoscaling.metrics | nindent 4 }}
{{- end }}
{{- end }}
```

### 6. PodDisruptionBudget
**File:** `charts/super-node/templates/pdb.yaml`

```yaml
{{- if .Values.podDisruptionBudget.enabled }}
{{- range .Values.global.regions }}
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: icn-super-node-{{ . }}
spec:
  minAvailable: {{ $.Values.podDisruptionBudget.minAvailable }}
  selector:
    matchLabels:
      app: icn-super-node
      region: {{ . }}
{{- end }}
{{- end }}
```

### 7. Istio VirtualService
**File:** `charts/super-node/templates/istio/virtualservice.yaml`

```yaml
{{- if .Values.serviceMesh.istio.enabled }}
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: icn-super-node
spec:
  hosts:
    - super-node.icn.svc.cluster.local
  http:
    - match:
        - headers:
            x-client-region:
              exact: na-west
      route:
        - destination:
            host: icn-super-node-na-west
          weight: 90
        - destination:
            host: icn-super-node-na-east
          weight: 10

    # ... similar rules for other regions ...

    # Default: route to lowest latency
    - route:
        {{- range .Values.global.regions }}
        - destination:
            host: icn-super-node-{{ . }}
          weight: {{ div 100 (len $.Values.global.regions) }}
        {{- end }}
{{- end }}
```

### 8. Deployment Script
**File:** `scripts/deploy-super-nodes.sh`

```bash
#!/bin/bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-icn}
ENVIRONMENT=${1:-prod}

echo "Deploying Super-Nodes to namespace: $NAMESPACE (environment: $ENVIRONMENT)"

# Create namespace if not exists
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy with environment-specific values
helm upgrade --install icn-super-nodes ./charts/super-node \
  --namespace $NAMESPACE \
  --values charts/super-node/values.yaml \
  --values charts/super-node/values-${ENVIRONMENT}.yaml \
  --wait \
  --timeout 10m

# Wait for all pods to be ready
echo "Waiting for all pods to be ready..."
kubectl wait --for=condition=Ready pod \
  -l app=icn-super-node \
  -n $NAMESPACE \
  --timeout=300s

echo "✅ Super-Nodes deployed successfully"
kubectl get pods -n $NAMESPACE -l app=icn-super-node -o wide
```

### Validation Commands

```bash
# Install Helm chart
helm install icn-super-nodes ./charts/super-node --namespace icn --create-namespace

# Check pod status
kubectl get pods -n icn -l app=icn-super-node

# Verify PVCs bound
kubectl get pvc -n icn

# Check HPA status
kubectl get hpa -n icn

# Test service mesh routing
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl -H "x-client-region: eu-west" http://icn-super-node.icn.svc.cluster.local/health

# Monitor pod logs
kubectl logs -n icn icn-super-node-na-west-0 --follow

# Simulate pod failure (test PDB)
kubectl delete pod icn-super-node-na-west-0 -n icn

# Upgrade to new version
helm upgrade icn-super-nodes ./charts/super-node --set image.tag=v1.1.0 -n icn
```

## Dependencies

**Hard Dependencies:**
- [T011] Super-Node Implementation - provides binary/image
- [T029] Director Docker Image - establishes image build patterns

**Soft Dependencies:**
- [T033] Observability Stack - for ServiceMonitor configuration

**External Dependencies:**
- Kubernetes 1.27+
- Helm 3.12+
- Istio 1.18+ (if service mesh enabled)
- Cloud provider storage class supporting 10TB+ volumes

## Design Decisions

**Decision 1: StatefulSet vs. Deployment**
- **Rationale:** StatefulSets provide stable network identities and persistent storage per pod, critical for P2P libp2p nodes that use PeerID derived from stable identity
- **Trade-offs:** (+) Stable PeerID, ordered deployment. (-) Slower rolling updates

**Decision 2: Per-Region StatefulSets vs. Single Global StatefulSet**
- **Rationale:** Separate StatefulSets per region enable region-specific configuration, independent scaling, and clearer blast radius for failures
- **Trade-offs:** (+) Region isolation, easier debugging. (-) More K8s objects (7 StatefulSets)

**Decision 3: Istio for Service Mesh vs. Native K8s Services**
- **Rationale:** Istio provides latency-based routing, mTLS, and observability out-of-box
- **Trade-offs:** (+) Advanced routing, security. (-) Additional complexity, resource overhead

**Decision 4: HPA Based on Bandwidth vs. CPU**
- **Rationale:** Super-Nodes are network-bound (serving video chunks), not CPU-bound. Bandwidth utilization is better scaling metric.
- **Trade-offs:** (+) Scales based on actual load. (-) Requires custom metrics server

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| PVC provisioning fails (10TB limit) | High | Medium | Pre-provision volumes, use cloud provider with high IOPS SSD support (AWS gp3, GCP SSD Persistent) |
| Cross-region latency >200ms | Medium | Low | Use cloud provider with global backbone (GCP, AWS), enable Istio circuit breaking |
| HPA thrashing (scale up/down loops) | Medium | Medium | Configure stabilizationWindowSeconds, set min 2 replicas per region |
| Node affinity conflict | Medium | Low | Use preferredDuringScheduling (soft constraint), not required |
| Storage costs exceed budget | High | Medium | Implement data retention policy, compress shards, monitor usage with alerts |

## Progress Log

### [2025-12-24] - Task Created

**Created By:** task-creator agent
**Reason:** Deploy Super-Node infrastructure across 7 regions for production mainnet
**Dependencies:** T011 (Super-Node Implementation), T029 (Docker image patterns)
**Estimated Complexity:** Complex (multi-region Kubernetes, StatefulSets, service mesh)

## Completion Checklist

### Code Complete
- [ ] Helm chart structure created
- [ ] StatefulSet templates for 7 regions
- [ ] HPA, PDB, ConfigMap, Secret templates
- [ ] Istio VirtualService and DestinationRule
- [ ] ServiceMonitor for Prometheus
- [ ] values.yaml and values-prod.yaml
- [ ] Deployment scripts

### Testing
- [ ] Helm lint passes
- [ ] Dry-run install succeeds
- [ ] Actual install deploys 14 pods
- [ ] All PVCs bound successfully
- [ ] HPA scales based on metrics
- [ ] PDB prevents simultaneous failures
- [ ] Rolling update preserves availability
- [ ] Service mesh routes correctly

### Documentation
- [ ] README.md in charts/super-node/
- [ ] Deployment guide for operators
- [ ] Troubleshooting common issues
- [ ] Cost estimation for cloud providers

### DevOps
- [ ] CI/CD pipeline tests Helm chart
- [ ] Automated deployment to staging cluster
- [ ] Monitoring alerts for pod failures
- [ ] Backup/restore procedure documented

**Definition of Done:**
Task is complete when `helm install icn-super-nodes ./charts/super-node` successfully deploys 14 Super-Node pods across 7 regions, all PVCs bound with 10TB storage, HPA configured for autoscaling, and Istio routing traffic to nearest healthy pods based on client region.
