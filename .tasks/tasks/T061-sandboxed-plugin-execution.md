# T061: Sandboxed Plugin Execution for Vortex

## Priority: P0 (Blocker)
## Complexity: 3-4 weeks
## Status: Pending
## Depends On: T049 (Docker Container Manager), T058 (Config Security Default)

---

## Objective

Replace direct in-process plugin execution with a sandboxed runtime that enforces filesystem, network, and resource isolation for untrusted plugins.

## Background

Current plugin loading executes arbitrary Python modules directly:

```python
spec.loader.exec_module(module)
```

This allows full host access. Timeouts do not provide security guarantees. We need hard isolation before production use.

## Implementation

### Step 1: Decide Sandbox Strategy (MVP)

Pick one primary sandbox for MVP and document trade-offs:
- **Option A (MVP)**: Docker-based sandbox using `node-core` container manager
- **Option B**: Wasmtime + WASI for Python (requires transpiling or plugin ABI rewrite)
- **Option C**: Firecracker/gVisor (stronger isolation, higher integration cost)

**Recommendation**: Start with Docker (Option A) for fastest path, keep an interface that allows Firecracker later.

### Step 2: Define a Sandbox Execution Interface

Create an internal interface so executor can switch implementations:

```python
class SandboxRunner(Protocol):
    async def run(self, plugin_id: str, payload: dict, budget_ms: int) -> SandboxResult: ...
```

### Step 3: Implement Docker Sandbox Runner

- Launch a minimal container image with a restricted runtime.
- Mount plugin bundle read-only.
- Disable outbound network by default.
- Apply cgroup limits (CPU, memory, GPU where applicable).
- Return stdout/stderr + exit code + runtime metadata.

### Step 4: Enforce Policy Gates

- If `allow_untrusted: true`, require sandbox runner available.
- If sandbox missing, fail fast with a security error.
- Require allowlist for any plugin that requests privileged capabilities.

### Step 5: Audit Logging and Metrics

- Log plugin ID, sandbox type, limits, exit codes.
- Record resource usage for diagnostics.
- Emit metrics for security review (sandbox failures, denials).

## Acceptance Criteria

- [ ] Untrusted plugins never run in-process
- [ ] Sandbox denies filesystem access outside mounted plugin bundle
- [ ] Sandbox denies network access by default (explicit allowlist only)
- [ ] Resource limits enforced (CPU/memory) and verified by tests
- [ ] `allow_untrusted: true` without sandbox fails fast
- [ ] Executor returns structured error when sandbox denies access
- [ ] Logs + metrics emitted for every plugin execution
- [ ] Documentation updated with sandbox security model

## Testing

- Unit tests for policy gating logic
- Integration test: plugin tries to read `/etc/passwd` (must fail)
- Integration test: plugin attempts outbound HTTP (must fail by default)
- Resource limit test: CPU-heavy plugin throttled or killed
- Regression test: trusted allowlisted plugin executes successfully

## Deliverables

1. `vortex/src/vortex/plugins/sandbox.py` (runner interface + implementation)
2. `vortex/src/vortex/plugins/executor.py` updates to use sandbox runner
3. Container image or runtime config for plugin sandbox
4. Security documentation + metrics hooks

---

**This task is a production blocker until complete.**
