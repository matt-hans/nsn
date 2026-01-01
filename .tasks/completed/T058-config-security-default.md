# T058: Change allow_untrusted Config Default to False

## Priority: P0 (IMMEDIATE)
## Complexity: 1 hour
## Status: Complete
## Completed: 2025-12-31

---

## Objective

Change the default value of `allow_untrusted` in the plugin configuration from `true` to `false` to prevent arbitrary plugin execution on nodes by default.

## Background

The current configuration in `config.yaml` defaults to allowing untrusted plugins:

```yaml
plugins:
  policy:
    allow_untrusted: true  # DANGEROUS DEFAULT
    allowlist: []
```

This means any node running with default configuration will execute arbitrary Python code from untrusted sources, creating a critical security vulnerability.

## Implementation

### Step 1: Update config.yaml

```yaml
plugins:
  policy:
    allow_untrusted: false  # SECURE DEFAULT
    allowlist: []
```

### Step 2: Update policy.py default

Ensure the Python policy code also defaults to `false` if the config is missing.

### Step 3: Update documentation

Add a note to operator documentation explaining how to enable untrusted plugins if explicitly desired.

## Acceptance Criteria

- [x] Default config has `allow_untrusted: false`
- [x] Policy code defaults to `false` if config missing (verified: policy.py line 63)
- [x] Existing tests pass (4/4 tests passed)
- [x] Documentation updated (README.md + config.yaml comments)

## Risk Assessment

**Risk of NOT doing this:** Critical - arbitrary code execution
**Risk of doing this:** Low - operators must explicitly opt-in to untrusted plugins

## Dependencies

None - this can be done immediately.

## Deliverables

1. Updated `config.yaml`
2. Updated `vortex/src/vortex/plugins/policy.py` (if needed)
3. Documentation note

---

## Implementation Notes

Files modified:
1. `vortex/config.yaml` - Changed `allow_untrusted: true` to `false`, added security comments
2. `node-core/sidecar/src/plugins.rs` - Changed Rust `Default` impl to use `false`
3. `vortex/src/vortex/plugins/README.md` - Updated example to show secure default

Note: Python `policy.py` already correctly defaulted to `False` when config was missing (line 63).

---

**This was a 1-hour quick win with major security improvement.**
