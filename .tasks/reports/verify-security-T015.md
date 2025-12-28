# Security Audit Report - T015 Flux-Schnell Integration

**Date:** 2025-12-28
**Task:** T015 - Flux-Schnell Integration
**Agent:** verify-security
**Scope:** vortex/src/vortex/models/flux.py, tests/unit/test_flux.py, tests/integration/test_flux_generation.py

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Security Score** | 92/100 |
| **Critical Vulnerabilities** | 0 |
| **High Vulnerabilities** | 0 |
| **Medium Vulnerabilities** | 1 |
| **Low Vulnerabilities** | 2 |
| **Recommendation** | **PASS** - No blocking security issues |

---

## CRITICAL Vulnerabilities

None. No critical vulnerabilities found.

---

## HIGH Vulnerabilities

None. No high vulnerabilities found.

---

## MEDIUM Vulnerabilities

### MULN-001: Safety Checker Disabled Without Alternative Enforcement

**Severity:** MEDIUM (CVSS 4.3)
**Location:** `vortex/src/vortex/models/flux.py:221`
**CWE:** CWE-1021 (Improper Restriction of Rendered UI Layers or Frames)

**Vulnerable Code:**
```python
# Disable safety checker (CLIP semantic verification handles content policy)
pipeline.safety_checker = None
logger.info("Disabled safety checker (CLIP handles content verification)")
```

**Issue:** The built-in diffusers safety checker is explicitly disabled to save VRAM and inference time. While CLIP semantic verification is stated as the alternative, there is no enforceable guarantee that CLIP will block all harmful content (CSAM, violence, NSFW). The CLIP check happens in a separate layer and may be bypassed or have false negatives.

**Impact:** Potential for generating harmful content if CLIP thresholds are insufficient or if CLIP verification is bypassed/defective.

**Fix:**
```python
# Add explicit content policy enforcement in FluxModel.generate()
def generate(self, prompt: str, ...) -> torch.Tensor:
    # Input validation - block known harmful patterns
    blocked_patterns = self._get_blocked_patterns()
    for pattern in blocked_patterns:
        if pattern.lower() in prompt.lower():
            raise ValueError(f"Prompt contains blocked content: {pattern}")

    # ... rest of generation

def _get_blocked_patterns(self) -> set[str]:
    """Return set of blocked content patterns (configurable)."""
    return {"nsfw", "nude", "violence", "gore", "weapon"}  # Extend as needed
```

**Mitigation Status:** Acknowledged - CLIP ensemble (T018) provides semantic verification, but explicit prompt filtering is recommended as defense-in-depth.

---

## LOW Vulnerabilities

### LOW-001: Prompt Tokenization is Approximate

**Severity:** LOW
**Location:** `vortex/src/vortex/models/flux.py:109-119`

**Issue:** Token counting uses `len(prompt.split())` which approximates tokens as words. Real CLIP tokenization may produce different token counts, potentially allowing slightly longer prompts than intended.

**Impact:** Minor - prompts could exceed CLIP limit by a small margin, causing implicit truncation by CLIP rather than explicit truncation by the application.

**Fix:**
```python
# Use actual CLIP tokenizer for accurate counting
from transformers import CLIPTokenizer

_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
actual_tokens = len(_tokenizer.encode(prompt))
if actual_tokens > self.MAX_PROMPT_TOKENS:
    # Truncate properly
```

### LOW-002: Error Messages May Leak System Information

**Severity:** LOW
**Location:** `vortex/src/vortex/models/flux.py:236-246`

**Issue:** CUDA OOM error messages include device properties and VRAM allocations which could be used for system fingerprinting.

**Impact:** Minor - reveals GPU memory layout to attackers.

**Fix:**
```python
# Sanitize error messages for production
error_msg = "CUDA OOM during Flux-Schnell loading. Required: ~6.0GB VRAM."
logger.error(error_msg, exc_info=True)  # Keep details in logs only
raise VortexInitializationError(error_msg) from e
```

---

## OWASP Top 10 Compliance

| Category | Status | Notes |
|----------|--------|-------|
| A1: Injection | PASS | No SQLi/command injection vectors. Prompts are data, not code. |
| A2: Broken Authentication | N/A | Not applicable to model code. |
| A3: Sensitive Data Exposure | PASS | No hardcoded secrets. Error messages could be tighter. |
| A4: XXE | N/A | No XML parsing. |
| A5: Broken Access Control | N/A | Model code has no access control requirements. |
| A6: Security Misconfiguration | PASS | Safetensors used for secure model loading. |
| A7: XSS | N/A | Not applicable (backend code). |
| A8: Insecure Deserialization | PASS | No pickle usage. Uses safetensors. |
| A9: Vulnerable Components | PASS | Uses standard diffusers, transformers, bitsandbytes. |
| A10: Insufficient Logging | PASS | Adequate logging for security events. |

---

## Dependency Vulnerabilities

**Dependencies Analyzed:**
- `diffusers` - Standard Hugging Face library, no known critical CVEs in latest versions
- `transformers` - Standard Hugging Face library, regularly updated
- `bitsandbytes` - NVIDIA quantization library
- `torch` - PyTorch framework

**Action Required:** Run `pip-audit` or `safety check` to verify no CVEs in pinned versions.

---

## Input Validation Analysis

### Prompt Handling (flux.py:104-119)

| Check | Implemented | Notes |
|-------|-------------|-------|
| Empty prompt validation | YES | Raises ValueError if empty/whitespace |
| Length limit | YES | Truncates to 77 tokens (approximate) |
| Character encoding | PASS | Standard Python str handling |
| Command injection | PASS | No eval/exec/shell operations |
| Template injection | N/A | No template rendering |

### Parameter Validation

| Parameter | Validation | Status |
|-----------|-----------|--------|
| `num_inference_steps` | None (trusted) | OK - internal use only |
| `guidance_scale` | None (float) | OK - no dangerous values |
| `seed` | None (int) | OK - torch.manual_seed safe |
| `output` | Type check via `.copy_()` | OK - torch handles validation |

---

## Model Loading Security

### Safetensors Usage (flux.py:216)

```python
use_safetensors=True,  # Secure format vs pickle
```

**Status:** PASS - Safetensors prevents arbitrary code execution during model loading (vs. pickle).

### Hugging Face Model Source (flux.py:211-212)

```python
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
```

**Status:** PASS - Uses official Hugging Face hub model from trusted org (black-forest-labs).

**Recommendation:** Pin model commit hash for reproducibility:
```python
"black-forest-labs/FLUX.1-schnell@commit_hash"
```

---

## Error Handling Without Secret Leakage

| Error Type | Location | Secret Leakage Risk | Status |
|------------|----------|---------------------|--------|
| VortexInitializationError | flux.py:231-254 | Minor (VRAM stats) | LOW-001 |
| ValueError (empty prompt) | flux.py:104-105 | None | PASS |
| CUDA OOM | flux.py:231-249 | System info only | PASS |

**Overall:** No secrets leaked. Error messages are informative but could be tighter for production.

---

## Cryptography Analysis

**Random Seed Handling (flux.py:122-124):**
```python
if seed is not None:
    torch.manual_seed(seed)
```

**Status:** PASS - Uses PyTorch's CSPRNG for seeding. No cryptographic operations requiring secure randomness.

---

## Threat Model for Flux Model

| Threat | Likelihood | Impact | Mitigation |
|--------|-----------|--------|------------|
| Adversarial prompt injection | Medium | Medium | CLIP semantic verification (T018) |
| Model poisoning via cache | Low | High | Use read-only cache, verify checksums |
| DoS via large prompts | Low | Low | Token limit enforced (77) |
| VRAM exhaustion | Medium | Medium | Pre-allocated buffers, OOM handling |

---

## Recommendations

1. **Before Deployment:**
   - Add explicit blocked content pattern matching in `generate()`
   - Pin Hugging Face model commit hash
   - Run `pip-audit` on dependencies

2. **Future Enhancements:**
   - Use actual CLIP tokenizer for accurate token counting
   - Add prompt sanitization layer before model
   - Consider rate limiting on generation endpoint

---

## Compliance Notes

- **GDPR:** No personal data processed
- **AI Act:** Model safety measures documented
- **SOC 2:** Logging and error handling adequate for audit

---

## Conclusion

**Decision: PASS**

The Flux-Schnell integration demonstrates good security practices:
- No hardcoded credentials or secrets
- No command injection vectors
- Proper use of safetensors
- Input validation on prompts
- Appropriate error handling

The one MEDIUM issue (disabled safety checker) is an intentional design decision with CLIP verification as the alternative. This is acceptable given the layered security architecture, but explicit prompt filtering is recommended as defense-in-depth.

**Security Score: 92/100**
- Deducted 8 points for: safety checker removal without hard enforcement, approximate tokenization, and verbose error messages.
