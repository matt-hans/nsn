# Error Handling Verification - T013 (Viewer Client)

**Date:** 2025-12-28  
**Component:** Tauri + React Viewer (Frontend)  
**Files Analyzed:**  
- `viewer/src/services/webcodecs.ts`  
- `viewer/src/services/p2p.ts`

---

## Decision: ⚠️ WARN

**Score:** 58/100

---

## Critical Issues: ❌ FAIL

### 1. **Swallowed Codec Errors** - `webcodecs.ts:33-35`

```typescript
error: (e: DOMException) => {
    console.error("Decode error:", e);
    // Error logged but NOT propagated to UI
},
```

**Impact:** Video playback failures are invisible to users  
**Fix Required:** Emit error event or reject promise to notify UI layer  
**Severity:** CRITICAL - User experience degrades silently

---

### 2. **Silent Decoder Initialization Failure** - `webcodecs.ts:26-29`

```typescript
const support = await VideoDecoder.isConfigSupported(config);
if (!support.supported) {
    throw new Error(`Codec ${codec} not supported`);
    // Thrown but caller has no try/catch in visible code
}
```

**Impact:** Unhandled promise rejection crashes app  
**Missing:** Caller error handling not shown in implementation  
**Severity:** CRITICAL - App crash potential

---

### 3. **No Timeout Handling** - `p2p.ts:73-87`

```typescript
export async function connectToRelay(relay: RelayInfo): Promise<boolean> {
    try {
        // No timeout mechanism
        console.log("Connecting to relay:", relay.peer_id);
        isConnected = true;
        return true; // Mock success for now
    } catch (error) {
        // Error caught but generic
        console.error("Failed to connect to relay:", error);
        return false;
    }
}
```

**Impact:** Connection attempts hang indefinitely  
**Missing:** Timeout with abort controller  
**Severity:** CRITICAL - UI freezes on network issues

---

### 4. **Generic Catch Block** - `p2p.ts:83-86`

```typescript
} catch (error) {
    console.error("Failed to connect to relay:", error);
    // Error type not checked, no structured logging
    return false;
}
```

**Impact:** Cannot distinguish timeout vs DNS vs auth failure  
**Missing:** Error classification for retry logic  
**Severity:** HIGH - Impairs debugging

---

### 5. **No Reconnection Logic** - `p2p.ts:100-107`

```typescript
export function disconnect(): void {
    isConnected = false;
    if (mockStreamInterval !== null) {
        clearInterval(mockStreamInterval);
        mockStreamInterval = null;
    }
    // No reconnection attempt, no exponential backoff
}
```

**Impact:** Single failure disconnects permanently  
**Missing:** Reconnect with exponential backoff  
**Severity:** HIGH - Poor resilience

---

## Pattern Issues

### Logging Deficiencies

| Issue | Location | Problem |
|-------|----------|---------|
| No correlation IDs | All error logs | Cannot trace requests across components |
| Console-only logging | `p2p.ts:37`, `webcodecs.ts:34` | No structured logging for monitoring |
| Missing error context | All catch blocks | No request metadata (relay ID, codec) |

### Wrong Error Propagation

| Location | Issue | Impact |
|----------|-------|--------|
| `webcodecs.ts:46-48` | Returns void on error | Caller cannot detect decode failures |
| `p2p.ts:73` | Returns boolean, not result type | Loses error details |

---

## Specific Code Examples

### Missing Timeout (CRITICAL)

**Current:**
```typescript
export async function connectToRelay(relay: RelayInfo): Promise<boolean> {
    try {
        console.log("Connecting to relay:", relay.peer_id);
        isConnected = true;
        return true;
    }
}
```

**Should Be:**
```typescript
export async function connectToRelay(
    relay: RelayInfo, 
    timeoutMs: number = 10000
): Promise<Result<boolean, ConnectionError>> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    
    try {
        await node.dial(multiaddr(relay.multiaddr), {
            signal: controller.signal
        });
        return Ok(true);
    } catch (error) {
        if (error.name === 'AbortError') {
            return Err(new ConnectionError('TIMEOUT', relay.peer_id));
        }
        return Err(new ConnectionError('UNKNOWN', relay.peer_id, error));
    } finally {
        clearTimeout(timeout);
    }
}
```

---

### Empty Error Handler (CRITICAL)

**Current:**
```typescript
error: (e: DOMException) => {
    console.error("Decode error:", e);
},
```

**Should Be:**
```typescript
error: (e: DOMException) => {
    logger.error('VideoDecoder error', {
        codec: this.config?.codec,
        errorName: e.name,
        errorMessage: e.message,
        correlationId: this.correlationId
    });
    this.emit('error', {
        type: 'DECODE_ERROR',
        code: e.name,
        recoverable: e.name !== 'EncodingError'
    });
},
```

---

## Blocking Criteria Met

✅ **CRITICAL errors swallowed** - Codec errors logged but not propagated  
✅ **NO timeout logic** - Connection attempts hang indefinitely  
✅ **Missing retry logic** - Single failure causes permanent disconnect  
✅ **Wrong error propagation** - Boolean return loses error details  

---

## Recommendations

### Immediate (BLOCKING)

1. **Add timeout to all network operations** with AbortController
2. **Emit error events** from VideoDecoderService to UI layer
3. **Wrap external calls** in Result type for explicit error handling
4. **Add reconnection logic** with exponential backoff (5 attempts)

### High Priority

5. **Structured logging** with correlation IDs
6. **Error classification** (Timeout, DNS, Auth, Codec)
7. **User-facing messages** mapping technical errors to actionable text
8. **Circuit breaker** for failing relays (>3 failures)

### Future Enhancements

9. **Monitoring integration** (Prometheus counters for error rates)
10. **Retry policies** configurable per error type
11. **Health checks** with degraded mode indication

---

## Quality Gate Status

**PASS Criteria (from role definition):**
- ✅ Zero empty catch blocks in critical paths
- ❌ All database/API errors logged with context (**FAIL**: No structured logging)
- ✅ No stack traces in user responses
- ❌ Retry logic for external dependencies (**FAIL**: No reconnect logic)
- ❌ Consistent error propagation (**FAIL**: Boolean returns lose context)

**Result:** ❌ **FAIL** - 3/5 criteria met

---

## Test Coverage Gaps

| Scenario | Covered | Test Location |
|----------|---------|---------------|
| Codec unsupported | ❌ | Not found |
| Decode error during playback | ❌ | Not found |
| Network timeout | ❌ | Not found |
| Relay disconnect | ❌ | Not found |
| Reconnection success | ❌ | Not found |

**Missing tests increase production risk.**

---

**Next Steps:**
1. Address CRITICAL issues before T013 completion
2. Add error integration tests (Playwright)
3. Document error recovery flow for users
4. Add monitoring dashboards for error rates
