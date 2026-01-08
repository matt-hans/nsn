# Testing Patterns

**Analysis Date:** 2026-01-08

## Test Framework

**Rust (nsn-chain, node-core):**
- Runner: Built-in Rust test framework
- Assertion: Standard `assert!`, `assert_eq!`, `assert_ne!`
- Config: Default cargo test configuration

**Python (vortex):**
- Runner: pytest 7.4+
- Assertion: Built-in assert with pytest introspection
- Config: `pyproject.toml` [tool.pytest] section
- Async: pytest-asyncio for async test support

**TypeScript (viewer):**
- Runner: Vitest 4.0
- Assertion: Vitest built-in expect
- Config: `vitest.config.ts` in viewer root

**E2E (viewer):**
- Runner: Playwright 1.49
- Config: `playwright.config.ts`
- Location: `viewer/e2e/`

**Run Commands:**

```bash
# Rust - all tests
cd nsn-chain && cargo test
cd node-core && cargo test

# Rust - specific pallet
cargo test -p pallet-nsn-stake

# Python - all tests
cd vortex && pytest

# Python - specific test
pytest tests/unit/test_flux.py

# TypeScript - all tests
cd viewer && pnpm test

# TypeScript - watch mode
pnpm test:watch

# TypeScript - coverage
pnpm test:coverage

# E2E tests
pnpm test:e2e
```

## Test File Organization

**Rust (nsn-chain pallets):**
- Location: `pallets/{name}/src/tests.rs` or inline `#[cfg(test)]` modules
- Pattern: `#[test]` attribute on test functions
- Mock runtime: Each pallet has `mock.rs` for test configuration

**Rust (node-core crates):**
- Location: `crates/{name}/src/tests/` directory or `tests.rs`
- Pattern: `#[cfg(test)]` module with `#[test]` functions

**Python (vortex):**
- Location: `tests/unit/` and `tests/integration/`
- Naming: `test_*.py` or `*_test.py`
- Pattern: `test_` prefix on test functions

**TypeScript (viewer):**
- Location: Co-located with source `*.test.ts`
- Naming: `{component}.test.tsx`, `{service}.test.ts`
- Pattern: `describe/it` blocks with Vitest

**Structure:**
```
nsn-chain/
  pallets/nsn-stake/src/
    lib.rs
    mock.rs       # Test runtime configuration
    tests.rs      # Unit tests

node-core/
  crates/p2p/src/
    lib.rs
    tests/
      mod.rs
      gossip_tests.rs

vortex/
  tests/
    unit/
      test_flux.py
      test_clip.py
    integration/
      test_pipeline.py

viewer/
  src/
    components/
      Player.tsx
      Player.test.tsx
    hooks/
      useStream.ts
      useStream.test.ts
```

## Test Structure

**Rust Test Organization:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::*;

    #[test]
    fn test_stake_deposit() {
        new_test_ext().execute_with(|| {
            // Arrange
            let account = 1u64;
            let amount = 100u128;

            // Act
            assert_ok!(Stake::deposit(RuntimeOrigin::signed(account), amount));

            // Assert
            assert_eq!(Stake::balance(account), amount);
        });
    }

    #[test]
    #[should_panic(expected = "InsufficientBalance")]
    fn test_stake_insufficient_balance() {
        // ...
    }
}
```

**Python Test Organization:**
```python
import pytest
from vortex.models.flux import FluxLoader

class TestFluxLoader:
    """Tests for Flux model loader."""

    @pytest.fixture
    def loader(self):
        return FluxLoader(device="cpu")

    def test_load_model_success(self, loader):
        """Should load model without errors."""
        # Arrange
        config = {"model_id": "test"}

        # Act
        result = loader.load(config)

        # Assert
        assert result is not None

    def test_load_model_invalid_config(self, loader):
        """Should raise ValueError for invalid config."""
        with pytest.raises(ValueError, match="missing required"):
            loader.load({})
```

**TypeScript Test Organization:**
```typescript
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Player } from './Player';

describe('Player', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('should render player controls', () => {
      render(<Player streamId="test-123" />);
      expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
    });

    it('should show loading state initially', () => {
      render(<Player streamId="test-123" />);
      expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });
  });

  describe('playback', () => {
    it('should start playback on play button click', async () => {
      // ...
    });
  });
});
```

**Patterns:**
- Arrange/Act/Assert structure
- One concept per test
- Descriptive test names
- beforeEach for shared setup
- afterEach to restore mocks

## Mocking

**Rust Mocking:**
- Pallet mocks: `mock.rs` with test runtime configuration
- Function mocking: mockall crate where needed
- External calls: Trait-based dependency injection

**Python Mocking:**
```python
from unittest.mock import Mock, patch

@patch('vortex.models.flux.torch.load')
def test_model_loading(mock_load):
    mock_load.return_value = Mock()
    # test code
    mock_load.assert_called_once()

# Fixture-based mocking
@pytest.fixture
def mock_gpu():
    with patch('vortex.utils.vram.get_available_vram') as mock:
        mock.return_value = 12_000_000_000  # 12GB
        yield mock
```

**TypeScript Mocking:**
```typescript
import { vi } from 'vitest';

// Mock module
vi.mock('@tauri-apps/api', () => ({
  invoke: vi.fn(),
}));

// Mock in test
const mockInvoke = vi.mocked(invoke);
mockInvoke.mockResolvedValue({ data: 'test' });
```

**What to Mock:**
- External APIs and services
- File system operations
- Network calls
- GPU/hardware interactions
- Time-sensitive operations

**What NOT to Mock:**
- Pure functions and utilities
- Internal business logic
- Type definitions

## Fixtures and Factories

**Rust Test Fixtures:**
```rust
// mock.rs
pub fn new_test_ext() -> sp_io::TestExternalities {
    let mut t = frame_system::GenesisConfig::default()
        .build_storage::<Test>()
        .unwrap();
    // Configure initial state
    t.into()
}
```

**Python Fixtures:**
```python
@pytest.fixture
def sample_prompt():
    return {
        "text": "A beautiful sunset over mountains",
        "negative": "blurry, low quality",
        "steps": 20,
    }

@pytest.fixture
def mock_pipeline(sample_prompt):
    pipeline = Mock()
    pipeline.generate.return_value = b"fake_image_data"
    return pipeline
```

**TypeScript Factories:**
```typescript
function createTestStream(overrides?: Partial<Stream>): Stream {
  return {
    id: 'test-stream-123',
    status: 'active',
    videoUrl: 'https://example.com/stream.m3u8',
    ...overrides,
  };
}
```

## Coverage

**Requirements:**
- No strict coverage target enforced
- Focus on critical paths (pallets, pipelines, services)
- Coverage tracked for awareness

**Configuration:**
- Rust: `cargo tarpaulin` or `cargo llvm-cov`
- Python: `pytest --cov=vortex`
- TypeScript: Vitest with `@vitest/coverage-v8`

**View Coverage:**
```bash
# Python
pytest --cov=vortex --cov-report=html
open htmlcov/index.html

# TypeScript
pnpm test:coverage
open coverage/index.html
```

## Test Types

**Unit Tests:**
- Scope: Single function/method in isolation
- Mocking: All external dependencies
- Speed: <100ms per test
- Location: Co-located or in tests/ directory

**Integration Tests:**
- Scope: Multiple modules together
- Mocking: Only external boundaries
- Examples: Full pipeline tests, pallet interactions

**E2E Tests (viewer):**
- Framework: Playwright
- Scope: Full user flows
- Location: `viewer/e2e/`
- Run: `pnpm test:e2e`

**Pallet Tests:**
- Use mock runtime (`mock.rs`)
- Test extrinsics and storage
- Verify events emitted
- Check error conditions

## Common Patterns

**Async Testing (Python):**
```python
@pytest.mark.asyncio
async def test_async_generation():
    result = await pipeline.generate_async(prompt)
    assert result is not None
```

**Async Testing (TypeScript):**
```typescript
it('should fetch stream data', async () => {
  const data = await fetchStream('test-id');
  expect(data).toMatchObject({ id: 'test-id' });
});
```

**Error Testing (Rust):**
```rust
#[test]
fn test_error_case() {
    new_test_ext().execute_with(|| {
        assert_noop!(
            Stake::withdraw(RuntimeOrigin::signed(1), 1000),
            Error::<Test>::InsufficientStake
        );
    });
}
```

**Error Testing (TypeScript):**
```typescript
it('should throw on invalid input', async () => {
  await expect(processStream(null)).rejects.toThrow('Invalid stream');
});
```

---

*Testing analysis: 2026-01-08*
*Update when test patterns change*
