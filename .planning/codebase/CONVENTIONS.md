# Coding Conventions

**Analysis Date:** 2026-01-08

## Naming Patterns

**Files:**
- `snake_case.rs` - Rust source files
- `snake_case.py` - Python source files
- `kebab-case.ts/tsx` - TypeScript/React files
- `UPPER_CASE.md` - Important project files (CLAUDE, README, CHANGELOG)
- `*.test.ts` - Test files co-located with source

**Functions:**
- Rust: `snake_case` for all functions
- Python: `snake_case` for all functions
- TypeScript: `camelCase` for functions, `handleEventName` for handlers

**Variables:**
- Rust: `snake_case`, `SCREAMING_SNAKE_CASE` for constants
- Python: `snake_case`, `UPPER_CASE` for module constants
- TypeScript: `camelCase`, `UPPER_SNAKE_CASE` for constants

**Types:**
- Rust: `PascalCase` for structs, enums, traits
- Python: `PascalCase` for classes
- TypeScript: `PascalCase` for types/interfaces, no `I` prefix

## Code Style

**Rust Formatting:**
- `cargo fmt` with default settings
- 100 character line length (rustfmt default)
- 4 space indentation
- Explicit lifetimes where required

**Rust Linting:**
- `cargo clippy -- -D warnings` (treat warnings as errors)
- All clippy lints enabled by default
- Run: `cargo clippy --release --workspace`

**Python Formatting:**
- Ruff for linting and formatting
- 100 character line length
- Target Python 3.11
- Rules: E, F, I, N, W, UP

**Python Linting:**
- `ruff check src/`
- mypy with strict mode for type checking
- Run: `ruff check src/ && mypy src/`

**TypeScript Formatting:**
- Biome for linting and formatting
- Run: `pnpm lint` or `biome check .`
- Fix: `pnpm lint:fix` or `biome check --write .`

**TypeScript Type Checking:**
- Strict mode enabled in tsconfig.json
- Run: `pnpm typecheck` or `tsc --noEmit`

## Import Organization

**Rust:**
1. `std` library imports
2. External crate imports
3. `crate::` internal imports
4. `super::` parent imports
- Blank line between groups

**Python:**
1. Standard library imports
2. Third-party imports
3. Local application imports
- Blank line between groups
- Sorted alphabetically within groups (enforced by ruff I)

**TypeScript:**
1. External packages (react, @tauri-apps/api)
2. Internal modules (@/ aliases if configured)
3. Relative imports (., ..)
4. Type imports (import type {})
- Blank line between groups

## Error Handling

**Rust Patterns:**
- `Result<T, E>` for fallible operations
- `thiserror` for custom error types in libraries
- `anyhow` for application-level error handling
- `?` operator for propagation
- Error types per crate: `crate::Error`

**Python Patterns:**
- Custom exception classes inheriting from `Exception`
- `try/except` with specific exception types
- Re-raise with context: `raise NewError(...) from e`
- Type hints for exceptions in docstrings

**TypeScript Patterns:**
- Throw errors, catch at boundaries
- Typed error classes where needed
- Async/await with try/catch (no .catch() chains)
- Error boundaries in React for UI errors

## Logging

**Rust:**
- `tracing` crate with spans and events
- Structured logging with fields
- Levels: trace, debug, info, warn, error
- Filter via `RUST_LOG` environment variable

**Python:**
- Standard `logging` module
- Structured logging with context
- Levels: debug, info, warning, error
- Prometheus metrics for observability

**TypeScript:**
- Console logging (Tauri handles persistence)
- Structured objects for complex data

## Comments

**When to Comment:**
- Explain "why", not "what"
- Document business rules and algorithms
- Note non-obvious edge cases
- Reference relevant tasks (T001, T002, etc.)

**Rust Documentation:**
- `///` for public API documentation
- `//!` for module-level documentation
- Examples in doc comments where helpful

**Python Documentation:**
- Docstrings for all public functions/classes
- NumPy or Google style docstrings
- Type hints in signatures (not in docstrings)

**TypeScript Documentation:**
- JSDoc for public APIs
- Minimal comments for self-explanatory code

**TODO Comments:**
- Format: `// TODO: description`
- Reference task if exists: `// TODO(T042): description`

## Function Design

**Rust:**
- Keep functions under 50 lines
- Extract helpers for complex logic
- Use `impl Trait` for return types where appropriate
- Prefer borrowing over ownership when possible

**Python:**
- Keep functions under 50 lines
- Type hints for all parameters and returns
- Default parameters for optional configuration
- `*args, **kwargs` sparingly and documented

**TypeScript:**
- Keep functions under 50 lines
- Max 3 parameters, use options object for more
- Explicit return types for public functions
- Arrow functions for callbacks, named for methods

## Module Design

**Rust:**
- One concept per module
- `lib.rs` exposes public API
- `mod.rs` for submodule organization
- Re-export common types at crate root

**Python:**
- `__init__.py` exports public API
- One class per file for complex classes
- Related functions can share a module
- `__all__` to control exports

**TypeScript:**
- Named exports preferred
- Default exports for React components only
- `index.ts` barrel files for directory exports
- Avoid circular dependencies

## Pallet Conventions (Substrate)

**Structure:**
```rust
#![cfg_attr(not(feature = "std"), no_std)]

#[frame::pallet]
pub mod pallet {
    use super::*;

    #[pallet::config]
    pub trait Config: frame_system::Config { ... }

    #[pallet::storage]
    pub type StorageItem<T> = StorageValue<...>;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> { ... }

    #[pallet::error]
    pub enum Error<T> { ... }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        #[pallet::call_index(0)]
        #[pallet::weight(...)]
        pub fn extrinsic_name(...) -> DispatchResult { ... }
    }
}
```

**Naming:**
- Storage: `PascalCase` (e.g., `StakeInfo`, `ReputationScores`)
- Events: Past tense verbs (e.g., `Staked`, `ReputationUpdated`)
- Errors: Descriptive conditions (e.g., `InsufficientStake`, `NotAuthorized`)
- Extrinsics: `snake_case` verbs (e.g., `stake`, `update_reputation`)

---

*Convention analysis: 2026-01-08*
*Update when patterns change*
