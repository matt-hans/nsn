# Repository Guidelines

## Project Structure & Module Organization
- `nsn-chain/`: Polkadot SDK chain (node, runtime, custom FRAME pallets).
- `node-core/`: Off-chain Rust node core (`nsn-node` binary + shared crates like `p2p`, `scheduler`, `lane0`, `lane1`, `storage`, `validator`).
- `vortex/`: Python AI generation engine (`src/vortex/` with `models/`, `pipeline/`, `utils/`).
- `viewer/`: Tauri + React desktop client (`src/` UI, `src-tauri/` Rust backend).
- `.tasks/`: Task manifest and specs; task IDs (e.g., `T013`) appear in docs.
- `.claude/`: Workflow, agents, plugins, rules, and hooks that govern task execution.

## Workflow & Agent System (.claude/)
- Task flows live in `.claude/commands/`: `/task-next` (health check + discovery), `/task-start <id>` (plugin discovery + agent selection), `/task-status` and `/task-health` (diagnostics), `/task-parallel` (worktrees), `/task-add` (new tasks).
- Authorized agents live in `.claude/agents/` (e.g., `task-developer`, `task-ui`, `task-completer`); avoid global agents outside this repo.
- Domain plugins are declared in `.claude/plugins/registry.json` (e.g., `substrate-architect`, `vram-oracle`, `bft-prover`) with constraints in `.claude/plugins/interfaces/`.
- `.claude/core/minion-engine.md` defines evidence-based execution and reliability labeling requirements.
- Project reference docs: `.claude/rules/architecture.md` and `.claude/rules/prd.md`.
- Hooks in `.claude/settings.json` auto-activate Python venvs and run Rust `cargo check`/`clippy` after Rust edits.

## Build, Test, and Development Commands
- Rust (chain/node-core): `cargo build --release`, `cargo test`, `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`.
- Vortex: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`, then `ruff check src/` and `pytest`.
- Viewer: `pnpm install`, `pnpm tauri:dev`/`pnpm tauri:build`, plus `pnpm lint` and `pnpm typecheck`.

## Coding Style & Naming Conventions
- **Rust**: format with `cargo fmt`, lint with `cargo clippy`; use `snake_case` for modules/functions and `nsn-*` / `pallet-nsn-*` naming.
- **Python**: `ruff` with `line-length = 100`; `mypy` is strict; `snake_case` functions and `PascalCase` classes.
- **TypeScript/React**: `pnpm lint` (Biome) and `pnpm typecheck`; `PascalCase` components and `camelCase` hooks.

## Testing Guidelines
- Rust tests run from each workspace (`nsn-chain`, `node-core`).
- Vortex tests use `pytest` (dev extras).
- Viewer has no test harness yet; add one only with project agreement.
- Integration tests (when present): `cd nsn-chain/test && pnpm test`.

## Commit & Pull Request Guidelines
- No established commit convention (only an initial commit exists); use short, imperative summaries (e.g., "add director slot validation").
- PRs should describe scope, link `.tasks/` IDs, and include screenshots for `viewer/` UI changes.

## Configuration & Toolchain Notes
- `nsn-chain/rust-toolchain.toml` pins Rust + WASM target.
- `viewer/package.json` requires Node >= 20 and pnpm >= 8.
- `vortex/` requires Python >= 3.11 and a CUDA-capable GPU for full workloads.
