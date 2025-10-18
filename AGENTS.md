# ChessBrain Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-10-18

## Active Technologies
- Python 3.10+, JavaScript (ES2021) + Flask, SQLAlchemy, PyTorch, python-chess, React + Vite, TensorBoard (001-chessbrain-rl-platform)

## Project Structure
```
frontend/
src/chessbrain/
tests/
scripts/demo/
```

## Commands
- `pytest` – run backend/unit suites (must fail first during TDD).
- `npx playwright test` – exercise SPA demos.
- `./scripts/demo/play_vs_ai.sh` – automate end-to-end gameplay demo.
- `./scripts/demo/run_training_cycle.sh` – queue and verify training job.
- `./scripts/demo/view_model_history.sh` – validate insights endpoints.

## Code Style
- Python: PEP 8 + type hints; format with `black` + `isort`.
- JavaScript: ESLint (recommended config) + Prettier via Vite tooling.
- Logs: JSON via `structlog` with trace IDs for observability compliance.

## Recent Changes
- 001-chessbrain-rl-platform: Added Python 3.10+, JavaScript (ES2021) + Flask, SQLAlchemy, PyTorch, python-chess, React + Vite, TensorBoard

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
