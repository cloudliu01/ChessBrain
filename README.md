# ChessBrain

ChessBrain is a modular chess experimentation platform built around the principles of demonstrability, testability, and explainability. The repository is organized using an Interface → Domain → Infrastructure layering so that presentation, core logic, and integrations evolve independently.

## Project Layout

- `src/chessbrain/interface/` – Entry points for humans and systems (CLI, APIs, visualizations).
- `src/chessbrain/domain/` – Pure chess logic and state management. This layer is deterministic and fully testable.
- `src/chessbrain/infrastructure/` – Service adapters, engines, and persistence boundaries.
- `tests/` – Unit and contract tests covering the public behavior of each layer.

## Getting Started

1. Create a virtual environment (Python 3.11+).
2. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run the test suite:
   ```bash
   pytest
   ```

## Usage

### e.g. spawn 4 CPU producers feeding a queue of 32 episodes
  PRODUCER_WORKERS=4 PRODUCER_QUEUE_SIZE=32 ./scripts/demo/run_training_cycle.sh

  Set PRODUCER_DEVICE if you want producers on a different device (default cpu). The training loop still supports synchronous mode when PRODUCER_WORKERS=0.

### Tests
  PYTHONPATH=. DATABASE_URL=sqlite+pysqlite:///:memory: ./venv312/bin/pytest \
    tests/unit/test_move_encoding.py \
    tests/unit/test_policy_value_network.py \
    tests/unit/test_training_loop.py \
    tests/integration/test_training_runner.py -q