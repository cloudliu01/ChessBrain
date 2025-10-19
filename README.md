# ChessBrain

ChessBrain is a modular chess experimentation platform built around the principles of demonstrability, testability, and explainability. The repository is organized using an Interface → Domain → Infrastructure layering so that presentation, core logic, and integrations evolve independently.

## Project Layout

- `src/chessbrain/interface/` – Entry points for humans and systems (CLI, APIs, visualizations).
- `src/chessbrain/domain/` – Pure chess logic and state management. This layer is deterministic and fully testable.
- `src/chessbrain/infrastructure/` – Service adapters, engines, and persistence boundaries.
- `tests/` – Unit and contract tests covering the public behavior of each layer.

## Getting Started

1. Create a virtual environment (Python 3.10+).
2. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run the test suite:
   ```bash
   pytest
   ```

## Next Steps

- Extend the domain layer with full chess move generation and validation.
- Add visualization/analysis tooling in the interface layer.
- Wire advanced engines or external services through the infrastructure layer.


## How to run 
* NVidia 1080Ti 
GRAD_ACCUM_STEPS=4  EPISODES=200 MCTS_SIMULATIONS=64 BATCH_SIZE=2048 TENSORBOARD_LOG_DIR=./tb_logs ./scripts/demo/run_training_cycle.sh

## How to view the tensorboard 
tensorboard --logdir ./tb_logs --port 6006