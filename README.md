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
GRAD_ACCUM_STEPS=4  EPISODES=200 MCTS_SIMULATIONS=64 BATCH_SIZE=16384 \
MICRO_BATCH_SIZE=2048 TENSORBOARD_LOG_DIR=./tb_logs \
DATABASE_URL=sqlite+pysqlite:///$(pwd)/.chessbrain-training.sqlite \
./scripts/demo/run_training_cycle.sh

## How to run with checkpoint
  1. Initial run (writes checkpoints to `MODEL_CHECKPOINT_DIR` and metrics to `TENSORBOARD_LOG_DIR`). The checkpoint now captures:
     - `model_state_dict`
     - `optimizer_state_dict` (Adam or whichever optimizer is active)
     - `scaler_state_dict` (when AMP is enabled)
     - `replay_buffer_state` (all stored self-play samples)

     Example:
     ```bash
     DATABASE_URL=sqlite+pysqlite:///$(pwd)/.chessbrain-training.sqlite \
     MODEL_CHECKPOINT_DIR=$(pwd)/models/checkpoints \
     TENSORBOARD_LOG_DIR=$(pwd)/tb_logs \
     TRAINING_LEARNING_RATE=0.0005 \
     TRAINING_L2_COEFFICIENT=0.00005 \
     python -m src.chessbrain.interface.cli.train \
       --episodes 200 \
       --batch-size 16384 \
       --checkpoint-interval 50 \
       --grad-accum-steps 4 \
       --mcts-simulations 64
     ```

  2. Resume from the latest checkpoint and add 20 more episodes. Pass the checkpoint path via `--resume-checkpoint` (or the `RESUME_CHECKPOINT` env var when using the helper script) and keep training hyper-parameters aligned with the original run so the restored optimizer and replay buffer continue seamlessly:
     ```bash
     RESUME_CHECKPOINT=models/checkpoints/<your-checkpoint>.pt \
     EPISODES=20 ./scripts/demo/run_training_cycle.sh
     # Or directly:
     TRAINING_LEARNING_RATE=0.0005 \
     TRAINING_L2_COEFFICIENT=0.00005 \
     python -m src.chessbrain.interface.cli.train \
       --episodes 20 \
       --resume-checkpoint models/checkpoints/<your-checkpoint>.pt \
       --batch-size 16384 \
       --grad-accum-steps 4 \
       --mcts-simulations 64
     ```

  3. Inspect progress live with TensorBoard:
     tensorboard --logdir ${TENSORBOARD_LOG_DIR:-data/tensorboard}

 4. Review exported games in `TENSORBOARD_LOG_DIR/<job-id>/games/`—`worst_loss_step_*.pgn` for the toughest cases and `checkmate_step_*.pgn` for decisive finishes.

### Training Hyperparameters via Environment

- `TRAINING_LEARNING_RATE` (default `0.001`) controls the optimizer step size used by the training loop.
- `TRAINING_L2_COEFFICIENT` (default `0.0001`) adjusts the L2 weight decay penalty applied during optimization.
- Export these variables before invoking the CLI or demo scripts to override the defaults without code edits.

## How to view the tensorboard 
tensorboard --logdir ./tb_logs --port 6006

## Browser Match Play
1. Launch the Flask API with your active model:
   ```bash
   export ACTIVE_MODEL_PATH=$(pwd)/models/checkpoints/c3ade8cf-step185-20251019171110.pt
   export DATABASE_URL=sqlite+pysqlite:///$(pwd)/.chessbrain-match.sqlite
   python -m flask --app src.chessbrain.interface.http.app:create_app run --port 5001
   ```
2. In another terminal, run the React client:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. Open http://localhost:5173/match to play against the AI. The match page auto-creates a session, enforces legal moves, and mirrors engine replies.
4. To validate the API loop headlessly, run `./scripts/demo/play_vs_ai.sh`; it starts the server, plays a short sequence, and prints the resulting move list.
