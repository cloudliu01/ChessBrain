# Implementation Plan: ChessBrain Reinforcement Learning Platform

**Branch**: `001-chessbrain-rl-platform` | **Date**: 2025-10-18 | **Spec**: [specs/001-chessbrain/spec.md](specs/001-chessbrain/spec.md)
**Input**: Feature specification from `/specs/001-chessbrain/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Deliver an end-to-end ChessBrain platform that combines a browser-based chess client, Flask backend, PostgreSQL persistence, and a PyTorch reinforcement-learning engine capable of self-play training and live inference. Training executes as a standalone flow (CLI/worker driven) that produces versioned checkpoints, while the public-facing stack only consumes published models via a plug-in inference interface. The system must support public gameplay, researcher-oriented training controls, and transparent storage of models, games, and metrics across macOS CPU and NVIDIA GPU environments.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.10+, JavaScript (ES2021)  
**Primary Dependencies**: Flask, SQLAlchemy, PyTorch, python-chess, React + Vite, TensorBoard  
**Storage**: PostgreSQL 14+ (primary), Local filesystem for `.pt` artifacts  
**Testing**: pytest, pytest-asyncio, React Testing Library, Playwright smoke demo  
**Target Platform**: macOS (Apple Silicon/Intel) dev hosts; Linux + NVIDIA CUDA for training  
**Project Type**: Web + backend + ML engine (multi-service with shared domain package)  
**Performance Goals**: Inference latency <=2 s per move on CPU; training parity across CPU/GPU; API responses <500 ms for history queries  
**Constraints**: Strict interface/domain/infrastructure separation; deterministic domain outcomes; explainable logging for inference/training; CLI demos per story; decoupled training jobs that publish models for hot-swappable inference  
**Scale/Scope**: Initial public demo + research stack; supports single-instance deployment with future horizontal scaling

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Layering**: Interface changes: new React SPA in `frontend/` consuming Flask endpoints; CLI demos in `scripts/demo/*.sh`. Domain changes: chess rule engine and training orchestration in `src/chessbrain/domain`. Infrastructure changes: Flask controllers, SQLAlchemy repositories, PyTorch device adapters, and standalone training runners in `src/chessbrain/infrastructure`. Published model artifacts move across boundaries only via an inference loader contract, keeping training flows outside UI/API pathways. Boundaries maintained via explicit service interfaces and DTOs—no cross-layer imports against constitution rules.
- **Determinism**: Domain services use `python-chess` and deterministic seeds for move validation and self-play simulations. Training loops separate stochastic exploration (in infrastructure) from domain evaluation to ensure identical inputs yield identical board states.
- **Demonstration**: Story demos executed via `scripts/demo/play_vs_ai.sh`, `scripts/demo/run_training_cycle.sh`, and `scripts/demo/view_model_history.sh`, each printing structured success logs.
- **Test-First**: For each story, author failing pytest suites (`tests/contract/test_gameplay.py`, `tests/integration/test_training_control.py`, `tests/integration/test_model_history.py`) plus Playwright UI automation before implementing. CI will run `pytest` and Playwright to enforce coverage.
- **Explainability**: Implement structured JSON logging (via `structlog`) for inference decisions, capture TensorBoard metrics for training, and expose trace IDs in API responses to correlate frontend events with backend logs.

**Gate Status**: PASS – All principles addressed with documented demos, deterministic domain controls, and observability guarantees. Re-validate after implementing contracts and data model alignments.

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```
frontend/
├── src/
│   ├── components/
│   ├── hooks/
│   └── pages/
└── public/

src/chessbrain/interface/
├── cli/
├── http/
└── telemetry/

src/chessbrain/domain/
├── chess/
├── training/
└── models/

src/chessbrain/infrastructure/
├── flask/
├── persistence/
├── rl/
└── observability/

scripts/demo/

tests/
├── unit/
├── contract/
└── integration/
```

**Structure Decision**: Interface code spans `frontend/` (React client) and `src/chessbrain/interface` (CLI + HTTP adapters). Domain logic, including chess rule enforcement and training orchestration contracts, resides in `src/chessbrain/domain`. Infrastructure adapters for Flask APIs, SQLAlchemy repositories, PyTorch device management, and offline training runners live in `src/chessbrain/infrastructure`, where training outputs publish checkpoints that the inference loader can hot-swap without UI/API coupling. Tests mirror this structure under `tests/unit`, `tests/contract`, and `tests/integration`.

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
