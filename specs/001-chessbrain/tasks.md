---
description: "Task list for ChessBrain RL platform delivery"
---

# Tasks: ChessBrain Reinforcement Learning Platform

**Input**: Design documents from `/specs/001-chessbrain/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Acceptance and unit tests are REQUIRED per constitution. Write the failing tests first, prove they fail, then implement to make them pass.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish baseline project structure and tooling

- [ ] T001 Scaffold backend package layout in `src/chessbrain/__init__.py` with interface/domain/infrastructure subpackages
- [ ] T002 Define backend dependencies (Flask, SQLAlchemy, PyTorch, structlog, python-chess) in `requirements.txt`
- [ ] T003 Configure Python formatting and linting (black, isort, flake8) in `pyproject.toml`
- [ ] T004 Initialize Alembic migrations configuration in `alembic.ini` and bootstrap `migrations/env.py`
- [ ] T005 Bootstrap React + Vite frontend with chessboard dependencies in `frontend/package.json`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T006 Implement environment configuration loader in `src/chessbrain/infrastructure/config.py`
- [ ] T007 Establish SQLAlchemy engine/session and Base metadata in `src/chessbrain/infrastructure/persistence/base.py`
- [ ] T008 Configure structlog JSON logging and trace ID helpers in `src/chessbrain/interface/telemetry/logging.py`
- [ ] T009 Implement Torch device selection helper honoring CPU/GPU in `src/chessbrain/infrastructure/rl/device.py`
- [ ] T010 Create Flask app factory with blueprint registration scaffold in `src/chessbrain/interface/http/app.py`
- [ ] T011 Provide shared pytest fixtures for database and app context in `tests/conftest.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Browser Match Play (Priority: P1) 🎯 MVP

**Goal**: Deliver deterministic human vs AI gameplay via web UI backed by persisted sessions and inference service.

**Independent Test**: Execute `pytest tests/integration/test_gameplay_flow.py` and `./scripts/demo/play_vs_ai.sh` to validate full game completion with legal move enforcement.
**Demo Command**: `./scripts/demo/play_vs_ai.sh`

### Test Harness for User Story 1 (MUST EXECUTE FIRST) ⚠️

- [ ] T012 [P] [US1] Author contract tests for session endpoints in `tests/contract/test_sessions.py`
- [ ] T013 [P] [US1] Create backend integration test covering session lifecycle in `tests/integration/test_gameplay_flow.py`
- [ ] T014 [P] [US1] Add Playwright automation for match flow in `frontend/tests/playwright/test_play_vs_ai.spec.ts`

### Implementation for User Story 1

- [ ] T015 [P] [US1] Implement session domain orchestrator using python-chess in `src/chessbrain/domain/chess/session_manager.py`
- [ ] T016 [P] [US1] Define inference gateway interface and model cache in `src/chessbrain/domain/models/inference_service.py`
- [ ] T017 [P] [US1] Implement SQLAlchemy repository for sessions in `src/chessbrain/infrastructure/persistence/game_session_repository.py`
- [ ] T018 [US1] Build PyTorch inference adapter honoring active model in `src/chessbrain/infrastructure/rl/inference_adapter.py`
- [ ] T019 [US1] Expose gameplay REST routes for sessions and moves in `src/chessbrain/interface/http/gameplay_routes.py`
- [ ] T020 [US1] Wire gameplay blueprint into Flask app factory in `src/chessbrain/interface/http/app.py`
- [ ] T021 [US1] Implement match UI page with move controls in `frontend/src/pages/MatchPage.tsx`
- [ ] T022 [US1] Create reusable chessboard and move list components in `frontend/src/components/Match/BoardView.tsx`
- [ ] T023 [US1] Update automation script for gameplay demo in `scripts/demo/play_vs_ai.sh`

**Checkpoint**: User Story 1 functional and testable end-to-end

---

## Phase 4: User Story 2 - Training Job Orchestration (Priority: P2)

**Goal**: Deliver standalone training pipeline that produces checkpoints, records metadata, and registers models without frontend/API coupling.

**Independent Test**: Run `pytest tests/integration/test_training_runner.py` followed by `./scripts/demo/run_training_cycle.sh` to confirm checkpoint creation and model registration.
**Demo Command**: `./scripts/demo/run_training_cycle.sh`

### Test Harness for User Story 2 (MUST EXECUTE FIRST) ⚠️

- [ ] T024 [P] [US2] Implement integration test for offline training runner in `tests/integration/test_training_runner.py`
- [ ] T025 [P] [US2] Add unit tests for training loop mechanics in `tests/unit/test_training_loop.py`

### Implementation for User Story 2

- [ ] T026 [P] [US2] Implement self-play training orchestrator in `src/chessbrain/domain/training/self_play_orchestrator.py`
- [ ] T027 [P] [US2] Build PyTorch training loop with checkpoint emission in `src/chessbrain/infrastructure/rl/training_loop.py`
- [ ] T028 [US2] Persist training job metadata and status transitions in `src/chessbrain/infrastructure/persistence/training_job_repository.py`
- [ ] T029 [US2] Provide CLI entry point for training runs in `src/chessbrain/interface/cli/train.py`
- [ ] T030 [US2] Implement checkpoint publisher that hands off artifacts to inference loader in `src/chessbrain/infrastructure/rl/checkpoint_publisher.py`
- [ ] T031 [US2] Expose `/models/import` endpoint for registering offline checkpoints in `src/chessbrain/interface/http/model_routes.py`
- [ ] T032 [US2] Update training demo script to trigger CLI pipeline and import endpoint in `scripts/demo/run_training_cycle.sh`

**Checkpoint**: User Story 2 pipeline produces usable checkpoints and registers them for inference

---

## Phase 5: User Story 3 - Model & Data Insights (Priority: P3)

**Goal**: Provide researchers visibility into stored games, metrics, and model versions through API and UI.

**Independent Test**: Run `pytest tests/integration/test_model_history.py` and execute `./scripts/demo/view_model_history.sh` to validate insights retrieval.
**Demo Command**: `./scripts/demo/view_model_history.sh`

### Test Harness for User Story 3 (MUST EXECUTE FIRST) ⚠️

- [ ] T033 [P] [US3] Add contract tests for model insights endpoints in `tests/contract/test_models.py`
- [ ] T034 [P] [US3] Create integration test verifying model/game history queries in `tests/integration/test_model_history.py`

### Implementation for User Story 3

- [ ] T035 [P] [US3] Implement model repository queries with filters in `src/chessbrain/infrastructure/persistence/model_repository.py`
- [ ] T036 [P] [US3] Build insights domain service aggregating metrics in `src/chessbrain/domain/models/model_insights_service.py`
- [ ] T037 [US3] Extend model routes with listings and game history in `src/chessbrain/interface/http/model_routes.py`
- [ ] T038 [US3] Implement researcher insights UI page in `frontend/src/pages/ModelHistoryPage.tsx`
- [ ] T039 [US3] Enhance insights demo script to assert response payloads in `scripts/demo/view_model_history.sh`

**Checkpoint**: User Story 3 insights available via API and UI with supporting telemetry

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Repository-wide refinements, documentation, and hardening

- [ ] T040 [P] Refresh offline training instructions in `specs/001-chessbrain/quickstart.md`
- [ ] T041 Update project overview with new flow in `README.md`
- [ ] T042 Harden observability docs and sampling guidance in `docs/observability.md`

---

## Dependencies & Execution Order

- **Phase Dependencies**
  - Phase 1 (Setup) → prerequisite for Phase 2
  - Phase 2 (Foundational) → prerequisite for all user story phases
  - User Story phases (3–5) can proceed in priority order (P1 → P2 → P3) once Phase 2 completes; US2 and US3 may run parallel after US1 artifacts stabilize
  - Phase 6 (Polish) follows completion of targeted user stories

- **User Story Dependencies**
  - **US1** requires foundational persistence, inference loader, and logging
  - **US2** depends on US1’s inference loader and model repository scaffolding for import
  - **US3** depends on US1 persistence for sessions and US2 model registration flow

---

## Parallel Execution Opportunities

- Setup tasks T002–T005 can proceed in parallel after T001
- Foundational tasks T006–T010 may run concurrently once configuration patterns are agreed; T011 follows database scaffold
- Within US1, tests T012–T014 can run in parallel; implementation tasks T015–T022 split between backend and frontend teams
- US2 tests T024–T025 can run concurrently; training loop (T027) and repository (T028) progress in parallel after orchestrator interface (T026)
- US3 backend tasks T035–T037 can parallelize with frontend task T038 once repository contract stabilizes
- Polish tasks T040–T042 are independent and can run simultaneously after feature completion

---

## Implementation Strategy

1. Deliver MVP via User Story 1 to unblock gameplay demos and establish inference contract
2. Layer in offline training pipeline (User Story 2) to start producing improved models without touching UI/API surfaces
3. Expand insights tooling (User Story 3) to visualize stored data and model performance
4. Finalize documentation and observability guidance to support ongoing research iterations

