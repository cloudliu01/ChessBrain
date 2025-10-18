# Feature Specification: ChessBrain Reinforcement Learning Platform

**Feature Branch**: `[001-chessbrain-rl-platform]`  
**Created**: 2025-10-18  
**Status**: Draft  
**Input**: User description: "PRD: ChessBrain interactive RL platform"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Browser Match Play (Priority: P1)

Public player visits the site, starts a match, and plays a legal chess game against the current AI model with move history and controls available.

**Why this priority**: Direct gameplay is the flagship experience and must be production-ready before any research tooling adds value.

**Independent Test**: Launch the stack locally, initiate a browser session, and verify a full game can be completed via automated UI flow with deterministic results.

**Demo Command**: `scripts/demo/play_vs_ai.sh`

**Acceptance Scenarios**:

1. **Given** the web UI is loaded, **When** the player selects "New Game", **Then** a session is created and the opening move is requested from the backend.
2. **Given** a running game, **When** the player attempts an illegal move, **Then** the UI blocks the action, displays an error, and the backend logs the rejection.

---

### User Story 2 - Training Job Orchestration (Priority: P2)

Researcher authenticates, configures, and launches self-play training jobs, monitoring progress and pausing or resuming as needed.

**Why this priority**: Continuous improvement of the AI requires a dependable training pipeline managed through the same backend.

**Independent Test**: Execute training control endpoints via integration tests to start, pause, resume, and checkpoint a self-play job on CPU with mocked GPU availability.

**Demo Command**: `scripts/demo/run_training_cycle.sh`

**Acceptance Scenarios**:

1. **Given** valid credentials, **When** a researcher POSTs a training configuration, **Then** the backend schedules the job and persists metadata in PostgreSQL.
2. **Given** an active job, **When** the researcher sends a pause request, **Then** the training loop halts gracefully and a checkpoint file is written.

---

### User Story 3 - Model & Data Insights (Priority: P3)

Researcher inspects stored games, training metrics, and model versions through the web interface or API to compare performance over time.

**Why this priority**: Transparent insight into learning progress is essential for research credibility and informs future iterations.

**Independent Test**: Seed the database with sample games and checkpoints, run API queries, and verify responses include required metadata and links to artifacts.

**Demo Command**: `scripts/demo/view_model_history.sh`

**Acceptance Scenarios**:

1. **Given** existing training records, **When** the researcher requests model history, **Then** the API returns ordered metadata with links to `.pt` checkpoints and evaluation stats.
2. **Given** completed games, **When** the researcher queries replay data, **Then** the system returns move lists, timestamps, and outcome labels within 500 ms.

---

### Edge Cases

- How does the system respond when PostgreSQL is unreachable during an active match or training job?
- What happens if the AI inference exceeds the 2 s budget or GPU resources are unavailable?
- How are simultaneous training and gameplay sessions isolated to prevent model file contention?
- How does the UI recover from a dropped WebSocket or REST connection mid-game while preserving state?
- How are observability/trace outputs captured and preserved if TensorBoard logging fails?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The web UI MUST enforce all chess rules, including promotion, en passant, repetition, and stalemate detection.
- **FR-002**: Players MUST be able to start, reset, undo, resign, and review move history within the browser session.
- **FR-003**: The backend MUST expose inference APIs that respond within <=2 seconds per move under CPU-only operation.
- **FR-004**: Game sessions MUST persist in PostgreSQL with unique identifiers, lifecycle timestamps, and move records.
- **FR-005**: The RL engine MUST support configurable self-play training (episodes, batch size, exploration rate) via PyTorch.
- **FR-006**: Training MUST emit checkpoints (.pt files) with semantic version tags and store metadata (hash, device, metrics) in the database.
- **FR-007**: Training control endpoints MUST require authenticated access and allow start, pause, resume, and status retrieval.
- **FR-008**: Observability MUST include TensorBoard metrics and JSON-structured logs streaming inference and training events.
- **FR-009**: Demo commands (`play_vs_ai.sh`, `run_training_cycle.sh`, `view_model_history.sh`) MUST execute the end-to-end scenarios for each user story.
- **FR-010**: The system MUST operate on both macOS CPU and NVIDIA GPU hosts without code changes, selecting the device dynamically.

### Key Entities *(include if feature involves data)*

- **GameSession**: Tracks game state, players, move list, status, timestamps, and linked model version.
- **TrainingJob**: Captures configuration parameters, current status, checkpoints, metrics path, and runtime environment.
- **ModelVersion**: Stores semantic version, evaluation metrics, checkpoint URI, training provenance, and activation flag.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The `scripts/demo/play_vs_ai.sh` flow completes a full game with zero illegal move regressions in automated tests.
- **SC-002**: Median inference latency during automated matches is <=2 seconds on macOS CPU with deterministic seeds.
- **SC-003**: A 100-episode self-play training run completes on CPU and GPU environments with identical configuration files.
- **SC-004**: Model history API returns results including checkpoint URIs and win/loss metrics in under 500 ms for 1k records.
- **SC-005**: TensorBoard dashboards update with loss and win-rate metrics within 60 seconds of training loop iterations.
