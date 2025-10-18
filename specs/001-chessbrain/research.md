# Research Findings

## Backend Framework & Layering
- **Decision**: Use Flask with Blueprints per service (gameplay, training, insights) and dependency inversion into `src/chessbrain/interface/http`.
- **Rationale**: Flask stays lightweight, integrates smoothly with SQLAlchemy, and allows explicit adapter-layer boundaries mandated by the constitution.
- **Alternatives considered**: FastAPI (strong typing but adds ASGI complexity beyond current needs); Django (heavier ORM and templating not required for SPA-backed API).

## Chess Rule Enforcement
- **Decision**: Adopt `python-chess` within `src/chessbrain/domain/chess` to encode legal move validation and board state transitions.
- **Rationale**: Library offers deterministic rule handling (promotion, en passant, repetitions) and integrates with PGN/FEN conversion for storage and demos.
- **Alternatives considered**: Custom move engine (too error-prone for timeline); Stockfish bindings (non-deterministic and violates explainability requirement).

## Reinforcement Learning Engine
- **Decision**: Implement AlphaZero-inspired self-play using PyTorch with separate policy and value heads, pluggable exploration schedule, and experience replay buffers persisted in PostgreSQL.
- **Rationale**: Matches research objectives, leverages PyTorch ecosystem for GPU acceleration, and keeps training loops transparent for explainability.
- **Alternatives considered**: Stable Baselines (optimized for RL environments but less suited to custom chess MCTS pipelines); TensorFlow (less alignment with existing tooling).

## Model Storage & Checkpointing
- **Decision**: Store `.pt` checkpoints on disk under `models/checkpoints/` with metadata rows in PostgreSQL referencing semantic versions and evaluation metrics.
- **Rationale**: Keeps large binaries out of the database while enabling transactional metadata queries for researcher insights.
- **Alternatives considered**: S3 object storage (future-ready but adds infra overhead now); storing binaries in DB (complicates backups and restores).

## Database Access Patterns
- **Decision**: Use SQLAlchemy ORM with Alembic migrations; repositories in `src/chessbrain/infrastructure/persistence` expose domain DTOs.
- **Rationale**: SQLAlchemy supports PostgreSQL JSONB columns for flexible metadata, Alembic ensures reproducible schema evolution.
- **Alternatives considered**: Direct psycopg access (faster but undermines maintainability); ORM-less query builders (harder to enforce layering).

## Frontend Stack
- **Decision**: Build the web UI with React + Vite, using the `chessboardjsx` successor `react-chessboard` for rendering and SWR for data fetching.
- **Rationale**: React ecosystem accelerates component reuse, Vite provides fast dev builds, and chessboard components reduce UI boilerplate.
- **Alternatives considered**: Vanilla JS/Canvas (lighter but slower to implement advanced UX); Next.js (SSR unnecessary for SPA).

## API Transport
- **Decision**: Start with RESTful JSON endpoints backed by Flask; add WebSocket channel only for real-time training updates in Phase 2 if needed.
- **Rationale**: REST satisfies gameplay latency constraints and simplifies demo scripts; training telemetry can stream through polling or SSE initially.
- **Alternatives considered**: Immediate WebSocket adoption (higher complexity and test burden); gRPC (overkill for browser client).

## Testing Strategy
- **Decision**: Combine pytest unit suites for domain/infrastructure, contract tests for HTTP APIs, and Playwright automation for SPA demos.
- **Rationale**: Aligns with constitution’s test-first rule, provides deterministic coverage across layers, and exercises CLI demos end-to-end.
- **Alternatives considered**: Selenium (heavier, less maintainable) or Cypress (JS-only, complicates Python-based test orchestration).

## Observability & Explainability
- **Decision**: Use `structlog` for JSON-formatted logs tagged with trace IDs, integrate TensorBoard for training metrics, and expose inference rationale via annotated move choices.
- **Rationale**: Structured logs enable correlation across frontend/backend; TensorBoard already suits PyTorch; explicit annotations satisfy explainability principle.
- **Alternatives considered**: Plain logging (insufficient structure); custom dashboard (duplication of TensorBoard features).

## Device Management
- **Decision**: Centralize device selection logic in `src/chessbrain/infrastructure/rl/device.py`, defaulting to CUDA when available and falling back to CPU with consistent seeds.
- **Rationale**: Guarantees identical code paths across macOS and NVIDIA environments and isolates torch device handling from domain services.
- **Alternatives considered**: Per-module device selection (risk of drift); environment-variable switching without abstraction (harder to test).
