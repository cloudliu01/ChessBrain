# Data Model

## Overview
The platform persists gameplay, training, and model metadata in PostgreSQL while storing large PyTorch checkpoints on disk. Entities below reflect domain objects exposed via repositories in `src/chessbrain/infrastructure/persistence`.

## Entities

### GameSession
- **Fields**
  - `id` (UUID, primary key)
  - `started_at` (TIMESTAMP WITH TIME ZONE, NOT NULL)
  - `ended_at` (TIMESTAMP WITH TIME ZONE, nullable until completion)
  - `status` (ENUM: `in_progress`, `white_won`, `black_won`, `drawn`, `aborted`)
  - `initial_fen` (TEXT, defaults to standard start position)
  - `current_fen` (TEXT, updated per move)
  - `moves` (JSONB array of SAN objects `{san, uci, evaluation, timestamp}`)
  - `active_model_version` (TEXT, FK → ModelVersion.version)
  - `player_color` (ENUM: `white`, `black`) indicating human side
  - `undo_count` (INTEGER, default 0, ensures undo limit)
  - `metadata` (JSONB for UI hints)
- **Relationships**
  - References `ModelVersion` for the engine used during play.
  - Emits `TrainingSample` (future) but not persisted directly in Phase 1.
- **Validation Rules**
  - `current_fen` MUST represent a position reachable from `initial_fen` via `moves`.
  - `moves` entries MUST include deterministic evaluation payloads when available.
- **State Transitions**
  - `in_progress` → terminal states triggered by legal checkmate/draw detection or resignation.
  - `in_progress` → `aborted` on session timeout or connectivity failure.

### TrainingJob
- **Fields**
  - `id` (UUID, primary key)
  - `created_at` / `updated_at` (TIMESTAMPS)
  - `status` (ENUM: `queued`, `running`, `paused`, `completed`, `failed`)
  - `config` (JSONB capturing hyperparameters, device, seeds)
  - `metrics_uri` (TEXT pointing to TensorBoard log directory)
  - `checkpoint_version` (TEXT, FK → ModelVersion.version, nullable until checkpointed)
  - `episodes_played` (INTEGER)
  - `last_heartbeat` (TIMESTAMP WITH TIME ZONE)
- **Relationships**
  - Optionally references `ModelVersion` when a checkpoint is produced.
  - Linked to generated replay datasets (stored in filesystem, referenced via config).
- **Validation Rules**
  - `config` MUST include seed and episode budget.
  - `status` transitions require presence of `last_heartbeat` for `running`.
- **State Transitions**
  - `queued` → `running` when training loop starts.
  - `running` → `paused` via API; requires checkpoint write.
  - `running` → `completed` when episode target reached and checkpoint stored.
  - Any state → `failed` on unhandled exception, capturing error details in metadata.

### ModelVersion
- **Fields**
  - `version` (TEXT, semantic version, primary key)
  - `created_at` (TIMESTAMP WITH TIME ZONE)
  - `source_job_id` (UUID, FK → TrainingJob.id)
  - `checkpoint_path` (TEXT relative to `models/checkpoints/`)
  - `policy_accuracy` (NUMERIC)
  - `value_mse` (NUMERIC)
  - `win_rate` (NUMERIC, 0..1)
  - `is_active` (BOOLEAN, indicates production model)
  - `notes` (TEXT)
- **Relationships**
  - Belongs to a `TrainingJob`.
  - Referenced by `GameSession`.
- **Validation Rules**
  - `is_active` TRUE allowed for exactly one row (enforced in application logic).
  - `win_rate` must be between 0 and 1 inclusive.
- **State Transitions**
  - Activation toggled via admin endpoint, ensuring previous active version is demoted.

## Derived & Supporting Structures

### TrainingMetricSnapshot (Materialized View candidate)
- Aggregates per-episode outcomes for dashboards.
- Fields: `job_id`, `episode`, `policy_loss`, `value_loss`, `win_rate`, `created_at`.

### AuditTrail (Append-only log)
- Captures training control commands (`start`, `pause`, `resume`) with operator identity.
- Enables reproducible research audits.

## Data Integrity Considerations
- Use PostgreSQL transactions for multi-entity updates (e.g., promoting a model and updating active sessions).
- Enforce foreign keys to prevent orphaned checkpoints.
- Implement unique constraints on `(job_id, episode)` within derived tables to guarantee deterministic analytics.
