
## 1. Overview

**ChessBrain** is an interactive web-based chess AI platform designed to both **demonstrate and apply reinforcement learning**.
It integrates a **web interface**, **Flask backend**, **database persistence**, and a **PyTorch RL engine** capable of self-training and playing full-rule chess.

It serves **two complementary audiences**:

* **Public users** who can play against the AI through a clean and responsive web interface.
* **Researchers and developers** who can explore, retrain, and evolve the AI model with full visibility into training data, results, and system logs.

---

## 2. Objectives

1. Build an **end-to-end chess system** combining human-facing playability with research-grade RL training.
2. Ensure **portability and maintainability** — runs on macOS (CPU) and NVIDIA GPU environments with minimal configuration changes.
3. Guarantee **testability, clarity, and modular structure** across code, model, and data layers.
4. Provide a **simple, self-contained stack** suitable for standalone demos or educational use.

---

## 3. User Profiles

### 3.1 Public User (Player)

* Access via browser; no installation required.
* Can start a new game, play against the trained AI, undo moves, and view basic status (move history, result, evaluation).
* No registration needed for play; session handled temporarily.

### 3.2 Researcher / Developer

* Runs backend locally or on GPU host.
* Can start, stop, and resume training jobs.
* Views or exports training logs, replay data, and model checkpoints.
* Uses the same database and web backend for persistence and experiment tracking.

---

## 4. Core Features

### 4.1 Web Gameplay Interface

* Modern, responsive chessboard UI.
* Legal move enforcement (promotion, en passant, castling, repetition draw).
* Turn indicator, move list, reset / resign / undo.
* Communication with Flask backend via REST or WebSocket.

### 4.2 Backend Server (Flask)

* Game session management (create, update, terminate).
* Model inference (select next move from policy / value net).
* Training control endpoints (start, pause, checkpoint).
* Basic authentication and admin endpoints (for research use).

### 4.3 Database

* Persistent storage for:

  * User sessions and game history.
  * Model metadata (version, training parameters, timestamp).
  * RL training records (replays, win/loss stats, training metrics).
* **Recommended choice:** **PostgreSQL**, because:

  * Reliable and easy to deploy cross-platform.
  * Integrates well with SQLAlchemy ORM.
  * Scalable enough for research data; simple enough for local use.
  * JSONB columns allow flexible metadata storage.

### 4.4 Reinforcement Learning Engine

* Self-play training loop (actor-critic or AlphaZero-style).
* Policy + value networks implemented in PyTorch.
* Configurable hyperparameters (learning rate, batch size, exploration, etc.).
* Periodic checkpoint saving and evaluation.
* Training can run on macOS CPU or NVIDIA GPU with identical code.

### 4.5 Model Inference

* Serve best available model to gameplay sessions.
* Convert board state → tensor → policy/value output → selected move.
* Optional deterministic or stochastic play modes (for difficulty levels).

---

## 5. System Architecture

### 5.1 High-Level Diagram

```
[Web UI]  <-->  [Flask Backend]  <-->  [PostgreSQL DB]
                       |
                       v
                [PyTorch RL Engine]
                       |
                       v
                [Model Artifacts (.pt)]
```

### 5.2 Layer Responsibilities

| Layer                 | Responsibility                                                 |
| --------------------- | -------------------------------------------------------------- |
| Frontend              | User interface for human vs AI chess play.                     |
| Backend (Flask)       | API for gameplay, inference, and training control.             |
| Database (PostgreSQL) | Store games, training records, model metadata.                 |
| RL Engine (PyTorch)   | Self-play training, policy/value evaluation.                   |
| Model Storage         | Organized directory for `.pt` checkpoints with metadata in DB. |

---

## 6. Functional Requirements

| ID   | Category       | Requirement                                                           |
| ---- | -------------- | --------------------------------------------------------------------- |
| F-01 | Gameplay       | The web interface enforces all chess rules via legal move checking.   |
| F-02 | Gameplay       | Players can play against the current best AI model.                   |
| F-03 | Gameplay       | Support move undo, reset, and new game.                               |
| F-04 | Inference      | Backend evaluates moves within a defined time budget (e.g., ≤2 s).    |
| F-05 | Training       | Self-play training from scratch, configurable parameters.             |
| F-06 | Training       | Save and reload checkpoints with version tags.                        |
| F-07 | Data           | Store all played games and training episodes.                         |
| F-08 | Admin          | Provide endpoints to list, load, and compare model versions.          |
| F-09 | Cross-Platform | Training and inference must run on both macOS (CPU) and NVIDIA (GPU). |

---

## 7. Non-Functional Requirements

### 7.1 Code Quality

* Follow **PEP 8** and **docstring standards**.
* Use **black** or **isort** for formatting.
* Type hints for all public functions.
* All modules include minimal unit tests.

### 7.2 Testability

* Core modules (engine, API, database access) independently testable.
* Mock data for RL and gameplay to allow offline testing.
* Deterministic random seeds for reproducibility.

### 7.3 Project Structure

```
chessbrain/
├── frontend/         # React or simple JS/HTML client
├── backend/          # Flask app + API routes
├── models/           # PyTorch model definitions
├── training/         # RL training scripts
├── data/             # Saved games, replays, logs
├── tests/            # Unit & integration tests
└── utils/            # Common utilities
```

### 7.4 Deployment & Configuration

* Local deployment via `python -m flask run`.
* Config via `.env` or YAML (for DB, model paths, device type).
* Requirements specified in `requirements.txt`.
* Docker optional for GPU/remote environments.

### 7.5 Observability

* Structured logging for Flask (JSON logs).
* Model training logs via `tensorboard`.
* Error reporting with clear tracebacks and context.

### 7.6 Performance

* Inference latency under 2 s (CPU acceptable).
* Training uses mini-batch updates, GPU acceleration optional.

---

## 8. Technology Stack

| Layer     | Technology               | Notes                                   |
| --------- | ------------------------ | --------------------------------------- |
| Frontend  | HTML / JS / CSS or React | Light, fast chessboard rendering.       |
| Backend   | Flask + SQLAlchemy       | Simplicity, readability, mainstream.    |
| Database  | PostgreSQL               | Reliable, easy ORM integration.         |
| AI Engine | PyTorch                  | Unified training + inference framework. |
| Logging   | TensorBoard + JSON logs  | For metrics and debugging.              |

---

## 9. Training & Inference Environment

| Platform       | Framework      | Hardware                 | Notes                                           |
| -------------- | -------------- | ------------------------ | ----------------------------------------------- |
| macOS          | PyTorch (CPU)  | Apple Silicon / Intel    | Dev and demo use.                               |
| Linux (GPU)    | PyTorch + CUDA | NVIDIA RTX/Compute       | Full RL training and high-speed inference.      |
| Cross-Platform | Same codebase  | Dynamic device switching | `torch.device("cuda" if available else "cpu")`. |

---

## 10. Development Phases

| Phase   | Milestone                       | Deliverables                             |
| ------- | ------------------------------- | ---------------------------------------- |
| Phase 1 | Web interface + basic Flask API | Playable demo with random/heuristic bot. |
| Phase 2 | RL engine integration           | Self-play + policy/value training loop.  |
| Phase 3 | Database persistence            | Store games, models, metrics.            |
| Phase 4 | Cross-platform optimization     | Verified on macOS + NVIDIA GPU.          |
| Phase 5 | Public demo packaging           | Stable release, documentation, examples. |

---

## 11. Governance

* All new code must comply with this PRD’s principles and folder structure.
* Any non-standard dependency must be justified and documented.
* Code reviews emphasize **clarity, testability, and maintainability**, not raw performance.
* Future PRD revisions must be versioned and changelogged.
