# Quickstart: ChessBrain RL Platform

This guide covers the minimum steps to launch the ChessBrain stack locally, play a match, and run a sample training cycle.

## Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 14+ running locally (`postgres://chessbrain:chessbrain@localhost:5432/chessbrain`)
- Poetry or pip (choose one)
- CUDA-capable GPU optional; CPU path works by default

## 1. Environment Setup
```bash
# Backend dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..

# Database (example)
createdb chessbrain
psql chessbrain -c "create user chessbrain with password 'chessbrain';"
psql chessbrain -c "grant all privileges on database chessbrain to chessbrain;"
alembic upgrade head
```

## 2. Configure Application
Create `.env` at repository root:
```env
FLASK_ENV=development
DATABASE_URL=postgresql+psycopg://chessbrain:chessbrain@localhost:5432/chessbrain
MODEL_CHECKPOINT_DIR=models/checkpoints
TENSORBOARD_LOG_DIR=data/tensorboard
PYTORCH_DEVICE=auto
```

## 3. Run Services
```bash
# Terminal 1 – backend
source .venv/bin/activate
python -m flask --app chessbrain.api run --port 5001

# Terminal 2 – frontend
cd frontend
npm run dev -- --port 5173
```

## 4. Play a Demo Match
```bash
./scripts/demo/play_vs_ai.sh \
  --frontend-url http://localhost:5173 \
  --backend-url http://localhost:5001/api/v1
```
The script launches Playwright to verify a full match and outputs structured logs containing trace IDs.

## 5. Launch Sample Training
```bash
./scripts/demo/run_training_cycle.sh \
  --episodes 50 \
  --batch-size 128 \
  --exploration 0.25 \
  --device auto
```
The CLI authenticates using environment variables, enqueues a training job, waits for checkpoint creation, and confirms TensorBoard log updates.

## 6. Inspect Model History
```bash
./scripts/demo/view_model_history.sh \
  --backend-url http://localhost:5001/api/v1 \
  --limit 5
```
This script queries the insights endpoint and prints model versions, win rates, and checkpoint paths.

## 7. Run Test Suite
```bash
pytest
npx playwright test
```
All tests MUST pass before merging changes per constitution.

## 8. Shut Down
Stop the frontend and backend processes. Optionally pause training jobs via:
```bash
curl -X POST http://localhost:5001/api/v1/training/jobs/{job_id}/pause \
  -H "Authorization: Bearer <token>"
```
