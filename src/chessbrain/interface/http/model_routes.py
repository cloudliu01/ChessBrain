from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, request

model_bp = Blueprint("models", __name__)


@model_bp.post("/import")
def import_model():
    """Register a checkpoint artifact for downstream inference loading."""
    payload: dict[str, Any] = request.get_json(silent=True) or {}

    checkpoint_path = payload.get("checkpoint_path")
    if not checkpoint_path:
        return jsonify({"error": "checkpoint_path is required"}), 400

    version = payload.get("version")
    source_job_id = payload.get("source_job_id")
    metrics = payload.get("metrics", {})

    registry_root = Path(current_app.config["MODEL_CHECKPOINT_DIR"])
    registry_root.mkdir(parents=True, exist_ok=True)
    registry_path = registry_root / "registry.json"

    entry = {
        "version": version,
        "source_job_id": source_job_id,
        "checkpoint_path": checkpoint_path,
        "metrics": metrics,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }

    existing: list[dict[str, Any]] = []
    if registry_path.exists():
        existing = json.loads(registry_path.read_text(encoding="utf-8"))

    # Replace any existing entry with the same version to keep registry idempotent.
    if version:
        existing = [row for row in existing if row.get("version") != version]

    existing.append(entry)
    registry_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    response_body = {"message": "model registered", "version": version, "entries": len(existing)}
    return jsonify(response_body), 201


__all__ = ["model_bp"]
