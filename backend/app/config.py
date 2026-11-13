"""Central configuration + persisted runtime settings (Table 6.1 of the report)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
DATASET_DIR = PROJECT_DIR / "dataset"
UPLOAD_DIR = DATA_DIR / "uploads"
ANALYSIS_DIR = DATA_DIR / "analyses"
MODEL_DIR = DATA_DIR / "models"
METRICS_DIR = DATA_DIR / "metrics"
FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"

for _d in (DATA_DIR, UPLOAD_DIR, ANALYSIS_DIR, MODEL_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov"}

# Report Table 6.1 / 5.2 defaults.
DEFAULTS: dict = {
    "image_size": 128,          # 128 x 128 x 3 input
    "seq_len": 10,              # T = 10
    "window_stride": 1,         # sliding-window stride used at analysis time
    "threshold": 0.35,          # regularity threshold tau (anomaly-score threshold = 1 - tau = 0.65)
    "lime_segments": 100,       # K (SLIC)
    "lime_samples": 1000,       # N
    "lime_top_k": 8,            # highlighted superpixels
    "lime_batch": 64,
    "seed": 42,
    "max_upload_mb": 500,
    "max_frames": 30000,        # safety cap per analysed video
    "merge_gap_sec": 1.0,       # flagged frames closer than this are merged into one event
    "min_event_frames": 3,
    "explain_events": True,     # run LIME on the peak frame of each event
    "max_events_explained": 20,
}


class Settings:
    """Thread-safe settings persisted to data/settings.json."""

    def __init__(self) -> None:
        self._path = DATA_DIR / "settings.json"
        self._lock = threading.Lock()
        self._values = dict(DEFAULTS)
        if self._path.exists():
            try:
                self._values.update(json.loads(self._path.read_text()))
            except Exception:
                pass

    def all(self) -> dict:
        with self._lock:
            return dict(self._values)

    def get(self, key: str):
        with self._lock:
            return self._values[key]

    def update(self, patch: dict) -> dict:
        with self._lock:
            for k, v in patch.items():
                if k in DEFAULTS and v is not None:
                    default = DEFAULTS[k]
                    self._values[k] = (v if isinstance(default, bool) else type(default)(v))
            self._path.write_text(json.dumps(self._values, indent=2))
            return dict(self._values)

    def reset(self) -> dict:
        with self._lock:
            self._values = dict(DEFAULTS)
            self._path.write_text(json.dumps(self._values, indent=2))
            return dict(self._values)


settings = Settings()
