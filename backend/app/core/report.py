"""ReportManager - logs alerts, scores, categories and explanations for operator review (Section 4.4)."""
from __future__ import annotations

import csv
import io
import json
import shutil
import threading
from pathlib import Path

from ..config import ANALYSIS_DIR

_lock = threading.Lock()


def analysis_dir(aid: str) -> Path:
    return ANALYSIS_DIR / aid


def event_dir(eid: str) -> Path:
    aid = eid.split("_e")[0]
    return ANALYSIS_DIR / aid / "events" / eid


class ReportManager:
    # -------------------------------------------------------------- analyses
    def record(self, result: dict) -> None:
        d = analysis_dir(result["id"])
        d.mkdir(parents=True, exist_ok=True)
        with _lock:
            (d / "result.json").write_text(json.dumps(result))

    def load(self, aid: str) -> dict | None:
        p = analysis_dir(aid) / "result.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def list_analyses(self) -> list[dict]:
        out = []
        for p in ANALYSIS_DIR.glob("*/result.json"):
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            out.append({k: r[k] for k in (
                "id", "video_id", "video_name", "created", "model", "tau", "theta", "duration",
                "frames", "fps", "n_events", "n_high", "elapsed") if k in r})
        return sorted(out, key=lambda r: r.get("created", ""), reverse=True)

    def delete(self, aid: str) -> bool:
        d = analysis_dir(aid)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            return True
        return False

    # -------------------------------------------------------------- events
    def list_events(self, severity: str | None = None, category: str | None = None,
                    analysis_id: str | None = None) -> list[dict]:
        events = []
        for a in self.list_analyses():
            if analysis_id and a["id"] != analysis_id:
                continue
            r = self.load(a["id"])
            if not r:
                continue
            for e in r["events"]:
                e = dict(e, video_name=r.get("video_name"), created=r.get("created"))
                if severity and e["severity"].lower() != severity.lower():
                    continue
                if category and e["category"] != category:
                    continue
                events.append(e)
        return events

    def get_event(self, eid: str) -> dict | None:
        r = self.load(eid.split("_e")[0])
        if not r:
            return None
        for e in r["events"]:
            if e["id"] == eid:
                return dict(e, video_name=r.get("video_name"), created=r.get("created"))
        return None

    def update_event(self, eid: str, patch: dict) -> dict | None:
        aid = eid.split("_e")[0]
        r = self.load(aid)
        if not r:
            return None
        for e in r["events"]:
            if e["id"] == eid:
                e.update(patch)
                self.record(r)
                return e
        return None

    # -------------------------------------------------------------- export
    def export_csv(self, events: list[dict]) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["event_id", "video", "timestamp", "duration_s", "anomaly_score", "threshold",
                    "severity", "estimated_category", "lime_top_region", "lime_local_r2"])
        for e in events:
            lime = e.get("lime") or {}
            top = (lime.get("top") or [{}])[0]
            w.writerow([e["id"], e.get("video_name", ""), e["timestamp"], e["duration"],
                        e["anomaly_score"], e["threshold"], e["severity"], e["category"],
                        top.get("location", ""), lime.get("r2", "")])
        return buf.getvalue()


reports = ReportManager()
