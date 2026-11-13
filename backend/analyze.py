"""Analyse one video from the command line (no UI):  python analyze.py path/to/video.mp4"""
import argparse
import json
from pathlib import Path

from app.core.pipeline import pipeline
from app.utils import new_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--threshold", type=float, default=None, help="regularity threshold tau")
    ap.add_argument("--no-explain", action="store_true")
    a = ap.parse_args()
    p = Path(a.video)
    res = pipeline.run(p, "cli", p.name, new_id(), a.threshold, not a.no_explain,
                       lambda f, m: print(f"[{f * 100:5.1f}%] {m}", flush=True))
    for e in res["events"]:
        print(f"{e['id']}  {e['timestamp']}  {e['duration']}s  score={e['anomaly_score']}  {e['severity']:6s}  {e['category']} (estimated)")
    print(json.dumps({k: res[k] for k in ("id", "frames", "n_events", "elapsed")}))
    print("Results stored under data/analyses/%s/" % res["id"])


if __name__ == "__main__":
    main()
