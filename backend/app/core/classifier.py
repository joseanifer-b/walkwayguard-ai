"""AnomalyClassifier - maps a flagged event to the six-category walkway vocabulary (Section 5.6).

The STAE is trained without anomaly-class labels, so this is NOT a learned classifier. It is a
transparent rule set over optical-flow cues inside the highlighted region, compared with the
statistics of normal footage measured at training time (data/models/normal_stats.json).
The result is always shown in the UI as "(estimated)". Thresholds are class constants below.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CATEGORIES = [
    "Running", "Wrong-direction movement", "Loitering",
    "Object throwing or dropping", "Non-pedestrian object", "Sudden crowd dispersal",
]
UNCLASSIFIED = "Unclassified anomaly"

# --- tunable heuristics -----------------------------------------------------------------
MOVE_MAG = 0.4            # px/frame (at 128 x 128) above which a pixel counts as moving
RUN_RATIO = 1.8           # speed / normal speed
RUN_Z = 2.0               # or z-score above normal
WRONG_COS = -0.3          # cosine to dominant flow below this = against the flow
STATIC_RATIO = 0.35       # speed / normal speed below this = (nearly) stationary
LOITER_MIN_SEC = 3.0
SMALL_AREA = 0.03         # bbox area fraction -> small object
SHORT_EVENT_SEC = 2.5
WIDE_ASPECT = 1.2         # bbox width / height typical for bicycles, carts, skateboards
DISPERSE_FRACTION = 0.25  # fraction of the frame that is moving
DISPERSE_OUTWARD = 0.3    # outward radial flow / speed


def _gray(f: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)


def _flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(_gray(a), _gray(b), None, 0.5, 2, 9, 2, 5, 1.1, 0)


def compute_normal_stats(frames_list: list[np.ndarray], max_pairs: int = 400, seed: int = 42) -> dict:
    """Speed / dominant-direction statistics of normal footage (1-frame steps, 128 px)."""
    rng = np.random.RandomState(seed)
    pairs = []
    for vi, fr in enumerate(frames_list):
        for i in range(0, max(len(fr) - 1, 0)):
            pairs.append((vi, i))
    if not pairs:
        return {"speed_mean": 0.5, "speed_std": 0.3, "dir": [0.0, 0.0]}
    sel = rng.choice(len(pairs), size=min(max_pairs, len(pairs)), replace=False)
    speeds, vecs = [], []
    for k in sel:
        vi, i = pairs[k]
        flow = _flow(frames_list[vi][i], frames_list[vi][i + 1])
        mag = np.linalg.norm(flow, axis=2)
        moving = mag > MOVE_MAG
        if moving.sum() < 20:
            continue
        speeds.append(float(np.percentile(mag[moving], 90)))
        vecs.append(flow[moving].mean(axis=0))
    if not speeds:
        return {"speed_mean": 0.5, "speed_std": 0.3, "dir": [0.0, 0.0]}
    v = np.mean(vecs, axis=0)
    n = float(np.linalg.norm(v))
    d = (v / n).tolist() if n > 1e-6 else [0.0, 0.0]
    return {"speed_mean": float(np.mean(speeds)), "speed_std": float(np.std(speeds) + 1e-3),
            "dir": d, "dir_strength": n, "pairs": len(speeds)}


class AnomalyClassifier:
    def __init__(self, stats_path: str | Path | None = None):
        self.stats = {"speed_mean": 0.5, "speed_std": 0.3, "dir": [0.0, 0.0]}
        self.calibrated = False
        if stats_path and Path(stats_path).exists():
            try:
                self.stats = json.loads(Path(stats_path).read_text())
                self.calibrated = True
            except Exception:
                pass

    def classify(self, frames: np.ndarray, box: tuple[int, int, int, int], fps: float,
                 duration_s: float, max_pairs: int = 12) -> dict:
        """frames: (N,128,128,3) uint8 frames of the event (plus a little margin)."""
        H, W = frames.shape[1:3]
        x0, y0, x1, y1 = box
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        area = (bw * bh) / float(H * W)
        aspect = bw / float(bh)

        idxs = np.linspace(0, max(len(frames) - 2, 0), num=min(max_pairs, max(len(frames) - 1, 1))).astype(int)
        speeds, vecs, move_frac, outward = [], [], [], []
        for i in idxs:
            if i + 1 >= len(frames):
                continue
            flow = _flow(frames[i], frames[i + 1])
            mag = np.linalg.norm(flow, axis=2)
            move_frac.append(float((mag > MOVE_MAG).mean()))
            reg_flow, reg_mag = flow[y0:y1, x0:x1], mag[y0:y1, x0:x1]
            moving = reg_mag > MOVE_MAG
            if moving.sum() >= 4:
                speeds.append(float(np.percentile(reg_mag[moving], 90)))
                vecs.append(reg_flow[moving].mean(axis=0))
            all_moving = mag > MOVE_MAG
            if all_moving.sum() > 30:
                ys, xs = np.nonzero(all_moving)
                cy, cx = ys.mean(), xs.mean()
                rv = np.stack([xs - cx, ys - cy], axis=1).astype(np.float32)
                rn = np.linalg.norm(rv, axis=1, keepdims=True) + 1e-6
                radial = ((flow[ys, xs] * (rv / rn)).sum(axis=1)).mean()
                outward.append(float(radial))
        speed = float(np.median(speeds)) if speeds else 0.0
        vec = np.mean(vecs, axis=0) if vecs else np.zeros(2)
        mu, sd = float(self.stats["speed_mean"]), float(self.stats["speed_std"])
        ratio, z = speed / (mu + 1e-6), (speed - mu) / (sd + 1e-6)
        d = np.asarray(self.stats.get("dir", [0.0, 0.0]), dtype=np.float64)
        vn = float(np.linalg.norm(vec))
        cos = float(np.dot(vec, d) / (vn * (np.linalg.norm(d)) + 1e-9)) if vn > 1e-6 and np.linalg.norm(d) > 1e-6 else 0.0
        mfrac = float(np.mean(move_frac)) if move_frac else 0.0
        out_ratio = (float(np.mean(outward)) / (speed + 1e-6)) if outward and speed > 0 else 0.0

        feats = {"speed": round(speed, 3), "speed_ratio": round(ratio, 2), "speed_z": round(z, 2),
                 "direction_cos": round(cos, 2), "moving_fraction": round(mfrac, 3),
                 "outward_ratio": round(out_ratio, 2), "bbox_area": round(area, 4),
                 "aspect": round(aspect, 2), "duration_s": round(duration_s, 2)}

        if mfrac > DISPERSE_FRACTION and out_ratio > DISPERSE_OUTWARD:
            label, why = "Sudden crowd dispersal", "large moving area with outward (diverging) flow"
        elif area < SMALL_AREA and duration_s < SHORT_EVENT_SEC and speed > 0:
            label, why = "Object throwing or dropping", "small, brief, fast-moving region"
        elif self.calibrated and cos < WRONG_COS and ratio > STATIC_RATIO:
            label, why = "Wrong-direction movement", "motion opposes the dominant flow of normal footage"
        elif ratio > RUN_RATIO or z > RUN_Z:
            label, why = "Running", "region speed far above normal walking speed"
        elif ratio < STATIC_RATIO and duration_s >= LOITER_MIN_SEC:
            label, why = "Loitering", "region nearly stationary for longer than the expected dwell time"
        elif aspect > WIDE_ASPECT and ratio > 1.0:
            label, why = "Non-pedestrian object", "wide, fast region typical of a bicycle, cart or skateboard"
        else:
            label, why = UNCLASSIFIED, "motion cues match none of the six categories"
        return {"category": label, "estimated": True, "rationale": why, "features": feats}
