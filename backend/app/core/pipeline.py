"""End-to-end STAE + LIME pipeline (Algorithm 1 / Section 5.7).

video -> sliding windows -> STAE -> reconstruction error -> regularity score s_t -> flag s_t < tau
      -> merge into events -> (estimated category) -> LIME on the event's peak frame -> heatmaps
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..config import MODEL_DIR, UPLOAD_DIR, settings
from ..models.stae import STAE, count_parameters, reconstruction_error
from ..utils import fmt_time, get_device, now_iso
from . import heatmap as hm
from .classifier import AnomalyClassifier
from .lime_explainer import LimeExplainer, LimeResult
from .report import event_dir, reports
from .scorer import AnomalyScorer
from .sequence_loader import SequenceLoader

NORMAL_STATS = MODEL_DIR / "normal_stats.json"
EXPLANATION_TEXT = (
    "The frame was flagged because its reconstruction error exceeded the calibrated threshold. "
    "LIME identified the highlighted region as strongly contributing to the anomaly score."
)


class ModelNotTrained(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
class ModelRegistry:
    """Loads / caches trained checkpoints (data/models/<name>.pt)."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[STAE, dict]] = {}
        self._lock = threading.Lock()
        self.device = get_device()

    @staticmethod
    def path(name: str) -> Path:
        return MODEL_DIR / f"{name}.pt"

    def available(self, name: str = "stae") -> bool:
        return self.path(name).exists()

    def meta(self, name: str = "stae") -> dict | None:
        if not self.available(name):
            return None
        return self.get(name)[1]

    def get(self, name: str = "stae") -> tuple[STAE, dict]:
        with self._lock:
            if name in self._cache:
                return self._cache[name]
            p = self.path(name)
            if not p.exists():
                raise ModelNotTrained(
                    f"No trained '{name}' checkpoint found. Train the model first "
                    f"(Model page, or `python train.py`)."
                )
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            meta = ckpt.get("meta", {})
            model = STAE(use_convlstm=meta.get("use_convlstm", name != "spatial"))
            model.load_state_dict(ckpt["state_dict"])
            model.to(self.device).eval()
            meta["params"] = count_parameters(model)
            self._cache[name] = (model, meta)
            return self._cache[name]

    def invalidate(self, name: str | None = None) -> None:
        with self._lock:
            if name:
                self._cache.pop(name, None)
            else:
                self._cache.clear()


registry = ModelRegistry()


# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def score_frames(model: STAE, frames: np.ndarray, seq_len: int, stride: int, device,
                 batch: int = 16, progress: Callable[[float], None] | None = None) -> np.ndarray:
    """Per-frame reconstruction error e_t. Each frame is reconstructed by every window that covers
    it (stride 1 = every frame sees full temporal context); the errors are averaged."""
    N = len(frames)
    loader = SequenceLoader(seq_len=seq_len, stride=stride)
    starts = loader.window_starts(N)
    if not starts:
        raise ValueError(f"Video too short: need at least {seq_len} frames, got {N}.")
    err_sum, err_cnt = np.zeros(N), np.zeros(N)
    model.eval()
    for bi in range(0, len(starts), batch):
        bs = starts[bi:bi + batch]
        x = loader.normalise(np.stack([frames[s:s + seq_len] for s in bs])).to(device)
        e = reconstruction_error(x, model(x)).cpu().numpy()  # (B, T)
        for s, row in zip(bs, e):
            err_sum[s:s + seq_len] += row
            err_cnt[s:s + seq_len] += 1
        if progress:
            progress(min(1.0, (bi + batch) / len(starts)))
    return err_sum / np.maximum(err_cnt, 1)


@torch.no_grad()
def reconstruct_at(model: STAE, context: np.ndarray, pos: int, device) -> tuple[np.ndarray, float]:
    """Reconstruction (uint8 RGB 128x128) and error of frame `pos` inside its T-frame window."""
    x = SequenceLoader.normalise(context[None]).to(device)
    xh = model(x)
    err = float(reconstruction_error(x, xh)[0, pos])
    rec = (xh[0, pos].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return rec, err


def window_for(n_frames: int, t: int, T: int) -> tuple[int, int]:
    """Window start and position of frame t so that every window has exactly T frames."""
    start = min(max(0, t - T + 1), n_frames - T)
    return start, t - start


# ---------------------------------------------------------------------------------------------
def render_event(edir: Path, display: np.ndarray, frame128: np.ndarray, recon128: np.ndarray,
                 lime: LimeResult | None) -> tuple[tuple[int, int, int, int], list[str]]:
    """Write all event images; returns (box in 128-px coords, image names)."""
    names = []
    dh = display.shape[:2]
    emap_img, emap_raw = hm.error_map(frame128, recon128, dh)
    box = hm.bbox_from_mask(lime.mask) if (lime is not None and lime.mask is not None) else None
    if box is None:
        box = hm.bbox_from_error(emap_raw)

    def put(name: str, img: np.ndarray) -> None:
        hm.save_jpg(edir / f"{name}.jpg", img)
        names.append(name)

    put("original", display)
    put("reconstruction", hm._up(recon128, dh))
    put("error_map", emap_img)
    put("highlight", hm.draw_box(display, box))
    put("zoom", hm.zoom_crop(display, box))
    if lime is not None:
        put("lime_heatmap", hm.lime_heatmap(display, lime.segments, lime.weights))
        put("superpixels", hm.superpixel_image(display, lime.segments))
    return box, names


def lime_payload(lime: LimeResult | None, box) -> dict | None:
    if lime is None:
        return None
    return {
        "top": lime.top, "r2": round(lime.r2, 4), "latency": round(lime.latency, 3),
        "n_samples": lime.n_samples, "n_segments": lime.n_segments, "top_k": lime.top_k,
        "base_error": lime.base_error,
        "text": EXPLANATION_TEXT,
    }


# ---------------------------------------------------------------------------------------------
class AnalysisPipeline:
    def __init__(self) -> None:
        self.classifier = AnomalyClassifier(NORMAL_STATS)

    def run(self, video_path: Path, video_id: str, video_name: str, analysis_id: str,
            threshold: float | None = None, explain: bool | None = None,
            progress: Callable[[float, str], None] | None = None, model_name: str = "stae") -> dict:
        cfg = settings.all()
        progress = progress or (lambda f, m: None)
        t_start = time.time()
        model, meta = registry.get(model_name)
        device = registry.device
        T, size = int(meta.get("seq_len", cfg["seq_len"])), int(meta.get("size", cfg["image_size"]))
        tau = float(threshold if threshold is not None else cfg["threshold"])
        explain = cfg["explain_events"] if explain is None else explain
        scorer = AnomalyScorer(tau)
        self.classifier = AnomalyClassifier(NORMAL_STATS)

        # 1-2. ingestion + resizing / [0,1] scaling, resampled to the frame-rate the model was trained at
        progress(0.02, "Decoding and resizing frames")
        loader = SequenceLoader(size=size, seq_len=T, stride=cfg["window_stride"])
        info = SequenceLoader.probe(video_path)
        frames, src_idx, fps = loader.load(video_path, target_fps=float(meta.get("fps", info["fps"])),
                                           max_frames=int(cfg["max_frames"]))
        N = len(frames)

        # 3-5. STAE reconstruction error -> regularity score
        progress(0.08, f"Running STAE on {N} frames")
        errors = score_frames(model, frames, T, int(cfg["window_stride"]), device,
                              progress=lambda p: progress(0.08 + 0.52 * p, "Running STAE"))
        s, a = scorer.score(errors)
        flags = scorer.flag(s)

        # 6. events
        events_raw = scorer.group_events(flags, a, fps, cfg["merge_gap_sec"], cfg["min_event_frames"])
        events_raw = sorted(events_raw, key=lambda e: -e["peak_score"])
        events_raw = sorted(events_raw[: max(int(cfg["max_events_explained"]), 1) * 3], key=lambda e: e["start_idx"])
        explainer = LimeExplainer(model, device, batch=int(cfg["lime_batch"]))

        events: list[dict] = []
        for n, ev in enumerate(events_raw):
            progress(0.60 + 0.38 * (n / max(len(events_raw), 1)), f"Explaining event {n + 1}/{len(events_raw)}")
            eid = f"{analysis_id}_e{n:02d}"
            edir = event_dir(eid)
            edir.mkdir(parents=True, exist_ok=True)
            peak = ev["peak_idx"]
            start, pos = window_for(N, peak, T)
            context = frames[start:start + T]
            np.save(edir / "context.npy", context)
            recon128, _ = reconstruct_at(model, context, pos, device)

            lime = None
            if explain and n < int(cfg["max_events_explained"]):
                lime = explainer.explain(context, pos, n_samples=cfg["lime_samples"],
                                         n_segments=cfg["lime_segments"], top_k=cfg["lime_top_k"],
                                         seed=int(cfg["seed"]))
            src = SequenceLoader.read_source_frame(video_path, src_idx[peak])
            display = hm.to_display(src if src is not None else hm._up(frames[peak], (360, 480)))
            box, names = render_event(edir, display, frames[peak], recon128, lime)

            t0, t1 = src_idx[ev["start_idx"]] / info["fps"], src_idx[ev["end_idx"]] / info["fps"]
            duration = max(t1 - t0, 1.0 / fps)
            lo, hi = max(0, ev["start_idx"] - 5), min(N, ev["end_idx"] + 6)
            cls = self.classifier.classify(frames[lo:hi], box, fps, duration)
            events.append({
                "id": eid, "analysis_id": analysis_id, "index": n,
                "timestamp": fmt_time(t0), "start_time": round(t0, 2), "end_time": round(t1, 2),
                "duration": round(duration, 1), "peak_time": round(src_idx[peak] / info["fps"], 2),
                "peak_frame": int(peak), "window_start": int(start), "window_pos": int(pos),
                "anomaly_score": round(ev["peak_score"], 4), "mean_score": round(ev["mean_score"], 4),
                "regularity": round(1 - ev["peak_score"], 4), "threshold": round(scorer.theta, 4),
                "severity": scorer.severity(ev["peak_score"]),
                "category": cls["category"], "category_estimated": True,
                "rationale": cls["rationale"], "features": cls["features"],
                "box": [round(box[0] / 128, 4), round(box[1] / 128, 4), round(box[2] / 128, 4), round(box[3] / 128, 4)],
                "lime": lime_payload(lime, box), "images": names,
            })

        # timeline for the UI (anomaly score a_t, time in seconds of the source video)
        times = [round(src_idx[i] / info["fps"], 3) for i in range(N)]
        step = max(1, N // 1500)
        result = {
            "id": analysis_id, "video_id": video_id, "video_name": video_name, "created": now_iso(),
            "model": model_name, "tau": tau, "theta": round(scorer.theta, 4),
            "fps": round(fps, 3), "frames": N, "duration": round(info["duration"], 2),
            "resolution": [info["width"], info["height"]], "n_events": len(events),
            "n_high": sum(1 for e in events if e["severity"] == "High"),
            "flagged_frames": int(flags.sum()), "elapsed": round(time.time() - t_start, 1),
            "timeline": {"t": times[::step], "anomaly": [round(float(v), 4) for v in a[::step]]},
            "events": events, "explained": bool(explain),
        }
        reports.record(result)
        progress(1.0, "Done")
        return result

    # ---- on-demand LIME for one event (Explainability page) -----------------------------------
    def explain_event(self, eid: str, n_samples: int, n_segments: int, top_k: int, seed: int = 42) -> dict:
        ev = reports.get_event(eid)
        if not ev:
            raise KeyError(eid)
        edir = event_dir(eid)
        context = np.load(edir / "context.npy")
        pos = int(ev["window_pos"])
        model, meta = registry.get(reports.load(ev["analysis_id"])["model"])
        explainer = LimeExplainer(model, registry.device, batch=int(settings.get("lime_batch")))
        lime = explainer.explain(context, pos, n_samples=n_samples, n_segments=n_segments, top_k=top_k, seed=seed)
        recon128, _ = reconstruct_at(model, context, pos, registry.device)
        import cv2
        display = cv2.cvtColor(cv2.imread(str(edir / "original.jpg")), cv2.COLOR_BGR2RGB)
        box, names = render_event(edir, display, context[pos], recon128, lime)
        patch = {"lime": lime_payload(lime, box), "images": names,
                 "box": [round(box[0] / 128, 4), round(box[1] / 128, 4), round(box[2] / 128, 4), round(box[3] / 128, 4)]}
        return reports.update_event(eid, patch)


pipeline = AnalysisPipeline()
