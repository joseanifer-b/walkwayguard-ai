"""Evaluation (Chapter 6): frame-level AUC / accuracy / precision / recall / F1 / FPR / EER / AP,
validation-tuned threshold, confusion matrix, and explanation metrics (pointing accuracy, mean IoU,
deletion AUC, stability, surrogate R^2, latency, N/K sensitivity).

Protocol
  * Test videos are split by video into a validation part (tau selection by max F1) and a held-out
    part on which every reported number is computed - test labels used for the final numbers are
    never used for tau (Section 5.5).
  * AUC is micro-averaged over all held-out frames after per-video min-max normalisation of the
    reconstruction error (Eq. 5); the mean per-video AUC is reported too.
  * Nothing here is hard-coded: every figure shown in the dashboard is measured by this module.
"""
from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from ..config import METRICS_DIR, MODEL_DIR, settings
from ..data.datasets import find_sequences, load_frame_labels, load_pixel_masks, load_sequence_frames
from .lime_explainer import LimeExplainer
from .pipeline import registry, score_frames
from .scorer import AnomalyScorer


def _downsample(x: np.ndarray, n: int = 600) -> list[float]:
    if len(x) <= n:
        return [round(float(v), 4) for v in x]
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return [round(float(x[i]), 4) for i in idx]


def _prf(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1), "precision": prec, "recall": rec,
            "f1": f1, "fpr": fp / max(fp + tn, 1)}


def best_f1_threshold(s: np.ndarray, y: np.ndarray) -> float:
    """tau maximising F1 on the validation frames (flag when s < tau)."""
    cands = np.unique(np.quantile(s, np.linspace(0.005, 0.995, 400)))
    best_t, best_f = 0.35, -1.0
    for t in cands:
        f = _prf(y, (s < t).astype(int))["f1"]
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def eer(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


# ---------------------------------------------------------------------------------------------
def evaluate_detection(model_name: str, dataset: str, val_fraction: float = 0.3, seed: int = 42,
                       max_frames: int | None = None, progress: Callable[[float, str], None] | None = None) -> dict:
    progress = progress or (lambda f, m: None)
    model, meta = registry.get(model_name)
    T, size, device = int(meta["seq_len"]), int(meta["size"]), registry.device
    seqs = [s for s in find_sequences(dataset)["test"]]
    if not seqs:
        raise FileNotFoundError(f"No test sequences found in dataset/{dataset}/.")

    per = []  # per-video: name, regularity, labels, frames
    for i, sq in enumerate(seqs):
        progress(0.7 * i / len(seqs), f"Scoring {sq.name} ({i + 1}/{len(seqs)})")
        y = load_frame_labels(dataset, sq)
        if y is None:
            continue
        frames, fps = load_sequence_frames(sq, size, T, max_frames=max_frames)
        n = min(len(frames), len(y))
        frames, y = frames[:n], y[:n]
        if n < T:
            continue
        e = score_frames(model, frames, T, 1, device)
        per.append({"seq": sq, "name": sq.name, "s": AnomalyScorer.regularity(e), "y": y, "frames": frames, "fps": fps})
    if not per:
        raise FileNotFoundError(
            f"No ground-truth labels found for the '{dataset}' test sequences. See dataset/README.md "
            f"for the expected label files.")

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(per))
    n_val = int(round(len(per) * val_fraction)) if len(per) >= 3 else 0
    val = [per[i] for i in order[:n_val]]
    test = [per[i] for i in order[n_val:]]
    sv = np.concatenate([p["s"] for p in val]) if val else None
    yv = np.concatenate([p["y"] for p in val]) if val else None
    if sv is not None and yv.sum() > 0:
        tau, tau_src = best_f1_threshold(sv, yv), "max F1 on held-out validation videos"
    else:  # not enough labelled validation data: fall back to configured tau
        tau, tau_src = float(settings.get("threshold")), "configured default (no labelled validation split)"

    s_all = np.concatenate([p["s"] for p in test])
    y_all = np.concatenate([p["y"] for p in test])
    pred = (s_all < tau).astype(int)
    m = _prf(y_all, pred)
    has_both = 0 < y_all.sum() < len(y_all)
    a_all = 1 - s_all
    res = {
        "model": model_name, "dataset": dataset, "tau": tau, "tau_source": tau_src, "theta": 1 - tau,
        "n_test_videos": len(test), "n_val_videos": len(val), "n_test_frames": int(len(y_all)),
        "anomalous_frame_pct": round(100 * float(y_all.mean()), 2),
        "trivial_all_normal_accuracy": round(100 * (1 - float(y_all.mean())), 2),
        "protocol": "AUC micro-averaged over held-out test frames, per-video min-max normalisation",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **{k: (round(v * 100, 2) if k in ("accuracy", "precision", "recall", "f1", "fpr") else v) for k, v in m.items()},
    }
    if has_both:
        fpr, tpr, _ = roc_curve(y_all, a_all)
        pr, rc, _ = precision_recall_curve(y_all, a_all)
        aucs = [roc_auc_score(p["y"], 1 - p["s"]) for p in test if 0 < p["y"].sum() < len(p["y"])]
        step = max(1, len(fpr) // 300)
        res.update({
            "auc": round(100 * float(roc_auc_score(y_all, a_all)), 2),
            "auc_macro": round(100 * float(np.mean(aucs)), 2) if aucs else None,
            "eer": round(100 * eer(fpr, tpr), 2),
            "ap": round(100 * float(average_precision_score(y_all, a_all)), 2),
            "roc": {"fpr": [round(float(v), 4) for v in fpr[::step]], "tpr": [round(float(v), 4) for v in tpr[::step]]},
            "pr": {"precision": [round(float(v), 4) for v in pr[::max(1, len(pr) // 300)]],
                   "recall": [round(float(v), 4) for v in rc[::max(1, len(rc) // 300)]]},
        })
    else:
        res.update({"auc": None, "warning": "Held-out frames contain only one class; AUC undefined."})
    # timelines for Figure 6.2 (regularity score with ground-truth intervals)
    res["timelines"] = [{"name": p["name"], "regularity": _downsample(p["s"]),
                         "labels": _downsample(p["y"].astype(float))} for p in test[:4]]
    progress(0.72, "Detection metrics computed")
    res["_per"] = per  # stripped before saving; used by the explanation evaluation
    res["_test_names"] = [p["name"] for p in test]
    return res


# ---------------------------------------------------------------------------------------------
def evaluate_explanations(model_name: str, dataset: str, det: dict, n_frames: int = 12,
                          progress: Callable[[float, str], None] | None = None) -> dict:
    """Pointing accuracy, mean IoU, deletion AUC, stability, local R^2, latency (Section 6.6)."""
    progress = progress or (lambda f, m: None)
    model, meta = registry.get(model_name)
    T, size, device = int(meta["seq_len"]), int(meta["size"]), registry.device
    cfg = settings.all()
    explainer = LimeExplainer(model, device, batch=int(cfg["lime_batch"]))
    tau = det["tau"]
    cands = []
    for p in det["_per"]:
        if p["name"] not in det["_test_names"]:
            continue
        idx = np.flatnonzero((p["s"] < tau) & (p["y"] == 1))
        idx = idx[idx >= T - 1]
        for i in idx[:: max(1, len(idx) // 4)][:4]:
            cands.append((p, int(i)))
    if not cands:
        return {"note": "No true-positive frames available for explanation evaluation."}
    step = max(1, len(cands) // n_frames)
    cands = cands[::step][:n_frames]

    def run_cfg(n, k, with_stability=False):
        lat, point, ious, dels, r2s, stab = [], [], [], [], [], []
        for ci, (p, t) in enumerate(cands):
            ctx = p["frames"][t - T + 1:t + 1]
            pos = T - 1
            r = explainer.explain(ctx, pos, n_samples=n, n_segments=k, top_k=cfg["lime_top_k"], seed=cfg["seed"])
            lat.append(r.latency)
            r2s.append(r.r2)
            if n == cfg["lime_samples"] and k == cfg["lime_segments"]:
                dels.append(explainer.deletion_auc(ctx, pos, r))
            gt = load_pixel_masks(dataset, p["seq"], t + 1, size)
            if gt is not None and t < len(gt) and gt[t].any() and r.top:
                best = max(r.weights.items(), key=lambda kv: kv[1])[0]
                point.append(float(gt[t][r.segments == best].mean() > 0.5))
                inter = np.logical_and(r.mask, gt[t]).sum()
                union = np.logical_or(r.mask, gt[t]).sum()
                ious.append(float(inter / union) if union else 0.0)
            if with_stability:
                masks = [r.mask]
                for sd in (1, 2):
                    masks.append(explainer.explain(ctx, pos, n_samples=n, n_segments=k, top_k=cfg["lime_top_k"], seed=sd).mask)
                pairs = [explainer.jaccard(masks[i], masks[j]) for i in range(3) for j in range(i + 1, 3)]
                stab.append(float(np.mean(pairs)))
        mean = lambda v: round(float(np.mean(v)), 3) if v else None  # noqa: E731
        return {"latency": mean(lat), "pointing_acc": round(100 * float(np.mean(point)), 1) if point else None,
                "mean_iou": mean(ious), "deletion_auc": mean(dels), "stability": mean(stab), "r2": mean(r2s),
                "frames": len(cands)}

    progress(0.75, "LIME metrics at configured N, K")
    main = run_cfg(cfg["lime_samples"], cfg["lime_segments"], with_stability=True)
    main["pointing_available"] = main["pointing_acc"] is not None
    sens = []
    for i, (n, k) in enumerate([(250, 50), (500, 100), (1000, 100)]):
        progress(0.85 + 0.12 * i / 3, f"Sensitivity N={n}, K={k}")
        r = run_cfg(n, k, with_stability=True)
        sens.append({"n": n, "k": k, "latency": r["latency"], "pointing_acc": r["pointing_acc"], "stability": r["stability"]})
    return {"metrics": main, "sensitivity": sens}


# ---------------------------------------------------------------------------------------------
def run_evaluation(model_name: str, dataset: str, explain: bool = True, apply_threshold: bool = False,
                   val_fraction: float = 0.3, max_frames: int | None = None,
                   progress: Callable[[float, str], None] | None = None) -> dict:
    progress = progress or (lambda f, m: None)
    det = evaluate_detection(model_name, dataset, val_fraction=val_fraction, max_frames=max_frames, progress=progress)
    exp = evaluate_explanations(model_name, dataset, det, progress=progress) if explain else None
    det.pop("_per", None)
    det.pop("_test_names", None)
    det["explainability"] = exp
    (METRICS_DIR / f"{model_name}_{dataset}.json").write_text(json.dumps(det))
    if apply_threshold and model_name == "stae":
        settings.update({"threshold": round(det["tau"], 4)})
        (MODEL_DIR / "calibration.json").write_text(json.dumps(
            {"tau": det["tau"], "dataset": dataset, "source": det["tau_source"], "created": det["created"]}, indent=2))
    progress(1.0, "Evaluation complete")
    return det


def load_all_results() -> list[dict]:
    out = []
    for p in sorted(METRICS_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out
