"""LimeExplainer - model-agnostic, post-hoc explanation of the STAE anomaly score (Section 5.4).

The trained STAE is treated as a black box f(x) = e_t, the reconstruction error of the flagged
frame x_t. The frame sits at position `pos` of its T-frame window; the other T-1 frames of the
window are held fixed while the superpixels of x_t are perturbed ("off" regions -> neutral grey).
The explainer only needs a scoring function, so the detector can be swapped without changes.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import slic

from ..models.stae import reconstruction_error

NEUTRAL = 127


@dataclass
class LimeResult:
    segments: np.ndarray                      # (H, W) int superpixel labels
    weights: dict[int, float]                 # superpixel -> surrogate coefficient
    top: list[dict] = field(default_factory=list)   # ranked top positive regions
    mask: np.ndarray | None = None            # (H, W) bool, union of top-k positive superpixels
    r2: float = 0.0                           # local R^2 of the surrogate
    latency: float = 0.0                      # seconds
    base_error: float = 0.0
    n_samples: int = 0
    n_segments: int = 0
    top_k: int = 8


def slic_segments(img: np.ndarray, n_segments: int = 100, compactness: float = 10) -> np.ndarray:
    return slic(img, n_segments=n_segments, compactness=compactness, start_label=0)


class LimeExplainer:
    def __init__(self, model: torch.nn.Module, device: torch.device, batch: int = 64):
        self.model = model
        self.device = device
        self.batch = batch

    # ------------------------------------------------------------------ scoring function f
    def make_score_fn(self, context: np.ndarray, pos: int):
        """context: (T,H,W,3) uint8 window; returns f(imgs[n,H,W,3]) -> errors [n,1]."""
        T = context.shape[0]
        ctx = torch.from_numpy(np.ascontiguousarray(context)).float().div_(255.0).permute(0, 3, 1, 2)  # (T,3,H,W)

        @torch.no_grad()
        def f(imgs: np.ndarray) -> np.ndarray:
            imgs = np.asarray(imgs)
            out = []
            for i in range(0, len(imgs), self.batch):
                chunk = torch.from_numpy(imgs[i:i + self.batch]).float().div_(255.0).permute(0, 3, 1, 2)
                n = chunk.shape[0]
                x = ctx.unsqueeze(0).repeat(n, 1, 1, 1, 1)
                x[:, pos] = chunk
                x = x.to(self.device)
                err = reconstruction_error(x, self.model(x))[:, pos]
                out.append(err.cpu().numpy())
            return np.concatenate(out).reshape(-1, 1).astype(np.float64)

        return f

    # ------------------------------------------------------------------ explanation
    def explain(self, context: np.ndarray, pos: int, n_samples: int = 1000, n_segments: int = 100,
                top_k: int = 8, seed: int = 42) -> LimeResult:
        frame = context[pos]
        f = self.make_score_fn(context, pos)
        t0 = time.time()
        explainer = lime_image.LimeImageExplainer(random_state=seed)
        kwargs = dict(
            classifier_fn=f,
            labels=(0,),
            top_labels=None,
            hide_color=NEUTRAL,
            num_samples=int(n_samples),
            batch_size=self.batch,
            segmentation_fn=lambda img: slic_segments(img, n_segments=n_segments),
            random_seed=seed,
        )
        if "progress_bar" in inspect.signature(explainer.explain_instance).parameters:  # newer lime only
            kwargs["progress_bar"] = False
        exp = explainer.explain_instance(frame, **kwargs)
        latency = time.time() - t0
        weights = {int(s): float(w) for s, w in exp.local_exp[0]}
        score = exp.score
        r2 = float(score.get(0, 0.0) if isinstance(score, dict) else score)
        segments = exp.segments

        ranked = sorted(((s, w) for s, w in weights.items() if w > 0), key=lambda t: -t[1])[:top_k]
        mask = np.zeros(segments.shape, dtype=bool)
        for s, _ in ranked:
            mask |= segments == s
        top = self._region_table(segments, ranked)
        return LimeResult(
            segments=segments, weights=weights, top=top, mask=mask, r2=r2, latency=latency,
            base_error=float(f(frame[None])[0, 0]), n_samples=int(n_samples),
            n_segments=int(segments.max()) + 1, top_k=top_k,
        )

    @staticmethod
    def _region_table(segments: np.ndarray, ranked: list[tuple[int, float]], rows: int = 4) -> list[dict]:
        """Top contributing regions with importance normalised over the listed regions (Fig. 6.8)."""
        H, W = segments.shape
        head = ranked[:rows]
        total = sum(w for _, w in head) or 1.0
        table = []
        for rank, (s, w) in enumerate(head, 1):
            ys, xs = np.nonzero(segments == s)
            cx, cy = float(xs.mean()) / W, float(ys.mean()) / H
            horiz = "left" if cx < 0.33 else "right" if cx > 0.66 else "centre"
            vert = "upper" if cy < 0.33 else "lower" if cy > 0.66 else "middle"
            table.append({
                "rank": rank, "region": int(s), "weight": float(w), "importance": round(w / total, 3),
                "location": f"{vert}-{horiz}", "cx": round(cx, 3), "cy": round(cy, 3),
                "area_pct": round(100.0 * len(xs) / (H * W), 2),
            })
        return table

    # ------------------------------------------------------------------ evaluation helpers
    def deletion_auc(self, context: np.ndarray, pos: int, result: LimeResult, steps: int = 20) -> float:
        """Remove superpixels in order of importance and track the (normalised) anomaly score.
        Lower area = the score drops quickly = more faithful explanation (Section 6.6b)."""
        f = self.make_score_fn(context, pos)
        order = [s for s, w in sorted(result.weights.items(), key=lambda t: -t[1]) if w > 0][:steps]
        if not order:
            return 1.0
        frame = context[pos].copy()
        base = max(float(f(frame[None])[0, 0]), 1e-12)
        batch = [frame.copy()]
        cur = frame.copy()
        for s in order:
            cur[result.segments == s] = NEUTRAL
            batch.append(cur.copy())
        vals = np.clip(f(np.stack(batch))[:, 0] / base, 0.0, 1.5)
        xs = np.linspace(0, 1, len(vals))
        integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        return float(integrate(vals, xs))

    @staticmethod
    def jaccard(a: np.ndarray, b: np.ndarray) -> float:
        inter, union = np.logical_and(a, b).sum(), np.logical_or(a, b).sum()
        return float(inter / union) if union else 1.0
