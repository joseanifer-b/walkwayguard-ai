"""AnomalyScorer - min-max normalisation, regularity score s_t, threshold tau, event grouping
(Sections 4.2, 4.4, 5.6).

Display convention (Section 4.2): dashboard shows the anomaly score a_t = 1 - s_t, so the
condition s_t < tau is the same as a_t > theta with theta = 1 - tau.
"""
from __future__ import annotations

import numpy as np


class AnomalyScorer:
    def __init__(self, tau: float = 0.35):
        self.tau = float(tau)

    @property
    def theta(self) -> float:
        return 1.0 - self.tau

    @staticmethod
    def regularity(errors: np.ndarray) -> np.ndarray:
        """Eq. (5): s_t = 1 - (e_t - min e) / (max e - min e)."""
        e = np.asarray(errors, dtype=np.float64)
        return 1.0 - (e - e.min()) / (e.max() - e.min() + 1e-8)

    def score(self, errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.regularity(errors)
        return s, 1.0 - s

    def flag(self, s: np.ndarray) -> np.ndarray:
        return np.asarray(s) < self.tau

    def group_events(self, flags: np.ndarray, anomaly: np.ndarray, fps: float,
                     merge_gap_sec: float = 1.0, min_len: int = 3) -> list[dict]:
        """Merge consecutive / closely occurring flagged frames into events (start, end, peak)."""
        idx = np.flatnonzero(flags)
        if idx.size == 0:
            return []
        gap = max(1, int(round(merge_gap_sec * fps)))
        groups, start, prev = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - prev > gap:
                groups.append((start, prev))
                start = i
            prev = i
        groups.append((start, prev))
        events = []
        for a, b in groups:
            if (b - a + 1) < min_len:
                continue
            seg = anomaly[a:b + 1]
            peak = int(a + np.argmax(seg))
            events.append({
                "start_idx": int(a), "end_idx": int(b), "peak_idx": peak,
                "peak_score": float(anomaly[peak]), "mean_score": float(seg.mean()),
                "flagged_frames": int(flags[a:b + 1].sum()),
            })
        return events

    def severity(self, peak_anomaly: float) -> str:
        """High / Medium / Low according to how far the peak exceeds theta."""
        margin = (peak_anomaly - self.theta) / max(1e-8, 1.0 - self.theta)
        if margin >= 0.5:
            return "High"
        if margin >= 0.25:
            return "Medium"
        return "Low"
