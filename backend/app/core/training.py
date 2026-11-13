"""Training strategy (Section 5.5): unsupervised, normal footage only, MSE loss, Adam,
early stopping on validation loss."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import MODEL_DIR
from ..data.datasets import find_sequences, load_sequence_frames
from ..models.stae import STAE, count_parameters, reconstruction_error
from ..utils import get_device, now_iso, seed_everything
from .classifier import compute_normal_stats
from .sequence_loader import SequenceLoader


class WindowDataset(Dataset):
    """Overlapping T-frame windows over a list of uint8 videos (kept as 128x128 uint8)."""

    def __init__(self, videos: list[np.ndarray], seq_len: int, stride: int):
        self.videos, self.T = videos, seq_len
        self.index = [(vi, s) for vi, v in enumerate(videos)
                      for s in range(0, len(v) - seq_len + 1, stride)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> torch.Tensor:
        vi, s = self.index[i]
        w = self.videos[vi][s:s + self.T]  # (T,H,W,3)
        return torch.from_numpy(w).float().div_(255.0).permute(0, 3, 1, 2)


def train_model(model_name: str = "stae", dataset: str = "avenue", epochs: int = 50,
                batch_size: int = 8, lr: float = 1e-4, stride: int = 1, size: int = 128,
                seq_len: int = 10, val_fraction: float = 0.1, patience: int = 5,
                max_videos: int | None = None, max_frames_per_video: int | None = None,
                seed: int = 42, progress: Callable[[float, str, dict], None] | None = None) -> dict:
    progress = progress or (lambda f, m, e: None)
    seed_everything(seed)
    device = get_device()
    seqs = find_sequences(dataset)["train"]
    if not seqs:
        raise FileNotFoundError(
            f"No training sequences found for '{dataset}'. Put the dataset inside dataset/{dataset}/ "
            f"(see dataset/README.md)."
        )
    if max_videos:
        seqs = seqs[:max_videos]

    videos, fps_list = [], []
    for i, sq in enumerate(seqs):
        progress(0.02 * i / len(seqs), f"Loading {sq.name} ({i + 1}/{len(seqs)})", {})
        fr, fps = load_sequence_frames(sq, size, seq_len, max_frames=max_frames_per_video)
        if len(fr) >= seq_len:
            videos.append(fr)
            fps_list.append(fps)
    if not videos:
        raise ValueError("Training videos are shorter than the sequence length.")

    # hold out whole videos for validation (early stopping + normal statistics)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(videos))
    n_val = max(1, int(round(len(videos) * val_fraction))) if len(videos) > 1 else 0
    val_ids = set(order[:n_val].tolist())
    train_vids = [v for i, v in enumerate(videos) if i not in val_ids] or videos
    val_vids = [v for i, v in enumerate(videos) if i in val_ids]
    train_ds = WindowDataset(train_vids, seq_len, stride)
    val_ds = WindowDataset(val_vids, seq_len, max(stride, seq_len // 2)) if val_vids else None
    if len(train_ds) == 0:
        raise ValueError("No training windows could be built.")

    model = STAE(use_convlstm=(model_name != "spatial")).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds and len(val_ds) else None

    history, best, best_state, bad, t0 = [], float("inf"), None, 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run, n = 0.0, 0
        for bi, x in enumerate(tl):
            x = x.to(device)
            loss = reconstruction_error(x, model(x)).mean()  # L_rec, Eq. (4)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * x.shape[0]
            n += x.shape[0]
            if bi % 20 == 0:
                frac = ((ep - 1) + bi / max(len(tl), 1)) / epochs
                progress(0.05 + 0.9 * frac, f"Epoch {ep}/{epochs} - batch {bi + 1}/{len(tl)}", {})
        train_loss = run / max(n, 1)
        val_loss = train_loss
        if vl is not None:
            model.eval()
            vs, vn = 0.0, 0
            with torch.no_grad():
                for x in vl:
                    x = x.to(device)
                    vs += reconstruction_error(x, model(x)).mean().item() * x.shape[0]
                    vn += x.shape[0]
            val_loss = vs / max(vn, 1)
        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})
        progress(0.05 + 0.9 * ep / epochs, f"Epoch {ep}/{epochs}  train {train_loss:.5f}  val {val_loss:.5f}",
                 {"history": history})
        if val_loss < best - 1e-7:
            best, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:  # early stopping on validation loss
                break

    meta = {
        "name": model_name, "use_convlstm": model_name != "spatial", "seq_len": seq_len, "size": size,
        "fps": float(np.median(fps_list)), "dataset": dataset, "epochs_run": len(history),
        "epochs_max": epochs, "best_val_loss": best, "batch_size": batch_size, "lr": lr, "stride": stride,
        "seed": seed, "train_videos": len(train_vids), "val_videos": len(val_vids),
        "train_windows": len(train_ds), "created": now_iso(), "device": str(device),
        "train_seconds": round(time.time() - t0, 1), "history": history,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state or model.state_dict(), "meta": meta}, MODEL_DIR / f"{model_name}.pt")

    if model_name == "stae":  # speed / direction statistics of normal footage for the category estimator
        progress(0.97, "Measuring normal-motion statistics", {})
        stats = compute_normal_stats(train_vids, seed=seed)
        (MODEL_DIR / "normal_stats.json").write_text(json.dumps(stats, indent=2))
    progress(1.0, "Training complete", {"history": history})
    return meta
