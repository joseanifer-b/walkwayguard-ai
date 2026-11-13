"""CUHK Avenue / ShanghaiTech Campus discovery + ground-truth loading.

Kaggle mirrors of these datasets are laid out differently, so the loader is deliberately
tolerant. Put the files anywhere inside:

    dataset/avenue/        dataset/shanghaitech/

It finds training / testing sequences (video files *or* folders of frames) by folder name
("train..." / "test..."), and matches ground truth by sequence name:

  * Avenue            :  testing_label_mask/<n>_label.mat   (variable `volLabel`, per-frame masks)
  * ShanghaiTech      :  test_frame_mask/<seq>.npy  (per-frame 0/1)  and  test_pixel_mask/<seq>.npy
  * any dataset       :  <seq>.npy / <seq>_label.npy (1-D per-frame labels, or N x H x W masks)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import DATASET_DIR
from ..core.sequence_loader import IMG_EXT, SequenceLoader

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv"}
DATASETS = {"avenue": "CUHK Avenue", "shanghaitech": "ShanghaiTech Campus", "demo": "Synthetic demo (smoke test)"}


@dataclass
class Sequence:
    name: str
    path: Path
    split: str  # "train" | "test"


def dataset_root(name: str) -> Path:
    return DATASET_DIR / name


def _has_images(folder: Path, minimum: int = 10) -> bool:
    n = 0
    try:
        for p in folder.iterdir():
            if p.suffix.lower() in IMG_EXT:
                n += 1
                if n >= minimum:
                    return True
    except OSError:
        pass
    return False


def _split_of(path: Path, root: Path) -> str | None:
    parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    joined = "/".join(parts)
    if any(w in joined for w in ("label", "mask", "gt", "ground", "annot", "pixel")):
        return None
    if "train" in joined:
        return "train"
    if "test" in joined:
        return "test"
    return None


def find_sequences(name: str) -> dict[str, list[Sequence]]:
    root = dataset_root(name)
    out: dict[str, list[Sequence]] = {"train": [], "test": []}
    if not root.exists():
        return out
    seen: set[Path] = set()
    for p in sorted(root.rglob("*")):
        if p.name.startswith("."):
            continue
        if p.is_file() and p.suffix.lower() in VIDEO_EXT:
            split = _split_of(p, root)
            if split:
                out[split].append(Sequence(p.stem, p, split))
        elif p.is_dir() and p not in seen and _has_images(p):
            split = _split_of(p, root)
            if split:
                seen.add(p)
                out[split].append(Sequence(p.name, p, split))
    for k in out:
        out[k].sort(key=lambda s: _natural(s.name))
    return out


def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def dataset_status() -> list[dict]:
    res = []
    for key, title in DATASETS.items():
        seqs = find_sequences(key)
        n_label = sum(1 for s in seqs["test"] if load_frame_labels(key, s) is not None)
        res.append({
            "key": key,
            "name": title,
            "path": str(dataset_root(key)),
            "train_sequences": len(seqs["train"]),
            "test_sequences": len(seqs["test"]),
            "labelled_test_sequences": n_label,
            "ready": len(seqs["train"]) > 0,
            "evaluable": len(seqs["test"]) > 0 and n_label > 0,
        })
    return res


# ---------------------------------------------------------------------- ground truth
def _key(s: str) -> str:
    s = s.lower()
    s = re.sub(r"(_?label(s)?|_?gt|_?mask|_?frame)$", "", s)
    s = s.strip("_- ")
    return str(int(s)) if s.isdigit() else s


def _find_label_files(root: Path, seq_name: str, exts: tuple[str, ...]) -> list[Path]:
    target = _key(seq_name)
    hits = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and _key(p.stem) == target:
            hits.append(p)
    return hits


def _frames_from_array(arr: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return (arr > 0).astype(np.uint8)
    if arr.ndim == 2 and 1 in arr.shape:
        return (arr.ravel() > 0).astype(np.uint8)
    if arr.ndim == 3:  # N x H x W masks
        return (arr.reshape(arr.shape[0], -1).max(axis=1) > 0).astype(np.uint8)
    return None


def load_frame_labels(dataset: str, seq: Sequence) -> np.ndarray | None:
    """Per-frame 0/1 ground truth for a test sequence, or None if not available."""
    root = dataset_root(dataset)
    for p in _find_label_files(root, seq.name, (".npy", ".mat")):
        try:
            if p.suffix.lower() == ".npy":
                if "pixel" in str(p).lower():
                    continue
                lab = _frames_from_array(np.load(p, allow_pickle=True))
                if lab is not None:
                    return lab
            else:
                lab = _labels_from_mat(p)
                if lab is not None:
                    return lab
        except Exception:
            continue
    # fall back to pixel masks (ShanghaiTech test_pixel_mask) -> frame labels
    for p in _find_label_files(root, seq.name, (".npy",)):
        if "pixel" in str(p).lower():
            try:
                return _frames_from_array(np.load(p, allow_pickle=True))
            except Exception:
                pass
    return None


def _labels_from_mat(path: Path) -> np.ndarray | None:
    from scipy.io import loadmat

    m = loadmat(str(path))
    for key in ("volLabel", "label", "labels", "gt", "frame_label"):
        if key in m:
            v = m[key]
            if v.dtype == object:  # Avenue: 1 x N cell array of H x W masks
                return np.array([int(np.asarray(f).any()) for f in v.ravel()], dtype=np.uint8)
            return _frames_from_array(v)
    return None


def load_pixel_masks(dataset: str, seq: Sequence, n_frames: int, size: int) -> np.ndarray | None:
    """Per-frame spatial ground truth resized to size x size -> bool (N,size,size), or None."""
    root = dataset_root(dataset)
    masks = None
    try:
        for p in _find_label_files(root, seq.name, (".npy",)):
            arr = np.load(p, allow_pickle=True)
            if arr.ndim == 3:
                masks = arr
                break
        if masks is None:
            for p in _find_label_files(root, seq.name, (".mat",)):
                from scipy.io import loadmat

                m = loadmat(str(p))
                if "volLabel" in m and m["volLabel"].dtype == object:
                    masks = np.stack([np.asarray(f) for f in m["volLabel"].ravel()])
                    break
    except Exception:
        return None
    if masks is None:
        return None
    out = np.zeros((min(len(masks), n_frames), size, size), dtype=bool)
    for i in range(len(out)):
        out[i] = cv2.resize((masks[i] > 0).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
    return out


# ---------------------------------------------------------------------- frame loading
def load_sequence_frames(seq: Sequence, size: int, seq_len: int,
                         max_frames: int | None = None) -> tuple[np.ndarray, float]:
    loader = SequenceLoader(size=size, seq_len=seq_len)
    frames, _, fps = loader.load(seq.path, max_frames=max_frames)
    return frames, fps
