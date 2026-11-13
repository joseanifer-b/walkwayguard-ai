"""SequenceLoader - splits video into overlapping sliding-window sub-sequences, resizes and
normalises pixel range (Section 4.4). Frames are decoded with OpenCV; complete videos are
kept only as small 128 x 128 uint8 arrays."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_FRAME_DIR_FPS = 25.0  # CUHK Avenue is 25 fps


class SequenceLoader:
    def __init__(self, size: int = 128, seq_len: int = 10, stride: int = 1):
        self.size = size
        self.seq_len = seq_len
        self.stride = stride

    # ------------------------------------------------------------------ probing
    @staticmethod
    def probe(path: str | Path) -> dict:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FRAME_DIR_FPS
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        return {
            "fps": float(fps),
            "frames": frames,
            "width": width,
            "height": height,
            "duration": float(frames / fps) if fps else 0.0,
        }

    # ------------------------------------------------------------------ loading
    def load(self, path: str | Path, target_fps: float | None = None,
             max_frames: int | None = None) -> tuple[np.ndarray, list[int], float]:
        """Return (frames uint8 [N,S,S,3] RGB, source frame index per kept frame, fps of kept frames)."""
        path = Path(path)
        if path.is_dir():
            return self._load_dir(path, max_frames)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FRAME_DIR_FPS
        out_fps = float(target_fps) if target_fps else float(src_fps)
        step = max(src_fps / out_fps, 1.0) if target_fps else 1.0
        frames: list[np.ndarray] = []
        src_idx: list[int] = []
        i, next_keep = 0, 0.0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i + 1e-6 >= next_keep:
                frames.append(self._prep(bgr))
                src_idx.append(i)
                next_keep += step
                if max_frames and len(frames) >= max_frames:
                    break
            i += 1
        cap.release()
        if not frames:
            raise ValueError(f"No frames decoded from {path}")
        return np.stack(frames), src_idx, (src_fps / step if target_fps else float(src_fps))

    def _load_dir(self, folder: Path, max_frames: int | None):
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)
        if max_frames:
            files = files[:max_frames]
        frames = []
        for f in files:
            bgr = cv2.imread(str(f))
            if bgr is not None:
                frames.append(self._prep(bgr))
        if not frames:
            raise ValueError(f"No readable images in {folder}")
        return np.stack(frames), list(range(len(frames))), DEFAULT_FRAME_DIR_FPS

    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)

    @staticmethod
    def read_source_frame(path: str | Path, index: int) -> np.ndarray | None:
        """Read one full-resolution RGB frame (used for display only)."""
        path = Path(path)
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXT)
            if 0 <= index < len(files):
                bgr = cv2.imread(str(files[index]))
                return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return None
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = cap.read()
        cap.release()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    # ------------------------------------------------------------------ windows
    def window_starts(self, n_frames: int, stride: int | None = None) -> list[int]:
        """Start index of each overlapping window; the last window always reaches the final frame."""
        T = self.seq_len
        stride = stride or self.stride
        if n_frames < T:
            return []
        starts = list(range(0, n_frames - T + 1, stride))
        if starts[-1] != n_frames - T:
            starts.append(n_frames - T)
        return starts

    @staticmethod
    def normalise(batch_uint8: np.ndarray):
        """uint8 (B,T,H,W,C) -> float tensor (B,T,C,H,W) scaled to [0, 1]."""
        import torch

        x = torch.from_numpy(np.ascontiguousarray(batch_uint8)).float().div_(255.0)
        return x.permute(0, 1, 4, 2, 3).contiguous()
