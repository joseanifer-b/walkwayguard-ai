"""Create a tiny SYNTHETIC walkway dataset in dataset/demo/ for smoke-testing the whole stack
(train -> evaluate -> analyse -> explain) without downloading anything.

Normal clips: grey path, people-like blobs walking left->right at a steady speed.
Test clips  : same, plus injected anomalies (fast runner, wrong-direction walker) with
              per-frame labels (<name>_label.npy) and pixel masks (<name>_pixel.npy).
This is NOT a benchmark - the numbers it produces mean nothing about CUHK Avenue / ShanghaiTech.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

H, W = 144, 192


def render(n, anomalies, seed):
    rng = np.random.RandomState(seed)
    people = [dict(x=rng.uniform(0, W), y=rng.uniform(70, 125), v=rng.uniform(1.4, 2.0), c=rng.randint(40, 255, 3).tolist())
              for _ in range(4)]
    frames, labels, masks = [], np.zeros(n, np.uint8), np.zeros((n, H, W), np.uint8)
    for t in range(n):
        img = np.full((H, W, 3), 120, np.uint8)
        img[:60] = (90, 110, 90)
        for p in people:
            p["x"] = (p["x"] + p["v"]) % W
            cv2.rectangle(img, (int(p["x"]), int(p["y"]) - 22), (int(p["x"]) + 8, int(p["y"])), p["c"], -1)
        for (a, b, kind) in anomalies:
            if a <= t < b:
                labels[t] = 1
                if kind == "run":
                    x = (t - a) * 7 % W
                else:
                    x = W - ((t - a) * 2) % W
                y = 100
                cv2.rectangle(img, (int(x), y - 22), (int(x) + 10, y), (255, 30, 30), -1)
                masks[t, y - 22:y, int(x):int(x) + 10] = 1
        img = np.clip(img + rng.randint(-3, 4, img.shape), 0, 255).astype(np.uint8)
        frames.append(img)
    return frames, labels, masks


def write(path, frames, fps=25):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "dataset" / "demo"))
    ap.add_argument("--train", type=int, default=4)
    ap.add_argument("--test", type=int, default=4)
    ap.add_argument("--frames", type=int, default=150)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "training_videos").mkdir(parents=True, exist_ok=True)
    (out / "testing_videos").mkdir(parents=True, exist_ok=True)
    (out / "testing_label_mask").mkdir(parents=True, exist_ok=True)
    for i in range(1, a.train + 1):
        f, _, _ = render(a.frames, [], seed=i)
        write(out / "training_videos" / f"{i:02d}.avi", f)
    for i in range(1, a.test + 1):
        n = a.frames
        an = [(40, 70, "run"), (100, 130, "wrong")] if i % 2 else [(60, 95, "run")]
        f, lab, msk = render(n, an, seed=100 + i)
        write(out / "testing_videos" / f"{i:02d}.avi", f)
        np.save(out / "testing_label_mask" / f"{i:02d}_label.npy", lab)
    print("Synthetic demo dataset written to", out)
