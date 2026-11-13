"""Train the STAE (or the spatial-only ablation) on NORMAL footage only.

    python train.py --dataset avenue                       # report settings: 50 epochs, bs 8, lr 1e-4
    python train.py --dataset avenue --epochs 20 --stride 2   # faster
    python train.py --dataset avenue --model spatial         # ablation (Table 6.7)
"""
import argparse

from app.core.training import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="avenue", choices=["avenue", "shanghaitech", "demo"])
    ap.add_argument("--model", default="stae", choices=["stae", "spatial"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--stride", type=int, default=1, help="window stride used to build training windows")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--max-frames-per-video", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    def prog(f, m, _):
        print(f"[{f * 100:5.1f}%] {m}", flush=True)

    meta = train_model(a.model, a.dataset, a.epochs, a.batch_size, a.lr, a.stride,
                       max_videos=a.max_videos, max_frames_per_video=a.max_frames_per_video, seed=a.seed, progress=prog)
    print("Saved data/models/%s.pt  (best val loss %.6f, %d epochs, %.0fs)" %
          (a.model, meta["best_val_loss"], meta["epochs_run"], meta["train_seconds"]))


if __name__ == "__main__":
    main()
