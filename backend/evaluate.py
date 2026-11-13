"""Evaluate a trained model: frame-level AUC / precision / recall / F1, tau tuning, LIME metrics.

    python evaluate.py --dataset avenue
    python evaluate.py --dataset avenue --model spatial --no-explain    # ablation row
"""
import argparse
import json

from app.core.evaluation import run_evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="avenue", choices=["avenue", "shanghaitech", "demo"])
    ap.add_argument("--model", default="stae", choices=["stae", "spatial"])
    ap.add_argument("--no-explain", action="store_true", help="skip the LIME explanation metrics")
    ap.add_argument("--apply-threshold", action="store_true", help="save the tuned tau as the system threshold")
    ap.add_argument("--val-fraction", type=float, default=0.3)
    a = ap.parse_args()
    res = run_evaluation(a.model, a.dataset, not a.no_explain, a.apply_threshold, a.val_fraction,
                         progress=lambda f, m: print(f"[{f * 100:5.1f}%] {m}", flush=True))
    keys = ["auc", "auc_macro", "accuracy", "precision", "recall", "f1", "fpr", "eer", "ap", "tau", "n_test_videos", "n_test_frames"]
    print(json.dumps({k: res.get(k) for k in keys}, indent=2))
    print("Full results: data/metrics/%s_%s.json" % (a.model, a.dataset))


if __name__ == "__main__":
    main()
