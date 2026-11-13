# WalkwayGuard AI - Explainable Anomaly Detection for Pedestrian Walkway Surveillance (STAE + LIME)

Implementation of the Project Phase-I report / IEEE paper "Explainable Anomaly Detection for Pedestrian
Walkway Surveillance using STAE and LIME".

* Detector: Spatio-Temporal Autoencoder - Conv-BN-ReLU encoder (32/64/128) -> ConvLSTM (128) -> transposed-conv decoder, trained on normal footage only (MSE, Adam, early stopping).
* Scoring: per-frame reconstruction error -> min-max regularity score s_t -> flag when s_t < tau (dashboard shows a_t = 1 - s_t against theta = 1 - tau) -> consecutive frames merged into events.
* Explanation: LIME (SLIC K=100, N=1000, top 8 superpixels) treats the STAE error as a black box; run on each event's peak frame.
* App: FastAPI backend + React/TypeScript dashboard (Dashboard, Analyze Video, Anomaly Events, Explainability, Model, Evaluation, Settings).

## Quick start

    # 1. Python 3.10-3.12
    cd backend
    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt                         # install PyTorch for your GPU first if needed

    # 2. Dataset: put CUHK Avenue (and optionally ShanghaiTech) in ../dataset/   (see dataset/README.md)

    # 3. Train (report settings: 50 epochs, batch 8, lr 1e-4, window stride 1)
    python train.py --dataset avenue
    #    quicker first try:  python train.py --dataset avenue --epochs 10 --stride 4

    # 4. Evaluate: AUC / precision / recall / F1, tunes tau on validation videos, LIME metrics
    python evaluate.py --dataset avenue --apply-threshold

    # 5. Run UI + API on http://localhost:8000
    cd .. && ./run.sh            # Windows: run.bat

Training and evaluation can also be started from the dashboard (Model and Evaluation pages).
Frontend dev mode: `cd frontend && npm install && npm run dev` (http://localhost:5173, proxies /api to :8000).
CLI analysis without the UI: `python analyze.py path/to/video.mp4`.
Smoke test without any dataset: `python scripts/make_demo_dataset.py` then `python train.py --dataset demo --epochs 3`
(synthetic data - the numbers mean nothing about Avenue/ShanghaiTech; delete dataset/demo and data/ afterwards).

## Layout

    backend/app/models/stae.py        STAE (report appendix code) + spatial-only ablation
    backend/app/core/                 sequence_loader, scorer, lime_explainer, heatmap, classifier,
                                      pipeline (Algorithm 1), training, evaluation, report, jobs
    backend/app/data/datasets.py      Avenue / ShanghaiTech discovery + ground truth
    backend/app/main.py               FastAPI endpoints
    backend/train.py evaluate.py analyze.py     command-line tools
    frontend/                         React + TypeScript dashboard (pre-built in frontend/dist)
    dataset/                          put the datasets here (blank placeholder files included)

## Read before the viva

* No results are hard-coded. The Evaluation page shows only what evaluate.py measures on your trained model. Your paper (Table IV: AUC 90.6) and report (Table 6.3: AUC 78.0) disagree - report only numbers you actually measure.
* Analysis needs a trained checkpoint; until then the UI says so instead of showing fake output.
* The behaviour category is "estimated": the STAE has no class labels, so core/classifier.py uses transparent optical-flow rules (thresholds at the top of that file). Per-category recall (report Table 6.4) cannot be computed - the datasets have no per-event category labels.
* Per-video min-max normalisation (Eq. 5) means every video contains a frame with s_t = 0; very short or entirely normal videos can still raise an alert near the threshold. State this as a limitation.
* Videos are resampled to the frame rate the model was trained at, so T=10 spans the same time in training and analysis.
* Validation protocol: test videos are split by video; tau is tuned (max F1) on the validation part and metrics are reported on the rest. Avenue has no separate labelled validation set, so this approximates "held-out validation".
* LIME cost: N=1000 scores 1000 windows per explained frame - fast on GPU/Apple MPS, slow on a weak CPU. Lower N in Settings if needed.
* LIME explains the anomaly score, not the network; highlights are decision support.
* "Published methods" rows on the Evaluation page are the values cited in your report, not re-run.
