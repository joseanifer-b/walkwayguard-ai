"""WalkwayGuard AI - FastAPI backend (application layer, Section 4.1).

Endpoints: upload, analysis, events, explanation, model (train / info), evaluation, settings.
If the React dashboard has been built (frontend/dist) it is served from the same port.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import ALLOWED_VIDEO_EXT, FRONTEND_DIST, MODEL_DIR, UPLOAD_DIR, settings
from .core import evaluation as ev_mod
from .core.classifier import CATEGORIES
from .core.jobs import jobs
from .core.pipeline import ModelNotTrained, pipeline, registry
from .core.report import event_dir, reports
from .core.sequence_loader import SequenceLoader
from .core.training import train_model
from .data.datasets import DATASETS, dataset_status
from .models.stae import ARCHITECTURE_TABLE
from .utils import new_id

app = FastAPI(title="WalkwayGuard AI", version="1.0.0",
              description="Explainable anomaly detection for pedestrian walkway surveillance (STAE + LIME)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

IMAGE_NAMES = {"original", "reconstruction", "error_map", "highlight", "zoom", "lime_heatmap", "superpixels"}


@app.exception_handler(ModelNotTrained)
async def _not_trained(_, exc: ModelNotTrained):
    return JSONResponse(status_code=409, content={"detail": str(exc), "code": "model_not_trained"})


# ------------------------------------------------------------------------------ status / settings
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/status")
def status():
    analyses = reports.list_analyses()
    events = reports.list_events()
    meta = registry.meta("stae") if registry.available("stae") else None
    sev = {"High": 0, "Medium": 0, "Low": 0}
    cats: dict[str, int] = {}
    for e in events:
        sev[e["severity"]] = sev.get(e["severity"], 0) + 1
        cats[e["category"]] = cats.get(e["category"], 0) + 1
    return {
        "device": str(registry.device),
        "model_ready": meta is not None,
        "model": {k: meta[k] for k in ("seq_len", "size", "fps", "dataset", "epochs_run", "best_val_loss", "created", "params") if k in meta} if meta else None,
        "spatial_ready": registry.available("spatial"),
        "threshold": settings.get("threshold"),
        "analyses": len(analyses),
        "events": len(events),
        "severity": sev,
        "categories": cats,
        "videos": len(list(UPLOAD_DIR.glob("*/*"))),
        "datasets": dataset_status(),
        "recent_analyses": analyses[:5],
    }


@app.get("/api/settings")
def get_settings():
    return settings.all()


@app.put("/api/settings")
def put_settings(patch: dict):
    return settings.update(patch)


@app.post("/api/settings/reset")
def reset_settings():
    return settings.reset()


@app.get("/api/categories")
def categories():
    return CATEGORIES


# ------------------------------------------------------------------------------ videos
def _video_path(vid: str) -> Path:
    d = UPLOAD_DIR / vid
    files = [p for p in d.glob("*") if p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXT] if d.exists() else []
    if not files:
        raise HTTPException(404, "Video not found")
    return files[0]


@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(400, "Unsupported file type. Allowed formats: MP4, AVI, MOV.")
    vid = new_id()
    d = UPLOAD_DIR / vid
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"video{ext}"
    limit = int(settings.get("max_upload_mb")) * 1024 * 1024
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"File exceeds the {settings.get('max_upload_mb')} MB limit.")
            out.write(chunk)
    try:
        info = SequenceLoader.probe(dest)
        if info["frames"] <= 0 or info["width"] <= 0:
            raise ValueError("no decodable frames")
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"The file could not be decoded as video ({exc}).") from exc
    meta = {"id": vid, "name": file.filename, "size_bytes": size, **info}
    (d / "meta.json").write_text(json.dumps(meta))
    return meta


@app.get("/api/videos")
def list_videos():
    out = []
    for p in UPLOAD_DIR.glob("*/meta.json"):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return sorted(out, key=lambda m: m["id"])


@app.get("/api/videos/{vid}")
def get_video(vid: str):
    p = UPLOAD_DIR / vid / "meta.json"
    if not p.exists():
        raise HTTPException(404, "Video not found")
    return json.loads(p.read_text())


@app.get("/api/videos/{vid}/stream")
def stream_video(vid: str):
    path = _video_path(vid)
    media = {".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo"}[path.suffix.lower()]
    return FileResponse(path, media_type=media)


@app.get("/api/videos/{vid}/thumbnail")
def thumbnail(vid: str):
    import cv2

    thumb = UPLOAD_DIR / vid / "thumb.jpg"
    if not thumb.exists():
        path = _video_path(vid)
        info = SequenceLoader.probe(path)
        frame = SequenceLoader.read_source_frame(path, int(info["frames"] * 0.1))
        if frame is None:
            raise HTTPException(404, "No thumbnail")
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (320, int(h * 320 / w)))
        cv2.imwrite(str(thumb), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return FileResponse(thumb, media_type="image/jpeg")


@app.delete("/api/videos/{vid}")
def delete_video(vid: str):
    shutil.rmtree(UPLOAD_DIR / vid, ignore_errors=True)
    return {"deleted": vid}


# ------------------------------------------------------------------------------ analysis
class AnalysisRequest(BaseModel):
    video_id: str
    threshold: float | None = None       # regularity threshold tau
    explain: bool | None = None


@app.post("/api/analyses")
def start_analysis(req: AnalysisRequest):
    if not registry.available("stae"):
        raise ModelNotTrained("No trained STAE checkpoint found. Train the model first (Model page, or `python train.py`).")
    path = _video_path(req.video_id)
    meta = json.loads((UPLOAD_DIR / req.video_id / "meta.json").read_text())
    aid = new_id()

    def work(job):
        def prog(f: float, m: str):
            job.progress, job.message = f, m
        res = pipeline.run(path, req.video_id, meta["name"], aid, req.threshold, req.explain, prog)
        return {"analysis_id": res["id"], "n_events": res["n_events"]}

    job = jobs.submit("analysis", work)
    return {"job_id": job.id, "analysis_id": aid}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return j.public()


@app.get("/api/analyses")
def list_analyses():
    return reports.list_analyses()


@app.get("/api/analyses/{aid}")
def get_analysis(aid: str):
    r = reports.load(aid)
    if not r:
        raise HTTPException(404, "Analysis not found")
    return r


@app.delete("/api/analyses/{aid}")
def delete_analysis(aid: str):
    return {"deleted": reports.delete(aid)}


# ------------------------------------------------------------------------------ events / explanations
@app.get("/api/events")
def list_events(severity: str | None = None, category: str | None = None, analysis_id: str | None = None):
    evs = reports.list_events(severity, category, analysis_id)
    return sorted(evs, key=lambda e: (e.get("created", ""), e["start_time"]), reverse=True)


@app.get("/api/events/export")
def export_events(format: str = Query("csv", pattern="^(csv|json)$")):
    evs = reports.list_events()
    if format == "json":
        return JSONResponse(evs, headers={"Content-Disposition": "attachment; filename=walkwayguard_events.json"})
    return PlainTextResponse(reports.export_csv(evs), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=walkwayguard_events.csv"})


@app.get("/api/events/{eid}")
def get_event(eid: str):
    e = reports.get_event(eid)
    if not e:
        raise HTTPException(404, "Event not found")
    return e


@app.get("/api/events/{eid}/image/{name}")
def event_image(eid: str, name: str):
    if name not in IMAGE_NAMES:
        raise HTTPException(404, "Unknown image")
    p = event_dir(eid) / f"{name}.jpg"
    if not p.exists():
        raise HTTPException(404, "Image not available (LIME not run for this event?)")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


class ExplainRequest(BaseModel):
    num_samples: int | None = None
    num_segments: int | None = None
    top_k: int | None = None


@app.post("/api/events/{eid}/explain")
def explain_event(eid: str, req: ExplainRequest):
    cfg = settings.all()
    try:
        return pipeline.explain_event(
            eid, req.num_samples or cfg["lime_samples"], req.num_segments or cfg["lime_segments"],
            req.top_k or cfg["lime_top_k"], cfg["seed"])
    except KeyError:
        raise HTTPException(404, "Event not found")


@app.delete("/api/events/{eid}")
def delete_event_note(eid: str):
    raise HTTPException(405, "Events are deleted with their analysis (DELETE /api/analyses/{id}).")


# ------------------------------------------------------------------------------ model
@app.get("/api/model/info")
def model_info():
    meta = registry.meta("stae") if registry.available("stae") else None
    spatial = registry.meta("spatial") if registry.available("spatial") else None
    cal = MODEL_DIR / "calibration.json"
    stats = MODEL_DIR / "normal_stats.json"
    last = jobs.latest("train")
    return {
        "architecture": ARCHITECTURE_TABLE,
        "device": str(registry.device),
        "stae": meta, "spatial": spatial,
        "calibration": json.loads(cal.read_text()) if cal.exists() else None,
        "normal_stats": json.loads(stats.read_text()) if stats.exists() else None,
        "hyperparameters": {
            "Framework": "PyTorch, OpenCV, scikit-image, LIME", "Input Resolution": "128 x 128 x 3",
            "Sequence Length (T)": "10 frames", "Window Stride": f"{settings.get('window_stride')} frame",
            "Encoder Filters": "32, 64, 128", "ConvLSTM Hidden Channels": 128, "Optimizer": "Adam",
            "Learning Rate": 0.0001, "Batch Size": 8, "Training Epochs": 50, "Loss Function": "Mean Squared Error (MSE)",
            "Regularity Threshold": settings.get("threshold"), "Threshold Selection": "Maximum validation F1-score",
            "Superpixel Segmentation": "SLIC", "Number of Superpixels (K)": settings.get("lime_segments"),
            "LIME Perturbation Samples (N)": settings.get("lime_samples"),
            "Highlighted Regions": f"Top {settings.get('lime_top_k')} superpixels", "Random Seed": settings.get("seed"),
        },
        "training_job": last.public() if last else None,
    }


class TrainRequest(BaseModel):
    model: str = "stae"                 # "stae" | "spatial" (ablation)
    dataset: str = "avenue"
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    stride: int = 1
    max_videos: int | None = None
    max_frames_per_video: int | None = None


@app.post("/api/model/train")
def start_training(req: TrainRequest):
    if req.model not in ("stae", "spatial"):
        raise HTTPException(400, "model must be 'stae' or 'spatial'")
    if req.dataset not in DATASETS:
        raise HTTPException(400, f"dataset must be one of {list(DATASETS)}")
    if jobs.running("train"):
        raise HTTPException(409, "A training job is already running.")

    def work(job):
        def prog(f, m, extra):
            job.progress, job.message = f, m
            if extra:
                job.extra.update(extra)
        meta = train_model(req.model, req.dataset, req.epochs, req.batch_size, req.lr, req.stride,
                           max_videos=req.max_videos, max_frames_per_video=req.max_frames_per_video,
                           seed=int(settings.get("seed")), progress=prog)
        registry.invalidate(req.model)
        return {k: v for k, v in meta.items() if k != "history"}

    return {"job_id": jobs.submit("train", work).id}


# ------------------------------------------------------------------------------ evaluation
class EvalRequest(BaseModel):
    model: str = "stae"
    dataset: str = "avenue"
    explain: bool = True
    apply_threshold: bool = True
    val_fraction: float = 0.3


@app.post("/api/evaluation/run")
def start_evaluation(req: EvalRequest):
    if not registry.available(req.model):
        raise ModelNotTrained(f"No trained '{req.model}' checkpoint. Train it first.")
    if jobs.running("evaluation"):
        raise HTTPException(409, "An evaluation job is already running.")

    def work(job):
        def prog(f, m):
            job.progress, job.message = f, m
        res = ev_mod.run_evaluation(req.model, req.dataset, req.explain, req.apply_threshold, req.val_fraction, progress=prog)
        return {"auc": res.get("auc"), "tau": res["tau"]}

    return {"job_id": jobs.submit("evaluation", work).id}


@app.get("/api/evaluation/results")
def evaluation_results():
    return ev_mod.load_all_results()


# ------------------------------------------------------------------------------ frontend
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        f = FRONTEND_DIST / full_path
        return FileResponse(f if f.is_file() else FRONTEND_DIST / "index.html")
