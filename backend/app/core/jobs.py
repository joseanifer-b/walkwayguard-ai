"""Tiny in-process background job manager (analysis, training, evaluation)."""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..utils import new_id


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    extra: dict = field(default_factory=dict)   # live data such as training history

    def public(self) -> dict:
        d = asdict(self)
        d["elapsed"] = round((self.finished or time.time()) - self.started, 1)
        return d


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[["Job"], Any]) -> Job:
        job = Job(id=new_id(), kind=kind)
        with self._lock:
            self._jobs[job.id] = job

        def runner() -> None:
            job.status = "running"
            try:
                job.result = fn(job)
                job.status, job.progress = "done", 1.0
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            finally:
                job.finished = time.time()

        threading.Thread(target=runner, daemon=True, name=f"job-{kind}-{job.id}").start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def running(self, kind: str) -> Job | None:
        with self._lock:
            for j in self._jobs.values():
                if j.kind == kind and j.status in ("queued", "running"):
                    return j
        return None

    def latest(self, kind: str) -> Job | None:
        with self._lock:
            js = [j for j in self._jobs.values() if j.kind == kind]
        return max(js, key=lambda j: j.started) if js else None


jobs = JobManager()
