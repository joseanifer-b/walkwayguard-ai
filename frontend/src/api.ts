import { useCallback, useEffect, useRef, useState } from "react";

export async function api<T = any>(path: string, opts?: RequestInit): Promise<T> {
  const r = await fetch("/api" + path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const j = await r.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {}
    throw new Error(detail);
  }
  return r.json();
}

export const post = <T = any>(path: string, body: any = {}) =>
  api<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
export const put = <T = any>(path: string, body: any = {}) =>
  api<T>(path, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

/** Load an endpoint once (and on demand). */
export function useApi<T = any>(path: string | null) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const reload = useCallback(async () => {
    if (!path) return;
    setLoading(true);
    try {
      setData(await api<T>(path));
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [path]);
  useEffect(() => {
    reload();
  }, [reload]);
  return { data, error, loading, reload, setData };
}

/** Poll a background job until it finishes. */
export function useJob(onDone?: (job: any) => void) {
  const [job, setJob] = useState<any>(null);
  const timer = useRef<number | null>(null);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  const stop = () => {
    if (timer.current) window.clearInterval(timer.current);
    timer.current = null;
  };
  const track = useCallback((jobId: string) => {
    stop();
    const tick = async () => {
      try {
        const j = await api("/jobs/" + jobId);
        setJob(j);
        if (j.status === "done" || j.status === "error") {
          stop();
          doneRef.current?.(j);
        }
      } catch {
        stop();
      }
    };
    tick();
    timer.current = window.setInterval(tick, 1200);
  }, []);
  useEffect(() => stop, []);
  const active = job && (job.status === "queued" || job.status === "running");
  return { job, track, active, setJob };
}

export const imgUrl = (eventId: string, name: string, bust?: any) =>
  `/api/events/${eventId}/image/${name}${bust ? "?v=" + bust : ""}`;

export const pct = (v: number | null | undefined, d = 1) => (v == null ? "—" : v.toFixed(d));
