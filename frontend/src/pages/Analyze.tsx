import { useRef, useState } from "react";
import { Upload } from "lucide-react";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, post, useApi, useJob } from "../api";
import { Card, Err, EventDetail, ModelBanner, PageHead, Progress } from "../components/ui";

const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export default function Analyze() {
  const { data: status } = useApi("/status");
  const { data: settings } = useApi("/settings");
  const [video, setVideo] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [explain, setExplain] = useState(true);
  const [tau, setTau] = useState<string>("");
  const [result, setResult] = useState<any>(null);
  const [sel, setSel] = useState<any>(null);
  const input = useRef<HTMLInputElement>(null);
  const { job, track, active } = useJob(async (j) => {
    if (j.status === "done") {
      const r = await api("/analyses/" + j.result.analysis_id);
      setResult(r);
      setSel(r.events[0] ?? null);
    }
  });

  async function upload(f: File) {
    setErr(null); setResult(null); setSel(null);
    const ext = f.name.split(".").pop()?.toLowerCase();
    if (!["mp4", "avi", "mov"].includes(ext || "")) return setErr("Unsupported file type. Allowed formats: MP4, AVI, MOV.");
    if (f.size > 500 * 1024 * 1024) return setErr("File exceeds the 500 MB limit.");
    setUploading(true);
    try {
      const fd = new FormData(); fd.append("file", f);
      const r = await fetch("/api/videos", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      setVideo(await r.json());
    } catch (e: any) { setErr(e.message); } finally { setUploading(false); }
  }

  async function start() {
    setErr(null); setResult(null); setSel(null);
    try {
      const body: any = { video_id: video.id, explain };
      if (tau !== "") body.threshold = parseFloat(tau);
      const r = await post("/analyses", body);
      track(r.job_id);
    } catch (e: any) { setErr(e.message); }
  }

  const timeline = result ? result.timeline.t.map((t: number, i: number) => ({ t, a: result.timeline.anomaly[i] })) : [];
  return (
    <>
      <PageHead title="Analyze Video" sub="Upload walkway footage, run the STAE detector, and review explained alerts" />
      <ModelBanner ready={status?.model_ready} />
      <Err msg={err} />
      <div className="grid g2">
        <Card>
          <div className={`drop ${over ? "over" : ""}`} onClick={() => input.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) upload(f); }}>
            <Upload size={42} color="#2563eb" />
            <h3>Upload Surveillance Video</h3>
            <p>Drag & drop your video file here, or click to <b>browse</b><br />Supported formats: MP4, AVI, MOV (max 500MB)</p>
            <button className="btn" disabled={uploading}>{uploading ? "Uploading…" : "Choose Video"}</button>
            <input ref={input} type="file" accept=".mp4,.avi,.mov" hidden onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
          </div>
          {video && (
            <div className="file-row">
              <img src={`/api/videos/${video.id}/thumbnail`} alt="" />
              <div><b>{video.name}</b><div className="muted small">{(video.size_bytes / 1048576).toFixed(1)} MB · {fmt(video.duration)}</div></div>
              <span className="tick">✓</span>
            </div>
          )}
        </Card>
        <Card title="Video Preview">
          {video ? (
            <>
              <video src={`/api/videos/${video.id}/stream`} controls />
              <h2 style={{ marginTop: 14 }}>Video Information</h2>
              <div className="grid g3">
                <div><div className="muted small">Duration</div><b>{fmt(video.duration)}</b></div>
                <div><div className="muted small">Resolution</div><b>{video.width} × {video.height}</b></div>
                <div><div className="muted small">FPS</div><b>{video.fps.toFixed(0)}</b></div>
              </div>
            </>
          ) : <div className="empty">Upload a video to preview it here.</div>}
        </Card>
      </div>

      {video && (
        <Card title="Run analysis">
          <div className="row" style={{ alignItems: "flex-end" }}>
            <div style={{ width: 220 }}>
              <label>Regularity threshold τ (default {settings?.threshold})</label>
              <input type="number" step="0.01" min="0" max="1" placeholder={String(settings?.threshold ?? 0.35)} value={tau} onChange={(e) => setTau(e.target.value)} />
            </div>
            <label className="row" style={{ margin: 0, color: "var(--text)" }}>
              <input type="checkbox" style={{ width: "auto" }} checked={explain} onChange={(e) => setExplain(e.target.checked)} /> Generate LIME explanations
            </label>
            <button className="btn" disabled={active || !status?.model_ready} onClick={start}>Start Analysis</button>
          </div>
          <Progress job={job} />
        </Card>
      )}

      {result && (
        <>
          <Card title="Anomaly timeline" right={<span className="muted small">{result.n_events} events · {result.frames} frames · {result.elapsed}s · threshold θ = {result.theta}</span>}>
            <div style={{ height: 220 }}>
              <ResponsiveContainer>
                <AreaChart data={timeline}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e3e7f0" />
                  <XAxis dataKey="t" tickFormatter={fmt} fontSize={12} />
                  <YAxis domain={[0, 1]} fontSize={12} />
                  <Tooltip formatter={(v: any) => Number(v).toFixed(3)} labelFormatter={(l) => fmt(Number(l))} />
                  <ReferenceLine y={result.theta} stroke="#e5484d" strokeDasharray="5 4" label={{ value: "θ", fill: "#e5484d" }} />
                  <Area type="monotone" dataKey="a" name="anomaly score" stroke="#2563eb" fill="#2563eb33" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <p className="muted small">Anomaly score a<sub>t</sub> = 1 − s<sub>t</sub> (per-video min-max normalised). A frame is flagged when a<sub>t</sub> &gt; θ = 1 − τ.</p>
          </Card>
          <Card title="Detected events">
            {result.events.length === 0 ? <div className="empty">No anomalies flagged at this threshold.</div> : (
              <table>
                <thead><tr><th>Time</th><th>Duration</th><th>Score</th><th>Severity</th><th>Estimated type</th></tr></thead>
                <tbody>
                  {result.events.map((e: any) => (
                    <tr key={e.id} className={`click ${sel?.id === e.id ? "sel" : ""}`} onClick={() => setSel(e)}>
                      <td>{e.timestamp}</td><td>{e.duration}s</td><td>{e.anomaly_score.toFixed(2)}</td>
                      <td><span className={`badge ${e.severity}`}>{e.severity}</span></td><td>{e.category}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
          {sel && <EventDetail ev={sel} />}
        </>
      )}
    </>
  );
}
