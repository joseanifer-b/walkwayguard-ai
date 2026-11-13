import { useEffect, useState } from "react";
import { post, put, useApi } from "../api";
import { Card, Err, PageHead } from "../components/ui";

const FIELDS: [string, string, string][] = [
  ["threshold", "Regularity threshold τ (anomaly score threshold θ = 1 − τ)", "number"],
  ["window_stride", "Analysis window stride (frames; 1 = report setting, larger = faster)", "number"],
  ["lime_segments", "LIME superpixels K (SLIC)", "number"],
  ["lime_samples", "LIME perturbation samples N", "number"],
  ["lime_top_k", "Highlighted superpixels (top-k)", "number"],
  ["lime_batch", "LIME scoring batch size", "number"],
  ["merge_gap_sec", "Merge flagged frames closer than (s)", "number"],
  ["min_event_frames", "Minimum flagged frames per event", "number"],
  ["max_events_explained", "Max events explained with LIME per video", "number"],
  ["max_frames", "Max frames analysed per video", "number"],
  ["max_upload_mb", "Max upload size (MB)", "number"],
  ["seed", "Random seed", "number"],
];

export default function SettingsPage() {
  const { data, reload } = useApi("/settings");
  const [v, setV] = useState<any>({});
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { if (data) setV(data); }, [data]);
  async function save() {
    setErr(null);
    try { await put("/settings", v); setMsg("Saved"); reload(); } catch (e: any) { setErr(e.message); }
  }
  async function reset() { await post("/settings/reset"); setMsg("Reset to report defaults"); reload(); }
  return (
    <>
      <PageHead title="Settings" sub="Runtime parameters (defaults follow Table 6.1 of the report)" />
      <Err msg={err} />
      <Card>
        <div className="grid g2">
          {FIELDS.map(([k, label, type]) => (
            <div className="field" key={k}><label>{label}</label>
              <input type={type} step="any" value={v[k] ?? ""} onChange={(e) => setV({ ...v, [k]: e.target.value === "" ? "" : +e.target.value })} /></div>
          ))}
        </div>
        <label className="row" style={{ color: "var(--text)" }}><input type="checkbox" style={{ width: "auto" }} checked={!!v.explain_events} onChange={(e) => setV({ ...v, explain_events: e.target.checked })} /> Run LIME automatically on each detected event</label>
        <div className="row" style={{ marginTop: 10 }}>
          <button className="btn" onClick={save}>Save</button><button className="btn ghost" onClick={reset}>Reset to defaults</button>
          {msg && <span className="muted">{msg}</span>}
        </div>
      </Card>
    </>
  );
}
