import { useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { post, useApi, useJob } from "../api";
import { Card, Err, PageHead, Progress } from "../components/ui";

export default function ModelPage() {
  const { data: info, reload } = useApi("/model/info");
  const { data: status } = useApi("/status");
  const [f, setF] = useState({ model: "stae", dataset: "avenue", epochs: 50, batch_size: 8, lr: 0.0001, stride: 1, max_videos: "" });
  const [err, setErr] = useState<string | null>(null);
  const { job, track, active } = useJob(() => reload());
  const set = (k: string, v: any) => setF({ ...f, [k]: v });
  const live = job ?? info?.training_job;
  const hist = live?.extra?.history ?? info?.stae?.history ?? [];

  async function train() {
    setErr(null);
    try {
      const body: any = { ...f, max_videos: f.max_videos === "" ? null : +f.max_videos };
      track((await post("/model/train", body)).job_id);
    } catch (e: any) { setErr(e.message); }
  }
  return (
    <>
      <PageHead title="Model" sub={`Spatio-Temporal Autoencoder (CNN encoder → ConvLSTM → decoder) · device: ${info?.device ?? "…"}`} />
      <Err msg={err} />
      <div className="grid g2">
        <Card title="Architecture (Table 5.2)">
          <table><thead><tr><th>Component</th><th>Parameter</th><th>Role</th></tr></thead>
            <tbody>{info?.architecture.map((r: any, i: number) => <tr key={i}><td>{r.component}</td><td>{r.parameter}</td><td>{r.role}</td></tr>)}</tbody></table>
        </Card>
        <Card title="Experimental setup (Table 6.1)">
          <table><tbody>{info && Object.entries(info.hyperparameters).map(([k, v]) => <tr key={k}><td className="muted">{k}</td><td>{String(v)}</td></tr>)}</tbody></table>
        </Card>
      </div>
      <div className="grid g2">
        <Card title="Trained checkpoints">
          {["stae", "spatial"].map((k) => {
            const m = info?.[k];
            return (
              <div key={k} style={{ marginBottom: 12 }}>
                <b>{k === "stae" ? "STAE (with ConvLSTM)" : "Spatial-only autoencoder (ablation)"}</b>{" "}
                <span className={`badge ${m ? "ok" : "warn"}`}>{m ? "trained" : "not trained"}</span>
                {m && <div className="kv" style={{ marginTop: 6 }}>
                  <span className="k">Dataset</span><span>{m.dataset}</span>
                  <span className="k">Epochs run</span><span>{m.epochs_run} / {m.epochs_max}</span>
                  <span className="k">Best val loss</span><span>{m.best_val_loss.toFixed(6)}</span>
                  <span className="k">Parameters</span><span>{m.params?.toLocaleString()}</span>
                  <span className="k">Training fps</span><span>{m.fps}</span>
                  <span className="k">Created</span><span>{m.created}</span>
                </div>}
              </div>
            );
          })}
          <div className="muted small">Threshold τ in use: <b>{status?.threshold}</b>{info?.calibration && <> (calibrated on {info.calibration.dataset}: {info.calibration.source})</>}</div>
        </Card>
        <Card title="Train">
          <div className="grid g2">
            <div className="field"><label>Model</label><select value={f.model} onChange={(e) => set("model", e.target.value)}><option value="stae">STAE</option><option value="spatial">Spatial-only (ablation)</option></select></div>
            <div className="field"><label>Dataset</label><select value={f.dataset} onChange={(e) => set("dataset", e.target.value)}>
              <option value="avenue">CUHK Avenue</option><option value="shanghaitech">ShanghaiTech Campus</option><option value="demo">Synthetic demo (smoke test)</option></select></div>
            <div className="field"><label>Epochs (early stopping on val loss)</label><input type="number" value={f.epochs} onChange={(e) => set("epochs", +e.target.value)} /></div>
            <div className="field"><label>Batch size</label><input type="number" value={f.batch_size} onChange={(e) => set("batch_size", +e.target.value)} /></div>
            <div className="field"><label>Learning rate</label><input type="number" step="0.00005" value={f.lr} onChange={(e) => set("lr", +e.target.value)} /></div>
            <div className="field"><label>Training window stride</label><input type="number" min={1} value={f.stride} onChange={(e) => set("stride", +e.target.value)} /></div>
            <div className="field"><label>Max videos (blank = all)</label><input type="number" value={f.max_videos} onChange={(e) => set("max_videos", e.target.value)} /></div>
          </div>
          <button className="btn" disabled={active} onClick={train}>{active ? "Training…" : "Start training"}</button>
          <Progress job={live} />
          <p className="muted small">Only normal footage is used. A larger stride trains faster; the report setting is stride 1 (batch 8, lr 1e-4, 50 epochs).</p>
        </Card>
      </div>
      {hist.length > 0 && (
        <Card title="Training loss (MSE)">
          <div style={{ height: 240 }}><ResponsiveContainer><LineChart data={hist}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e3e7f0" /><XAxis dataKey="epoch" /><YAxis fontSize={12} /><Tooltip /><Legend />
            <Line dataKey="train_loss" name="train" stroke="#2563eb" dot={false} /><Line dataKey="val_loss" name="validation" stroke="#e5484d" dot={false} />
          </LineChart></ResponsiveContainer></div>
        </Card>
      )}
    </>
  );
}
