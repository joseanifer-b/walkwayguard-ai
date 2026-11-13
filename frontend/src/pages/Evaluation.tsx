import { useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { pct, post, useApi, useJob } from "../api";
import { Card, Err, PageHead, Progress } from "../components/ui";

// Frame-level AUC (%) published by the original papers, as cited in the report (Table 6.6).
const PUBLISHED = [
  ["Hasan et al. — convolutional autoencoder", 70.2, 60.9],
  ["Luo et al. — ConvLSTM autoencoder", 77.0, 68.0],
  ["Liu et al. — future-frame prediction", 84.9, 72.8],
  ["Gong et al. — memory-augmented autoencoder", 83.3, 71.2],
] as const;

export default function Evaluation() {
  const { data: results, reload } = useApi<any[]>("/evaluation/results");
  const [dataset, setDataset] = useState("avenue");
  const [explain, setExplain] = useState(true);
  const [model, setModel] = useState("stae");
  const [err, setErr] = useState<string | null>(null);
  const { job, track, active } = useJob(() => reload());
  const get = (m: string) => results?.find((r) => r.model === m && r.dataset === dataset);
  const r = get("stae");
  const sp = get("spatial");
  const ex = r?.explainability;
  const rows = (d: any) => d?.roc ? d.roc.fpr.map((x: number, i: number) => ({ fpr: x, tpr: d.roc.tpr[i] })) : [];

  async function run() {
    setErr(null);
    try { track((await post("/evaluation/run", { model, dataset, explain, apply_threshold: model === "stae" })).job_id); }
    catch (e: any) { setErr(e.message); }
  }
  const otherSets = ["avenue", "shanghaitech"].map((d) => results?.find((x) => x.model === "stae" && x.dataset === d));
  return (
    <>
      <PageHead title="Evaluation" sub="Every number on this page is measured by an evaluation run — nothing is pre-filled" />
      <Err msg={err} />
      <Card title="Run evaluation">
        <div className="row" style={{ alignItems: "flex-end" }}>
          <div style={{ width: 220 }}><label>Dataset</label><select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            <option value="avenue">CUHK Avenue</option><option value="shanghaitech">ShanghaiTech Campus</option><option value="demo">Synthetic demo</option></select></div>
          <div style={{ width: 220 }}><label>Model</label><select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="stae">STAE</option><option value="spatial">Spatial-only (ablation)</option></select></div>
          <label className="row" style={{ margin: 0, color: "var(--text)" }}><input type="checkbox" style={{ width: "auto" }} checked={explain} onChange={(e) => setExplain(e.target.checked)} /> LIME explanation metrics (slow)</label>
          <button className="btn" disabled={active} onClick={run}>{active ? "Running…" : "Run evaluation"}</button>
        </div>
        <Progress job={job} />
        <p className="muted small">Protocol: test videos are split by video; τ is tuned (max F1) on the validation part and all reported numbers come from the held-out part. AUC is micro-averaged over frames after per-video min-max normalisation. STAE evaluation also saves τ as the system threshold.</p>
      </Card>

      {!r ? <div className="card empty">No STAE results for this dataset yet. Train the model, then run an evaluation.</div> : (
        <>
          {r.warning && <div className="alert warn">{r.warning}</div>}
          <div className="grid g2">
            <Card title="Frame-level detection (Table 6.3)">
              <table><tbody>
                {[["AUC (%)", pct(r.auc, 2)], ["AUC, mean per video (%)", pct(r.auc_macro, 2)], ["Accuracy (%)", pct(r.accuracy, 2)], ["Precision (%)", pct(r.precision, 2)],
                  ["Recall (%)", pct(r.recall, 2)], ["F1 (%)", pct(r.f1, 2)], ["FPR (%)", pct(r.fpr, 2)], ["EER (%)", pct(r.eer, 2)], ["Average precision (%)", pct(r.ap, 2)],
                  ["τ (regularity threshold)", r.tau.toFixed(4)]].map(([k, v]) => <tr key={k}><td className="muted">{k}</td><td><b>{v}</b></td></tr>)}
              </tbody></table>
              <p className="muted small">{r.n_test_videos} held-out test videos · {r.n_test_frames} frames · {r.anomalous_frame_pct}% anomalous. A classifier that labels everything normal would score {r.trivial_all_normal_accuracy}% accuracy — compare accuracy with that, and rely on F1 and AUC. τ source: {r.tau_source}.</p>
            </Card>
            <Card title="ROC curve (Figure 6.1)">
              <div style={{ height: 280 }}><ResponsiveContainer><LineChart data={rows(r)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e3e7f0" /><XAxis type="number" dataKey="fpr" domain={[0, 1]} fontSize={12} /><YAxis domain={[0, 1]} fontSize={12} /><Tooltip />
                <Line dataKey="tpr" stroke="#2563eb" dot={false} strokeWidth={2} /></LineChart></ResponsiveContainer></div>
            </Card>
          </div>
          <div className="grid g2">
            <Card title="Confusion matrix (Figure 6.3)">
              <div className="cm">
                <div /><div className="h">Pred. normal</div><div className="h">Pred. anomalous</div>
                <div className="h">Actual normal</div><div style={{ background: "#dcfce7" }}>TN {r.tn}</div><div style={{ background: "#fee2e2" }}>FP {r.fp}</div>
                <div className="h">Actual anomalous</div><div style={{ background: "#fee2e2" }}>FN {r.fn}</div><div style={{ background: "#dcfce7" }}>TP {r.tp}</div>
              </div>
            </Card>
            <Card title="Regularity timelines (Figure 6.2)">
              {r.timelines.slice(0, 2).map((t: any) => (
                <div key={t.name} style={{ height: 120 }}>
                  <div className="muted small">Video {t.name} (shaded = ground-truth anomaly)</div>
                  <ResponsiveContainer><LineChart data={t.regularity.map((v: number, i: number) => ({ i, s: v, y: t.labels[i] }))}>
                    <XAxis dataKey="i" hide /><YAxis domain={[0, 1]} fontSize={11} /><Tooltip />
                    <Line dataKey="y" stroke="#f5a524" dot={false} strokeWidth={1} name="ground truth" /><Line dataKey="s" stroke="#2563eb" dot={false} name="regularity s_t" /></LineChart></ResponsiveContainer>
                </div>))}
            </Card>
          </div>
          <div className="grid g2">
            <Card title="Comparison with published methods (frame-level AUC, %)">
              <table><thead><tr><th>Method</th><th>Avenue</th><th>ShanghaiTech</th></tr></thead><tbody>
                {PUBLISHED.map(([n, a, s]) => <tr key={n}><td>{n}</td><td>{a}</td><td>{s}</td></tr>)}
                <tr><td><b>This system (STAE + LIME, measured)</b></td><td><b>{pct(otherSets[0]?.auc, 2)}</b></td><td><b>{pct(otherSets[1]?.auc, 2)}</b></td></tr>
              </tbody></table>
              <p className="muted small">Published values are taken from the original papers as cited in the report; protocols differ, so this is context, not an exact ranking.</p>
            </Card>
            <Card title="Ablation: spatial-only vs full STAE (Table 6.7)">
              {!sp ? <div className="muted">Train the spatial-only model (Model page) and evaluate it to fill this table.</div> : (
                <table><thead><tr><th>Model</th><th>AUC (%)</th><th>F1 (%)</th></tr></thead><tbody>
                  <tr><td>Spatial-only autoencoder</td><td>{pct(sp.auc, 2)}</td><td>{pct(sp.f1, 2)}</td></tr>
                  <tr><td>STAE (with ConvLSTM)</td><td>{pct(r.auc, 2)}</td><td>{pct(r.f1, 2)}</td></tr></tbody></table>)}
            </Card>
          </div>
          <Card title="Explainability metrics (Table 6.8) and LIME sensitivity (Table 6.9)">
            {!ex || ex.note ? <div className="muted">{ex?.note ?? "Run the evaluation with LIME explanation metrics enabled."}</div> : (
              <div className="grid g2">
                <div>
                  <table><tbody>
                    {[["Pointing accuracy (%)", ex.metrics.pointing_acc ?? "n/a (no spatial ground truth)"], ["Mean IoU (top-k superpixels)", ex.metrics.mean_iou ?? "n/a"],
                      ["Deletion AUC (lower is better)", ex.metrics.deletion_auc], ["Top-k stability (Jaccard)", ex.metrics.stability], ["Surrogate local R²", ex.metrics.r2],
                      ["Latency per frame (s)", ex.metrics.latency], ["Frames evaluated", ex.metrics.frames]].map(([k, v]) => <tr key={k as string}><td className="muted">{k}</td><td><b>{String(v)}</b></td></tr>)}
                  </tbody></table>
                </div>
                <div>
                  <table><thead><tr><th>N</th><th>K</th><th>Latency (s)</th><th>Pointing (%)</th><th>Stability</th></tr></thead><tbody>
                    {ex.sensitivity.map((s: any) => <tr key={`${s.n}-${s.k}`}><td>{s.n}</td><td>{s.k}</td><td>{s.latency}</td><td>{s.pointing_acc ?? "n/a"}</td><td>{s.stability}</td></tr>)}
                  </tbody></table>
                </div>
              </div>)}
            <p className="muted small">Measured on a sample of true-positive frames. Per-category recall (Table 6.4) is not computed: Avenue and ShanghaiTech provide no per-event category labels.</p>
          </Card>
        </>
      )}
    </>
  );
}
