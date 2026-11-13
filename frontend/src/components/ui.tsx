import { ReactNode } from "react";
import { imgUrl } from "../api";

export const Card = ({ title, children, right }: { title?: string; children: ReactNode; right?: ReactNode }) => (
  <div className="card">
    {(title || right) && (
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 10 }}>
        {title && <h2 style={{ margin: 0 }}>{title}</h2>}
        {right}
      </div>
    )}
    {children}
  </div>
);

export const Stat = ({ label, value, color }: { label: string; value: ReactNode; color?: string }) => (
  <div className="card stat">
    <span className="v" style={{ color }}>{value}</span>
    <span className="l">{label}</span>
  </div>
);

export const PageHead = ({ title, sub }: { title: string; sub?: string }) => (
  <>
    <h1>{title}</h1>
    {sub && <p className="sub">{sub}</p>}
  </>
);

export const Err = ({ msg }: { msg?: string | null }) => (msg ? <div className="alert err">{msg}</div> : null);

export const ModelBanner = ({ ready }: { ready: boolean | undefined }) =>
  ready === false ? (
    <div className="alert warn">
      No trained STAE checkpoint yet. Put the dataset in <code>dataset/</code> and train on the <b>Model</b> page (or run{" "}
      <code>python train.py</code>). Analysis stays disabled until a model exists — no placeholder results are shown.
    </div>
  ) : null;

export const Progress = ({ job }: { job: any }) =>
  job ? (
    <div style={{ marginTop: 12 }}>
      <div className="progress"><div style={{ width: `${Math.round((job.progress || 0) * 100)}%` }} /></div>
      <div className="msg">
        {job.status === "error" ? <span style={{ color: "var(--red)" }}>{job.error}</span> : `${job.message || job.status} · ${Math.round((job.progress || 0) * 100)}% · ${job.elapsed}s`}
      </div>
    </div>
  ) : null;

export function SeverityBadge({ s }: { s: string }) {
  return <span className={`badge ${s}`}>{s} Severity</span>;
}

/** Figure 6.6 - anomaly event details with four evidence panels. */
export function EventDetail({ ev, bust }: { ev: any; bust?: any }) {
  const has = (n: string) => ev.images?.includes(n);
  return (
    <div>
      <div className="event-head">
        <div className="kv" style={{ gridTemplateColumns: "130px 1fr", color: "#fff" }}>
          <span style={{ opacity: 0.7 }}>Timestamp</span><b>{ev.timestamp}</b>
          <span style={{ opacity: 0.7 }}>Duration</span><b>{ev.duration} seconds</b>
          <span style={{ opacity: 0.7 }}>Anomaly Score</span><b>{ev.anomaly_score.toFixed(2)}</b>
          <span style={{ opacity: 0.7 }}>Threshold</span><b>{ev.threshold.toFixed(2)}</b>
          <span style={{ opacity: 0.7 }}>Type</span><b>{ev.category} (estimated)</b>
        </div>
        <SeverityBadge s={ev.severity} />
      </div>
      <div className="event-body">
        <div className="panels">
          {[["original", "Original Frame"], ["reconstruction", "Reconstructed"], ["error_map", "Error Map"], ["highlight", "Highlighted Region"]].map(([k, t]) => (
            <div className="panel" key={k}>
              <div className="cap">{t}</div>
              <img src={imgUrl(ev.id, k, bust)} alt={t} />
            </div>
          ))}
        </div>
        <div className="expl">
          <b>Explanation</b>
          <div>
            {has("lime_heatmap")
              ? ev.lime?.text
              : "The frame was flagged because its reconstruction error exceeded the calibrated threshold. LIME has not been run for this event — open the Explainability page to generate it."}
          </div>
          <div className="muted small" style={{ marginTop: 6 }}>
            Category is estimated from motion cues ({ev.rationale}); the STAE itself has no class labels. The error map comes from the detector and the highlighted region from LIME — their agreement is a consistency check, not proof.
          </div>
        </div>
      </div>
    </div>
  );
}
