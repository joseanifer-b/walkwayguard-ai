import { Link } from "react-router-dom";
import { useApi } from "../api";
import { Card, Err, ModelBanner, PageHead, Stat } from "../components/ui";

export default function Dashboard() {
  const { data: s, error } = useApi("/status");
  return (
    <>
      <PageHead title="Dashboard" sub="Explainable anomaly detection for pedestrian walkway surveillance" />
      <Err msg={error} />
      <ModelBanner ready={s?.model_ready} />
      {s && (
        <>
          <div className="grid g4">
            <Stat label="Videos analysed" value={s.analyses} />
            <Stat label="Anomaly events" value={s.events} />
            <Stat label="High severity" value={s.severity.High} color="var(--red)" />
            <Stat label="Regularity threshold τ" value={s.threshold} />
          </div>
          <div className="grid g2">
            <Card title="Recent analyses" right={<Link to="/analyze">New analysis →</Link>}>
              {s.recent_analyses.length === 0 ? (
                <div className="empty">No videos analysed yet.</div>
              ) : (
                <table>
                  <thead><tr><th>Video</th><th>Events</th><th>High</th><th>Time</th></tr></thead>
                  <tbody>
                    {s.recent_analyses.map((a: any) => (
                      <tr key={a.id}>
                        <td><Link to={`/events?analysis=${a.id}`}>{a.video_name}</Link></td>
                        <td>{a.n_events}</td><td>{a.n_high}</td><td>{a.elapsed}s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
            <Card title="Events by estimated category">
              {Object.keys(s.categories).length === 0 ? (
                <div className="empty">No events yet.</div>
              ) : (
                <table>
                  <tbody>
                    {Object.entries(s.categories).map(([k, v]) => (
                      <tr key={k}><td>{k} <span className="muted small">(estimated)</span></td><td>{v as number}</td></tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
          </div>
          <div className="grid g2">
            <Card title="Model">
              {s.model ? (
                <div className="kv">
                  <span className="k">Trained on</span><span>{s.model.dataset}</span>
                  <span className="k">Epochs run</span><span>{s.model.epochs_run}</span>
                  <span className="k">Best val loss</span><span>{s.model.best_val_loss?.toFixed(6)}</span>
                  <span className="k">Parameters</span><span>{s.model.params?.toLocaleString()}</span>
                  <span className="k">Device</span><span>{s.device}</span>
                </div>
              ) : <div className="muted">Not trained yet.</div>}
            </Card>
            <Card title="Datasets">
              <table>
                <thead><tr><th>Dataset</th><th>Train seq.</th><th>Test seq.</th><th>Labelled</th></tr></thead>
                <tbody>
                  {s.datasets.filter((d: any) => d.key !== "demo" || d.train_sequences > 0).map((d: any) => (
                    <tr key={d.key}>
                      <td>{d.name}</td><td>{d.train_sequences}</td><td>{d.test_sequences}</td>
                      <td><span className={`badge ${d.evaluable ? "ok" : "warn"}`}>{d.evaluable ? d.labelled_test_sequences : "missing"}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted small">Place files in <code>dataset/avenue/</code> and <code>dataset/shanghaitech/</code>.</p>
            </Card>
          </div>
        </>
      )}
    </>
  );
}
