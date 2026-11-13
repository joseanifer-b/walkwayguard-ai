import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { imgUrl, post, useApi } from "../api";
import { Card, Err, PageHead } from "../components/ui";

export default function Explainability() {
  const [params, setParams] = useSearchParams();
  const { data: events } = useApi<any[]>("/events");
  const { data: settings } = useApi("/settings");
  const eid = params.get("event");
  const [ev, setEv] = useState<any>(null);
  const [n, setN] = useState(1000);
  const [k, setK] = useState(100);
  const [top, setTop] = useState(8);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [bust, setBust] = useState(0);

  useEffect(() => { if (events && !eid && events.length) setParams({ event: events[0].id }, { replace: true }); }, [events, eid]);
  useEffect(() => { if (events && eid) setEv(events.find((e) => e.id === eid) ?? null); }, [events, eid]);
  useEffect(() => { if (settings) { setN(settings.lime_samples); setK(settings.lime_segments); setTop(settings.lime_top_k); } }, [settings]);

  async function rerun() {
    if (!ev) return;
    setBusy(true); setErr(null);
    try {
      const r = await post(`/events/${ev.id}/explain`, { num_samples: n, num_segments: k, top_k: top });
      setEv({ ...ev, ...r }); setBust(Date.now());
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }
  const lime = ev?.lime;
  const hasLime = ev?.images?.includes("lime_heatmap");
  return (
    <>
      <PageHead title="Explainability" sub="LIME saliency for the peak frame of an event: which regions drive the STAE reconstruction error" />
      <Err msg={err} />
      {!events?.length ? <div className="card empty">No events yet. Analyse a video first.</div> : (
        <>
          <Card>
            <div className="row" style={{ alignItems: "flex-end" }}>
              <div style={{ minWidth: 300 }}><label>Event</label>
                <select value={eid ?? ""} onChange={(e) => setParams({ event: e.target.value })}>
                  {events.map((e) => <option key={e.id} value={e.id}>{e.video_name} · {e.timestamp} · {e.category}</option>)}
                </select></div>
              <div style={{ width: 120 }}><label>Samples N</label><input type="number" value={n} min={50} step={50} onChange={(e) => setN(+e.target.value)} /></div>
              <div style={{ width: 120 }}><label>Superpixels K</label><input type="number" value={k} min={10} step={10} onChange={(e) => setK(+e.target.value)} /></div>
              <div style={{ width: 120 }}><label>Top regions</label><input type="number" value={top} min={1} max={30} onChange={(e) => setTop(+e.target.value)} /></div>
              <button className="btn" disabled={busy} onClick={rerun}>{busy ? "Running LIME…" : hasLime ? "Re-run LIME" : "Run LIME"}</button>
            </div>
            <p className="muted small" style={{ marginBottom: 0 }}>Larger N and K give a more stable explanation at higher latency (default N=1000, K=100, top 8).</p>
          </Card>
          {ev && (
            <Card title={`LIME explanation — ${ev.timestamp} · ${ev.category} (estimated)`}>
              <div className="panels three">
                <div className="panel"><div className="cap">Original Frame</div><img src={imgUrl(ev.id, "original", bust)} /></div>
                <div className="panel"><div className="cap">LIME Heatmap (red = high importance)</div>{hasLime ? <img src={imgUrl(ev.id, "lime_heatmap", bust)} /> : <div className="empty">Not generated</div>}</div>
                <div className="panel"><div className="cap">Superpixels</div>{hasLime ? <img src={imgUrl(ev.id, "superpixels", bust)} /> : <div className="empty">Not generated</div>}</div>
              </div>
              <div className="grid g2">
                <div>
                  <h2>Top Contributing Regions</h2>
                  {lime?.top?.length ? (
                    <table>
                      <thead><tr><th>#</th><th>Region</th><th>Location</th><th>Importance</th></tr></thead>
                      <tbody>{lime.top.map((r: any) => (
                        <tr key={r.rank}><td>{r.rank}</td><td>{r.region}</td><td>{r.location}</td><td>{r.importance.toFixed(2)}</td></tr>))}</tbody>
                    </table>
                  ) : <div className="muted">No positive-importance regions.</div>}
                  <p className="muted small">Importance is normalised over the listed regions (surrogate coefficients of the local linear model).</p>
                </div>
                <div>
                  <h2>Zoomed view</h2>
                  <div className="panel"><img src={imgUrl(ev.id, "zoom", bust)} style={{ maxHeight: 220, objectFit: "contain" }} /></div>
                  {lime && <div className="kv" style={{ marginTop: 12 }}>
                    <span className="k">Local R²</span><span>{lime.r2}</span>
                    <span className="k">Latency</span><span>{lime.latency}s</span>
                    <span className="k">Samples / segments</span><span>{lime.n_samples} / {lime.n_segments}</span>
                  </div>}
                </div>
              </div>
              <div className="expl" style={{ marginTop: 14 }}>
                LIME explains the <b>anomaly score</b>, not the network. A highlighted region means that masking it changes the reconstruction error locally; it does not prove why the behaviour is abnormal. Highlights can fall on background or edges when error is caused by illumination or noise. Use it to decide where to look.
              </div>
            </Card>
          )}
        </>
      )}
    </>
  );
}
