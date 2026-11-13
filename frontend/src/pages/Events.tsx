import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Download } from "lucide-react";
import { useApi } from "../api";
import { Card, Err, EventDetail, PageHead } from "../components/ui";

export default function Events() {
  const [params] = useSearchParams();
  const analysis = params.get("analysis");
  const [sev, setSev] = useState("");
  const [cat, setCat] = useState("");
  const { data: cats } = useApi<string[]>("/categories");
  const q = useMemo(() => {
    const p = new URLSearchParams();
    if (sev) p.set("severity", sev);
    if (cat) p.set("category", cat);
    if (analysis) p.set("analysis_id", analysis);
    return "/events?" + p.toString();
  }, [sev, cat, analysis]);
  const { data: events, error } = useApi<any[]>(q);
  const [sel, setSel] = useState<any>(null);
  return (
    <>
      <PageHead title="Anomaly Events" sub="All flagged events across analysed videos. Categories are estimates from motion cues." />
      <Err msg={error} />
      <Card right={
        <div className="row">
          <a className="btn ghost sm" href="/api/events/export?format=csv"><Download size={14} /> CSV</a>
          <a className="btn ghost sm" href="/api/events/export?format=json"><Download size={14} /> JSON</a>
        </div>}>
        <div className="row" style={{ marginBottom: 12 }}>
          <div style={{ width: 180 }}><label>Severity</label>
            <select value={sev} onChange={(e) => setSev(e.target.value)}><option value="">All</option><option>High</option><option>Medium</option><option>Low</option></select></div>
          <div style={{ width: 260 }}><label>Estimated category</label>
            <select value={cat} onChange={(e) => setCat(e.target.value)}><option value="">All</option>{cats?.map((c) => <option key={c}>{c}</option>)}<option>Unclassified anomaly</option></select></div>
          {analysis && <span className="badge Low">Filtered to one video · <Link style={{ color: "#fff" }} to="/events">clear</Link></span>}
        </div>
        {!events || events.length === 0 ? <div className="empty">No events found.</div> : (
          <table>
            <thead><tr><th>Video</th><th>Time</th><th>Duration</th><th>Score</th><th>Severity</th><th>Estimated type</th><th></th></tr></thead>
            <tbody>
              {events.map((e) => (
                <tr key={e.id} className={`click ${sel?.id === e.id ? "sel" : ""}`} onClick={() => setSel(e)}>
                  <td>{e.video_name}</td><td>{e.timestamp}</td><td>{e.duration}s</td><td>{e.anomaly_score.toFixed(2)}</td>
                  <td><span className={`badge ${e.severity}`}>{e.severity}</span></td><td>{e.category}</td>
                  <td><Link to={`/explainability?event=${e.id}`} onClick={(x) => x.stopPropagation()}>Explain →</Link></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
      {sel && <EventDetail ev={sel} />}
    </>
  );
}
