import { NavLink, Route, Routes } from "react-router-dom";
import { BarChart3, Brain, Gauge, Home, ListChecks, ScanSearch, Settings as Cog, Upload } from "lucide-react";
import { useApi } from "./api";
import Dashboard from "./pages/Dashboard";
import Analyze from "./pages/Analyze";
import Events from "./pages/Events";
import Explainability from "./pages/Explainability";
import ModelPage from "./pages/ModelPage";
import Evaluation from "./pages/Evaluation";
import SettingsPage from "./pages/SettingsPage";

const nav = [
  ["/", "Dashboard", Home],
  ["/analyze", "Analyze Video", Upload],
  ["/events", "Anomaly Events", ListChecks],
  ["/explainability", "Explainability", ScanSearch],
  ["/model", "Model", Brain],
  ["/evaluation", "Evaluation", BarChart3],
  ["/settings", "Settings", Cog],
] as const;

export default function App() {
  const { data: status } = useApi("/status");
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          AI Surveillance Dashboard
          <small>WalkwayGuard AI · STAE + LIME</small>
        </div>
        <nav className="nav">
          {nav.map(([to, label, Icon]) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => (isActive ? "active" : "")}>
              <Icon size={18} /> {label}
            </NavLink>
          ))}
        </nav>
        <div className="side-foot">
          <Gauge size={13} style={{ verticalAlign: -2 }} /> {status ? `${status.device.toUpperCase()} · model ${status.model_ready ? "ready" : "not trained"}` : "connecting…"}
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/events" element={<Events />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/model" element={<ModelPage />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  );
}
