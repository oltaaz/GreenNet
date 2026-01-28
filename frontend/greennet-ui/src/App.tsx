import { useEffect, useState } from "react";
import { API, type RunSummary, type PerStepRow } from "./lib/api";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from "recharts";
import "./app.css";

export default function App() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRun, setSelectedRun] = useState<RunSummary | null>(null);
  const [rows, setRows] = useState<PerStepRow[]>([]);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    API.listRuns()
      .then(setRuns)
      .catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (!selectedRun) return;
    API.getRunPerStep(selectedRun.run_id)
      .then(setRows)
      .catch((e) => setErr(String(e)));
  }, [selectedRun]);

  return (
    <div className="page">
      <header className="topbar">
        <div>
          <h1>GreenNet Dashboard</h1>
          <p className="muted">Compare energy, drops, reward across runs</p>
        </div>
      </header>

      {err && <div className="error">{err}</div>}

      <div className="grid">
        <section className="card">
          <h2>Runs</h2>
          <div className="list">
            {runs.map((r) => (
              <button
                key={r.run_id}
                className={`row ${selectedRun?.run_id === r.run_id ? "active" : ""}`}
                onClick={() => setSelectedRun(r)}
              >
                <div className="rowTitle">{r.run_id}</div>
                <div className="rowMeta">
                  {r.policy ?? "?"} · {r.scenario ?? "?"} · seed {r.topology_seed ?? "?"}
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className="card">
          <h2>Run detail</h2>
          {!selectedRun ? (
            <p className="muted">Select a run to see charts.</p>
          ) : (
            <>
              <div className="muted" style={{ marginBottom: 12 }}>
                <b>{selectedRun.run_id}</b> — {selectedRun.policy} / {selectedRun.scenario}
              </div>

              <div className="chartBlock">
                <h3>Energy</h3>
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={rows}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="t" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="energy" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="chartBlock">
                <h3>Dropped</h3>
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={rows}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="t" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="dropped" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {"reward" in (rows[0] ?? {}) && (
                <div className="chartBlock">
                  <h3>Reward</h3>
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={rows}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="t" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="reward" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  );
}