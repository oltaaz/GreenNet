import { useEffect, useMemo, useState } from "react";
import ChartCard from "../components/ChartCard";
import KpiCard from "../components/KpiCard";
import RunControls from "../components/RunControls";
import { ErrorNotice, InfoNotice, LoadingNotice } from "../components/StatusState";
import TopologyPanel from "../components/TopologyPanel";
import { getLinkState, getRunPerStep, getTopology, listRuns, startRun } from "../lib/api";
import { isDemoRunId } from "../lib/demo";
import {
  chartRows,
  fallbackTopology,
  fmt,
  inferPolicy,
  kpiFromRows,
  latestRunByPolicy,
  linkStateFromRatio,
  normalizePerStep,
  toMetrics,
} from "../lib/data";
import type { LinkStateMap, PerStepRow, RunSummary, TopologyData } from "../lib/types";

type PolicySeries = Record<string, PerStepRow[]>;

export default function DashboardPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [rows, setRows] = useState<PerStepRow[]>([]);
  const [topology, setTopology] = useState<TopologyData>(fallbackTopology("dashboard"));
  const [policySeries, setPolicySeries] = useState<PolicySeries>({});
  const [linkState, setLinkState] = useState<LinkStateMap | null>(null);

  const [loadingRuns, setLoadingRuns] = useState(false);
  const [loadingData, setLoadingData] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string>("");

  const [policy, setPolicy] = useState("ppo");
  const [scenario, setScenario] = useState("normal");
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(300);
  const [topologyStepIndex, setTopologyStepIndex] = useState(0);

  useEffect(() => {
    let alive = true;

    async function loadRuns() {
      setLoadingRuns(true);
      setError("");

      try {
        const runItems = await listRuns();
        if (!alive) {
          return;
        }

        setRuns(runItems);
        if (!selectedRunId && runItems.length > 0) {
          setSelectedRunId(runItems[0].run_id);
        } else if (selectedRunId && !runItems.find((run) => run.run_id === selectedRunId)) {
          setSelectedRunId(runItems[0]?.run_id ?? "");
        }
      } catch (apiError) {
        if (alive) {
          setError(apiError instanceof Error ? apiError.message : "Failed to load runs");
        }
      } finally {
        if (alive) {
          setLoadingRuns(false);
        }
      }
    }

    void loadRuns();

    return () => {
      alive = false;
    };
  }, [selectedRunId]);

  useEffect(() => {
    if (!selectedRunId) {
      setRows([]);
      setTopology(fallbackTopology("dashboard-empty"));
      return;
    }

    let alive = true;

    async function loadRunData() {
      setLoadingData(true);
      setError("");

      try {
        const [stepRows, topo] = await Promise.all([getRunPerStep(selectedRunId), getTopology(selectedRunId)]);
        if (!alive) {
          return;
        }

        const normalized = normalizePerStep(stepRows);
        setRows(normalized);
        setTopology(topo ?? fallbackTopology(selectedRunId));
        setTopologyStepIndex(Math.max(0, normalized.length - 1));
      } catch (apiError) {
        if (alive) {
          setError(apiError instanceof Error ? apiError.message : "Failed to load run data");
          setRows([]);
          setTopology(fallbackTopology(selectedRunId));
        }
      } finally {
        if (alive) {
          setLoadingData(false);
        }
      }
    }

    void loadRunData();

    return () => {
      alive = false;
    };
  }, [selectedRunId]);

  useEffect(() => {
    let alive = true;

    async function loadPolicySeries() {
      const policies = ["baseline", "ppo"];
      const result: PolicySeries = {};

      await Promise.all(
        policies.map(async (policyName) => {
          const run = latestRunByPolicy(runs, policyName);
          if (!run) {
            return;
          }

          try {
            const data = normalizePerStep(await getRunPerStep(run.run_id));
            if (data.length > 0) {
              result[policyName] = data;
            }
          } catch {
            // Optional comparison series. Ignore failures.
          }
        }),
      );

      if (alive) {
        setPolicySeries(result);
      }
    }

    if (runs.length > 0) {
      void loadPolicySeries();
    }

    return () => {
      alive = false;
    };
  }, [runs]);

  useEffect(() => {
    if (!selectedRunId || rows.length === 0) {
      setLinkState(null);
      return;
    }

    const row = rows[topologyStepIndex] ?? rows[rows.length - 1];
    let alive = true;

    async function loadLinkState() {
      try {
        const state = await getLinkState(selectedRunId, row.t);
        if (alive) {
          setLinkState(state?.links_on ?? null);
        }
      } catch {
        if (alive) {
          setLinkState(null);
        }
      }
    }

    void loadLinkState();
    return () => {
      alive = false;
    };
  }, [rows, selectedRunId, topologyStepIndex]);

  const currentRow = rows[topologyStepIndex] ?? rows[rows.length - 1] ?? null;
  const previousRow =
    rows[Math.max(0, Math.min(topologyStepIndex, rows.length - 1) - 1)] ??
    rows[rows.length - 1] ??
    null;

  const activeLinkState = useMemo(() => {
    if (linkState) {
      return linkState;
    }
    if (!currentRow) {
      return linkStateFromRatio(topology.edges, 1, 0);
    }
    return linkStateFromRatio(topology.edges, currentRow.active_ratio ?? 1, currentRow.t);
  }, [currentRow, linkState, topology.edges]);

  const previousLinkState = useMemo(() => {
    if (!previousRow) {
      return linkStateFromRatio(topology.edges, 1, 0);
    }
    return linkStateFromRatio(topology.edges, previousRow.active_ratio ?? 1, previousRow.t);
  }, [previousRow, topology.edges]);

  const kpis = useMemo(() => kpiFromRows(rows), [rows]);
  const timeData = useMemo(() => chartRows(rows), [rows]);

  const energyData = useMemo(() => {
    const merged = new Map<number, Record<string, number>>();

    for (const row of timeData) {
      const t = Number(row.t);
      merged.set(t, { t, selected: Number(row.energy_kwh) });
    }

    for (const [policyName, policyRows] of Object.entries(policySeries)) {
      for (const row of chartRows(policyRows)) {
        const t = Number(row.t);
        const existing = merged.get(t) ?? { t, selected: Number.NaN };
        existing[policyName] = Number(row.energy_kwh);
        merged.set(t, existing);
      }
    }

    return [...merged.values()].sort((a, b) => a.t - b.t);
  }, [policySeries, timeData]);

  async function handleRun(): Promise<void> {
    setRunning(true);
    setError("");

    try {
      const started = await startRun({ policy, scenario, seed, steps });
      setSelectedRunId(started.run_id);
      const refreshed = await listRuns();
      setRuns(refreshed);
    } catch (apiError) {
      setError(apiError instanceof Error ? apiError.message : "Unable to start run on backend");
    } finally {
      setRunning(false);
    }
  }

  function handleReset(): void {
    setPolicy("ppo");
    setScenario("normal");
    setSeed(42);
    setSteps(300);
    setTopologyStepIndex(Math.max(0, rows.length - 1));
  }

  const selectedRun = runs.find((run) => run.run_id === selectedRunId);
  const demoMode = selectedRun ? isDemoRunId(selectedRun.run_id) : false;

  return (
    <div className="page dashboard-page">
      <section className="page-title-row">
        <div>
          <p className="page-eyebrow">Network Simulation Dashboard</p>
          <h1>Analyze energy, delay, drops, and link activity per step</h1>
        </div>
        {selectedRun ? (
          <div className="meta-chip">
            <span>Run</span>
            <strong>{selectedRun.run_id}</strong>
            <small>{inferPolicy(selectedRun)}</small>
          </div>
        ) : null}
      </section>

      {loadingRuns ? <LoadingNotice title="Loading runs" description="Fetching available simulation outputs." /> : null}
      {error ? <ErrorNotice title="Data Error" description={error} /> : null}
      {demoMode ? (
        <InfoNotice
          title="Demo Data Mode"
          description="Backend data is unavailable or empty. Showing generated simulation data so the dashboard remains fully interactive."
        />
      ) : null}

      {kpis.length > 0 ? (
        <section className="kpi-grid">
          {kpis.map((metric) => (
            <KpiCard key={metric.label} metric={metric} />
          ))}
        </section>
      ) : null}

      <section className="dashboard-grid">
        <div className="left-stack">
          {loadingData ? <LoadingNotice title="Loading run metrics" description="Reading per-step data." /> : null}

          <ChartCard
            title="Energy Consumption Over Time"
            subtitle="kWh"
            data={energyData.length > 0 ? energyData : timeData}
            lines={[
              { dataKey: "selected", label: "Selected Run", color: "#55ffaa" },
              { dataKey: "baseline", label: "Baseline", color: "#5dc8ff", dashed: true },
              { dataKey: "ppo", label: "PPO", color: "#00f2bf" },
            ]}
          />

          <ChartCard
            title="Delay Over Time"
            subtitle="ms"
            data={timeData}
            lines={[{ dataKey: "avg_delay_ms", label: "Avg Delay", color: "#66d2ff" }]}
          />

          <ChartCard
            title="Packet Drop Rate Over Time"
            subtitle="packets"
            data={timeData}
            lines={[{ dataKey: "dropped", label: "Dropped", color: "#ff7f96" }]}
          />

          <ChartCard
            title="Active Links Ratio Over Time"
            subtitle="%"
            data={timeData}
            lines={[{ dataKey: "active_ratio", label: "Active Ratio", color: "#50f7b7" }]}
            yAxisFormatter={(value) => `${fmt(value, 0)}%`}
          />
        </div>

        <aside className="right-stack">
          <RunControls
            policy={policy}
            scenario={scenario}
            seed={seed}
            steps={steps}
            loading={running}
            runId={selectedRunId}
            runs={runs}
            onPolicyChange={setPolicy}
            onScenarioChange={setScenario}
            onSeedChange={setSeed}
            onStepsChange={setSteps}
            onRun={handleRun}
            onReset={handleReset}
            onRunSelect={setSelectedRunId}
          />

          <section className="glass-card">
            <div className="card-heading compact">
              <p>Topology Step</p>
              <h3>
                t = {currentRow?.t ?? 0} | active {fmt((currentRow?.active_ratio ?? 1) * 100, 1)}%
              </h3>
            </div>
            <input
              type="range"
              min={0}
              max={Math.max(0, rows.length - 1)}
              value={Math.min(topologyStepIndex, Math.max(0, rows.length - 1))}
              onChange={(event) => setTopologyStepIndex(Number(event.target.value))}
              disabled={rows.length <= 1}
            />
          </section>

          <TopologyPanel
            topology={topology}
            linkState={activeLinkState}
            previousLinkState={previousLinkState}
            stepIndex={currentRow?.t ?? topologyStepIndex}
            metrics={currentRow ? toMetrics(currentRow) : undefined}
            previousMetrics={previousRow ? toMetrics(previousRow) : undefined}
            title="Network Topology"
          />
        </aside>
      </section>
    </div>
  );
}
