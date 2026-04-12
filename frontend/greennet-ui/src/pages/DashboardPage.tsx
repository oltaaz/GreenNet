import { useEffect, useMemo, useState } from "react";
import ChartCard from "../components/ChartCard";
import KpiCard from "../components/KpiCard";
import OfficialResultCard from "../components/OfficialResultCard";
import RunControls from "../components/RunControls";
import { ErrorNotice, InfoNotice, LoadingNotice } from "../components/StatusState";
import TopologyPanel from "../components/TopologyPanel";
import {
  getLinkState,
  getOfficialLockedResults,
  getRunPerStep,
  getRunSummary,
  getTopology,
  listRuns,
  startRun,
} from "../lib/api";
import { isDemoRunId } from "../lib/demo";
import {
  chartRows,
  fallbackTopology,
  formatPolicyLabel,
  formatScenarioLabel,
  fmt,
  inferPolicy,
  kpiFromOverall,
  latestMatchingRunByPolicy,
  latestRunByPolicy,
  linkStateFromRatio,
  normalizePerStep,
  toMetrics,
} from "../lib/data";
import type {
  LinkStateMap,
  OfficialLockedResult,
  PerStepRow,
  RunOverallSummary,
  RunSummary,
  TopologyData,
} from "../lib/types";

type PolicySeries = Record<string, PerStepRow[]>;

const REFERENCE_POLICY_STYLES: Record<string, { label: string; color: string }> = {
  all_on: { label: "All-On", color: "#f7bf5e" },
  heuristic: { label: "Heuristic", color: "#5dc8ff" },
  ppo: { label: "PPO", color: "#00f2bf" },
};

function upsertRun(runs: RunSummary[], nextRun: RunSummary): RunSummary[] {
  const existing = runs.findIndex((run) => run.run_id === nextRun.run_id);
  if (existing === 0) {
    return runs;
  }
  if (existing > 0) {
    return [nextRun, ...runs.filter((run) => run.run_id !== nextRun.run_id)];
  }
  return [nextRun, ...runs];
}

export default function DashboardPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [rows, setRows] = useState<PerStepRow[]>([]);
  const [overallSummary, setOverallSummary] = useState<RunOverallSummary | null>(null);
  const [officialResults, setOfficialResults] = useState<OfficialLockedResult[]>([]);
  const [topology, setTopology] = useState<TopologyData>(fallbackTopology("dashboard"));
  const [policySeries, setPolicySeries] = useState<PolicySeries>({});
  const [linkState, setLinkState] = useState<LinkStateMap | null>(null);

  const [loadingRuns, setLoadingRuns] = useState(false);
  const [loadingData, setLoadingData] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string>("");

  const [policy, setPolicy] = useState("ppo");
  const [scenario, setScenario] = useState("normal");
  const [seed, setSeed] = useState("");
  const [steps, setSteps] = useState("");
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
  }, []);

  useEffect(() => {
    const selectedRun = runs.find((run) => run.run_id === selectedRunId);
    if (!selectedRun) {
      return;
    }

    setPolicy(selectedRun.policy ? String(selectedRun.policy).toLowerCase() : inferPolicy(selectedRun));
    setScenario(selectedRun.scenario ? String(selectedRun.scenario).toLowerCase() : "normal");
    setSeed(selectedRun.seed != null ? String(selectedRun.seed) : selectedRun.topology_seed != null ? String(selectedRun.topology_seed) : "");
    setSteps(selectedRun.max_steps != null ? String(selectedRun.max_steps) : "");
  }, [runs, selectedRunId]);

  useEffect(() => {
    if (!selectedRunId) {
      setRows([]);
      setOverallSummary(null);
      setTopology(fallbackTopology("dashboard-empty"));
      return;
    }

    let alive = true;

    async function loadRunData() {
      setLoadingData(true);
      setError("");

      try {
        const [stepRows, topo, summary] = await Promise.all([
          getRunPerStep(selectedRunId),
          getTopology(selectedRunId),
          getRunSummary(selectedRunId),
        ]);
        if (!alive) {
          return;
        }

        const normalized = normalizePerStep(stepRows);
        setRows(normalized);
        setOverallSummary(summary);
        setTopology(topo ?? fallbackTopology(selectedRunId));
        setTopologyStepIndex(Math.max(0, normalized.length - 1));
      } catch (apiError) {
        if (alive) {
          setError(apiError instanceof Error ? apiError.message : "Failed to load run data");
          setRows([]);
          setOverallSummary(null);
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

    async function loadOfficialResults() {
      try {
        const items = await getOfficialLockedResults(["burst", "hotspot"]);
        if (alive) {
          setOfficialResults(items);
        }
      } catch {
        if (alive) {
          setOfficialResults([]);
        }
      }
    }

    void loadOfficialResults();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    let alive = true;

    async function loadPolicySeries() {
      const policies = ["all_on", "heuristic", "ppo"];
      const result: PolicySeries = {};
      const selectedRun = runs.find((run) => run.run_id === selectedRunId) ?? null;
      const criteria = selectedRun
        ? {
            scenario: selectedRun.scenario ?? null,
            seed: selectedRun.seed ?? null,
            max_steps: selectedRun.max_steps ?? null,
            topology_name: selectedRun.topology_name ?? null,
            topology_seed: selectedRun.topology_seed ?? null,
            tag: selectedRun.tag ?? null,
          }
        : {};

      await Promise.all(
        policies.map(async (policyName) => {
          const run = selectedRun ? latestMatchingRunByPolicy(runs, policyName, criteria) : latestRunByPolicy(runs, policyName);
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
  }, [runs, selectedRunId]);

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

  const selectedRun = runs.find((run) => run.run_id === selectedRunId);
  const demoMode = selectedRun ? isDemoRunId(selectedRun.run_id) : false;
  const activeOfficialScenario = (selectedRun?.scenario ?? "").toLowerCase();
  const highlightedOfficialResult =
    officialResults.find((item) => item.scenario === activeOfficialScenario) ?? null;

  const kpis = useMemo(() => kpiFromOverall(overallSummary), [overallSummary]);
  const timeData = useMemo(() => chartRows(rows), [rows]);
  const energyLines = useMemo(() => {
    const lines: Array<{ dataKey: string; label: string; color: string; dashed?: boolean }> = [
      { dataKey: "selected", label: "Selected Run", color: "#55ffaa" },
    ];

    for (const [policyName, style] of Object.entries(REFERENCE_POLICY_STYLES)) {
      if ((policySeries[policyName] ?? []).length === 0) {
        continue;
      }
      lines.push({
        dataKey: policyName,
        label: `${style.label} (latest)`,
        color: style.color,
        dashed: true,
      });
    }

    return lines;
  }, [policySeries]);

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
    const seedValue = seed === "" ? Number.NaN : Number(seed);
    const stepsValue = steps === "" ? Number.NaN : Number(steps);

    if (!Number.isFinite(seedValue) || !Number.isFinite(stepsValue)) {
      setError("Enter both seed and steps before starting a run");
      return;
    }

    setRunning(true);
    setError("");

    try {
      const started = await startRun({ policy, scenario, seed: seedValue, steps: stepsValue });
      const refreshed = await listRuns();
      const startedRun: RunSummary = {
        run_id: started.run_id,
        policy,
        scenario,
        seed: seedValue,
        max_steps: stepsValue,
        tag: "dashboard",
      };

      setRuns(upsertRun(refreshed, startedRun));
      setSelectedRunId(started.run_id);
    } catch (apiError) {
      setError(apiError instanceof Error ? apiError.message : "Unable to start run on backend");
    } finally {
      setRunning(false);
    }
  }

  function handleReset(): void {
    const selectedRun = runs.find((run) => run.run_id === selectedRunId);
    setPolicy(selectedRun?.policy ? String(selectedRun.policy).toLowerCase() : "ppo");
    setScenario(selectedRun?.scenario ? String(selectedRun.scenario).toLowerCase() : "normal");
    setSeed(
      selectedRun?.seed != null
        ? String(selectedRun.seed)
        : selectedRun?.topology_seed != null
          ? String(selectedRun.topology_seed)
          : "",
    );
    setSteps(selectedRun?.max_steps != null ? String(selectedRun.max_steps) : "");
    setTopologyStepIndex(Math.max(0, rows.length - 1));
  }

  return (
    <div className="page dashboard-page">
      <section className="page-title-row">
        <div>
          <p className="page-eyebrow">Current Run Overview</p>
          <h1>Inspect the selected run with the same energy, QoS, and activity labels used in reporting</h1>
        </div>
        {selectedRun ? (
          <div className="meta-chip">
            <span>Run</span>
            <strong>{selectedRun.run_id}</strong>
            <small>{formatPolicyLabel(inferPolicy(selectedRun))}</small>
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
      {highlightedOfficialResult ? (
        <InfoNotice
          title="Scenario Validation Available"
          description={`An official locked validation bundle is available for ${formatScenarioLabel(highlightedOfficialResult.scenario)}. The card below keeps the acceptance result visible while this page stays focused on the selected live run.`}
        />
      ) : null}

      {selectedRun ? (
        <section className="glass-card run-overview-card">
          <div className="card-heading">
            <p>Current Run</p>
            <h3>Selection details</h3>
          </div>
          <div className="report-stat-grid compact">
            <article className="report-stat">
              <span>Policy</span>
              <strong>{formatPolicyLabel(inferPolicy(selectedRun))}</strong>
            </article>
            <article className="report-stat">
              <span>Scenario</span>
              <strong>{formatScenarioLabel(selectedRun.scenario ?? scenario)}</strong>
            </article>
            <article className="report-stat">
              <span>Seed</span>
              <strong>{selectedRun.seed ?? selectedRun.topology_seed ?? "-"}</strong>
            </article>
            <article className="report-stat">
              <span>Topology</span>
              <strong>{selectedRun.topology_name ?? "seeded-random"}</strong>
            </article>
            <article className="report-stat">
              <span>Steps</span>
              <strong>{selectedRun.max_steps ?? overallSummary?.steps_mean ?? "-"}</strong>
            </article>
          </div>
        </section>
      ) : null}

      {kpis.length > 0 ? (
        <section className="kpi-grid">
          {kpis.map((metric) => (
            <KpiCard key={metric.label} metric={metric} />
          ))}
        </section>
      ) : null}
      {officialResults.length > 0 ? (
        <section className="official-results-section">
          <div className="card-heading">
            <p>Official Locked Results</p>
            <h3>Burst and hotspot acceptance bundles</h3>
          </div>

          <div className="official-results-grid">
            {officialResults.map((result) => (
              <OfficialResultCard
                key={`${result.scenario}-${result.bundle_id}`}
                result={result}
                active={result.scenario === activeOfficialScenario}
              />
            ))}
          </div>
        </section>
      ) : null}

      <section className="dashboard-grid">
        <div className="left-stack">
          {loadingData ? <LoadingNotice title="Loading run metrics" description="Reading per-step data." /> : null}

          <ChartCard
            title="Energy by Step"
            subtitle="kWh per step"
            data={energyData.length > 0 ? energyData : timeData}
            lines={energyLines}
          />

          <ChartCard
            title="Path Latency by Step"
            subtitle="routing latency in ms"
            data={timeData}
            lines={[{ dataKey: "avg_path_latency_ms", label: "Path Latency", color: "#66d2ff" }]}
          />

          <ChartCard
            title="Dropped Traffic by Step"
            subtitle="packets per step"
            data={timeData}
            lines={[{ dataKey: "dropped", label: "Dropped", color: "#ff7f96" }]}
          />

          <ChartCard
            title="Active Link Ratio by Step"
            subtitle="average share of controllable links that are active"
            data={timeData}
            lines={[{ dataKey: "active_ratio", label: "Active Link Ratio", color: "#50f7b7" }]}
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
