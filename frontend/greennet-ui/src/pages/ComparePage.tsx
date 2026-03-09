import { useEffect, useMemo, useState } from "react";
import ChartCard from "../components/ChartCard";
import { ErrorNotice, InfoNotice, LoadingNotice } from "../components/StatusState";
import { getRunPerStep, getRunSummary, listRuns, startRun } from "../lib/api";
import { chartRows, compareSummary, fmt, formatPolicyLabel, inferPolicy, normalizePerStep } from "../lib/data";
import { isDemoRunId } from "../lib/demo";
import type { PerStepRow, RunOverallSummary, RunSummary } from "../lib/types";

const policyColors: Record<string, string> = {
  baseline: "#5dc8ff",
  noop: "#f7bf5e",
  ppo: "#55ffaa",
};
const POLICY_LIST = ["baseline", "noop", "ppo"] as const;

export default function ComparePage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(300);
  const [scenario, setScenario] = useState("normal");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [rowsByPolicy, setRowsByPolicy] = useState<Record<string, PerStepRow[]>>({});
  const [summariesByPolicy, setSummariesByPolicy] = useState<Record<string, RunOverallSummary | null>>({});

  const demoMode = runs.length > 0 && runs.every((run) => isDemoRunId(run.run_id));

  useEffect(() => {
    let alive = true;

    async function loadRuns() {
      try {
        const runItems = await listRuns();
        if (alive) {
          setRuns(runItems);
        }
      } catch {
        // Non-blocking in compare mode.
      }
    }

    void loadRuns();
    return () => {
      alive = false;
    };
  }, []);

  async function resolveRunId(policy: string): Promise<string | null> {
    try {
      const started = await startRun({ policy, scenario, seed, steps });
      return started.run_id;
    } catch {
      const fallbackRun = runs.find(
        (run) =>
          inferPolicy(run) === policy &&
          (run.scenario ?? "").toLowerCase() === scenario.toLowerCase() &&
          run.seed === seed,
      );
      return fallbackRun?.run_id ?? null;
    }
  }

  async function handleRunComparison(): Promise<void> {
    setLoading(true);
    setError("");

    try {
      const entries = await Promise.all(
        POLICY_LIST.map(async (policy) => {
          const runId = await resolveRunId(policy);
          if (!runId) {
            throw new Error(`No ${formatPolicyLabel(policy)} run is available for scenario=${scenario}, seed=${seed}`);
          }

          const [rowsRaw, summary] = await Promise.all([getRunPerStep(runId), getRunSummary(runId)]);
          return [policy, { rows: normalizePerStep(rowsRaw), summary }] as const;
        }),
      );

      setRowsByPolicy(Object.fromEntries(entries.map(([policy, value]) => [policy, value.rows])));
      setSummariesByPolicy(Object.fromEntries(entries.map(([policy, value]) => [policy, value.summary])));
    } catch (apiError) {
      setError(apiError instanceof Error ? apiError.message : "Comparison run failed");
    } finally {
      setLoading(false);
    }
  }

  const summaries = useMemo(
    () =>
      POLICY_LIST.map((policy) => ({
        policy,
        summary: compareSummary(rowsByPolicy[policy] ?? [], summariesByPolicy[policy]),
      })),
    [rowsByPolicy, summariesByPolicy],
  );

  const mergedData = useMemo(() => {
    const rowMap = new Map<number, Record<string, number>>();

    for (const policy of POLICY_LIST) {
      const rows = chartRows(rowsByPolicy[policy] ?? []);
      for (const row of rows) {
        const t = Number(row.t);
        const existing = rowMap.get(t) ?? { t };
        existing[`${policy}_energy`] = Number(row.energy_kwh);
        existing[`${policy}_dropped`] = Number(row.dropped);
        rowMap.set(t, existing);
      }
    }

    return [...rowMap.values()].sort((a, b) => a.t - b.t);
  }, [rowsByPolicy]);

  return (
    <div className="page compare-page">
      <section className="page-title-row">
        <div>
          <p className="page-eyebrow">Policy Comparison</p>
          <h1>Baseline vs No-Op vs PPO under same seed and steps</h1>
        </div>
      </section>

      <section className="glass-card compare-controls">
        <label>
          Scenario
          <select value={scenario} onChange={(event) => setScenario(event.target.value)}>
            <option value="normal">Normal</option>
            <option value="burst">Burst</option>
            <option value="hotspot">Hotspot</option>
          </select>
        </label>

        <label>
          Seed
          <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
        </label>

        <label>
          Steps
          <input type="number" min={1} value={steps} onChange={(event) => setSteps(Number(event.target.value))} />
        </label>

        <button className="btn-primary" onClick={handleRunComparison} disabled={loading}>
          {loading ? "Running..." : "Run Comparison"}
        </button>
      </section>

      {loading ? <LoadingNotice title="Comparing policies" description="Running or loading policy runs." /> : null}
      {error ? <ErrorNotice title="Comparison Error" description={error} /> : null}
      {demoMode ? (
        <InfoNotice
          title="Demo Data Mode"
          description="Backend runs were not found, so comparison uses generated policy traces."
        />
      ) : null}

      <section className="glass-card table-card">
        <div className="card-heading">
          <p>Summary</p>
          <h3>Policy Metrics</h3>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Policy</th>
                <th>energy_kwh_total</th>
                <th>carbon_g_total</th>
                <th>path_latency_ms_mean</th>
                <th>packets_sent_total</th>
                <th>dropped_total</th>
                <th>delivered_total</th>
                <th>active_ratio_mean (%)</th>
                <th>reward_total</th>
              </tr>
            </thead>
            <tbody>
              {summaries.map(({ policy, summary }) => (
                <tr key={policy}>
                  <td>{formatPolicyLabel(policy)}</td>
                  <td>{fmt(summary.energy_kwh, 3)}</td>
                  <td>{fmt(summary.carbon_g)}</td>
                  <td>{fmt(summary.avg_delay_ms)}</td>
                  <td>{fmt(summary.packets_sent)}</td>
                  <td>{fmt(summary.dropped)}</td>
                  <td>{fmt(summary.delivered)}</td>
                  <td>{fmt(summary.active_ratio, 1)}</td>
                  <td>{fmt(summary.reward)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="compare-charts">
        <ChartCard
          title="Step Energy Overlay"
          subtitle="kWh per step"
          data={mergedData}
          lines={POLICY_LIST.map((policy) => ({
            dataKey: `${policy}_energy`,
            label: formatPolicyLabel(policy),
            color: policyColors[policy],
          }))}
        />

        <ChartCard
          title="Dropped Packets Overlay"
          subtitle="packets"
          data={mergedData}
          lines={POLICY_LIST.map((policy) => ({
            dataKey: `${policy}_dropped`,
            label: formatPolicyLabel(policy),
            color: policyColors[policy],
          }))}
        />
      </section>
    </div>
  );
}
