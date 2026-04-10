import { useEffect, useMemo, useState } from "react";
import ChartCard from "../components/ChartCard";
import OfficialResultCard from "../components/OfficialResultCard";
import StatusBadge from "../components/StatusBadge";
import { ErrorNotice, InfoNotice, LoadingNotice } from "../components/StatusState";
import {
  getFinalEvaluationReport,
  getOfficialLockedResults,
  getRunPerStep,
  getRunSummary,
  listRunsWithSource,
  startRun,
  type RunCatalogSource,
} from "../lib/api";
import { useBackendStatus } from "../hooks/useBackendStatus";
import {
  bestAiScenarioRows,
  chartRows,
  compareSummary,
  finalEvaluationRowsForScope,
  fmt,
  formatPolicyLabel,
  formatScenarioLabel,
  formatSignedValue,
  formatStatusLabel,
  inferPolicy,
  normalizePerStep,
  normalizePolicy,
  selectQosAcceptanceMissing,
  selectQosAcceptanceStatus,
  selectQosAcceptanceThresholds,
  selectStabilityMissing,
  selectStabilityStatus,
  statusTone,
} from "../lib/data";
import type {
  FinalEvaluationReport,
  FinalEvaluationSummaryRow,
  OfficialLockedResult,
  PerStepRow,
  RunOverallSummary,
  RunSummary,
} from "../lib/types";

const policyColors: Record<string, string> = {
  all_on: "#f7bf5e",
  heuristic: "#5dc8ff",
  ppo: "#55ffaa",
};
const POLICY_LIST = ["all_on", "heuristic", "ppo"] as const;

type PolicyKey = (typeof POLICY_LIST)[number];

function ReportStat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <article className="report-stat">
      <span>{label}</span>
      <strong>{value}</strong>
      {hint ? <small>{hint}</small> : null}
    </article>
  );
}

export default function ComparePage() {
  const { status: backendStatus, message: backendMessage } = useBackendStatus();
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(300);
  const [scenario, setScenario] = useState("normal");
  const [loadingComparison, setLoadingComparison] = useState(false);
  const [loadingReport, setLoadingReport] = useState(false);
  const [runCatalogSource, setRunCatalogSource] = useState<RunCatalogSource>("backend");
  const [comparisonError, setComparisonError] = useState("");
  const [reportError, setReportError] = useState("");
  const [rowsByPolicy, setRowsByPolicy] = useState<Record<string, PerStepRow[]>>({});
  const [summariesByPolicy, setSummariesByPolicy] = useState<Record<string, RunOverallSummary | null>>({});
  const [finalReport, setFinalReport] = useState<FinalEvaluationReport | null>(null);
  const [officialResults, setOfficialResults] = useState<OfficialLockedResult[]>([]);

  useEffect(() => {
    let alive = true;

    async function loadRuns() {
      try {
        const catalog = await listRunsWithSource();
        if (alive) {
          setRuns(catalog.runs);
          setRunCatalogSource(catalog.source);
        }
      } catch {
        if (alive) {
          setRuns([]);
          setRunCatalogSource("backend");
        }
      }
    }

    void loadRuns();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    let alive = true;

    async function loadReportingContext() {
      setLoadingReport(true);
      setReportError("");

      const [reportResult, officialResult] = await Promise.allSettled([
        getFinalEvaluationReport(),
        getOfficialLockedResults(["burst", "hotspot"]),
      ]);

      if (!alive) {
        return;
      }

      if (reportResult.status === "fulfilled") {
        setFinalReport(reportResult.value);
        if (!reportResult.value) {
          setReportError("Final evaluation report was not found on the backend.");
        }
      } else {
        setFinalReport(null);
        setReportError(reportResult.reason instanceof Error ? reportResult.reason.message : "Failed to load final evaluation report.");
      }

      if (officialResult.status === "fulfilled") {
        setOfficialResults(officialResult.value);
      } else {
        setOfficialResults([]);
      }

      setLoadingReport(false);
    }

    void loadReportingContext();
    return () => {
      alive = false;
    };
  }, []);

  async function resolveRunId(policy: PolicyKey): Promise<string | null> {
    const canonicalPolicy = normalizePolicy(policy);
    const existingRun = runs.find(
      (run) =>
        inferPolicy(run) === canonicalPolicy &&
        (run.scenario ?? "").toLowerCase() === scenario.toLowerCase() &&
        (run.seed ?? run.topology_seed) === seed,
    );

    if (existingRun) {
      return existingRun.run_id;
    }

    try {
      const started = await startRun({ policy: canonicalPolicy, scenario, seed, steps });
      return started.run_id;
    } catch {
      return null;
    }
  }

  async function handleRunComparison(): Promise<void> {
    if (!backendOnline) {
      setComparisonError(backendMessage);
      return;
    }

    setLoadingComparison(true);
    setComparisonError("");

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
      setComparisonError(apiError instanceof Error ? apiError.message : "Comparison run failed");
    } finally {
      setLoadingComparison(false);
    }
  }

  const liveComparisonReady = useMemo(
    () => POLICY_LIST.some((policy) => (rowsByPolicy[policy] ?? []).length > 0),
    [rowsByPolicy],
  );

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

  const overallRows = useMemo(() => {
    const rows = finalEvaluationRowsForScope(finalReport, "overall");
    return [...rows].sort((left, right) => {
      const leftRank = left.is_primary_baseline ? 0 : left.is_best_ai_policy_for_scope ? 1 : 2;
      const rightRank = right.is_primary_baseline ? 0 : right.is_best_ai_policy_for_scope ? 1 : 2;
      return leftRank - rightRank || left.policy.localeCompare(right.policy);
    });
  }, [finalReport]);

  const scenarioRows = useMemo(() => bestAiScenarioRows(finalReport), [finalReport]);
  const bestAiOverall = finalReport?.best_ai_policy ?? overallRows.find((row) => row.is_best_ai_policy_for_scope) ?? null;
  const bestBaselineOverall =
    overallRows.find((row) => row.is_primary_baseline) ??
    finalReport?.best_policy ??
    overallRows.find((row) => row.is_best_policy_for_scope) ??
    null;
  const reportQosThresholds = selectQosAcceptanceThresholds(finalReport);
  const hasCentralQosThresholds = Boolean(finalReport?.qos_thresholds);
  const bestAiQosStatus = selectQosAcceptanceStatus(bestAiOverall) || finalReport?.qos_acceptance_status || "";
  const bestAiQosMissing = selectQosAcceptanceMissing(bestAiOverall) || finalReport?.qos_acceptance_missing || "";
  const bestAiStabilityStatus = selectStabilityStatus(bestAiOverall) || finalReport?.overall_stability_status || "";
  const bestAiStabilityMissing = selectStabilityMissing(bestAiOverall) || "";

  const generatedAtLabel = useMemo(() => {
    if (!finalReport?.generated_at_utc) {
      return null;
    }

    try {
      return new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(new Date(finalReport.generated_at_utc));
    } catch {
      return finalReport.generated_at_utc;
    }
  }, [finalReport]);

  const backendOnline = backendStatus === "online";

  const headline = useMemo(() => {
    if (!bestAiOverall || !bestBaselineOverall) {
      return null;
    }

    return `${formatPolicyLabel(bestAiOverall.policy)} is the best AI policy against ${formatPolicyLabel(
      bestBaselineOverall.policy,
    )} in the final evaluation.`;
  }, [bestAiOverall, bestBaselineOverall]);

  return (
    <div className="page compare-page">
      <section className="page-title-row">
        <div>
          <p className="page-eyebrow">Reporting</p>
          <h1>Unified results for the traditional baseline, heuristic baseline, and AI policy story</h1>
        </div>
        {finalReport ? (
          <div className="meta-chip">
            <span>Final report</span>
            <strong>{generatedAtLabel ?? "Available"}</strong>
            <small>
              {finalReport.classification?.official_traditional_baseline_policy
                ? `traditional ${formatPolicyLabel(finalReport.classification.official_traditional_baseline_policy)}`
                : "report loaded"}
            </small>
          </div>
        ) : null}
      </section>

      {loadingReport ? <LoadingNotice title="Loading reporting context" description="Reading final evaluation and scenario validation bundles." /> : null}
      {reportError ? <InfoNotice title="Final Evaluation Unavailable" description={reportError} /> : null}
      {!backendOnline ? (
        <InfoNotice
          title="Live comparison disabled"
          description={`${backendMessage} Stored reporting artifacts can still load, but starting comparison runs is disabled until the backend is available.`}
        />
      ) : null}
      {runCatalogSource === "demo" ? (
        <InfoNotice
          title="Demo Data Mode"
          description="The backend run catalog was unavailable, so live policy comparison is using generated traces. Final evaluation and official validation still use backend artifacts when available."
        />
      ) : null}

      {finalReport && bestAiOverall ? (
        <section className="report-summary-grid">
          <article className="glass-card report-hero-card">
            <div className="card-heading">
              <p>Final Comparison</p>
              <h3>{headline ?? "Best AI outcome"}</h3>
            </div>
            <p className="report-lead">
              {finalReport.source?.selected_run_count ?? overallRows.reduce((sum, row) => sum + (row.run_count ?? 0), 0)} runs
              across {finalReport.source?.selected_scenarios?.length ?? scenarioRows.length} scenarios feed the final evaluation artifact.
            </p>
            <div className="status-badge-row">
              <StatusBadge
                label={`Operational ${formatStatusLabel(bestAiOverall.stability_qualified_hypothesis_status ?? bestAiOverall.hypothesis_status ?? "")}`}
                tone={statusTone(bestAiOverall.stability_qualified_hypothesis_status ?? bestAiOverall.hypothesis_status ?? "")}
              />
              {bestAiQosStatus ? (
                <StatusBadge label={`QoS ${formatStatusLabel(bestAiQosStatus)}`} tone={statusTone(bestAiQosStatus)} />
              ) : null}
              {bestAiQosMissing ? <StatusBadge label={`QoS ${formatStatusLabel(bestAiQosMissing)}`} tone="warning" /> : null}
              {bestAiStabilityStatus ? (
                <StatusBadge
                  label={`Stability ${formatStatusLabel(bestAiStabilityStatus)}`}
                  tone={statusTone(bestAiStabilityStatus)}
                />
              ) : null}
              {bestAiStabilityMissing ? (
                <StatusBadge label={`Stability ${formatStatusLabel(bestAiStabilityMissing)}`} tone="warning" />
              ) : null}
              <StatusBadge label={`AI policy ${formatPolicyLabel(bestAiOverall.policy)}`} tone="neutral" />
            </div>
            <div className="report-stat-grid">
              <ReportStat
                label="Energy reduction vs traditional"
                value={formatSignedValue(bestAiOverall.energy_reduction_pct_vs_baseline, 1, "%")}
              />
              <ReportStat
                label="Energy reduction vs heuristic"
                value={formatSignedValue(bestAiOverall.energy_reduction_pct_vs_heuristic_baseline, 1, "%")}
              />
              <ReportStat
                label="Delivered traffic change"
                value={formatSignedValue(bestAiOverall.delivered_traffic_change_pct_vs_baseline, 1, "%")}
              />
              <ReportStat
                label="Dropped traffic change"
                value={formatSignedValue(bestAiOverall.dropped_traffic_change_pct_vs_baseline, 1, "%")}
              />
              <ReportStat
                label="QoS violation rate delta"
                value={formatSignedValue(
                  bestAiOverall.qos_violation_rate_delta_vs_baseline == null
                    ? undefined
                    : bestAiOverall.qos_violation_rate_delta_vs_baseline * 100,
                  2,
                  " pts",
                )}
              />
            </div>
            {finalReport.artifact?.summary_path ? (
              <p className="card-caption">Source artifact: {finalReport.artifact.summary_path}</p>
            ) : null}
          </article>

          <article className="glass-card report-threshold-card">
            <div className="card-heading">
              <p>{hasCentralQosThresholds ? "QoS Acceptance" : "Hypothesis Gate"}</p>
              <h3>
                {hasCentralQosThresholds
                  ? "Centralized QoS thresholds used in the final report"
                  : "Acceptance thresholds used in the final report"}
              </h3>
            </div>
            <div className="threshold-list">
              <ReportStat
                label="Energy reduction target"
                value={`${fmt((reportQosThresholds ?? finalReport.hypothesis_thresholds)?.energy_target_pct ?? 0, 1)}%`}
              />
              <ReportStat
                label="Max delivered loss"
                value={`${fmt((reportQosThresholds ?? finalReport.hypothesis_thresholds)?.max_delivered_loss_pct ?? 0, 1)}%`}
              />
              <ReportStat
                label="Max dropped increase"
                value={`${fmt((reportQosThresholds ?? finalReport.hypothesis_thresholds)?.max_dropped_increase_pct ?? 0, 1)}%`}
              />
              <ReportStat
                label="Max delay increase"
                value={`${fmt((reportQosThresholds ?? finalReport.hypothesis_thresholds)?.max_delay_increase_pct ?? 0, 1)}%`}
              />
              <ReportStat
                label="Max path latency increase"
                value={`${fmt((reportQosThresholds ?? finalReport.hypothesis_thresholds)?.max_path_latency_increase_pct ?? 0, 1)}%`}
              />
              <ReportStat
                label="Max QoS violation rate increase"
                value={`${fmt(((reportQosThresholds ?? finalReport.hypothesis_thresholds)?.max_qos_violation_rate_increase_abs ?? 0) * 100, 2)} pts`}
              />
            </div>
          </article>
        </section>
      ) : null}

      {overallRows.length > 0 ? (
        <section className="glass-card table-card">
          <div className="card-heading">
            <p>Traditional vs Heuristic vs AI</p>
            <h3>Overall final evaluation summary</h3>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Policy</th>
                  <th>Class</th>
                  <th>Runs</th>
                  <th>Total Energy</th>
                  <th>Energy vs Traditional</th>
                  <th>Delivered Traffic</th>
                  <th>Delivered vs Traditional</th>
                  <th>Dropped Traffic</th>
                  <th>Average Delay</th>
                  <th>QoS Violation Rate</th>
                  <th>Stability</th>
                  <th>Operational</th>
                  <th>QoS Acceptance</th>
                </tr>
              </thead>
              <tbody>
                {overallRows.map((row) => (
                  <tr
                    key={`${row.scope_type}-${row.policy}`}
                    className={row.is_best_ai_policy_for_scope || row.is_primary_baseline ? "report-highlight-row" : ""}
                  >
                    <td>
                      <div className="table-policy-cell">
                        <strong>{formatPolicyLabel(row.policy)}</strong>
                        <div className="table-badge-row">
                          {row.is_primary_baseline ? <StatusBadge label="Traditional" tone="neutral" /> : null}
                          {row.is_heuristic_baseline ? <StatusBadge label="Heuristic" tone="neutral" /> : null}
                          {row.is_best_ai_policy_for_scope ? <StatusBadge label="Best AI" tone="success" /> : null}
                        </div>
                      </div>
                    </td>
                    <td>{formatStatusLabel(row.policy_class)}</td>
                    <td>{fmt(row.run_count ?? NaN, 0)}</td>
                    <td>{fmt(row.energy_kwh_mean ?? NaN, 3)}</td>
                    <td>{formatSignedValue(row.energy_reduction_pct_vs_baseline, 1, "%")}</td>
                    <td>{fmt(row.delivered_traffic_mean ?? NaN, 0)}</td>
                    <td>{formatSignedValue(row.delivered_traffic_change_pct_vs_baseline, 1, "%")}</td>
                    <td>{fmt(row.dropped_traffic_mean ?? NaN, 0)}</td>
                    <td>{fmt(row.avg_delay_ms_mean ?? NaN, 1)}</td>
                    <td>{fmt((row.qos_violation_rate_mean ?? NaN) * 100, 1)}%</td>
                    <td>
                      <StatusBadge
                        label={formatStatusLabel(selectStabilityStatus(row))}
                        tone={statusTone(selectStabilityStatus(row))}
                      />
                    </td>
                    <td>
                      <StatusBadge
                        label={formatStatusLabel(row.stability_qualified_hypothesis_status ?? row.hypothesis_status ?? "")}
                        tone={statusTone(row.stability_qualified_hypothesis_status ?? row.hypothesis_status ?? "")}
                      />
                    </td>
                    <td>
                      <StatusBadge
                        label={formatStatusLabel(selectQosAcceptanceStatus(row))}
                        tone={statusTone(selectQosAcceptanceStatus(row))}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {scenarioRows.length > 0 ? (
        <section className="scenario-report-section">
          <div className="card-heading">
            <p>Scenario Results</p>
            <h3>Best AI result per scenario against the official traditional baseline</h3>
          </div>

          <div className="scenario-results-grid">
            {scenarioRows.map((row: FinalEvaluationSummaryRow) => (
              <article key={`${row.scope_type}-${row.scope}-${row.policy}`} className="glass-card scenario-report-card">
                <div className="scenario-report-head">
                  <div className="card-heading">
                    <p>Scenario</p>
                    <h3>{formatScenarioLabel(row.scope)}</h3>
                  </div>
                  <div className="status-badge-row">
                    <StatusBadge
                      label={formatStatusLabel(row.stability_qualified_hypothesis_status ?? row.hypothesis_status ?? "")}
                      tone={statusTone(row.stability_qualified_hypothesis_status ?? row.hypothesis_status ?? "")}
                    />
                    <StatusBadge
                      label={`Stability ${formatStatusLabel(selectStabilityStatus(row))}`}
                      tone={statusTone(selectStabilityStatus(row))}
                    />
                    <StatusBadge
                      label={formatStatusLabel(selectQosAcceptanceStatus(row))}
                      tone={statusTone(selectQosAcceptanceStatus(row))}
                    />
                    {selectStabilityMissing(row) ? (
                      <StatusBadge label={`Stability ${formatStatusLabel(selectStabilityMissing(row))}`} tone="warning" />
                    ) : null}
                    {selectQosAcceptanceMissing(row) ? (
                      <StatusBadge label={`QoS ${formatStatusLabel(selectQosAcceptanceMissing(row))}`} tone="warning" />
                    ) : null}
                  </div>
                </div>
                <p className="card-caption">
                  {formatPolicyLabel(row.policy)} vs {formatPolicyLabel(row.comparison_baseline_policy ?? "")}
                  {row.comparison_heuristic_baseline_policy
                    ? ` | heuristic ${formatPolicyLabel(row.comparison_heuristic_baseline_policy)}`
                    : ""}
                </p>
                <div className="report-stat-grid compact">
                  <ReportStat label="Energy vs traditional" value={formatSignedValue(row.energy_reduction_pct_vs_baseline, 1, "%")} />
                  <ReportStat
                    label="Energy vs heuristic"
                    value={formatSignedValue(row.energy_reduction_pct_vs_heuristic_baseline, 1, "%")}
                  />
                  <ReportStat label="Delivered change" value={formatSignedValue(row.delivered_traffic_change_pct_vs_baseline, 1, "%")} />
                  <ReportStat label="Dropped change" value={formatSignedValue(row.dropped_traffic_change_pct_vs_baseline, 1, "%")} />
                  <ReportStat label="Delay change" value={formatSignedValue(row.avg_delay_ms_change_pct_vs_baseline, 1, "%")} />
                  <ReportStat
                    label="QoS rate delta"
                    value={formatSignedValue(
                      row.qos_violation_rate_delta_vs_baseline == null ? undefined : row.qos_violation_rate_delta_vs_baseline * 100,
                      2,
                      " pts",
                    )}
                  />
                  <ReportStat label="Run count" value={fmt(row.run_count ?? NaN, 0)} />
                </div>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      {officialResults.length > 0 ? (
        <section className="official-results-section">
          <div className="card-heading">
            <p>Scenario Acceptance</p>
            <h3>Official locked validation bundles</h3>
          </div>

          <div className="official-results-grid">
            {officialResults.map((result) => (
              <OfficialResultCard key={`${result.scenario}-${result.bundle_id}`} result={result} />
            ))}
          </div>
        </section>
      ) : null}

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

        <button className="btn-primary" onClick={handleRunComparison} disabled={loadingComparison || !backendOnline}>
          {loadingComparison ? "Loading..." : "Load Live Comparison"}
        </button>
      </section>

      {loadingComparison ? (
        <LoadingNotice title="Comparing policies" description="Running or loading all-on, heuristic, and PPO for the selected scenario." />
      ) : null}
      {comparisonError ? <ErrorNotice title="Live Comparison Error" description={comparisonError} /> : null}

      {!liveComparisonReady ? (
        <InfoNotice
          title="Live policy comparison"
          description="Load a scenario/seed pair to compare all-on, heuristic, and PPO with the same reporting labels used in the final summary."
        />
      ) : (
        <>
          <section className="glass-card table-card">
            <div className="card-heading">
              <p>Policy Comparison</p>
              <h3>Live comparison for the selected scenario</h3>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Policy</th>
                    <th>Total Energy</th>
                    <th>Carbon Emissions</th>
                    <th>Average Delay</th>
                    <th>Path Latency</th>
                    <th>Delivered Traffic</th>
                    <th>Dropped Traffic</th>
                    <th>Drop Rate</th>
                    <th>QoS Acceptance</th>
                    <th>Active Links</th>
                    <th>Total Reward</th>
                  </tr>
                </thead>
                <tbody>
                  {summaries.map(({ policy, summary }) => (
                    (() => {
                      const qosStatus = selectQosAcceptanceStatus(summariesByPolicy[policy]);
                      return (
                        <tr key={policy}>
                          <td>{formatPolicyLabel(policy)}</td>
                          <td>{fmt(summary.energy_kwh, 3)}</td>
                          <td>{fmt(summary.carbon_g, 3)}</td>
                          <td>{fmt(summary.avg_delay_ms, 1)}</td>
                          <td>{fmt(summary.avg_path_latency_ms, 2)}</td>
                          <td>{fmt(summary.delivered, 0)}</td>
                          <td>{fmt(summary.dropped, 0)}</td>
                          <td>{fmt(summary.drop_rate, 1)}%</td>
                          <td>
                            {qosStatus ? (
                              <StatusBadge label={formatStatusLabel(qosStatus)} tone={statusTone(qosStatus)} />
                            ) : (
                              "-"
                            )}
                          </td>
                          <td>{fmt(summary.active_ratio, 1)}%</td>
                          <td>{fmt(summary.reward, 1)}</td>
                        </tr>
                      );
                    })()
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
              title="Dropped Traffic Overlay"
              subtitle="packets per step"
              data={mergedData}
              lines={POLICY_LIST.map((policy) => ({
                dataKey: `${policy}_dropped`,
                label: formatPolicyLabel(policy),
                color: policyColors[policy],
              }))}
            />
          </section>
        </>
      )}
    </div>
  );
}
