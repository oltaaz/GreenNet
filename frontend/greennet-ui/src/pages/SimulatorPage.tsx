import { useEffect, useMemo, useState } from "react";
import SimulatorCanvas from "../components/SimulatorCanvas";
import { ErrorNotice, InfoNotice, LoadingNotice } from "../components/StatusState";
import { getRunPerStep, getSteps, getTopology, listRuns, startRun } from "../lib/api";
import {
  fallbackTopology,
  formatPolicyLabel,
  formatRunOptionLabel,
  fmt,
  inferPolicy,
  latestRunByPolicy,
  linkStateFromRatio,
  normalizePerStep,
  timelineFromRows,
} from "../lib/data";
import { isDemoRunId } from "../lib/demo";
import type { RunSummary, StepState, TopologyData } from "../lib/types";

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

export default function SimulatorPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [policy, setPolicy] = useState("ppo");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [topology, setTopology] = useState<TopologyData>(fallbackTopology("simulator"));
  const [timeline, setTimeline] = useState<StepState[]>([]);

  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [currentStep, setCurrentStep] = useState(0);
  const [showDropped, setShowDropped] = useState(true);

  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;

    async function loadRuns() {
      try {
        const runItems = await listRuns();
        if (alive) {
          setRuns(runItems);
        }
      } catch (apiError) {
        if (alive) {
          setError(apiError instanceof Error ? apiError.message : "Failed to load runs");
        }
      }
    }

    void loadRuns();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (selectedRunId && runs.some((run) => run.run_id === selectedRunId)) {
      return;
    }
    const candidate = latestRunByPolicy(runs, policy) ?? runs[0] ?? null;
    if (candidate?.run_id) {
      setSelectedRunId(candidate.run_id);
    }
  }, [policy, runs, selectedRunId]);

  useEffect(() => {
    if (!selectedRunId) {
      setTimeline([]);
      setTopology(fallbackTopology("simulator-empty"));
      return;
    }

    let alive = true;

    async function loadTimeline() {
      setLoading(true);
      setError("");

      try {
        const [rowsRaw, topo, stepsFromApi] = await Promise.all([
          getRunPerStep(selectedRunId),
          getTopology(selectedRunId),
          getSteps(selectedRunId),
        ]);

        if (!alive) {
          return;
        }

        const normalizedRows = normalizePerStep(rowsRaw);
        const graph = topo ?? fallbackTopology(`${selectedRunId}-${policy}`);
        const fallbackTimeline = timelineFromRows(normalizedRows, graph);

        const mergedTimeline = fallbackTimeline.map((step, index) => {
          const fromApi = stepsFromApi?.[index];
          const hasLinks = Boolean(fromApi?.links_on && Object.keys(fromApi.links_on).length > 0);

          return {
            t: step.t,
            metrics: fromApi?.metrics ?? step.metrics,
            links_on: hasLinks ? fromApi?.links_on : step.links_on,
            packet_events: fromApi?.packet_events,
          };
        });

        setTopology(graph);
        setTimeline(mergedTimeline);
        setCurrentStep(0);
      } catch (apiError) {
        if (alive) {
          setError(apiError instanceof Error ? apiError.message : "Failed to load simulator timeline");
          setTimeline([]);
          setTopology(fallbackTopology(`${selectedRunId}-${policy}-fallback`));
        }
      } finally {
        if (alive) {
          setLoading(false);
        }
      }
    }

    void loadTimeline();

    return () => {
      alive = false;
    };
  }, [policy, selectedRunId]);

  async function handleStartPolicyRun(): Promise<void> {
    setRunning(true);
    setError("");

    try {
      const started = await startRun({ policy, scenario: "normal", seed: 42, steps: 300 });
      const refreshed = await listRuns();
      setRuns(
        upsertRun(refreshed, {
          run_id: started.run_id,
          policy,
          scenario: "normal",
          seed: 42,
          tag: "dashboard",
        }),
      );
      setSelectedRunId(started.run_id);
    } catch (apiError) {
      setError(apiError instanceof Error ? apiError.message : "Could not start run from simulator");
    } finally {
      setRunning(false);
    }
  }

  const maxStep = Math.max(0, timeline.length - 1);
  const step = timeline[currentStep] ?? timeline[maxStep] ?? null;
  const previousStep = timeline[Math.max(0, currentStep - 1)] ?? step;
  const displayedStep = step?.t ?? currentStep;
  const displayedMaxStep = timeline[maxStep]?.t ?? maxStep;

  const derivedPolicy = useMemo(() => {
    const run = runs.find((item) => item.run_id === selectedRunId);
    return run ? inferPolicy(run) : policy;
  }, [policy, runs, selectedRunId]);

  const demoMode = isDemoRunId(selectedRunId);

  const linkTelemetry = useMemo(() => {
    if (!step) {
      return { onCount: 0, offCount: 0, switched: 0, energyDelta: 0 };
    }

    const currentState =
      step.links_on ??
      linkStateFromRatio(topology.edges, step.metrics.active_ratio, step.t);

    const previousState =
      previousStep?.links_on ??
      linkStateFromRatio(
        topology.edges,
        previousStep?.metrics.active_ratio ?? step.metrics.active_ratio,
        previousStep?.t ?? step.t,
      );

    let onCount = 0;
    let offCount = 0;
    let switched = 0;

    for (const edge of topology.edges) {
      const current = Boolean(currentState[edge.id]);
      const previous = Boolean(previousState[edge.id]);

      if (current) {
        onCount += 1;
      } else {
        offCount += 1;
      }

      if (current !== previous) {
        switched += 1;
      }
    }

    return {
      onCount,
      offCount,
      switched,
      energyDelta: (step.metrics.energy_kwh ?? 0) - (previousStep?.metrics.energy_kwh ?? 0),
    };
  }, [previousStep, step, topology.edges]);

  return (
    <div className="page simulator-page">
      <section className="page-title-row">
        <div>
          <p className="page-eyebrow">Packet Simulator</p>
          <h1>Topology playback with simulated packet flow and real link states</h1>
        </div>
      </section>

      {loading ? <LoadingNotice title="Loading simulator" description="Fetching topology and per-step states." /> : null}
      {error ? <ErrorNotice title="Simulator Error" description={error} /> : null}
      {demoMode ? (
        <InfoNotice
          title="Demo Data Mode"
          description="This run is generated locally because backend run data is missing or unavailable."
        />
      ) : null}

      <section className="simulator-layout">
        <div className="glass-card simulator-canvas-card">
          <SimulatorCanvas
            topology={topology}
            steps={timeline}
            currentStep={currentStep}
            playing={playing}
            speed={speed}
            showDropped={showDropped}
            onStepChange={setCurrentStep}
          />
        </div>

        <aside className="glass-card simulator-controls">
          <div className="card-heading">
            <p>Controls</p>
            <h3>Simulator Playback</h3>
          </div>

          <label>
            Policy
            <select value={policy} onChange={(event) => setPolicy(event.target.value)}>
              <option value="all_on">All-On</option>
              <option value="heuristic">Heuristic</option>
              <option value="ppo">PPO</option>
            </select>
          </label>

          <label>
            Run
            <select value={selectedRunId} onChange={(event) => setSelectedRunId(event.target.value)}>
              {runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {formatRunOptionLabel(run)}
                </option>
              ))}
            </select>
          </label>

          <div className="button-row">
            <button className="btn-primary" onClick={() => setPlaying((prev) => !prev)}>
              {playing ? "Pause" : "Play"}
            </button>
            <button className="btn-muted" onClick={handleStartPolicyRun} disabled={running}>
              {running ? "Starting..." : "Run Policy"}
            </button>
          </div>

          <label>
            Speed ({fmt(speed, 2)}x)
            <input
              type="range"
              min={0.25}
              max={4}
              step={0.25}
              value={speed}
              onChange={(event) => setSpeed(Number(event.target.value))}
            />
          </label>

          <label>
            Timeline (step {displayedStep}/{displayedMaxStep})
            <input
              type="range"
              min={0}
              max={maxStep}
              value={Math.min(currentStep, maxStep)}
              onChange={(event) => setCurrentStep(Number(event.target.value))}
              disabled={timeline.length <= 1}
            />
          </label>

          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showDropped}
              onChange={(event) => setShowDropped(event.target.checked)}
            />
            Show dropped packets
          </label>

          <section className="stats-grid-mini">
            <article>
              <span>Sent (step)</span>
              <strong>{fmt((step?.metrics.delivered ?? 0) + (step?.metrics.dropped ?? 0), 0)}</strong>
            </article>
            <article>
              <span>Delivered (step)</span>
              <strong>{fmt(step?.metrics.delivered ?? 0, 0)}</strong>
            </article>
            <article>
              <span>Dropped (step)</span>
              <strong>{fmt(step?.metrics.dropped ?? 0, 0)}</strong>
            </article>
            <article>
              <span>Path Latency (ms)</span>
              <strong>{fmt(step?.metrics.avg_delay_ms ?? 0)}</strong>
            </article>
            <article>
              <span>Energy (kWh)</span>
              <strong>{fmt(step?.metrics.energy_kwh ?? 0, 3)}</strong>
            </article>
            <article>
              <span>Active Ratio (%)</span>
              <strong>{fmt((step?.metrics.active_ratio ?? 0) * 100, 1)}</strong>
            </article>
            <article>
              <span>Links ON</span>
              <strong>{linkTelemetry.onCount}</strong>
            </article>
            <article>
              <span>Links OFF</span>
              <strong>{linkTelemetry.offCount}</strong>
            </article>
            <article>
              <span>Switched (step)</span>
              <strong>{linkTelemetry.switched}</strong>
            </article>
            <article>
              <span>Energy Delta</span>
              <strong>
                {linkTelemetry.energyDelta >= 0 ? "+" : ""}
                {fmt(linkTelemetry.energyDelta, 3)}
              </strong>
            </article>
          </section>

          <p className="card-caption">Policy in focus: {formatPolicyLabel(derivedPolicy)}</p>
        </aside>
      </section>
    </div>
  );
}
