import { formatRunOptionLabel } from "../lib/data";
import type { RunSummary } from "../lib/types";

type RunControlsProps = {
  policy: string;
  scenario: string;
  seed: string;
  steps: string;
  loading?: boolean;
  runId?: string;
  runs?: RunSummary[];
  onPolicyChange: (value: string) => void;
  onScenarioChange: (value: string) => void;
  onSeedChange: (value: string) => void;
  onStepsChange: (value: string) => void;
  onRun: () => void;
  onReset: () => void;
  onRunSelect?: (runId: string) => void;
};

function sanitizeIntegerInput(value: string): string {
  const digitsOnly = value.replace(/\D/g, "");
  if (!digitsOnly) {
    return "";
  }
  return String(Number(digitsOnly));
}

export default function RunControls({
  policy,
  scenario,
  seed,
  steps,
  loading,
  runId,
  runs,
  onPolicyChange,
  onScenarioChange,
  onSeedChange,
  onStepsChange,
  onRun,
  onReset,
  onRunSelect,
}: RunControlsProps) {
  const parsedSteps = steps === "" ? Number.NaN : Number(steps);
  const showLongRunWarning = Number.isFinite(parsedSteps) && parsedSteps >= 4000;
  const canRun =
    seed !== "" &&
    steps !== "" &&
    Number.isFinite(parsedSteps) &&
    parsedSteps >= 1 &&
    parsedSteps <= 5000;

  function handleSeedInputChange(value: string): void {
    onSeedChange(sanitizeIntegerInput(value));
  }

  function handleSeedBlur(): void {
    if (seed === "") {
      return;
    }
    onSeedChange(String(Math.max(0, Number(seed))));
  }

  function handleStepsInputChange(value: string): void {
    const next = sanitizeIntegerInput(value);
    onStepsChange(next);
  }

  function handleStepsBlur(): void {
    if (steps === "") {
      return;
    }

    const normalized = String(Math.min(5000, Math.max(1, Number(steps))));
    onStepsChange(normalized);
  }

  return (
    <section className="glass-card control-card">
      <div className="card-heading">
        <p>Run Controls</p>
        <h3>Configure Simulation</h3>
      </div>

      <div className="control-form">
        {onRunSelect && runs && runs.length > 0 ? (
          <label>
            Existing Run
            <select value={runId ?? ""} onChange={(event) => onRunSelect(event.target.value)}>
              {runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {formatRunOptionLabel(run)}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        <label>
          Policy
          <select value={policy} onChange={(event) => onPolicyChange(event.target.value)}>
            <option value="all_on">All-On</option>
            <option value="heuristic">Heuristic</option>
            <option value="ppo">PPO</option>
          </select>
        </label>

        <label>
          Traffic Scenario
          <select value={scenario} onChange={(event) => onScenarioChange(event.target.value)}>
            <option value="normal">Normal</option>
            <option value="burst">Burst</option>
            <option value="hotspot">Hotspot</option>
          </select>
        </label>

        <label>
          Seed
          <input
            type="text"
            inputMode="numeric"
            pattern="[0-9]*"
            value={seed}
            onChange={(event) => handleSeedInputChange(event.target.value)}
            onBlur={handleSeedBlur}
          />
        </label>

        <label>
          Steps
          <input
            type="text"
            inputMode="numeric"
            pattern="[0-9]*"
            value={steps}
            onChange={(event) => handleStepsInputChange(event.target.value)}
            onBlur={handleStepsBlur}
          />
        </label>
      </div>

      {showLongRunWarning ? (
        <div className="control-warning" role="status" aria-live="polite">
          <strong>High step count selected</strong>
          <p>Runs near the current 5000-step limit can take a bit longer to finish in the live dashboard.</p>
        </div>
      ) : null}

      <div className="button-row">
        <button className="btn-primary" onClick={onRun} disabled={loading || !canRun}>
          {loading ? "Running..." : "Run Simulation"}
        </button>
        <button className="btn-muted" onClick={onReset}>
          Reset
        </button>
      </div>
    </section>
  );
}
