import type { RunSummary } from "../lib/types";

type RunControlsProps = {
  policy: string;
  scenario: string;
  seed: number;
  steps: number;
  loading?: boolean;
  runId?: string;
  runs?: RunSummary[];
  onPolicyChange: (value: string) => void;
  onScenarioChange: (value: string) => void;
  onSeedChange: (value: number) => void;
  onStepsChange: (value: number) => void;
  onRun: () => void;
  onReset: () => void;
  onRunSelect?: (runId: string) => void;
};

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
                  {run.run_id}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        <label>
          Policy
          <select value={policy} onChange={(event) => onPolicyChange(event.target.value)}>
            <option value="baseline">Baseline</option>
            <option value="noop">No-Op</option>
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
            type="number"
            min={0}
            value={seed}
            onChange={(event) => onSeedChange(Number(event.target.value))}
          />
        </label>

        <label>
          Steps
          <input
            type="number"
            min={1}
            value={steps}
            onChange={(event) => onStepsChange(Number(event.target.value))}
          />
        </label>
      </div>

      <div className="button-row">
        <button className="btn-primary" onClick={onRun} disabled={loading}>
          {loading ? "Running..." : "Run Simulation"}
        </button>
        <button className="btn-muted" onClick={onReset}>
          Reset
        </button>
      </div>
    </section>
  );
}
