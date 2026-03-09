import { fmt, officialLockedScenarioMetrics } from "../lib/data";
import type { OfficialLockedResult } from "../lib/types";

type OfficialResultCardProps = {
  result: OfficialLockedResult;
  active?: boolean;
};

function formatScenarioLabel(value: string): string {
  if (!value) {
    return "Unknown";
  }
  return value.charAt(0).toUpperCase() + value.slice(1).toLowerCase();
}

export default function OfficialResultCard({ result, active = false }: OfficialResultCardProps) {
  const metrics = officialLockedScenarioMetrics(result);
  const selectedRow = result.summary;
  const decision = result.delta_summary;
  const metaItems = [
    selectedRow?.off_level ? selectedRow.off_level.toUpperCase() : null,
    selectedRow?.episodes != null ? `${selectedRow.episodes} eps` : null,
    selectedRow?.seeds ? `seeds ${selectedRow.seeds}` : null,
    selectedRow?.cap_used ? `cap ${selectedRow.cap_used}` : null,
  ].filter((item): item is string => Boolean(item));

  return (
    <article className={`glass-card official-result-card${active ? " active" : ""}`}>
      <div className="official-result-head">
        <div className="card-heading">
          <p>Official Locked Result</p>
          <h3>{formatScenarioLabel(result.scenario)}</h3>
        </div>

        <div className="official-result-badges">
          {active ? <span className="official-pill active">Live Scenario</span> : null}
          <span className={`official-pill ${result.pass_all ? "pass" : "check"}`}>
            {result.pass_all ? "PASS" : "CHECK"}
          </span>
        </div>
      </div>

      <p className="official-result-note">
        Locked acceptance bundle <strong>{result.bundle_id}</strong> with controller-evaluated held-out traffic.
      </p>

      {metaItems.length > 0 ? (
        <div className="official-result-meta">
          {metaItems.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      ) : null}

      <div className="official-result-kpis">
        {metrics.map((metric) => (
          <div key={`${result.scenario}-${metric.label}`} className="official-result-kpi">
            <span>{metric.label}</span>
            <strong>{fmt(metric.value, metric.unit === "kWh" ? 3 : 2)}</strong>
            <small>{metric.unit}</small>
          </div>
        ))}
      </div>

      {decision || result.notes ? (
        <p className="card-caption">
          {decision?.reason ? `${decision.reason}. ` : ""}
          {decision?.delta_reward != null ? `Reward Delta vs No-Op ${fmt(decision.delta_reward, 2)}. ` : ""}
          {result.notes ?? ""}
        </p>
      ) : null}
    </article>
  );
}
