import type { KpiMetric } from "../lib/types";
import { fmt } from "../lib/data";

type KpiCardProps = {
  metric: KpiMetric;
};

export default function KpiCard({ metric }: KpiCardProps) {
  return (
    <article className="glass-card kpi-card">
      <p>{metric.label}</p>
      <div>
        <strong>{fmt(metric.value)}</strong>
        <span>{metric.unit}</span>
      </div>
    </article>
  );
}
