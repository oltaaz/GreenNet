import type { KpiMetric } from "../lib/types";
import { fmt } from "../lib/data";

type KpiCardProps = {
  metric: KpiMetric;
};

export default function KpiCard({ metric }: KpiCardProps) {
  const digits = metric.unit === "kWh" ? 3 : metric.unit === "%" ? 1 : metric.unit === "steps" ? 0 : 2;

  return (
    <article className="glass-card kpi-card">
      <p>{metric.label}</p>
      <div>
        <strong>{fmt(metric.value, digits)}</strong>
        <span>{metric.unit}</span>
      </div>
    </article>
  );
}
