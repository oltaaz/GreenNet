import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type ChartLine = {
  dataKey: string;
  label: string;
  color: string;
  dashed?: boolean;
};

type ChartCardProps = {
  title: string;
  subtitle: string;
  data: Array<Record<string, number | string>>;
  lines: ChartLine[];
  yAxisFormatter?: (value: number) => string;
};

export default function ChartCard({
  title,
  subtitle,
  data,
  lines,
  yAxisFormatter,
}: ChartCardProps) {
  return (
    <section className="glass-card chart-card">
      <div className="card-heading">
        <p>{subtitle}</p>
        <h3>{title}</h3>
      </div>
      <div className="chart-wrap">
        <ResponsiveContainer width="99%" height={210}>
          <LineChart data={data} margin={{ top: 10, right: 12, left: 6, bottom: 8 }}>
            <CartesianGrid stroke="rgba(169, 189, 222, 0.16)" vertical={false} />
            <XAxis dataKey="t" stroke="rgba(169,189,222,0.8)" fontSize={12} />
            <YAxis
              stroke="rgba(169,189,222,0.8)"
              fontSize={12}
              tickFormatter={yAxisFormatter}
              width={52}
            />
            <Tooltip
              contentStyle={{
                background: "rgba(5, 16, 34, 0.96)",
                border: "1px solid rgba(115, 146, 185, 0.4)",
                borderRadius: "12px",
              }}
            />
            <Legend wrapperStyle={{ fontSize: "12px" }} />
            {lines.map((line) => (
              <Line
                key={line.dataKey}
                type="monotone"
                dataKey={line.dataKey}
                name={line.label}
                stroke={line.color}
                strokeWidth={2.1}
                strokeDasharray={line.dashed ? "5 5" : undefined}
                dot={false}
                isAnimationActive
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
