import type {
  KpiMetric,
  LinkStateMap,
  OfficialLockedResult,
  PerStepRow,
  RunOverallSummary,
  RunSummary,
  StepMetrics,
  StepState,
  TopologyData,
  TopologyEdge,
  TopologyNode,
} from "./types";

export function edgeId(source: string, target: string): string {
  return source < target ? `${source}__${target}` : `${target}__${source}`;
}

function toNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
}

function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function hashSeed(input: string): number {
  let hash = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function toMetrics(row: PerStepRow): StepMetrics {
  return {
    t: toNumber(row.t, 0),
    energy_kwh: toNumber(row.energy_kwh, 0),
    carbon_g: toNumber(row.carbon_g, 0),
    avg_delay_ms: toNumber(row.avg_delay_ms, 0),
    avg_path_latency_ms: toNumber(row.avg_path_latency_ms, 0),
    congestion_delay_ms: toNumber(row.congestion_delay_ms, 0),
    dropped: toNumber(row.dropped, 0),
    delivered: toNumber(row.delivered, 0),
    active_ratio: toNumber(row.active_ratio, 1),
    reward: toNumber(row.reward, 0),
  };
}

export function normalizePerStep(rows: PerStepRow[]): PerStepRow[] {
  return [...rows]
    .map((row, index) => {
      const tValue = Number.isFinite(Number(row.t)) ? Number(row.t) : index;
      return {
        ...row,
        t: tValue,
        energy_kwh: toNumber(row.delta_energy_kwh ?? row.energy_kwh, 0),
        carbon_g: toNumber(row.delta_carbon_g ?? row.carbon_g, 0),
        avg_delay_ms: toNumber(row.avg_delay_ms, 0),
        avg_path_latency_ms: toNumber(row.avg_path_latency_ms, 0),
        congestion_delay_ms: toNumber(row.congestion_delay_ms, 0),
        dropped: toNumber(row.delta_dropped ?? row.dropped, 0),
        delivered: toNumber(row.delta_delivered ?? row.delivered, 0),
        active_ratio: toNumber(row.active_ratio, 1),
        reward: toNumber(row.reward, 0),
      };
    })
    .sort((a, b) => a.t - b.t);
}

export function latestRow(rows: PerStepRow[]): PerStepRow | null {
  if (!rows.length) {
    return null;
  }
  return rows[rows.length - 1];
}

function average(values: number[]): number {
  return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function sumMetric(rows: PerStepRow[], primaryKey: string, fallbackKey?: keyof PerStepRow): number {
  const hasPrimary = rows.some((row) => row[primaryKey] != null);
  return rows.reduce((sum, row) => {
    const value = hasPrimary
      ? row[primaryKey]
      : fallbackKey == null
        ? 0
        : row[fallbackKey];
    return sum + toNumber(value, 0);
  }, 0);
}

export function deriveOverallFromRows(rows: PerStepRow[]): RunOverallSummary | null {
  if (!rows.length) {
    return null;
  }

  const avgDelayValues = rows.map((row) => toNumber(row.avg_delay_ms, 0));
  const avgPathLatencyValues = rows.map((row) =>
    row.avg_path_latency_ms == null ? toNumber(row.avg_delay_ms, 0) : toNumber(row.avg_path_latency_ms, 0),
  );

  return {
    reward_total_mean: rows.reduce((sum, row) => sum + toNumber(row.reward, 0), 0),
    delivered_total_mean: sumMetric(rows, "delta_delivered", "delivered"),
    dropped_total_mean: sumMetric(rows, "delta_dropped", "dropped"),
    energy_kwh_total_mean: sumMetric(rows, "delta_energy_kwh", "energy_kwh"),
    carbon_g_total_mean: sumMetric(rows, "delta_carbon_g", "carbon_g"),
    avg_utilization_mean: 0,
    active_ratio_mean: average(rows.map((row) => toNumber(row.active_ratio, 1))),
    avg_delay_ms_mean: average(avgDelayValues),
    avg_path_latency_ms_mean: average(avgPathLatencyValues),
    steps_mean: rows.length,
  };
}

export function kpiFromOverall(summary: RunOverallSummary | null): KpiMetric[] {
  if (!summary) {
    return [];
  }

  const steps = Math.max(1, toNumber(summary.steps_mean, 1));
  const delivered = toNumber(summary.delivered_total_mean, 0);
  const dropped = toNumber(summary.dropped_total_mean, 0);
  const totalPacketsSent =
    delivered + dropped;
  const dropRate = totalPacketsSent > 0 ? (dropped / totalPacketsSent) * 100 : 0;

  return [
    { label: "Energy Usage", value: toNumber(summary.energy_kwh_total_mean, 0), unit: "kWh" },
    { label: "Carbon Emissions", value: toNumber(summary.carbon_g_total_mean, 0), unit: "g CO2" },
    {
      label: "Path Latency",
      value: toNumber(summary.avg_path_latency_ms_mean ?? summary.avg_delay_ms_mean, 0),
      unit: "ms",
    },
    { label: "Packets / Step", value: totalPacketsSent / steps, unit: "pkts" },
    { label: "Dropped / Step", value: dropped / steps, unit: "pkts" },
    { label: "Drop Rate", value: dropRate, unit: "%" },
    { label: "Active Links Ratio", value: toNumber(summary.active_ratio_mean, 0) * 100, unit: "%" },
    { label: "Run Length", value: steps, unit: "steps" },
  ];
}

export function chartRows(rows: PerStepRow[]): Array<Record<string, number>> {
  return rows.map((row) => ({
    t: toNumber(row.t, 0),
    energy_kwh: toNumber(row.energy_kwh, 0),
    avg_delay_ms: toNumber(row.avg_delay_ms, 0),
    avg_path_latency_ms: toNumber(row.avg_path_latency_ms, 0),
    congestion_delay_ms: toNumber(row.congestion_delay_ms, 0),
    dropped: toNumber(row.dropped, 0),
    active_ratio: toNumber(row.active_ratio, 1) * 100,
    delivered: toNumber(row.delivered, 0),
    carbon_g: toNumber(row.carbon_g, 0),
    reward: toNumber(row.reward, 0),
  }));
}

export function inferPolicy(run: RunSummary): string {
  if (run.policy && run.policy.trim()) {
    return run.policy.toLowerCase();
  }

  const runId = run.run_id.toLowerCase();
  if (runId.includes("baseline") || runId.includes("all_on")) {
    return "baseline";
  }
  if (runId.includes("noop") || runId.includes("heuristic")) {
    return "noop";
  }
  if (runId.includes("ppo") || runId.includes("rl")) {
    return "ppo";
  }
  return "unknown";
}

export function latestRunByPolicy(runs: RunSummary[], policy: string): RunSummary | null {
  const filtered = runs.filter((run) => inferPolicy(run) === policy.toLowerCase());
  if (!filtered.length) {
    return null;
  }
  return filtered[0];
}

function radialLayout(nodeCount: number): Array<{ x: number; y: number }> {
  return Array.from({ length: nodeCount }, (_, idx) => {
    const angle = (Math.PI * 2 * idx) / nodeCount;
    const radius = idx % 2 === 0 ? 0.38 : 0.3;
    const x = 0.5 + Math.cos(angle) * radius;
    const y = 0.5 + Math.sin(angle) * radius;
    return { x, y };
  });
}

export function fallbackTopology(seedSource: string, nodeCount = 12): TopologyData {
  const seed = hashSeed(seedSource || "greennet-default");
  const rng = seeded(seed);

  const basePositions = radialLayout(nodeCount);
  const nodes: TopologyNode[] = Array.from({ length: nodeCount }, (_, idx) => {
    const jitterX = (rng() - 0.5) * 0.08;
    const jitterY = (rng() - 0.5) * 0.08;
    return {
      id: `N${idx + 1}`,
      label: `N${idx + 1}`,
      x: Math.min(0.92, Math.max(0.08, basePositions[idx].x + jitterX)),
      y: Math.min(0.9, Math.max(0.1, basePositions[idx].y + jitterY)),
    };
  });

  const edgeSet = new Set<string>();
  const edges: TopologyEdge[] = [];

  function addEdge(sourceIdx: number, targetIdx: number): void {
    if (sourceIdx === targetIdx) {
      return;
    }

    const source = nodes[sourceIdx].id;
    const target = nodes[targetIdx].id;
    const id = edgeId(source, target);

    if (edgeSet.has(id)) {
      return;
    }

    edgeSet.add(id);
    edges.push({ id, source, target });
  }

  for (let idx = 0; idx < nodeCount; idx += 1) {
    addEdge(idx, (idx + 1) % nodeCount);
  }

  for (let idx = 0; idx < nodeCount; idx += 1) {
    addEdge(idx, (idx + 3) % nodeCount);
  }

  for (let idx = 0; idx < nodeCount; idx += 1) {
    if (rng() > 0.6) {
      addEdge(idx, Math.floor(rng() * nodeCount));
    }
  }

  return { nodes, edges };
}

export function withLayout(topology: TopologyData): TopologyData {
  const nodesMissingCoordinates = topology.nodes.some((node) => node.x == null || node.y == null);
  if (!nodesMissingCoordinates) {
    return topology;
  }

  const positions = radialLayout(topology.nodes.length || 1);
  return {
    ...topology,
    nodes: topology.nodes.map((node, index) => ({
      ...node,
      x: node.x ?? positions[index].x,
      y: node.y ?? positions[index].y,
    })),
  };
}

export function linkStateFromRatio(edges: TopologyEdge[], ratio: number, step: number): LinkStateMap {
  if (!edges.length) {
    return {};
  }

  const targetOn = Math.max(1, Math.round(edges.length * Math.min(1, Math.max(0, ratio))));
  const offset = step % edges.length;

  const states: LinkStateMap = {};
  for (let idx = 0; idx < edges.length; idx += 1) {
    const edge = edges[(idx + offset) % edges.length];
    states[edge.id] = idx < targetOn;
  }

  return states;
}

export function timelineFromRows(rows: PerStepRow[], topology: TopologyData): StepState[] {
  return rows.map((row) => {
    const metrics = toMetrics(row);
    return {
      t: metrics.t,
      metrics,
      links_on: linkStateFromRatio(topology.edges, metrics.active_ratio, metrics.t),
    };
  });
}

export function compareSummary(rows: PerStepRow[], overall?: RunOverallSummary | null): Record<string, number> {
  const summary = overall ?? deriveOverallFromRows(rows);
  const delivered = toNumber(summary?.delivered_total_mean, 0);
  const dropped = toNumber(summary?.dropped_total_mean, 0);

  return {
    energy_kwh: toNumber(summary?.energy_kwh_total_mean, 0),
    carbon_g: toNumber(summary?.carbon_g_total_mean, 0),
    avg_delay_ms: toNumber(summary?.avg_path_latency_ms_mean ?? summary?.avg_delay_ms_mean, 0),
    congestion_delay_ms: 0,
    packets_sent: delivered + dropped,
    dropped,
    delivered,
    active_ratio: toNumber(summary?.active_ratio_mean, 0) * 100,
    reward: toNumber(summary?.reward_total_mean, 0),
  };
}

export function officialLockedScenarioMetrics(result: OfficialLockedResult): KpiMetric[] {
  const trained = result.trained_det;
  const delta = result.summary ?? result.delta_summary;

  if (!trained) {
    return [];
  }

  return [
    { label: "Delivered (mean)", value: toNumber(trained.delivered_mean, 0), unit: "pkts" },
    { label: "Dropped (mean)", value: toNumber(trained.dropped_mean, 0), unit: "pkts" },
    { label: "Drop Rate", value: toNumber(trained.drop_rate, 0) * 100, unit: "%" },
    { label: "Energy (mean)", value: toNumber(trained.energy_kwh_mean, 0), unit: "kWh" },
    { label: "Drop Delta vs No-Op", value: toNumber(delta?.delta_dropped, 0), unit: "pkts" },
    { label: "Energy Delta vs No-Op", value: toNumber(delta?.delta_energy_kwh, 0), unit: "kWh" },
  ];
}

export function fmt(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "-";
}

export function formatPolicyLabel(policy: string): string {
  const normalized = policy.toLowerCase();
  if (normalized === "ppo") {
    return "PPO";
  }
  if (normalized === "noop") {
    return "No-Op";
  }
  if (normalized === "baseline") {
    return "Baseline";
  }
  return policy;
}
