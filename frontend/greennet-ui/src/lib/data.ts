import type {
  KpiMetric,
  LinkStateMap,
  PerStepRow,
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
        energy_kwh: toNumber(row.energy_kwh, 0),
        carbon_g: toNumber(row.carbon_g, 0),
        avg_delay_ms: toNumber(row.avg_delay_ms, 0),
        dropped: toNumber(row.dropped, 0),
        delivered: toNumber(row.delivered, 0),
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

export function kpiFromRows(rows: PerStepRow[]): KpiMetric[] {
  const last = latestRow(rows);
  if (!last) {
    return [];
  }

  return [
    { label: "Energy Usage", value: toNumber(last.energy_kwh, 0), unit: "kWh" },
    { label: "Carbon Emissions", value: toNumber(last.carbon_g, 0), unit: "g CO2" },
    { label: "Avg Delay", value: toNumber(last.avg_delay_ms, 0), unit: "ms" },
    { label: "Drop Rate", value: toNumber(last.dropped, 0), unit: "pkts" },
    { label: "Active Links Ratio", value: toNumber(last.active_ratio, 0) * 100, unit: "%" },
    { label: "Delivered", value: toNumber(last.delivered, 0), unit: "pkts" },
  ];
}

export function chartRows(rows: PerStepRow[]): Array<Record<string, number>> {
  return rows.map((row) => ({
    t: toNumber(row.t, 0),
    energy_kwh: toNumber(row.energy_kwh, 0),
    avg_delay_ms: toNumber(row.avg_delay_ms, 0),
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

export function compareSummary(rows: PerStepRow[]): Record<string, number> {
  const last = latestRow(rows);
  const rewardTotal = rows.reduce((sum, row) => sum + toNumber(row.reward, 0), 0);

  return {
    energy_kwh: toNumber(last?.energy_kwh, 0),
    carbon_g: toNumber(last?.carbon_g, 0),
    avg_delay_ms: toNumber(last?.avg_delay_ms, 0),
    dropped: toNumber(last?.dropped, 0),
    delivered: toNumber(last?.delivered, 0),
    active_ratio: toNumber(last?.active_ratio, 1) * 100,
    reward: rewardTotal,
  };
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
