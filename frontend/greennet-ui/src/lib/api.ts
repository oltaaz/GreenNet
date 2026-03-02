import type {
  LinkStateMap,
  PacketEvent,
  PerStepRow,
  RunSummary,
  StartRunParams,
  StepState,
  TopologyData,
  TopologyEdge,
  TopologyNode,
} from "./types";
import {
  createDemoRun,
  getDemoPerStep,
  getDemoSteps,
  getDemoTopology,
  isDemoRunId,
  listDemoRuns,
} from "./demo";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

function withBase(path: string): string {
  if (!API_BASE_URL) {
    return path;
  }
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(withBase(path), {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
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

function toStringSafe(value: unknown, fallback = ""): string {
  if (typeof value === "string" && value.trim() !== "") {
    return value;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return fallback;
}

function edgeId(source: string, target: string): string {
  return source < target ? `${source}__${target}` : `${target}__${source}`;
}

function normalizeRun(row: unknown): RunSummary {
  const item = (row ?? {}) as Record<string, unknown>;
  return {
    run_id: toStringSafe(item.run_id ?? item.id, ""),
    started_at: (item.started_at as string | null | undefined) ?? null,
    policy: (item.policy as string | null | undefined) ?? null,
    scenario: (item.scenario as string | null | undefined) ?? null,
    topology_seed: item.topology_seed == null ? null : toNumber(item.topology_seed, 0),
  };
}

function normalizePerStepRow(row: unknown, index: number): PerStepRow {
  const item = (row ?? {}) as Record<string, unknown>;

  const tRaw = item.t ?? item.step ?? item.time;
  const t = Number.isFinite(Number(tRaw)) ? Number(tRaw) : index;

  const energyKwh = item.energy_kwh ?? item.energy ?? item.delta_energy_kwh;

  return {
    t,
    energy_kwh: toNumber(energyKwh, 0),
    carbon_g: toNumber(item.carbon_g ?? item.delta_carbon_g, 0),
    avg_delay_ms: toNumber(item.avg_delay_ms ?? item.average_delay_ms, 0),
    dropped: toNumber(item.dropped ?? item.delta_dropped, 0),
    delivered: toNumber(item.delivered ?? item.delta_delivered, 0),
    active_ratio: toNumber(item.active_ratio, 1),
    reward: toNumber(item.reward ?? item.total_reward, 0),
    ...item,
  };
}

function normalizeNodes(rawNodes: unknown): TopologyNode[] {
  if (!Array.isArray(rawNodes)) {
    return [];
  }

  const nodes: TopologyNode[] = [];
  for (const node of rawNodes) {
    const item = (node ?? {}) as Record<string, unknown>;
    const id = toStringSafe(item.id ?? item.node_id, "");
    if (!id) {
      continue;
    }

    nodes.push({
      id,
      label: toStringSafe(item.label, id),
      x: Number.isFinite(Number(item.x)) ? Number(item.x) : undefined,
      y: Number.isFinite(Number(item.y)) ? Number(item.y) : undefined,
    });
  }

  return nodes;
}

function normalizeEdges(rawEdges: unknown): TopologyEdge[] {
  if (!Array.isArray(rawEdges)) {
    return [];
  }

  const edges: TopologyEdge[] = [];
  for (const edge of rawEdges) {
    const item = (edge ?? {}) as Record<string, unknown>;
    const source = toStringSafe(item.source ?? item.u ?? item.from ?? item.src, "");
    const target = toStringSafe(item.target ?? item.v ?? item.to ?? item.dst, "");

    if (!source || !target) {
      continue;
    }

    edges.push({
      id: toStringSafe(item.id, edgeId(source, target)),
      source,
      target,
    });
  }

  return edges;
}

function normalizeLinkState(payload: unknown): LinkStateMap {
  const item = (payload ?? {}) as Record<string, unknown>;

  if (item.links_on && typeof item.links_on === "object" && !Array.isArray(item.links_on)) {
    const linksOn = item.links_on as Record<string, unknown>;
    return Object.fromEntries(Object.entries(linksOn).map(([key, value]) => [key, Boolean(value)]));
  }

  if (Array.isArray(item.links_on)) {
    const entries = item.links_on as Array<Record<string, unknown>>;
    const normalized: LinkStateMap = {};
    for (const entry of entries) {
      const source = toStringSafe(entry.source ?? entry.u, "");
      const target = toStringSafe(entry.target ?? entry.v, "");
      const id = toStringSafe(entry.id, source && target ? edgeId(source, target) : "");
      if (!id) {
        continue;
      }
      normalized[id] = Boolean(entry.on ?? entry.active ?? entry.enabled);
    }
    return normalized;
  }

  return {};
}

async function tryEndpoints<T>(requests: Array<() => Promise<T>>): Promise<T> {
  let lastError: unknown = new Error("No endpoints attempted");

  for (const request of requests) {
    try {
      return await request();
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError;
}

async function tryOptionalEndpoints<T>(requests: Array<() => Promise<T>>): Promise<T | null> {
  try {
    return await tryEndpoints(requests);
  } catch {
    return null;
  }
}

export async function listRuns(): Promise<RunSummary[]> {
  try {
    const rows = await tryEndpoints<unknown[]>([
      () => requestJson<unknown[]>("/api/runs"),
      () => requestJson<unknown[]>("/runs"),
    ]);

    const normalized = rows.map(normalizeRun).filter((run) => run.run_id);
    if (normalized.length > 0) {
      return normalized;
    }
  } catch {
    // Fall back to demo runs.
  }

  return listDemoRuns();
}

export async function getRunPerStep(runId: string): Promise<PerStepRow[]> {
  if (isDemoRunId(runId)) {
    return getDemoPerStep(runId);
  }

  try {
    const rows = await tryEndpoints<unknown[]>([
      () => requestJson<unknown[]>(`/api/runs/${runId}/per_step`),
      () => requestJson<unknown[]>(`/runs/${runId}/per_step`),
      () => requestJson<unknown[]>(`/api/runs/${runId}/steps`),
    ]);

    return rows.map((row, index) => normalizePerStepRow(row, index));
  } catch {
    return getDemoPerStep("demo-ppo-normal-42-300");
  }
}

export async function startRun(params: StartRunParams): Promise<{ run_id: string }> {
  const payload = {
    policy: params.policy,
    scenario: params.scenario,
    seed: params.seed,
    steps: params.steps,
  };

  try {
    return await tryEndpoints<{ run_id: string }>([
      () => requestJson<{ run_id: string }>("/api/runs/start", { method: "POST", body: JSON.stringify(payload) }),
      () => requestJson<{ run_id: string }>("/api/start_run", { method: "POST", body: JSON.stringify(payload) }),
      () => requestJson<{ run_id: string }>("/api/simulate/start", { method: "POST", body: JSON.stringify(payload) }),
    ]);
  } catch {
    return createDemoRun(params);
  }
}

export async function getTopology(runId?: string): Promise<TopologyData | null> {
  if (runId && isDemoRunId(runId)) {
    return getDemoTopology(runId);
  }

  const payload = await tryOptionalEndpoints<Record<string, unknown>>([
    () => requestJson<Record<string, unknown>>(`/api/runs/${runId ?? "latest"}/topology`),
    () => requestJson<Record<string, unknown>>(`/api/topology${runId ? `?run_id=${encodeURIComponent(runId)}` : ""}`),
    () => requestJson<Record<string, unknown>>("/api/network/topology"),
  ]);

  if (!payload) {
    return getDemoTopology(runId ?? "demo-ppo-normal-42-300");
  }

  const nodes = normalizeNodes(payload.nodes ?? payload.vertices ?? payload.graph_nodes);
  const edges = normalizeEdges(payload.edges ?? payload.links ?? payload.graph_edges);

  if (!nodes.length || !edges.length) {
    return getDemoTopology(runId ?? "demo-ppo-normal-42-300");
  }

  return { nodes, edges };
}

export async function getLinkState(runId: string, step: number): Promise<{ links_on: LinkStateMap } | null> {
  if (isDemoRunId(runId)) {
    const steps = getDemoSteps(runId);
    const index = Math.max(0, Math.min(steps.length - 1, Math.round(step)));
    return { links_on: steps[index]?.links_on ?? {} };
  }

  const payload = await tryOptionalEndpoints<Record<string, unknown>>([
    () => requestJson<Record<string, unknown>>(`/api/runs/${runId}/link_state?step=${step}`),
    () => requestJson<Record<string, unknown>>(`/api/runs/${runId}/links?step=${step}`),
    () => requestJson<Record<string, unknown>>(`/api/link_state?run_id=${encodeURIComponent(runId)}&step=${step}`),
  ]);

  if (!payload) {
    return null;
  }

  return { links_on: normalizeLinkState(payload) };
}

export async function getSteps(runId: string): Promise<StepState[] | null> {
  if (isDemoRunId(runId)) {
    return getDemoSteps(runId);
  }

  const rows = await tryOptionalEndpoints<unknown[]>([
    () => requestJson<unknown[]>(`/api/runs/${runId}/steps`),
    () => requestJson<unknown[]>(`/api/runs/${runId}/timeline`),
  ]);

  if (!rows) {
    return getDemoSteps("demo-ppo-normal-42-300");
  }

  return rows.map((row, index) => {
    const item = (row ?? {}) as Record<string, unknown>;
    const metrics = normalizePerStepRow(item.metrics ?? item, index);
    const tRaw = item.t ?? item.step ?? metrics.t;

    return {
      t: Number.isFinite(Number(tRaw)) ? Number(tRaw) : index,
      metrics: {
        t: metrics.t,
        energy_kwh: toNumber(metrics.energy_kwh, 0),
        carbon_g: toNumber(metrics.carbon_g, 0),
        avg_delay_ms: toNumber(metrics.avg_delay_ms, 0),
        dropped: toNumber(metrics.dropped, 0),
        delivered: toNumber(metrics.delivered, 0),
        active_ratio: toNumber(metrics.active_ratio, 1),
        reward: toNumber(metrics.reward, 0),
      },
      links_on: normalizeLinkState(item),
      packet_events: Array.isArray(item.packet_events) ? (item.packet_events as PacketEvent[]) : undefined,
    };
  });
}

export async function getPacketEvents(runId: string, step: number): Promise<PacketEvent[] | null> {
  if (isDemoRunId(runId)) {
    return [];
  }

  const payload = await tryOptionalEndpoints<Record<string, unknown>>([
    () => requestJson<Record<string, unknown>>(`/api/runs/${runId}/packet_events?step=${step}`),
    () => requestJson<Record<string, unknown>>(`/api/packets?run_id=${encodeURIComponent(runId)}&step=${step}`),
  ]);

  if (!payload) {
    return null;
  }

  if (Array.isArray(payload.events)) {
    return payload.events as PacketEvent[];
  }

  if (Array.isArray(payload.packet_events)) {
    return payload.packet_events as PacketEvent[];
  }

  return null;
}
