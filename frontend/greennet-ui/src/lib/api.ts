import type {
  LinkStateMap,
  OfficialLockedDeltaSummary,
  OfficialLockedEvalRow,
  OfficialLockedResult,
  OfficialLockedStats,
  PacketEvent,
  PerStepRow,
  RunOverallSummary,
  RunSummary,
  StartRunParams,
  StepState,
  TopologyData,
  TopologyEdge,
  TopologyNode,
} from "./types";
import { deriveOverallFromRows, fallbackTopology } from "./data";
import {
  getDemoPerStep,
  getDemoSteps,
  getDemoTopology,
  isDemoRunId,
  listDemoRuns,
} from "./demo";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const CURATED_RUN_TAG = (import.meta.env.VITE_RUN_TAG ?? "dashboard").trim();

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

function toOptionalNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
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
  const highlights =
    item.highlights && typeof item.highlights === "object"
      ? (item.highlights as Record<string, unknown>)
      : {};

  return {
    run_id: toStringSafe(item.run_id ?? item.id, ""),
    started_at: (item.started_at as string | null | undefined) ?? null,
    policy: (item.policy as string | null | undefined) ?? null,
    scenario: (item.scenario as string | null | undefined) ?? null,
    seed: item.seed == null ? null : toNumber(item.seed, 0),
    topology_seed: item.topology_seed == null ? null : toNumber(item.topology_seed, 0),
    max_steps: item.max_steps == null ? null : toNumber(item.max_steps, 0),
    tag: (item.tag as string | null | undefined) ?? null,
    reward_total_mean: toOptionalNumber(item.reward_total_mean ?? highlights.reward_total_mean),
    dropped_total_mean: toOptionalNumber(item.dropped_total_mean ?? highlights.dropped_total_mean),
    energy_kwh_total_mean: toOptionalNumber(item.energy_kwh_total_mean ?? highlights.energy_kwh_total_mean),
  };
}

function normalizeOverallSummary(payload: unknown): RunOverallSummary {
  const root = (payload ?? {}) as Record<string, unknown>;
  const item = (root.overall && typeof root.overall === "object" ? root.overall : root) as Record<string, unknown>;
  const avgPathLatency = item.avg_path_latency_ms_mean ?? item.avg_path_latency_ms;

  return {
    reward_total_mean: toNumber(item.reward_total_mean ?? item.reward_total, 0),
    delivered_total_mean: toNumber(item.delivered_total_mean ?? item.delivered_total, 0),
    dropped_total_mean: toNumber(item.dropped_total_mean ?? item.dropped_total, 0),
    energy_kwh_total_mean: toNumber(item.energy_kwh_total_mean ?? item.energy_kwh_total, 0),
    carbon_g_total_mean: toNumber(item.carbon_g_total_mean ?? item.carbon_g_total, 0),
    avg_utilization_mean: toNumber(item.avg_utilization_mean ?? item.avg_utilization, 0),
    active_ratio_mean: toNumber(item.active_ratio_mean ?? item.active_ratio, 0),
    avg_delay_ms_mean: toNumber(item.avg_delay_ms_mean ?? item.avg_delay_ms, 0),
    avg_path_latency_ms_mean: avgPathLatency == null ? undefined : toNumber(avgPathLatency, 0),
    steps_mean: item.steps_mean == null ? undefined : toNumber(item.steps_mean, 0),
  };
}

function normalizeOfficialEvalRow(payload: unknown): OfficialLockedEvalRow | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }

  const item = payload as Record<string, unknown>;
  return {
    scenario: toStringSafe(item.scenario, "").toLowerCase(),
    off_level: toStringSafe(item.off_level, ""),
    pass: Boolean(item.pass),
    delta_reward: toOptionalNumber(item.delta_reward),
    delta_energy_kwh: toOptionalNumber(item.delta_energy_kwh),
    delta_dropped: toOptionalNumber(item.delta_dropped),
    on_edges_mean: toOptionalNumber(item.on_edges_mean),
    toggles_applied_mean: toOptionalNumber(item.toggles_applied_mean),
    blocked_on_actions_mean: toOptionalNumber(item.blocked_on_actions_mean),
    cap_used: (item.cap_used as string | null | undefined) ?? null,
    seeds: (item.seeds as string | null | undefined) ?? null,
    episodes: item.episodes == null ? null : toNumber(item.episodes, 0),
    log_file: (item.log_file as string | null | undefined) ?? null,
  };
}

function normalizeOfficialStats(payload: unknown): OfficialLockedStats | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }

  const item = payload as Record<string, unknown>;
  return {
    reward_mean: toNumber(item.reward_mean, 0),
    energy_kwh_mean: toNumber(item.energy_kwh_mean, 0),
    dropped_mean: toNumber(item.dropped_mean, 0),
    delivered_mean: toNumber(item.delivered_mean, 0),
    sent_mean: toNumber(item.sent_mean, 0),
    drop_rate: toNumber(item.drop_rate, 0),
    toggles_applied_mean: toOptionalNumber(item.toggles_applied_mean),
    on_edges_mean: toOptionalNumber(item.on_edges_mean),
    blocked_on_actions_mean: toOptionalNumber(item.blocked_on_actions_mean),
  };
}

function normalizeOfficialDeltaSummary(payload: unknown): OfficialLockedDeltaSummary | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }

  const item = payload as Record<string, unknown>;
  return {
    better: Boolean(item.better),
    reason: (item.reason as string | null | undefined) ?? null,
    delta_reward: toOptionalNumber(item.delta_reward),
    delta_energy_kwh: toOptionalNumber(item.delta_energy_kwh),
    delta_dropped: toOptionalNumber(item.delta_dropped),
  };
}

function normalizeOfficialResult(payload: unknown): OfficialLockedResult | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }

  const item = payload as Record<string, unknown>;
  const summary = normalizeOfficialEvalRow(item.summary);
  const evalRows = toArrayPayload(item.eval_rows)
    .map(normalizeOfficialEvalRow)
    .filter((row): row is OfficialLockedEvalRow => row !== null);

  return {
    scenario: toStringSafe(item.scenario, "").toLowerCase(),
    bundle_id: toStringSafe(item.bundle_id, ""),
    bundle_path: (item.bundle_path as string | null | undefined) ?? null,
    pass_all: item.pass_all == null ? null : Boolean(item.pass_all),
    summary,
    eval_rows: evalRows,
    trained_det: normalizeOfficialStats(item.trained_det),
    noop_det: normalizeOfficialStats(item.noop_det),
    delta_summary: normalizeOfficialDeltaSummary(item.delta_summary),
    notes: (item.notes as string | null | undefined) ?? null,
  };
}

function normalizePerStepRow(row: unknown, index: number): PerStepRow {
  const item = (row ?? {}) as Record<string, unknown>;

  const tRaw = item.t ?? item.step ?? item.time;
  const t = Number.isFinite(Number(tRaw)) ? Number(tRaw) : index;

  // Prefer per-step delta fields when present so demo and backend runs use the same semantics.
  const energyKwh = item.delta_energy_kwh ?? item.energy_kwh ?? item.energy;
  const carbonG = item.delta_carbon_g ?? item.carbon_g;
  const rawAvgDelayMs = item.avg_delay_ms ?? item.average_delay_ms;
  const avgPathLatencyMs = item.avg_path_latency_ms ?? item.path_latency_ms;
  const explicitCongestionDelayMs = item.congestion_delay_ms ?? item.queue_delay_ms ?? item.congestion_ms;
  const computedCongestionDelayMs =
    explicitCongestionDelayMs ??
    (rawAvgDelayMs != null && avgPathLatencyMs != null
      ? Math.max(toNumber(rawAvgDelayMs, 0) - toNumber(avgPathLatencyMs, 0), 0)
      : 0);
  const displayDelayMs = avgPathLatencyMs ?? rawAvgDelayMs;

  return {
    ...item,
    t,
    energy_kwh: toNumber(energyKwh, 0),
    carbon_g: toNumber(carbonG, 0),
    avg_delay_ms: toNumber(displayDelayMs, 0),
    avg_path_latency_ms: toNumber(avgPathLatencyMs, 0),
    congestion_delay_ms: toNumber(computedCongestionDelayMs, 0),
    dropped: toNumber(item.delta_dropped ?? item.dropped, 0),
    delivered: toNumber(item.delta_delivered ?? item.delivered, 0),
    active_ratio: toNumber(item.active_ratio, 1),
    reward: toNumber(item.reward ?? item.total_reward, 0),
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

function toArrayPayload(payload: unknown): unknown[] {
  if (Array.isArray(payload)) {
    return payload;
  }

  const item = (payload ?? {}) as Record<string, unknown>;
  if (Array.isArray(item.items)) {
    return item.items;
  }
  if (Array.isArray(item.runs)) {
    return item.runs;
  }
  if (Array.isArray(item.data)) {
    return item.data;
  }

  return [];
}

function collapsePerStepRows(rows: PerStepRow[]): PerStepRow[] {
  const buckets = new Map<
    number,
    {
      count: number;
      energy_kwh: number;
      carbon_g: number;
      avg_delay_ms: number;
      avg_path_latency_ms: number;
      congestion_delay_ms: number;
      dropped: number;
      delivered: number;
      active_ratio: number;
      reward: number;
    }
  >();

  for (const row of rows) {
    const t = toNumber(row.t, 0);
    const bucket = buckets.get(t) ?? {
      count: 0,
      energy_kwh: 0,
      carbon_g: 0,
      avg_delay_ms: 0,
      avg_path_latency_ms: 0,
      congestion_delay_ms: 0,
      dropped: 0,
      delivered: 0,
      active_ratio: 0,
      reward: 0,
    };

    bucket.count += 1;
    bucket.energy_kwh += toNumber(row.energy_kwh, 0);
    bucket.carbon_g += toNumber(row.carbon_g, 0);
    bucket.avg_delay_ms += toNumber(row.avg_delay_ms, 0);
    bucket.avg_path_latency_ms += toNumber(row.avg_path_latency_ms, 0);
    bucket.congestion_delay_ms += toNumber(row.congestion_delay_ms, 0);
    bucket.dropped += toNumber(row.dropped, 0);
    bucket.delivered += toNumber(row.delivered, 0);
    bucket.active_ratio += toNumber(row.active_ratio, 1);
    bucket.reward += toNumber(row.reward, 0);
    buckets.set(t, bucket);
  }

  return [...buckets.entries()]
    .map(([t, bucket]) => ({
      t,
      energy_kwh: bucket.energy_kwh / bucket.count,
      carbon_g: bucket.carbon_g / bucket.count,
      avg_delay_ms: bucket.avg_delay_ms / bucket.count,
      avg_path_latency_ms: bucket.avg_path_latency_ms / bucket.count,
      congestion_delay_ms: bucket.congestion_delay_ms / bucket.count,
      dropped: bucket.dropped / bucket.count,
      delivered: bucket.delivered / bucket.count,
      active_ratio: bucket.active_ratio / bucket.count,
      reward: bucket.reward / bucket.count,
    }))
    .sort((a, b) => a.t - b.t);
}

function dedupeCuratedRuns(runs: RunSummary[]): RunSummary[] {
  const selected = new Map<string, RunSummary>();
  const order: string[] = [];

  for (const run of runs) {
    const policy = toStringSafe(run.policy, "unknown").toLowerCase();
    const scenario = toStringSafe(run.scenario, "unknown").toLowerCase();
    const key = `${policy}::${scenario}`;
    const current = selected.get(key);

    if (!current) {
      order.push(key);
      selected.set(key, run);
      continue;
    }

    const currentDropped = current.dropped_total_mean ?? Number.POSITIVE_INFINITY;
    const nextDropped = run.dropped_total_mean ?? Number.POSITIVE_INFINITY;
    const currentEnergy = current.energy_kwh_total_mean ?? Number.POSITIVE_INFINITY;
    const nextEnergy = run.energy_kwh_total_mean ?? Number.POSITIVE_INFINITY;
    const currentReward = current.reward_total_mean ?? Number.NEGATIVE_INFINITY;
    const nextReward = run.reward_total_mean ?? Number.NEGATIVE_INFINITY;

    const isBetter =
      nextDropped < currentDropped ||
      (nextDropped === currentDropped && nextEnergy < currentEnergy) ||
      (nextDropped === currentDropped && nextEnergy === currentEnergy && nextReward > currentReward);

    if (isBetter) {
      selected.set(key, run);
    }
  }

  return order.map((key) => selected.get(key)).filter((run): run is RunSummary => Boolean(run));
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
  const requests: Array<{ curated: boolean; load: () => Promise<unknown> }> = [];

  if (CURATED_RUN_TAG) {
    const query = `?tag=${encodeURIComponent(CURATED_RUN_TAG)}&limit=200`;
    requests.push(
      { curated: true, load: () => requestJson<unknown>(`/api/runs${query}`) },
      { curated: true, load: () => requestJson<unknown>(`/api/runs_flat${query}`) },
    );
  }

  requests.push(
    { curated: false, load: () => requestJson<unknown>("/api/runs?limit=200") },
    { curated: false, load: () => requestJson<unknown>("/api/runs_flat?limit=200") },
    { curated: false, load: () => requestJson<unknown>("/runs") },
  );

  for (const request of requests) {
    try {
      const payload = await request.load();
      const rows = toArrayPayload(payload);
      const normalized = rows.map(normalizeRun).filter((run) => run.run_id);
      if (normalized.length > 0) {
        return request.curated ? dedupeCuratedRuns(normalized) : normalized;
      }
    } catch {
      // Try the next source.
    }
  }

  return listDemoRuns();
}

export async function getOfficialLockedResults(scenarios: string[] = []): Promise<OfficialLockedResult[]> {
  const normalizedScenarios = scenarios.map((value) => value.trim().toLowerCase()).filter(Boolean);
  const query =
    normalizedScenarios.length > 0
      ? `?scenario=${encodeURIComponent(normalizedScenarios.join(","))}`
      : "";

  const payload = await requestJson<unknown>(`/api/official_results${query}`);
  return toArrayPayload(payload)
    .map(normalizeOfficialResult)
    .filter((result): result is OfficialLockedResult => result !== null && Boolean(result.bundle_id));
}

export async function getRunPerStep(runId: string): Promise<PerStepRow[]> {
  if (isDemoRunId(runId)) {
    return getDemoPerStep(runId);
  }

  const payload = await tryEndpoints<unknown>([
    () => requestJson<unknown>(`/api/runs/${runId}/per_step`),
    () => requestJson<unknown>(`/runs/${runId}/per_step`),
  ]);

  const rows = toArrayPayload(payload).map((row, index) => normalizePerStepRow(row, index));
  if (!rows.length) {
    throw new Error(`No per-step data found for run ${runId}`);
  }

  return collapsePerStepRows(rows);
}

export async function getRunSummary(runId: string): Promise<RunOverallSummary | null> {
  if (isDemoRunId(runId)) {
    return deriveOverallFromRows(getDemoPerStep(runId));
  }

  const payload = await tryOptionalEndpoints<unknown>([
    () => requestJson<unknown>(`/api/runs/${runId}/summary?mode=overall`),
    () => requestJson<unknown>(`/api/runs/${runId}/summary`),
  ]);

  if (payload) {
    return normalizeOverallSummary(payload);
  }

  try {
    return deriveOverallFromRows(await getRunPerStep(runId));
  } catch {
    return null;
  }
}

export async function startRun(params: StartRunParams): Promise<{ run_id: string }> {
  const payload = {
    policy: params.policy,
    scenario: params.scenario,
    seed: params.seed,
    steps: params.steps,
  };

  return await tryEndpoints<{ run_id: string }>([
    () => requestJson<{ run_id: string }>("/api/runs/start", { method: "POST", body: JSON.stringify(payload) }),
    () => requestJson<{ run_id: string }>("/api/start_run", { method: "POST", body: JSON.stringify(payload) }),
    () => requestJson<{ run_id: string }>("/api/simulate/start", { method: "POST", body: JSON.stringify(payload) }),
  ]);
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
    if (runId) {
      const envPayload = await tryOptionalEndpoints<Record<string, unknown>>([
        () => requestJson<Record<string, unknown>>(`/api/runs/${runId}/env`),
      ]);

      if (envPayload) {
        const nodeCount = Math.max(2, toNumber(envPayload.node_count, 12));
        return fallbackTopology(`${runId}-${toStringSafe(envPayload.topology_seed, runId)}`, nodeCount);
      }
    }

    return null;
  }

  const nodes = normalizeNodes(payload.nodes ?? payload.vertices ?? payload.graph_nodes);
  const edges = normalizeEdges(payload.edges ?? payload.links ?? payload.graph_edges);

  if (!nodes.length || !edges.length) {
    return null;
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
    return null;
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
        avg_path_latency_ms: toNumber(metrics.avg_path_latency_ms, 0),
        congestion_delay_ms: toNumber(metrics.congestion_delay_ms, 0),
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
