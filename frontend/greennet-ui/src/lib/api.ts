import type {
  FinalEvaluationArtifact,
  FinalEvaluationClassification,
  FinalEvaluationReport,
  FinalEvaluationSource,
  FinalEvaluationSummaryRow,
  FinalEvaluationThresholds,
  LinkStateMap,
  BackendHealth,
  OfficialLockedDeltaSummary,
  OfficialLockedEvalRow,
  OfficialLockedResult,
  OfficialLockedStats,
  PacketEvent,
  PerStepRow,
  RunOverallSummary,
  RunSummary,
  StabilityThresholds,
  StartRunParams,
  StepState,
  TopologyData,
  TopologyEdge,
  TopologyNode,
} from "./types";
import { deriveOverallFromRows, fallbackTopology, normalizePolicy } from "./data";
import {
  getDemoPerStep,
  getDemoSteps,
  getDemoTopology,
  isDemoRunId,
  listDemoRuns,
} from "./demo";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
export const EXPECTED_BACKEND_URL = API_BASE_URL || "http://127.0.0.1:8000";

export class BackendUnavailableError extends Error {
  expectedUrl: string;

  constructor(expectedUrl = EXPECTED_BACKEND_URL) {
    super(`Backend API unavailable. Expected GreenNet FastAPI at ${expectedUrl}. Start the backend first.`);
    this.name = "BackendUnavailableError";
    this.expectedUrl = expectedUrl;
  }
}

function withBase(path: string): string {
  if (!API_BASE_URL) {
    return path;
  }
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(withBase(path), {
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
      ...init,
    });
  } catch {
    throw new BackendUnavailableError();
  }

  if (!response.ok) {
    if ([502, 503, 504].includes(response.status)) {
      throw new BackendUnavailableError();
    }
    throw new Error(`${response.status} ${response.statusText}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export function isBackendUnavailableError(error: unknown): error is BackendUnavailableError {
  return error instanceof BackendUnavailableError;
}

export async function getBackendHealth(): Promise<BackendHealth> {
  const payload = await requestJson<Record<string, unknown>>("/api/health");
  return {
    status: toStringSafe(payload.status, "ok"),
    apiBaseUrl: API_BASE_URL,
    expectedBackendUrl: EXPECTED_BACKEND_URL,
  };
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

function normalizeQosAcceptanceThresholds(payload: unknown): FinalEvaluationThresholds | undefined {
  if (!payload || typeof payload !== "object") {
    return undefined;
  }

  const item = payload as Record<string, unknown>;
  return {
    energy_target_pct: toNumber(item.energy_target_pct, 0),
    max_delivered_loss_pct: toNumber(item.max_delivered_loss_pct, 0),
    max_dropped_increase_pct: toNumber(item.max_dropped_increase_pct, 0),
    max_delay_increase_pct: toNumber(item.max_delay_increase_pct, 0),
    max_path_latency_increase_pct: toNumber(item.max_path_latency_increase_pct, 0),
    max_qos_violation_rate_increase_abs: toNumber(item.max_qos_violation_rate_increase_abs, 0),
  };
}

function normalizeStabilityThresholds(payload: unknown): StabilityThresholds | undefined {
  if (!payload || typeof payload !== "object") {
    return undefined;
  }

  const item = payload as Record<string, unknown>;
  return {
    decision_interval_steps: toOptionalNumber(item.decision_interval_steps),
    toggle_cooldown_steps: toOptionalNumber(item.toggle_cooldown_steps),
    global_toggle_cooldown_steps: toOptionalNumber(item.global_toggle_cooldown_steps),
    off_calm_steps_required: toOptionalNumber(item.off_calm_steps_required),
    max_off_toggles_per_episode: toOptionalNumber(item.max_off_toggles_per_episode),
    max_total_toggles_per_episode: toOptionalNumber(item.max_total_toggles_per_episode),
    max_emergency_on_toggles_per_episode: toOptionalNumber(item.max_emergency_on_toggles_per_episode),
    emergency_on_bypasses_cooldown:
      item.emergency_on_bypasses_cooldown == null ? undefined : Boolean(item.emergency_on_bypasses_cooldown),
    reversal_window_steps: toOptionalNumber(item.reversal_window_steps),
    reversal_penalty: toOptionalNumber(item.reversal_penalty),
    min_steps_for_assessment: toOptionalNumber(item.min_steps_for_assessment),
    max_transition_rate: toOptionalNumber(item.max_transition_rate),
    max_flap_rate: toOptionalNumber(item.max_flap_rate),
    max_flap_count: toOptionalNumber(item.max_flap_count),
  };
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
    policy: normalizePolicy(item.policy),
    scenario: (item.scenario as string | null | undefined) ?? null,
    seed: item.seed == null ? null : toNumber(item.seed, 0),
    topology_seed: item.topology_seed == null ? null : toNumber(item.topology_seed, 0),
    max_steps: item.max_steps == null ? null : toNumber(item.max_steps, 0),
    tag: (item.tag as string | null | undefined) ?? null,
    source: (item.source as string | null | undefined) ?? null,
    episodes: item.episodes == null ? null : toNumber(item.episodes, 0),
    deterministic: item.deterministic == null ? null : Boolean(item.deterministic),
    has:
      item.has && typeof item.has === "object"
        ? {
            per_step: Boolean((item.has as Record<string, unknown>).per_step),
            summary: Boolean((item.has as Record<string, unknown>).summary),
            meta: Boolean((item.has as Record<string, unknown>).meta),
            env_config: Boolean((item.has as Record<string, unknown>).env_config),
          }
        : null,
    reward_total_mean: toOptionalNumber(item.reward_total_mean ?? highlights.reward_total_mean),
    dropped_total_mean: toOptionalNumber(item.dropped_total_mean ?? highlights.dropped_total_mean),
    energy_kwh_total_mean: toOptionalNumber(item.energy_kwh_total_mean ?? highlights.energy_kwh_total_mean),
    transition_rate_mean: toOptionalNumber(item.transition_rate_mean ?? highlights.transition_rate_mean),
    flap_rate_mean: toOptionalNumber(item.flap_rate_mean ?? highlights.flap_rate_mean),
    qos_acceptance_status: toStringSafe(
      item.qos_acceptance_status ?? highlights.qos_acceptance_status ?? item.qos_acceptability_status ?? highlights.qos_acceptability_status,
      "",
    ),
    qos_acceptance_missing: toStringSafe(
      item.qos_acceptance_missing ?? highlights.qos_acceptance_missing ?? item.qos_acceptability_missing ?? highlights.qos_acceptability_missing,
      "",
    ),
    stability_status: toStringSafe(item.stability_status, ""),
    stability_missing: toStringSafe(item.stability_missing, ""),
  };
}

function normalizeOverallSummary(payload: unknown): RunOverallSummary {
  const root = (payload ?? {}) as Record<string, unknown>;
  const item = (root.overall && typeof root.overall === "object" ? root.overall : root) as Record<string, unknown>;
  const avgPathLatency = item.avg_path_latency_ms_mean ?? item.avg_path_latency_ms;
  const qosThresholds = normalizeQosAcceptanceThresholds(
    item.qos_thresholds ?? item.qos_acceptance_thresholds ?? item.qos_acceptability_thresholds ?? item.hypothesis_thresholds,
  );
  const stabilityThresholds = normalizeStabilityThresholds(item.stability_thresholds);

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
    transition_count_total_mean: toOptionalNumber(item.transition_count_total_mean),
    transition_rate_mean: toOptionalNumber(item.transition_rate_mean),
    flap_event_count_total_mean: toOptionalNumber(item.flap_event_count_total_mean),
    flap_rate_mean: toOptionalNumber(item.flap_rate_mean),
    qos_acceptance_status: toStringSafe(
      item.qos_acceptance_status ?? item.qos_acceptability_status ?? root.qos_acceptance_status ?? root.qos_acceptability_status,
      "",
    ),
    qos_acceptance_missing: toStringSafe(
      item.qos_acceptance_missing ?? item.qos_acceptability_missing ?? root.qos_acceptance_missing ?? root.qos_acceptability_missing,
      "",
    ),
    stability_status: toStringSafe(item.stability_status ?? root.stability_status, ""),
    stability_missing: toStringSafe(item.stability_missing ?? root.stability_missing, ""),
    qos_acceptance_thresholds: qosThresholds,
    qos_thresholds: qosThresholds,
    stability_thresholds: stabilityThresholds,
    qos_violation_rate_mean: toOptionalNumber(item.qos_violation_rate_mean ?? root.qos_violation_rate_mean),
    qos_violation_rate_std: toOptionalNumber(item.qos_violation_rate_std ?? root.qos_violation_rate_std),
    qos_violation_rate_count: toOptionalNumber(item.qos_violation_rate_count ?? root.qos_violation_rate_count),
    qos_violation_count_mean: toOptionalNumber(item.qos_violation_count_mean ?? root.qos_violation_count_mean),
    qos_violation_count_std: toOptionalNumber(item.qos_violation_count_std ?? root.qos_violation_count_std),
    qos_violation_count_count: toOptionalNumber(item.qos_violation_count_count ?? root.qos_violation_count_count),
    qos_violation_count_total: toOptionalNumber(item.qos_violation_count_total ?? root.qos_violation_count_total),
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
    qos_acceptance_status: toStringSafe(item.qos_acceptance_status ?? item.qos_acceptability_status, ""),
    qos_acceptance_missing: toStringSafe(item.qos_acceptance_missing ?? item.qos_acceptability_missing, ""),
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
    qos_acceptance_status: toStringSafe(item.qos_acceptance_status ?? item.qos_acceptability_status, ""),
    qos_acceptance_missing: toStringSafe(item.qos_acceptance_missing ?? item.qos_acceptability_missing, ""),
    summary,
    eval_rows: evalRows,
    trained_det: normalizeOfficialStats(item.trained_det),
    noop_det: normalizeOfficialStats(item.noop_det),
    delta_summary: normalizeOfficialDeltaSummary(item.delta_summary),
    notes: (item.notes as string | null | undefined) ?? null,
  };
}

function normalizeStringList(payload: unknown): string[] {
  if (!Array.isArray(payload)) {
    return [];
  }

  return payload
    .map((item) => toStringSafe(item, "").trim())
    .filter((item): item is string => item.length > 0);
}

function normalizeFinalEvaluationSource(payload: unknown): FinalEvaluationSource | undefined {
  if (!payload || typeof payload !== "object") {
    return undefined;
  }

  const item = payload as Record<string, unknown>;
  return {
    mode: toStringSafe(item.mode, "unknown"),
    description: toStringSafe(item.description, ""),
    selected_run_count: toOptionalNumber(item.selected_run_count),
    selected_policies: normalizeStringList(item.selected_policies),
    selected_scenarios: normalizeStringList(item.selected_scenarios),
  };
}

function normalizeFinalEvaluationClassification(payload: unknown): FinalEvaluationClassification | undefined {
  if (!payload || typeof payload !== "object") {
    return undefined;
  }

  const item = payload as Record<string, unknown>;
  return {
    primary_baseline_policy: toStringSafe(item.primary_baseline_policy, ""),
    official_traditional_baseline_policy: toStringSafe(item.official_traditional_baseline_policy, ""),
    strongest_heuristic_baseline_policy: toStringSafe(item.strongest_heuristic_baseline_policy, ""),
    baseline_policies: normalizeStringList(item.baseline_policies),
    ai_policies: normalizeStringList(item.ai_policies),
  };
}

function normalizeFinalEvaluationThresholds(payload: unknown): FinalEvaluationThresholds | undefined {
  if (!payload || typeof payload !== "object") {
    return undefined;
  }

  const item = payload as Record<string, unknown>;
  return {
    energy_target_pct: toNumber(item.energy_target_pct, 0),
    max_qos_violation_rate_increase_abs: toNumber(item.max_qos_violation_rate_increase_abs, 0),
    max_delivered_loss_pct: toNumber(item.max_delivered_loss_pct, 0),
    max_dropped_increase_pct: toNumber(item.max_dropped_increase_pct, 0),
    max_delay_increase_pct: toNumber(item.max_delay_increase_pct, 0),
    max_path_latency_increase_pct: toNumber(item.max_path_latency_increase_pct, 0),
  };
}

type FinalEvaluationNumberField =
  | "run_count"
  | "seed_count"
  | "episodes_total"
  | "steps_total"
  | "energy_kwh_mean"
  | "energy_kwh_std"
  | "energy_kwh_count"
  | "energy_kwh_delta_vs_baseline"
  | "energy_reduction_pct_vs_baseline"
  | "energy_kwh_delta_vs_heuristic_baseline"
  | "energy_reduction_pct_vs_heuristic_baseline"
  | "delivered_traffic_mean"
  | "delivered_traffic_std"
  | "delivered_traffic_count"
  | "delivered_traffic_delta_vs_baseline"
  | "delivered_traffic_change_pct_vs_baseline"
  | "delivered_traffic_delta_vs_heuristic_baseline"
  | "delivered_traffic_change_pct_vs_heuristic_baseline"
  | "dropped_traffic_mean"
  | "dropped_traffic_std"
  | "dropped_traffic_count"
  | "dropped_traffic_delta_vs_baseline"
  | "dropped_traffic_change_pct_vs_baseline"
  | "dropped_traffic_delta_vs_heuristic_baseline"
  | "dropped_traffic_change_pct_vs_heuristic_baseline"
  | "avg_delay_ms_mean"
  | "avg_delay_ms_std"
  | "avg_delay_ms_count"
  | "avg_delay_ms_delta_vs_baseline"
  | "avg_delay_ms_change_pct_vs_baseline"
  | "avg_delay_ms_delta_vs_heuristic_baseline"
  | "avg_delay_ms_change_pct_vs_heuristic_baseline"
  | "avg_path_latency_ms_mean"
  | "avg_path_latency_ms_std"
  | "avg_path_latency_ms_count"
  | "avg_path_latency_ms_delta_vs_baseline"
  | "avg_path_latency_ms_change_pct_vs_baseline"
  | "avg_path_latency_ms_delta_vs_heuristic_baseline"
  | "avg_path_latency_ms_change_pct_vs_heuristic_baseline"
  | "qos_violation_rate_mean"
  | "qos_violation_rate_std"
  | "qos_violation_rate_count"
  | "qos_violation_rate_delta_vs_baseline"
  | "qos_violation_rate_delta_vs_heuristic_baseline"
  | "qos_violation_count_mean"
  | "qos_violation_count_std"
  | "qos_violation_count_count"
  | "qos_violation_count_total"
  | "qos_violation_count_delta_vs_baseline"
  | "qos_violation_count_delta_vs_heuristic_baseline"
  | "transition_count_total_mean"
  | "transition_count_total_std"
  | "transition_count_total_count"
  | "transition_rate_mean"
  | "transition_rate_std"
  | "transition_rate_count"
  | "flap_event_count_total_mean"
  | "flap_event_count_total_std"
  | "flap_event_count_total_count"
  | "flap_rate_mean"
  | "flap_rate_std"
  | "flap_rate_count"
  | "carbon_g_mean"
  | "carbon_g_std"
  | "carbon_g_count"
  | "carbon_g_delta_vs_baseline"
  | "carbon_reduction_pct_vs_baseline"
  | "carbon_g_delta_vs_heuristic_baseline"
  | "carbon_reduction_pct_vs_heuristic_baseline";

const FINAL_EVALUATION_NUMBER_FIELDS: FinalEvaluationNumberField[] = [
  "run_count",
  "seed_count",
  "episodes_total",
  "steps_total",
  "energy_kwh_mean",
  "energy_kwh_std",
  "energy_kwh_count",
  "energy_kwh_delta_vs_baseline",
  "energy_reduction_pct_vs_baseline",
  "energy_kwh_delta_vs_heuristic_baseline",
  "energy_reduction_pct_vs_heuristic_baseline",
  "delivered_traffic_mean",
  "delivered_traffic_std",
  "delivered_traffic_count",
  "delivered_traffic_delta_vs_baseline",
  "delivered_traffic_change_pct_vs_baseline",
  "delivered_traffic_delta_vs_heuristic_baseline",
  "delivered_traffic_change_pct_vs_heuristic_baseline",
  "dropped_traffic_mean",
  "dropped_traffic_std",
  "dropped_traffic_count",
  "dropped_traffic_delta_vs_baseline",
  "dropped_traffic_change_pct_vs_baseline",
  "dropped_traffic_delta_vs_heuristic_baseline",
  "dropped_traffic_change_pct_vs_heuristic_baseline",
  "avg_delay_ms_mean",
  "avg_delay_ms_std",
  "avg_delay_ms_count",
  "avg_delay_ms_delta_vs_baseline",
  "avg_delay_ms_change_pct_vs_baseline",
  "avg_delay_ms_delta_vs_heuristic_baseline",
  "avg_delay_ms_change_pct_vs_heuristic_baseline",
  "avg_path_latency_ms_mean",
  "avg_path_latency_ms_std",
  "avg_path_latency_ms_count",
  "avg_path_latency_ms_delta_vs_baseline",
  "avg_path_latency_ms_change_pct_vs_baseline",
  "avg_path_latency_ms_delta_vs_heuristic_baseline",
  "avg_path_latency_ms_change_pct_vs_heuristic_baseline",
  "qos_violation_rate_mean",
  "qos_violation_rate_std",
  "qos_violation_rate_count",
  "qos_violation_rate_delta_vs_baseline",
  "qos_violation_rate_delta_vs_heuristic_baseline",
  "qos_violation_count_mean",
  "qos_violation_count_std",
  "qos_violation_count_count",
  "qos_violation_count_total",
  "qos_violation_count_delta_vs_baseline",
  "qos_violation_count_delta_vs_heuristic_baseline",
  "transition_count_total_mean",
  "transition_count_total_std",
  "transition_count_total_count",
  "transition_rate_mean",
  "transition_rate_std",
  "transition_rate_count",
  "flap_event_count_total_mean",
  "flap_event_count_total_std",
  "flap_event_count_total_count",
  "flap_rate_mean",
  "flap_rate_std",
  "flap_rate_count",
  "carbon_g_mean",
  "carbon_g_std",
  "carbon_g_count",
  "carbon_g_delta_vs_baseline",
  "carbon_reduction_pct_vs_baseline",
  "carbon_g_delta_vs_heuristic_baseline",
  "carbon_reduction_pct_vs_heuristic_baseline",
];

function normalizeFinalEvaluationRow(payload: unknown): FinalEvaluationSummaryRow | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }

  const item = payload as Record<string, unknown>;
  const row: FinalEvaluationSummaryRow = {
    scope_type: toStringSafe(item.scope_type, ""),
    scope: toStringSafe(item.scope, ""),
    scenario: toStringSafe(item.scenario, ""),
    policy: toStringSafe(item.policy, ""),
    policy_class: toStringSafe(item.policy_class, ""),
    seed_list: toStringSafe(item.seed_list, ""),
    comparison_baseline_policy: toStringSafe(item.comparison_baseline_policy, ""),
    comparison_available: item.comparison_available == null ? undefined : Boolean(item.comparison_available),
    comparison_official_baseline_policy: toStringSafe(item.comparison_official_baseline_policy, ""),
    comparison_official_baseline_available:
      item.comparison_official_baseline_available == null ? undefined : Boolean(item.comparison_official_baseline_available),
    comparison_heuristic_baseline_policy: toStringSafe(item.comparison_heuristic_baseline_policy, ""),
    comparison_heuristic_baseline_available:
      item.comparison_heuristic_baseline_available == null ? undefined : Boolean(item.comparison_heuristic_baseline_available),
    is_primary_baseline: item.is_primary_baseline == null ? undefined : Boolean(item.is_primary_baseline),
    is_official_traditional_baseline:
      item.is_official_traditional_baseline == null ? undefined : Boolean(item.is_official_traditional_baseline),
    is_heuristic_baseline: item.is_heuristic_baseline == null ? undefined : Boolean(item.is_heuristic_baseline),
    is_best_policy_for_scope:
      item.is_best_policy_for_scope == null ? undefined : Boolean(item.is_best_policy_for_scope),
    is_best_ai_policy_for_scope:
      item.is_best_ai_policy_for_scope == null ? undefined : Boolean(item.is_best_ai_policy_for_scope),
    qos_acceptability_status: toStringSafe(item.qos_acceptability_status, ""),
    qos_acceptability_missing: toStringSafe(item.qos_acceptability_missing, ""),
    qos_acceptance_status: toStringSafe(item.qos_acceptance_status ?? item.qos_acceptability_status, ""),
    qos_acceptance_missing: toStringSafe(item.qos_acceptance_missing ?? item.qos_acceptability_missing, ""),
    stability_status: toStringSafe(item.stability_status, ""),
    stability_missing: toStringSafe(item.stability_missing, ""),
    hypothesis_status: toStringSafe(item.hypothesis_status, ""),
    stability_qualified_hypothesis_status: toStringSafe(item.stability_qualified_hypothesis_status, ""),
    qos_thresholds: normalizeQosAcceptanceThresholds(
      item.qos_thresholds ?? item.qos_acceptance_thresholds ?? item.qos_acceptability_thresholds ?? item.hypothesis_thresholds,
    ),
    stability_thresholds: normalizeStabilityThresholds(item.stability_thresholds),
  };

  for (const field of FINAL_EVALUATION_NUMBER_FIELDS) {
    const value = toOptionalNumber(item[field]);
    if (value !== undefined) {
      row[field] = value;
    }
  }

  if (!row.scope_type || !row.scope || !row.policy) {
    return null;
  }

  return row;
}

function normalizeFinalEvaluationArtifact(payload: unknown): FinalEvaluationArtifact | undefined {
  if (!payload || typeof payload !== "object") {
    return undefined;
  }

  const item = payload as Record<string, unknown>;
  const summaryPath = toStringSafe(item.summary_path, "");
  if (!summaryPath) {
    return undefined;
  }

  return {
    summary_path: summaryPath,
    report_path: toStringSafe(item.report_path, "") || null,
  };
}

function normalizeFinalEvaluationReport(payload: unknown): FinalEvaluationReport | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }

  const item = payload as Record<string, unknown>;
  const summaryRows = toArrayPayload(item.summary_rows)
    .map(normalizeFinalEvaluationRow)
    .filter((row): row is FinalEvaluationSummaryRow => row !== null);

  if (!summaryRows.length) {
    return null;
  }

  return {
    generated_at_utc: toStringSafe(item.generated_at_utc, ""),
    source: normalizeFinalEvaluationSource(item.source),
    classification: normalizeFinalEvaluationClassification(item.classification),
    hypothesis_thresholds: normalizeFinalEvaluationThresholds(item.hypothesis_thresholds),
    qos_thresholds: normalizeQosAcceptanceThresholds(
      item.qos_thresholds ?? item.qos_acceptance_thresholds ?? item.qos_acceptability_thresholds ?? item.hypothesis_thresholds,
    ),
    qos_acceptance_status: toStringSafe(item.qos_acceptance_status ?? item.qos_acceptability_status ?? item.overall_qos_status, ""),
    qos_acceptance_missing: toStringSafe(item.qos_acceptance_missing ?? item.qos_acceptability_missing, ""),
    stability_thresholds: normalizeStabilityThresholds(item.stability_thresholds),
    overall_stability_status: toStringSafe(item.overall_stability_status, "") || null,
    overall_operational_status: toStringSafe(item.overall_operational_status, "") || null,
    best_policy: normalizeFinalEvaluationRow(item.best_policy),
    best_ai_policy: normalizeFinalEvaluationRow(item.best_ai_policy),
    overall_hypothesis_status: toStringSafe(item.overall_hypothesis_status, "") || null,
    summary_rows: summaryRows,
    artifact: normalizeFinalEvaluationArtifact(item.artifact),
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

type ListRunsOptions = {
  tag?: string;
  policy?: string;
  scenario?: string;
  seed?: number;
  limit?: number;
};

export type RunCatalogSource = "backend" | "demo";

export type RunCatalog = {
  runs: RunSummary[];
  source: RunCatalogSource;
};

function buildRunsQuery(options: ListRunsOptions): string {
  const params = new URLSearchParams();
  params.set("limit", String(options.limit ?? 200));

  if (options.tag) {
    params.set("tag", options.tag);
  }
  if (options.policy) {
    params.set("policy", normalizePolicy(options.policy));
  }
  if (options.scenario) {
    params.set("scenario", options.scenario.trim().toLowerCase());
  }
  if (options.seed != null) {
    params.set("seed", String(options.seed));
  }

  return `?${params.toString()}`;
}

export async function listRunsWithSource(options: ListRunsOptions = {}): Promise<RunCatalog> {
  const query = buildRunsQuery(options);
  const requests: Array<() => Promise<unknown>> = [
    () => requestJson<unknown>(`/api/runs${query}`),
    () => requestJson<unknown>(`/api/runs_flat${query}`),
    () => requestJson<unknown>("/runs"),
  ];

  for (const load of requests) {
    try {
      const payload = await load();
      const normalized = toArrayPayload(payload).map(normalizeRun).filter((run) => run.run_id);
      if (normalized.length > 0) {
        return { runs: normalized, source: "backend" };
      }
    } catch {
      // Try the next source.
    }
  }

  return { runs: listDemoRuns(), source: "demo" };
}

export async function listRuns(options: ListRunsOptions = {}): Promise<RunSummary[]> {
  return (await listRunsWithSource(options)).runs;
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

export async function getFinalEvaluationReport(): Promise<FinalEvaluationReport | null> {
  const payload = await tryOptionalEndpoints<unknown>([() => requestJson<unknown>("/api/final_evaluation")]);
  if (!payload) {
    return null;
  }

  return normalizeFinalEvaluationReport(payload);
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
    policy: normalizePolicy(params.policy),
    scenario: params.scenario.trim().toLowerCase(),
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
