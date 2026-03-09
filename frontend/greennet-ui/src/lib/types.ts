export type PolicyType = "baseline" | "noop" | "ppo" | string;

export type RunSummary = {
  run_id: string;
  started_at?: string | null;
  policy?: PolicyType | null;
  scenario?: string | null;
  seed?: number | null;
  topology_seed?: number | null;
  max_steps?: number | null;
  tag?: string | null;
  reward_total_mean?: number;
  dropped_total_mean?: number;
  energy_kwh_total_mean?: number;
};

export type RunOverallSummary = {
  reward_total_mean: number;
  delivered_total_mean: number;
  dropped_total_mean: number;
  energy_kwh_total_mean: number;
  carbon_g_total_mean: number;
  avg_utilization_mean: number;
  active_ratio_mean: number;
  avg_delay_ms_mean: number;
  avg_path_latency_ms_mean?: number;
  steps_mean?: number;
};

export type OfficialLockedEvalRow = {
  scenario: string;
  off_level: string;
  pass: boolean;
  delta_reward?: number;
  delta_energy_kwh?: number;
  delta_dropped?: number;
  on_edges_mean?: number;
  toggles_applied_mean?: number;
  blocked_on_actions_mean?: number;
  cap_used?: string | null;
  seeds?: string | null;
  episodes?: number | null;
  log_file?: string | null;
};

export type OfficialLockedStats = {
  reward_mean: number;
  energy_kwh_mean: number;
  dropped_mean: number;
  delivered_mean: number;
  sent_mean: number;
  drop_rate: number;
  toggles_applied_mean?: number;
  on_edges_mean?: number;
  blocked_on_actions_mean?: number;
};

export type OfficialLockedDeltaSummary = {
  better: boolean;
  reason?: string | null;
  delta_reward?: number;
  delta_energy_kwh?: number;
  delta_dropped?: number;
};

export type OfficialLockedResult = {
  scenario: string;
  bundle_id: string;
  bundle_path?: string | null;
  pass_all?: boolean | null;
  summary?: OfficialLockedEvalRow | null;
  eval_rows: OfficialLockedEvalRow[];
  trained_det?: OfficialLockedStats | null;
  noop_det?: OfficialLockedStats | null;
  delta_summary?: OfficialLockedDeltaSummary | null;
  notes?: string | null;
};

export type PerStepRow = {
  t: number;
  energy_kwh?: number;
  carbon_g?: number;
  avg_delay_ms?: number;
  avg_path_latency_ms?: number;
  congestion_delay_ms?: number;
  dropped?: number;
  delivered?: number;
  active_ratio?: number;
  reward?: number;
  [key: string]: number | string | null | undefined;
};

export type StartRunParams = {
  policy: PolicyType;
  scenario: string;
  seed: number;
  steps: number;
};

export type TopologyNode = {
  id: string;
  label?: string;
  x?: number;
  y?: number;
};

export type TopologyEdge = {
  id: string;
  source: string;
  target: string;
};

export type TopologyData = {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
};

export type LinkStateMap = Record<string, boolean>;

export type PacketEvent = {
  packet_id?: string;
  edge_id?: string;
  source?: string;
  target?: string;
  progress?: number;
  status?: "in_transit" | "delivered" | "dropped" | string;
};

export type StepMetrics = {
  t: number;
  energy_kwh: number;
  carbon_g: number;
  avg_delay_ms: number;
  avg_path_latency_ms?: number;
  congestion_delay_ms?: number;
  dropped: number;
  delivered: number;
  active_ratio: number;
  reward: number;
};

export type StepState = {
  t: number;
  metrics: StepMetrics;
  links_on?: LinkStateMap;
  packet_events?: PacketEvent[];
};

export type KpiMetric = {
  label: string;
  value: number;
  unit: string;
};
