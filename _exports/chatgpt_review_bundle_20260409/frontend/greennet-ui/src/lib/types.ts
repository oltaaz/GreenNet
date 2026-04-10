export type PolicyType = "all_on" | "heuristic" | "ppo" | string;

export type RunFileFlags = {
  per_step: boolean;
  summary: boolean;
  meta: boolean;
  env_config: boolean;
};

export type RunSummary = {
  run_id: string;
  started_at?: string | null;
  policy?: PolicyType | null;
  scenario?: string | null;
  seed?: number | null;
  topology_seed?: number | null;
  max_steps?: number | null;
  tag?: string | null;
  source?: string | null;
  episodes?: number | null;
  deterministic?: boolean | null;
  has?: RunFileFlags | null;
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
  digits?: number;
};

export type FinalEvaluationSource = {
  mode: string;
  description: string;
  selected_run_count?: number;
  selected_policies?: string[];
  selected_scenarios?: string[];
};

export type FinalEvaluationClassification = {
  primary_baseline_policy: string;
  baseline_policies: string[];
  ai_policies: string[];
};

export type FinalEvaluationThresholds = {
  energy_target_pct: number;
  max_qos_violation_rate_increase_abs: number;
  max_delivered_loss_pct: number;
  max_dropped_increase_pct: number;
  max_delay_increase_pct: number;
  max_path_latency_increase_pct: number;
};

export type FinalEvaluationSummaryRow = {
  scope_type: string;
  scope: string;
  scenario: string;
  policy: string;
  policy_class: string;
  run_count?: number;
  seed_count?: number;
  episodes_total?: number;
  steps_total?: number;
  seed_list?: string;
  comparison_baseline_policy?: string;
  comparison_available?: boolean;
  is_primary_baseline?: boolean;
  is_best_policy_for_scope?: boolean;
  is_best_ai_policy_for_scope?: boolean;
  qos_acceptability_status?: string;
  qos_acceptability_missing?: string;
  hypothesis_status?: string;
  energy_kwh_mean?: number;
  energy_kwh_std?: number;
  energy_kwh_count?: number;
  energy_kwh_delta_vs_baseline?: number;
  energy_reduction_pct_vs_baseline?: number;
  delivered_traffic_mean?: number;
  delivered_traffic_std?: number;
  delivered_traffic_count?: number;
  delivered_traffic_delta_vs_baseline?: number;
  delivered_traffic_change_pct_vs_baseline?: number;
  dropped_traffic_mean?: number;
  dropped_traffic_std?: number;
  dropped_traffic_count?: number;
  dropped_traffic_delta_vs_baseline?: number;
  dropped_traffic_change_pct_vs_baseline?: number;
  avg_delay_ms_mean?: number;
  avg_delay_ms_std?: number;
  avg_delay_ms_count?: number;
  avg_delay_ms_delta_vs_baseline?: number;
  avg_delay_ms_change_pct_vs_baseline?: number;
  avg_path_latency_ms_mean?: number;
  avg_path_latency_ms_std?: number;
  avg_path_latency_ms_count?: number;
  avg_path_latency_ms_delta_vs_baseline?: number;
  avg_path_latency_ms_change_pct_vs_baseline?: number;
  qos_violation_rate_mean?: number;
  qos_violation_rate_std?: number;
  qos_violation_rate_count?: number;
  qos_violation_rate_delta_vs_baseline?: number;
  qos_violation_count_mean?: number;
  qos_violation_count_std?: number;
  qos_violation_count_count?: number;
  qos_violation_count_total?: number;
  qos_violation_count_delta_vs_baseline?: number;
  carbon_g_mean?: number;
  carbon_g_std?: number;
  carbon_g_count?: number;
  carbon_g_delta_vs_baseline?: number;
  carbon_reduction_pct_vs_baseline?: number;
};

export type FinalEvaluationArtifact = {
  summary_path: string;
  report_path?: string | null;
};

export type FinalEvaluationReport = {
  generated_at_utc: string;
  source?: FinalEvaluationSource;
  classification?: FinalEvaluationClassification;
  hypothesis_thresholds?: FinalEvaluationThresholds;
  best_policy?: FinalEvaluationSummaryRow | null;
  best_ai_policy?: FinalEvaluationSummaryRow | null;
  overall_hypothesis_status?: string | null;
  summary_rows: FinalEvaluationSummaryRow[];
  artifact?: FinalEvaluationArtifact;
};
