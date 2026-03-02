export type PolicyType = "baseline" | "noop" | "ppo" | string;

export type RunSummary = {
  run_id: string;
  started_at?: string | null;
  policy?: PolicyType | null;
  scenario?: string | null;
  topology_seed?: number | null;
};

export type PerStepRow = {
  t: number;
  energy_kwh?: number;
  carbon_g?: number;
  avg_delay_ms?: number;
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
