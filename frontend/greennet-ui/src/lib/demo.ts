import { fallbackTopology, linkStateFromRatio, normalizePolicy, timelineFromRows } from "./data";
import type { PerStepRow, RunSummary, StartRunParams, StepState, TopologyData } from "./types";

type DemoRunConfig = {
  run_id: string;
  policy: string;
  scenario: string;
  seed: number;
  steps: number;
  started_at?: string | null;
};

const DEMO_KEY = "greennet_demo_runs_v1";

function normalizeDemoPolicy(policy: string): string {
  return normalizePolicy(policy);
}

function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function parseNumber(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function baseDemoRuns(): DemoRunConfig[] {
  return [
    {
      run_id: "demo-ppo-normal-42-300",
      policy: "ppo",
      scenario: "normal",
      seed: 42,
      steps: 300,
      started_at: null,
    },
    {
      run_id: "demo-all_on-normal-42-300",
      policy: "all_on",
      scenario: "normal",
      seed: 42,
      steps: 300,
      started_at: null,
    },
    {
      run_id: "demo-heuristic-normal-42-300",
      policy: "heuristic",
      scenario: "normal",
      seed: 42,
      steps: 300,
      started_at: null,
    },
  ];
}

function readStoredRuns(): DemoRunConfig[] {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(DEMO_KEY);
    if (!raw) {
      return [];
    }

    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }

    const runs: DemoRunConfig[] = [];
    for (const item of parsed) {
      const row = item as Partial<DemoRunConfig>;
      if (!row.run_id || !row.policy || !row.scenario) {
        continue;
      }

      runs.push({
        run_id: row.run_id,
        policy: normalizeDemoPolicy(row.policy),
        scenario: row.scenario,
        seed: Number.isFinite(Number(row.seed)) ? Number(row.seed) : 42,
        steps: Number.isFinite(Number(row.steps)) ? Number(row.steps) : 300,
        started_at: row.started_at ?? null,
      });
    }

    return runs;
  } catch {
    return [];
  }
}

function writeStoredRuns(runs: DemoRunConfig[]): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.setItem(DEMO_KEY, JSON.stringify(runs));
  } catch {
    // Storage is optional.
  }
}

function parseRunId(runId: string): DemoRunConfig {
  const match = runId.match(/^demo-([a-z_]+)-([a-z_]+)-(\d+)-(\d+)(?:-(\d+))?$/i);
  if (!match) {
    return {
      run_id: runId,
      policy: "ppo",
      scenario: "normal",
      seed: 42,
      steps: 300,
      started_at: null,
    };
  }

  return {
    run_id: runId,
    policy: normalizeDemoPolicy(match[1]),
    scenario: match[2].toLowerCase(),
    seed: parseNumber(match[3], 42),
    steps: parseNumber(match[4], 300),
    started_at: null,
  };
}

function configFromRunId(runId: string): DemoRunConfig {
  const predefined = listDemoRunConfigs().find((run) => run.run_id === runId);
  if (predefined) {
    return predefined;
  }

  return parseRunId(runId);
}

function scenarioTraffic(scenario: string, t: number, noise: number): number {
  if (scenario === "burst") {
    const burst = t % 42 < 8 ? 19 : 0;
    return 16 + Math.sin(t / 9) * 3.4 + burst + noise * 3.5;
  }
  if (scenario === "hotspot") {
    const wave = Math.sin(t / 7.2) * 5.4 + Math.sin(t / 17) * 2.2;
    return 18 + wave + noise * 4.5;
  }
  return 14 + Math.sin(t / 14) * 2.8 + noise * 2.8;
}

function activeRatio(policy: string, trafficNorm: number, prev: number, noise: number): number {
  if (policy === "all_on") {
    return 1;
  }

  if (policy === "heuristic") {
    const raw = 0.66 + Math.sin(trafficNorm * 3.1) * 0.14 + (noise - 0.5) * 0.12;
    return clamp(raw, 0.44, 0.84);
  }

  const target = 0.5 + trafficNorm * 0.34 + (noise - 0.5) * 0.07;
  const smooth = prev * 0.64 + target * 0.36;
  return clamp(smooth, 0.43, 0.9);
}

function deliveredRatio(policy: string, trafficNorm: number, active: number, noise: number): number {
  const activeDeficit = Math.max(0, 0.82 - active);

  if (policy === "all_on") {
    return clamp(0.94 - Math.max(0, trafficNorm - 0.84) * 0.06 + (noise - 0.5) * 0.01, 0.84, 0.97);
  }

  if (policy === "heuristic") {
    return clamp(0.88 - activeDeficit * 0.31 - Math.max(0, trafficNorm - 0.9) * 0.08 + (noise - 0.5) * 0.02, 0.66, 0.93);
  }

  return clamp(0.91 - activeDeficit * 0.17 - Math.max(0, trafficNorm - 0.9) * 0.05 + (noise - 0.5) * 0.03, 0.73, 0.96);
}

function delayMs(policy: string, trafficNorm: number, active: number, switched: number, noise: number): number {
  const base = policy === "all_on" ? 10.8 : policy === "heuristic" ? 12.2 : 10.1;
  const pressure = trafficNorm * (policy === "heuristic" ? 11.2 : 8.8);
  const activePenalty = Math.max(0, 0.78 - active) * (policy === "heuristic" ? 17 : 10);
  const switchPenalty = switched * (policy === "heuristic" ? 3.4 : 1.7);
  return Math.max(3.6, base + pressure + activePenalty + switchPenalty + (noise - 0.5) * 2.2);
}

function energyStepKwh(policy: string, trafficNorm: number, active: number): number {
  const factor = policy === "all_on" ? 1.12 : policy === "heuristic" ? 0.95 : 0.84;
  const raw = (0.026 + active * 0.062 + trafficNorm * 0.018) * factor;
  return Math.max(0.0012, raw);
}

function configToSummary(config: DemoRunConfig): RunSummary {
  return {
    run_id: config.run_id,
    started_at: config.started_at ?? null,
    policy: normalizeDemoPolicy(config.policy),
    scenario: config.scenario,
    seed: config.seed,
    topology_seed: config.seed,
    max_steps: config.steps,
    source: "demo",
    tag: "demo",
  };
}

export function isDemoRunId(runId: string): boolean {
  return runId.startsWith("demo-");
}

export function listDemoRunConfigs(): DemoRunConfig[] {
  const merged = [...readStoredRuns(), ...baseDemoRuns()];
  const unique = new Map<string, DemoRunConfig>();

  for (const run of merged) {
    if (!unique.has(run.run_id)) {
      unique.set(run.run_id, run);
    }
  }

  return [...unique.values()].sort((a, b) => {
    if (a.started_at && b.started_at) {
      return b.started_at.localeCompare(a.started_at);
    }
    if (a.started_at) {
      return -1;
    }
    if (b.started_at) {
      return 1;
    }
    return a.run_id.localeCompare(b.run_id);
  });
}

export function listDemoRuns(): RunSummary[] {
  return listDemoRunConfigs().map(configToSummary);
}

export function createDemoRun(params: StartRunParams): { run_id: string } {
  const now = Date.now();
  const policy = normalizeDemoPolicy(String(params.policy));
  const runId = `demo-${policy}-${params.scenario}-${params.seed}-${params.steps}-${now}`;

  const created: DemoRunConfig = {
    run_id: runId,
    policy,
    scenario: params.scenario.toLowerCase(),
    seed: params.seed,
    steps: params.steps,
    started_at: new Date(now).toISOString(),
  };

  const existing = readStoredRuns().filter((run) => run.run_id !== runId);
  const next = [created, ...existing].slice(0, 20);
  writeStoredRuns(next);

  return { run_id: runId };
}

export function getDemoPerStep(runId: string): PerStepRow[] {
  const config = configFromRunId(runId);
  const rng = seeded(config.seed);

  let cumulativeEnergy = 0;
  let cumulativeCarbon = 0;
  let cumulativeDelivered = 0;
  let cumulativeDropped = 0;
  let prevActive = config.policy === "all_on" ? 1 : 0.72;

  const rows: PerStepRow[] = [];

  for (let t = 0; t < config.steps; t += 1) {
    const noise = rng();
    const traffic = Math.max(4, scenarioTraffic(config.scenario, t, noise));
    const trafficNorm = clamp(traffic / 36, 0, 1.35);

    const active = activeRatio(config.policy, trafficNorm, prevActive, rng());
    const switched = Math.abs(active - prevActive);

    const deliverRatio = deliveredRatio(config.policy, trafficNorm, active, rng());
    const deliveredStep = Math.max(0, traffic * deliverRatio);
    const droppedStep = Math.max(0, traffic - deliveredStep);

    const avgDelayMs = delayMs(config.policy, trafficNorm, active, switched, rng());

    const deltaEnergy = energyStepKwh(config.policy, trafficNorm, active);
    const deltaCarbon = deltaEnergy * (390 + noise * 45);

    cumulativeEnergy += deltaEnergy;
    cumulativeCarbon += deltaCarbon;
    cumulativeDelivered += deliveredStep;
    cumulativeDropped += droppedStep;

    const reward =
      deliveredStep * 0.44 -
      droppedStep * 1.74 -
      deltaEnergy * 67 -
      avgDelayMs * 0.026 -
      switched * 4.2;

    rows.push({
      t,
      energy_kwh: cumulativeEnergy,
      carbon_g: cumulativeCarbon,
      avg_delay_ms: avgDelayMs,
      dropped: cumulativeDropped,
      delivered: cumulativeDelivered,
      active_ratio: active,
      reward,
      delta_energy_kwh: deltaEnergy,
      delta_dropped: droppedStep,
      delta_delivered: deliveredStep,
      traffic,
    });

    prevActive = active;
  }

  return rows;
}

export function getDemoTopology(runId: string): TopologyData {
  const config = configFromRunId(runId);
  const size = config.scenario === "hotspot" ? 14 : config.scenario === "burst" ? 12 : 11;
  return fallbackTopology(`${config.policy}-${config.scenario}-${config.seed}`, size);
}

export function getDemoSteps(runId: string): StepState[] {
  const rows = getDemoPerStep(runId);
  const topology = getDemoTopology(runId);
  const steps = timelineFromRows(rows, topology);

  return steps.map((step, index) => {
    const row = rows[index];
    const linksOn = linkStateFromRatio(topology.edges, step.metrics.active_ratio, step.t);

    return {
      ...step,
      links_on: linksOn,
      packet_events: [],
      metrics: {
        ...step.metrics,
        energy_kwh: row.energy_kwh ?? step.metrics.energy_kwh,
        carbon_g: row.carbon_g ?? step.metrics.carbon_g,
        avg_delay_ms: row.avg_delay_ms ?? step.metrics.avg_delay_ms,
        dropped: row.dropped ?? step.metrics.dropped,
        delivered: row.delivered ?? step.metrics.delivered,
      },
    };
  });
}
