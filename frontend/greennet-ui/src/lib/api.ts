export type RunSummary = {
  run_id: string;
  started_at?: string;
  policy?: string;
  scenario?: string;
  topology_seed?: number;
};

export type PerStepRow = {
  t: number;
  energy: number;
  dropped: number;
  reward?: number;
  active_ratio?: number;
};

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

// Adjust these endpoints to whatever your FastAPI exposes.
export const API = {
  listRuns: () => apiGet<RunSummary[]>("/api/runs"),
  getRunPerStep: (runId: string) => apiGet<PerStepRow[]>(`/api/runs/${runId}/per_step`),
};