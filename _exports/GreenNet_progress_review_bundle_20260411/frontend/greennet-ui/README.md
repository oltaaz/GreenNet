# GreenNet React UI

This package contains the implementation of the official GreenNet frontend.

Primary pages:
- `/dashboard`: current run overview with standardized KPIs, per-step charts, topology playback, and scenario validation context
- `/results`: unified reporting page for final baseline-vs-AI comparison, per-scenario outcomes, locked acceptance bundles, and live policy comparison
- `/simulator`: replay packet/link behavior for a selected run

Data strategy:
- `src/lib/api.ts` is the single typed client for backend data
- run lists now default to the full backend catalog so experiment inspection is preserved
- policy names are normalized to `all_on`, `heuristic`, and `ppo` so older `noop`/`baseline` aliases still render correctly
- final reporting uses `/api/final_evaluation` for the latest thesis/demo-ready evaluation artifact
- locked acceptance bundles use `/api/official_results`
- live run detail views continue to use `/api/runs/*`

Reporting structure:
- Current run overview lives on `/dashboard`
- Final research/demo story lives on `/results`
- Shared KPI names across both pages are: `Total Energy`, `Carbon Emissions`, `Delivered Traffic`, `Dropped Traffic`, `Drop Rate`, `Average Delay`, `Path Latency`, and `Active Links`
- Hypothesis and QoS acceptance statuses are presented directly from the final evaluation artifact instead of being re-derived in the UI

How to run:
- official combined demo path: run `npm run dev` from `/Users/oltazagraxha/Desktop/GreenNet`
- frontend-only path: run `npm run dev` from `/Users/oltazagraxha/Desktop/GreenNet/frontend`
- direct package path: run `npm run dev` from `/Users/oltazagraxha/Desktop/GreenNet/frontend/greennet-ui`
