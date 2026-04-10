# GreenNet Frontend / Dashboard / Presentation Audit

## Executive Verdict

- Readiness score: **6.8/10**
- Verdict: **Ready with minor fixes** for internal review, but **not fully submission-polished** for an honors board if the frontend is meant to carry part of the public-facing demo burden.

The React UI is real, buildable, and meaningfully integrated with the backend. It does fulfill the basic dashboard/visualization promise better than the retained Streamlit tooling. The main weaknesses are reviewer trust and polish: lint is failing, fallback/demo behavior still weakens presentation credibility, there is no frontend test coverage, and the simulator/dashboard visuals include inferred/generated elements that can be mistaken for measured network state unless the reviewer reads the notices carefully.

## Scope audited

- Public React UI: `frontend/greennet-ui/`
- Frontend shell wrappers: `frontend/`, repo-root `package.json`
- Internal Streamlit tooling: `dashboard/`
- API consumption and reviewer-facing startup path

## What is clearly strong

### 1. There is a real reviewer-facing React surface, not just a mockup

Evidence:
- The repo explicitly positions React as the official public/demo UI in [README.md](/Users/enionismaili/Desktop/GreenNet/README.md#L5), [frontend/greennet-ui/README.md](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/README.md#L1), and [dashboard/README.md](/Users/enionismaili/Desktop/GreenNet/dashboard/README.md#L3).
- The React app has distinct pages for home, dashboard, results, simulator, and about in [App.tsx](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/App.tsx).
- The repo-root command starts backend + frontend together via [package.json](/Users/enionismaili/Desktop/GreenNet/package.json#L8).

Assessment:
- This is not “presentationware.” There is a genuine app structure and a plausible reviewer path.

### 2. API integration is centralized and more mature than a typical capstone UI

Evidence:
- `src/lib/api.ts` is a single typed client layer, matching the docs claim in [frontend/greennet-ui/README.md](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/README.md#L10).
- It consumes real endpoints for health, runs, summaries, per-step metrics, topology, steps, official locked results, and final evaluation in [api.ts](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/lib/api.ts#L910).
- The backend endpoints it expects actually exist in `api_app.py`.

Assessment:
- This is a strong engineering decision. It makes the frontend easier to audit and reduces the chance of page-level drift.

### 3. The results page is the strongest reviewer-facing screen

Evidence:
- The `/results` page explicitly frames the final story around traditional baseline vs heuristic vs AI in [ComparePage.tsx](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/pages/ComparePage.tsx#L280).
- It loads final evaluation artifacts plus official locked results rather than pretending the whole story comes from ad hoc live runs.
- It surfaces thresholds, QoS/stability badges, and source artifact paths in [ComparePage.tsx](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/pages/ComparePage.tsx#L313).

Assessment:
- For an honors reviewer, this is the most credible page because it ties UI content to final evaluation artifacts instead of only flashy visuals.

### 4. The React app is buildable right now

Verified:
- `npm --prefix frontend/greennet-ui run build` passed.

Assessment:
- This matters. The official UI exists and compiles. That is materially better than many capstone frontends.

## Requirement-by-requirement evaluation

### Visualization / dashboard promise

- **PASS**: A real dashboard exists in React with KPI cards, charts, official-result cards, topology visualization, and a simulator page.
- **PARTIAL**: Some of the visuals are inferred/generated rather than direct backend truth. Topology can be generated when missing, and demo traces can be synthesized when run listing fails in [api.ts](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/lib/api.ts#L910) and [api.ts](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/lib/api.ts#L1013).
- **PARTIAL**: The simulator promises “real link states” in [SimulatorPage.tsx](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/pages/SimulatorPage.tsx#L243), but packet motion and some topology/layout behavior are still presentational approximations rather than fully measured network animation.

### Public React UI vs internal Streamlit separation

- **PASS**: The repo now clearly says React is official and Streamlit is internal in [dashboard/README.md](/Users/enionismaili/Desktop/GreenNet/dashboard/README.md#L3), [frontend/greennet-ui/README.md](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/README.md#L3), and [README.md](/Users/enionismaili/Desktop/GreenNet/README.md#L5).
- **PARTIAL**: The internal Streamlit app still contains an experiment launcher and a filesystem-first run browser in [dashboard/app.py](/Users/enionismaili/Desktop/GreenNet/dashboard/app.py#L41). That is fine for internal tooling, but it remains a second presentation surface that could confuse reviewers if they discover it.

### API consumption and integration quality

- **PASS**: The public UI uses real backend endpoints and a Vite proxy to `localhost:8000` in [vite.config.ts](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/vite.config.ts).
- **PARTIAL**: The client has a strong fallback strategy, but the fallback strategy is presentation-risky. If `/api/runs` cannot load, the app silently substitutes generated demo runs before showing a banner in [api.ts](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/lib/api.ts#L910) and [DashboardPage.tsx](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/pages/DashboardPage.tsx#L447).

### Reviewer-facing clarity

- **PASS**: There is one obvious startup command, `npm run dev`, in [README.md](/Users/enionismaili/Desktop/GreenNet/README.md#L7).
- **PARTIAL**: The React UI is understandable, but the distinction between “live run view,” “official locked validation,” and “final evaluation artifact” still requires attention from the reviewer. It is better than before, but not fully idiot-proof.
- **PARTIAL**: The simulator is visually engaging, but it is the easiest page for a reviewer to over-interpret as a faithful network playback rather than a mixed playback-plus-visualization layer.

### Buildability / polish

- **PASS**: Production build succeeds.
- **FAIL**: Lint fails. That is a real polish gap for a “final submission” software surface.
- **FAIL**: There are no frontend tests, no component tests, and no UI smoke tests discoverable in `frontend/greennet-ui/`.

## Public React UI audit

### Strengths

- Stronger reviewer story than the Streamlit dashboard.
- Reasonable route structure: home, dashboard, results, simulator, about.
- Consistent visual system and deliberate styling in [app.css](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/styles/app.css).
- Results page is anchored in formal artifacts, not only ephemeral runs.
- Backend status is visible globally in navigation.

### Weaknesses

#### 1. Demo fallback is still a trust hazard

Evidence:
- If run listing fails, the UI falls back to generated runs in [api.ts](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/lib/api.ts#L910).
- The dashboard then renders “generated simulation data” with a notice in [DashboardPage.tsx](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/pages/DashboardPage.tsx#L447).
- The compare page can mix generated live traces with backend final artifacts in [ComparePage.tsx](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/pages/ComparePage.tsx#L306).

Why it matters:
- This is honest enough for developers, but still risky in a final demo. A reviewer can easily miss the banner and assume all visible traces are official evidence.

Judgment:
- **Present but weak**. Acceptable as a fallback mechanism, not ideal as a capstone presentation choice.

#### 2. Generated topology/layout weakens evidentiary clarity

Evidence:
- Missing topology triggers `fallbackTopology(...)` in [api.ts](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/lib/api.ts#L1013).
- The dashboard and simulator expose notices when generated layout is used in [DashboardPage.tsx](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/pages/DashboardPage.tsx#L453) and [SimulatorPage.tsx](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/pages/SimulatorPage.tsx#L262).

Why it matters:
- A generated layout is fine for developer resilience, but weaker for final-review credibility. Reviewers prefer concrete topology provenance.

Judgment:
- **Partially implemented**. Good fallback engineering, weaker submission presentation.

#### 3. Build passes, but lint failure undercuts “polished final frontend”

Verified:
- Build passed.
- `npm --prefix frontend/greennet-ui run lint` failed with:
  - `react-hooks/set-state-in-effect` and `react-refresh/only-export-components` in [useBackendStatus.tsx](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/hooks/useBackendStatus.tsx#L38)
  - unused variable in [api.ts](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/lib/api.ts#L65)
  - hook dependency warnings in [TopologyPanel.tsx](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/components/TopologyPanel.tsx) and [DashboardPage.tsx](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/pages/DashboardPage.tsx)

Why it matters:
- Reviewers may not run lint, but if they do, a failing frontend quality gate damages the “complete/polished” impression.

Judgment:
- **Fail** on polish.

#### 4. No frontend verification layer

Evidence:
- No tests or specs were found under `frontend/greennet-ui/`.
- No frontend test runner dependency was found in package manifests.

Why it matters:
- The UI presents final results and acceptance status. Having zero UI-level verification is a weakness, especially because the app contains fallback and normalization logic.

Judgment:
- **Missing**.

## Internal Streamlit tooling audit

### What it is

- A separate analyst/developer tool, explicitly described that way in [dashboard/README.md](/Users/enionismaili/Desktop/GreenNet/dashboard/README.md#L3).
- It provides single-run inspection, policy comparison, and an internal run launcher in [dashboard/app.py](/Users/enionismaili/Desktop/GreenNet/dashboard/app.py#L119).

### Strengths

- Useful for internal analysis.
- More filesystem-oriented and pragmatic for ad hoc exploration.
- The title/caption correctly disclaim it as internal in [dashboard/app.py](/Users/enionismaili/Desktop/GreenNet/dashboard/app.py#L120).

### Weaknesses

- It is visibly a second dashboard product, not just a hidden utility.
- The launcher runs experiments directly from the UI in [dashboard/app.py](/Users/enionismaili/Desktop/GreenNet/dashboard/app.py#L41), which is fine internally but not something you want reviewers mistaking for the official submission experience.
- It is tied to `results/` browsing and developer workflows, which conflicts with the repo’s newer “curated artifact” story.

Judgment:
- **Internal-only tooling, not submission-facing.**
- Keep it, but do not let it become the reviewer’s first impression.

## Reviewer experience audit

### What will work well

- A reviewer can start the official frontend from the repo root using [README.md](/Users/enionismaili/Desktop/GreenNet/README.md#L7).
- The React UI has clearer narrative separation than the Streamlit app.
- The `/results` page is the best place to communicate the thesis-facing claim.

### What will confuse reviewers

- Seeing “dashboard,” “results,” “simulator,” and a separate Streamlit “dashboard” folder in the same repo.
- Discovering that some views can show generated demo runs or generated topology if backend data is missing.
- Understanding which visuals are direct evidence versus reviewer-friendly renderings.

### Reviewer-safe interpretation

- The React UI fulfills the visualization promise.
- The Streamlit app should be treated as internal analyst tooling, not the main deliverable.
- The strongest submission-facing visual artifact is the React `/results` route, not the live dashboard and not the Streamlit launcher.

## Honesty / overclaim audit for presentation

### Supported

- “Official React frontend” is supported.
- “Backend-integrated dashboard and results UI” is supported.
- “Internal Streamlit tooling” is supported.
- “Unified reporting page for baseline-vs-AI comparison” is supported.

### Partially supported / should be softened

- “Topology playback with simulated packet flow and real link states” should be softened unless you make the simulated/rendered parts more explicit on-page. The current wording in [SimulatorPage.tsx](/Users/enionismaili/Desktop/GreenNet/frontend/greennet-ui/src/pages/SimulatorPage.tsx#L243) reads slightly stronger than the implementation warrants.
- Any claim implying the live dashboard always reflects stored official evidence should be softened because generated/demo fallback exists.

### Misleading if phrased carelessly

- Calling the simulator a faithful replay of measured packet flow.
- Presenting generated demo traces during a board review without explicitly saying they are generated.
- Letting reviewers infer that the Streamlit dashboard is the same thing as the official frontend.

## Final judgment

### What is fulfilled

- The repository does fulfill the dashboard/visualization promise.
- The public React UI is real, integrated, and presentation-capable.
- The internal Streamlit tooling is now correctly demoted from official demo status.

### What is still weak

- Frontend polish is not complete because lint fails.
- Reviewer trust is weakened by demo and topology fallbacks.
- There is no frontend test/smoke harness.
- The simulator is visually effective but evidentially weaker than the results page.

## Priority fixes

### Critical before final submission

- Fix all frontend lint errors and warnings.
- Make demo-mode state impossible to miss on all routes, especially `/dashboard` and `/simulator`.
- Make generated topology / generated packet-flow language more explicit in the simulator UI.

### Important but not blocking

- Add at least one frontend smoke test covering load of `/results` with backend available.
- Add one short reviewer note telling them the recommended order is `Home -> Results -> Dashboard -> Simulator`.
- Ensure the Streamlit dashboard is never presented as the main demo path in any final submission document or oral demo.

### Nice-to-have polish

- Split the large JS bundle if you want stronger frontend engineering polish.
- Add clearer provenance badges such as `artifact-backed`, `live-run`, `generated-demo`, `generated-layout`.

