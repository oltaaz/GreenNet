# Backend Submission Checklist

## Critical before submission
- [ ] Add a short reviewer-facing note that this is a forwarding-level network simulator with simplified power/QoS abstractions, not a packet-level or routing-protocol control-plane simulator.
- [ ] Rewrite any “AI-driven optimization” language so it accurately describes the evaluated policy as PPO-based control with hard QoS/stability safeguards and hybrid override logic.
- [ ] Mark `official_acceptance_v1` as the only canonical final benchmark path and explicitly point reviewers to the one command, one manifest, and one output folder that matter.
- [ ] Remove, exclude, or clearly label `_exports/` and older “official_matrix_*` result trees so they do not compete with the canonical final path.

## Important but not blocking
- [ ] Add one paragraph explaining what forecasting and the impact predictor actually are in this repo, and whether they are optional or central to the final evaluation story.
- [ ] Add a policy taxonomy table mapping `all_on/noop`, `heuristic/baseline`, and `ppo` to their canonical experimental roles.
- [ ] State clearly that file artifacts are the primary source of truth and SQLite is a secondary indexing/query layer that may lag if best-effort persistence fails.

## Nice-to-have polish
- [ ] Refactor the PPO safety wrapper and heuristic override logic out of `run_experiment.py` into a dedicated controller module after submission.
- [ ] Add a small diagram of the backend flow: acceptance manifest -> run matrix -> run artifacts -> final pipeline -> SQLite/API.
- [ ] Add one or two invariant-based integration tests for hypothesis classification and canonical filtering, not just endpoint/file success.
