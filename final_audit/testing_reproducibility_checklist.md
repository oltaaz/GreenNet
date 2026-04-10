# GreenNet Testing, Reproducibility, and Evidence Checklist

## Critical before submission

- [ ] Unify the canonical rerun output path and the canonical shipped bundle.
  What is wrong: [`greennet/evaluation/reproduction.py`](/Users/enionismaili/Desktop/GreenNet/greennet/evaluation/reproduction.py) defaults to `artifacts/final_pipeline/latest`, while docs present [`artifacts/final_pipeline/official_acceptance_v1/`](/Users/enionismaili/Desktop/GreenNet/artifacts/final_pipeline/official_acceptance_v1) as canonical.
  Why it matters: reviewers will not know which bundle is the real final one.
  Done looks like: one default output directory everywhere.

- [ ] Fix `/api/final_evaluation` so it does not depend on ambient repository DB state when artifact-local state is intended.
  What is wrong: [`api_app.py`](/Users/enionismaili/Desktop/GreenNet/api_app.py) always prefers DB payloads, and [`tests/integration/test_api_app.py`](/Users/enionismaili/Desktop/GreenNet/tests/integration/test_api_app.py) currently fails because of that.
  Why it matters: reviewer-visible behavior is not cleanly reproducible.
  Done looks like: the failing integration test passes deterministically.

- [ ] Refresh the verification note and supporting evidence.
  What is wrong: [`final_audit/verification/README.md`](/Users/enionismaili/Desktop/GreenNet/final_audit/verification/README.md) claims a fully passing state that is not what the current audit observed.
  Why it matters: stale verification notes undermine trust more than missing notes.
  Done looks like: the note matches the current repo exactly, including failures if any remain.

- [ ] Reduce or exclude the 33G SQLite DB from the final reviewer package.
  What is wrong: [`artifacts/db/greennet.sqlite3`](/Users/enionismaili/Desktop/GreenNet/artifacts/db/greennet.sqlite3) is too large and historically mixed.
  Why it matters: reviewer UX, storage burden, and provenance clarity all suffer.
  Done looks like: a compact canonical DB or no DB dependency in the reviewer-critical path.

## Important but not blocking

- [ ] Add one compact end-to-end reproducibility test that actually runs a minimized final pipeline.
  What is wrong: current reproduction tests are mostly smoke and `--check-only`.
  Why it matters: plumbing is tested better than the final claim path.
  Done looks like: one integration test proves bundle generation from scratch on a small case.

- [ ] Normalize reviewer commands to `.venv/bin/python -m pytest` and similar repo-local invocations.
  What is wrong: global `pytest` is not available in this environment.
  Why it matters: commands should work exactly as written for reviewers.
  Done looks like: docs use only environment-local commands.

- [ ] Mark archive-only evidence paths more aggressively.
  What is wrong: `latest`, `archive`, `official_acceptance_v1`, `official_matrix_v6`, `results`, and `final_audit/verification` all look important.
  Why it matters: too many semi-official paths reduce reviewer confidence.
  Done looks like: one reviewer-first path, all others clearly secondary.

## Nice-to-have polish

- [ ] Curate `artifacts/locked/` so it contains final evidence rather than mixed debugging history.
  What is wrong: scenario-locked artifacts are mixed with exploratory logs and baseline debugging traces.
  Why it matters: the evidence layer looks less intentional than it should.
  Done looks like: locked artifacts are concise and clearly purposeful.

- [ ] Add a short provenance note comparing current runnable results and historical promoted results.
  What is wrong: the repo explains the discrepancy in multiple places, but not in one compact comparison.
  Why it matters: this will likely be asked in review or defense.
  Done looks like: one table explains `1.33%` vs `1.49%`, source paths, and which is canonical.
