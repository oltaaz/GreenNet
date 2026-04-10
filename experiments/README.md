# Experiments Directory Guide

This directory mixes active experiment entrypoints with preserved historical experiment bundles.

Canonical final-submission runner:

- `run_official_acceptance_matrix.py`

Historical preserved bundles:

- `official_matrix_v1/`
- `official_matrix_v2/`
- `official_matrix_v3/`
- `official_matrix_v4/`
- `official_matrix_v6/`
- `official_matrix_accept/`
- `official_matrix_test/`
- `official_matrix_test_ppo/`

These preserved folders are useful for development history and traceability, but they are not the primary reviewer-facing final evidence path.

For the final submission story, use:

- `configs/acceptance_matrices/official_acceptance_v1.json`
- `artifacts/final_pipeline/official_acceptance_v1/`
