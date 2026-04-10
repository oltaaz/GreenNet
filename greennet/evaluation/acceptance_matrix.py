from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from greennet.policy_taxonomy import canonical_experiment_policy_name


@dataclass(frozen=True)
class AcceptanceMatrixCase:
    case_id: str
    label: str
    scenario: str
    topology_name: str | None = None
    topology_path: str | None = None
    topology_seed: int | None = None
    traffic_name: str | None = None
    traffic_path: str | None = None
    traffic_model: str | None = None
    traffic_scenario: str | None = None
    traffic_scenario_version: int | None = None
    traffic_scenario_intensity: float | None = None
    traffic_scenario_duration: float | None = None
    traffic_scenario_frequency: float | None = None
    notes: str | None = None


@dataclass(frozen=True)
class AcceptanceMatrix:
    schema_version: int
    matrix_id: str
    matrix_name: str
    tag: str
    description: str
    policies: tuple[str, ...]
    seeds: tuple[int, ...]
    episodes: int
    steps: int
    deterministic: bool
    baseline_policies: tuple[str, ...]
    ai_policies: tuple[str, ...]
    primary_baseline_policy: str
    routing_baseline: str | None
    routing_link_cost_model: str | None
    cases: tuple[AcceptanceMatrixCase, ...]
    manifest_path: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def official_acceptance_matrix_path() -> Path:
    return _repo_root() / "configs" / "acceptance_matrices" / "official_acceptance_v1.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Acceptance matrix must be a JSON object: {path}")
    return data


def _clean_text(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Acceptance matrix field '{field_name}' must be non-empty")
    return text


def _clean_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def load_acceptance_matrix(path: str | Path | None = None) -> AcceptanceMatrix:
    resolved = official_acceptance_matrix_path() if path is None else Path(path).expanduser().resolve()
    payload = _read_json(resolved)

    schema_version = int(payload.get("schema_version", 0))
    if schema_version != 1:
        raise ValueError(f"Unsupported acceptance matrix schema_version={schema_version} in {resolved}")

    policies_raw = payload.get("policies")
    if not isinstance(policies_raw, list) or not policies_raw:
        raise ValueError(f"Acceptance matrix must define a non-empty 'policies' list: {resolved}")
    policies = tuple(canonical_experiment_policy_name(_clean_text(item, field_name="policies[]")) for item in policies_raw)

    seeds_raw = payload.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError(f"Acceptance matrix must define a non-empty 'seeds' list: {resolved}")
    seeds = tuple(int(seed) for seed in seeds_raw)

    cases_raw = payload.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError(f"Acceptance matrix must define a non-empty 'cases' list: {resolved}")

    seen_case_ids: set[str] = set()
    cases: list[AcceptanceMatrixCase] = []
    for index, raw_case in enumerate(cases_raw):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Acceptance matrix case #{index} must be an object: {resolved}")
        case_id = _clean_text(raw_case.get("id"), field_name=f"cases[{index}].id")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate acceptance matrix case id '{case_id}' in {resolved}")
        seen_case_ids.add(case_id)

        topology_name = _clean_optional_text(raw_case.get("topology_name"))
        topology_path = _clean_optional_text(raw_case.get("topology_path"))
        if topology_name and topology_path:
            raise ValueError(
                f"Acceptance matrix case '{case_id}' cannot define both topology_name and topology_path"
            )
        if not topology_name and not topology_path:
            raise ValueError(
                f"Acceptance matrix case '{case_id}' must define topology_name or topology_path"
            )

        traffic_name = _clean_optional_text(raw_case.get("traffic_name"))
        traffic_path = _clean_optional_text(raw_case.get("traffic_path"))
        if traffic_name and traffic_path:
            raise ValueError(
                f"Acceptance matrix case '{case_id}' cannot define both traffic_name and traffic_path"
            )

        scenario = _clean_text(raw_case.get("scenario"), field_name=f"cases[{index}].scenario")
        if (traffic_name or traffic_path) and scenario != "custom":
            raise ValueError(
                f"Acceptance matrix case '{case_id}' uses traffic replay input and must set scenario='custom'"
            )

        cases.append(
            AcceptanceMatrixCase(
                case_id=case_id,
                label=_clean_text(raw_case.get("label", case_id), field_name=f"cases[{index}].label"),
                scenario=scenario,
                topology_name=topology_name,
                topology_path=str((resolved.parent / topology_path).resolve()) if topology_path else None,
                topology_seed=int(raw_case["topology_seed"]) if raw_case.get("topology_seed") is not None else None,
                traffic_name=traffic_name,
                traffic_path=str((resolved.parent / traffic_path).resolve()) if traffic_path else None,
                traffic_model=_clean_optional_text(raw_case.get("traffic_model")),
                traffic_scenario=_clean_optional_text(raw_case.get("traffic_scenario")),
                traffic_scenario_version=(
                    int(raw_case["traffic_scenario_version"])
                    if raw_case.get("traffic_scenario_version") is not None
                    else None
                ),
                traffic_scenario_intensity=(
                    float(raw_case["traffic_scenario_intensity"])
                    if raw_case.get("traffic_scenario_intensity") is not None
                    else None
                ),
                traffic_scenario_duration=(
                    float(raw_case["traffic_scenario_duration"])
                    if raw_case.get("traffic_scenario_duration") is not None
                    else None
                ),
                traffic_scenario_frequency=(
                    float(raw_case["traffic_scenario_frequency"])
                    if raw_case.get("traffic_scenario_frequency") is not None
                    else None
                ),
                notes=_clean_optional_text(raw_case.get("notes")),
            )
        )

    baseline_policies_raw = payload.get("baseline_policies", ["all_on", "heuristic"])
    ai_policies_raw = payload.get("ai_policies", ["ppo"])

    matrix = AcceptanceMatrix(
        schema_version=schema_version,
        matrix_id=_clean_text(payload.get("matrix_id"), field_name="matrix_id"),
        matrix_name=_clean_text(payload.get("matrix_name"), field_name="matrix_name"),
        tag=_clean_text(payload.get("tag"), field_name="tag"),
        description=_clean_text(payload.get("description"), field_name="description"),
        policies=policies,
        seeds=seeds,
        episodes=int(payload.get("episodes", 1)),
        steps=int(payload.get("steps", 300)),
        deterministic=bool(payload.get("deterministic", True)),
        baseline_policies=tuple(
            canonical_experiment_policy_name(_clean_text(item, field_name="baseline_policies[]"))
            for item in baseline_policies_raw
        ),
        ai_policies=tuple(
            canonical_experiment_policy_name(_clean_text(item, field_name="ai_policies[]"))
            for item in ai_policies_raw
        ),
        primary_baseline_policy=canonical_experiment_policy_name(
            _clean_text(payload.get("primary_baseline_policy", "all_on"), field_name="primary_baseline_policy")
        ),
        routing_baseline=_clean_optional_text(payload.get("routing_baseline")),
        routing_link_cost_model=_clean_optional_text(payload.get("routing_link_cost_model")),
        cases=tuple(cases),
        manifest_path=str(resolved),
    )

    if matrix.primary_baseline_policy not in matrix.baseline_policies:
        raise ValueError(
            f"Acceptance matrix primary_baseline_policy='{matrix.primary_baseline_policy}' must be included in baseline_policies"
        )
    if not set(matrix.ai_policies).issubset(set(matrix.policies)):
        raise ValueError("Acceptance matrix ai_policies must be a subset of policies")
    if not set(matrix.baseline_policies).issubset(set(matrix.policies)):
        raise ValueError("Acceptance matrix baseline_policies must be a subset of policies")

    return matrix


def acceptance_matrix_metadata(matrix: AcceptanceMatrix) -> dict[str, Any]:
    return {
        "matrix_id": matrix.matrix_id,
        "matrix_name": matrix.matrix_name,
        "matrix_manifest": matrix.manifest_path,
        "matrix_tag": matrix.tag,
        "matrix_policies": list(matrix.policies),
        "matrix_seeds": list(matrix.seeds),
        "matrix_episodes": matrix.episodes,
        "matrix_steps": matrix.steps,
        "matrix_case_count": len(matrix.cases),
        "matrix_cases": [
            {
                "id": case.case_id,
                "label": case.label,
                "scenario": case.scenario,
                "topology_name": case.topology_name,
                "topology_path": case.topology_path,
                "traffic_name": case.traffic_name,
                "traffic_path": case.traffic_path,
                "traffic_model": case.traffic_model,
                "traffic_scenario": case.traffic_scenario,
            }
            for case in matrix.cases
        ],
    }
