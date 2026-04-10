"""Canonical controller-policy taxonomy for GreenNet.

This module separates three concepts that were previously conflated:

- official traditional baseline: static all-on control over the routing baseline
- energy-aware heuristic baseline: handcrafted utilization-threshold controller
- AI policies: learned controllers such as PPO
"""
from __future__ import annotations

TRADITIONAL_BASELINE_POLICY_ALIASES = {
    "all_on": "all_on",
    "noop": "all_on",
}

HEURISTIC_BASELINE_POLICY_ALIASES = {
    "heuristic": "heuristic",
    "baseline": "heuristic",
}

AI_POLICY_ALIASES = {
    "ppo": "ppo",
}

EXPERIMENT_POLICY_ALIASES = {
    **TRADITIONAL_BASELINE_POLICY_ALIASES,
    **HEURISTIC_BASELINE_POLICY_ALIASES,
    **AI_POLICY_ALIASES,
}

CONTROLLER_POLICY_ALIASES = {
    "all_on": "all_on",
    "noop": "all_on",
    "heuristic": "utilization_threshold",
    "baseline": "utilization_threshold",
    "ppo": "ppo",
}

DEFAULT_TRADITIONAL_BASELINE_POLICIES = ("all_on", "noop")
DEFAULT_HEURISTIC_BASELINE_POLICIES = ("heuristic", "baseline", "utilization_threshold")
DEFAULT_NON_AI_BASELINE_POLICIES = (
    *DEFAULT_TRADITIONAL_BASELINE_POLICIES,
    *DEFAULT_HEURISTIC_BASELINE_POLICIES,
)
DEFAULT_AI_POLICIES = ("ppo",)


def canonical_experiment_policy_name(policy: str | None) -> str:
    key = str(policy or "").strip().lower()
    return EXPERIMENT_POLICY_ALIASES.get(key, key or "unknown")


def canonical_controller_policy_name(policy: str | None) -> str:
    key = str(policy or "").strip().lower()
    return CONTROLLER_POLICY_ALIASES.get(key, key or "unknown")


def controller_policy_class(policy: str | None) -> str:
    canonical = canonical_controller_policy_name(policy)
    if canonical == "all_on":
        return "traditional_baseline"
    if canonical == "utilization_threshold":
        return "energy_aware_heuristic"
    if canonical == "ppo":
        return "ai_policy"
    return "other"


def experiment_policy_class(policy: str | None) -> str:
    canonical = canonical_experiment_policy_name(policy)
    if canonical == "all_on":
        return "traditional_baseline"
    if canonical == "heuristic":
        return "heuristic_baseline"
    if canonical == "ppo":
        return "ai_policy"
    return "other"


def is_traditional_baseline_policy(policy: str | None) -> bool:
    return canonical_experiment_policy_name(policy) == "all_on"


def is_heuristic_baseline_policy(policy: str | None) -> bool:
    return canonical_experiment_policy_name(policy) == "heuristic"


def is_ai_policy(policy: str | None) -> bool:
    return canonical_experiment_policy_name(policy) == "ppo"


def reviewer_policy_label(policy: str | None) -> str:
    canonical = canonical_experiment_policy_name(policy)
    if canonical == "all_on":
        return "Traditional (All-On)"
    if canonical == "heuristic":
        return "Energy-Aware Heuristic"
    if canonical == "ppo":
        return "PPO-Based Hybrid (AI)"
    return str(policy or "")


def reviewer_policy_descriptor(policy: str | None) -> str:
    canonical = canonical_experiment_policy_name(policy)
    if canonical == "ppo":
        return (
            "PPO-based hybrid controller with rule-based safety, recovery, and calm-off "
            "overrides over the fixed routing baseline"
        )
    if canonical == "heuristic":
        return "Handcrafted energy-aware heuristic controller over the fixed routing baseline"
    if canonical == "all_on":
        return "Traditional all-on control over the fixed routing baseline"
    return str(policy or "")
