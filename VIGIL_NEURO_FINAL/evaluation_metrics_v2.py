"""Section 5/8 Disaggregated Navigation Evaluation Metrics & Exact Binomial Confidence Intervals.

Replaces single-scalar 'safe_success' with explicit disaggregated episode classifications,
scenario-specific goal regions, path efficiency, switching counts, exact Clopper-Pearson
binomial confidence bounds (general k >= 0), and world-family clustered summaries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import beta


# Scenario-specific feasible navigation targets & tolerance specs
# (target_x_m, reference_path_m, outage_abstention_expected, bypass_or_stop_ok)
SCENARIO_SPECS: Dict[str, dict] = {
    "open_healthy": {"goal_x": 0.90, "ref_dist": 1.35, "abstain_expected": False, "stop_valid": False},
    "near_miss_post": {"goal_x": 0.90, "ref_dist": 1.35, "abstain_expected": False, "stop_valid": False},
    "looming_post": {"goal_x": 0.85, "ref_dist": 1.10, "abstain_expected": False, "stop_valid": True},
    "freeze_outage_clear": {"goal_x": 0.55, "ref_dist": 1.00, "abstain_expected": False, "stop_valid": True},
    "blocked_outage": {"goal_x": 0.80, "ref_dist": 0.90, "abstain_expected": True, "stop_valid": True},
    "occluded_hazard": {"goal_x": 0.80, "ref_dist": 1.00, "abstain_expected": False, "stop_valid": True},
    "corridor_pedestrian": {"goal_x": 0.55, "ref_dist": 1.10, "abstain_expected": False, "stop_valid": False},
    "warehouse_forklift_cross": {"goal_x": 0.60, "ref_dist": 1.20, "abstain_expected": False, "stop_valid": False},
    "outdoor_pothole": {"goal_x": 0.55, "ref_dist": 1.05, "abstain_expected": False, "stop_valid": False},
    "lab_glass_wall": {"goal_x": 0.44, "ref_dist": 0.45, "abstain_expected": False, "stop_valid": True},
    "ramp_descent": {"goal_x": 0.50, "ref_dist": 1.15, "abstain_expected": False, "stop_valid": False},
    "night_mode": {"goal_x": 0.50, "ref_dist": 1.05, "abstain_expected": False, "stop_valid": False},
}


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (1 - alpha) two-sided Clopper-Pearson binomial confidence interval; one-sided upper at k=0."""
    if n <= 0:
        return (0.0, 1.0)
    if k == 0:
        return (0.0, float(1.0 - (alpha) ** (1.0 / n)))
    if k == n:
        return (float((alpha) ** (1.0 / n)), 1.0)
    lo = float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def world_clustered_mean_ci(values_by_cluster: Dict[str, List[float]], z: float = 1.96) -> dict:
    """Compute mean and world-family cluster-robust normal-approx 95% CI across correlated worlds."""
    cluster_means = [float(np.mean(v)) for v in values_by_cluster.values() if v]
    c = len(cluster_means)
    if c == 0:
        return {"mean": 0.0, "clusters": 0, "se": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    grand_mean = float(np.mean(cluster_means))
    if c == 1:
        return {"mean": grand_mean, "clusters": 1, "se": 0.0, "ci95_low": grand_mean, "ci95_high": grand_mean}
    se = float(np.std(cluster_means, ddof=1) / math.sqrt(c))
    return {
        "mean": grand_mean,
        "clusters": c,
        "se": se,
        "ci95_low": float(grand_mean - z * se),
        "ci95_high": float(grand_mean + z * se),
    }


def classify_episode(
    scenario: str,
    x_m: float,
    y_m: float,
    body_collisions: int,
    whisker_events: int,
    abstained: bool,
    outage_steps: int,
    actions_seq: List[str],
    u_task_start: float,
    u_task_end: float,
) -> dict:
    """Produce complete disaggregated navigation and safety record for one closed-loop episode."""
    spec = SCENARIO_SPECS.get(scenario, {"goal_x": 0.60, "ref_dist": 1.00, "abstain_expected": False, "stop_valid": False})
    dist_traveled = math.hypot(x_m, y_m)
    goal_reached = bool(x_m >= spec["goal_x"])
    collided = body_collisions > 0
    col_free_goal = bool(goal_reached and not collided)

    # Appropriate vs unnecessary abstention evaluated strictly against scenario feasibility
    app_abstain = bool(abstained and spec["abstain_expected"] and not collided)
    unn_abstain = bool(abstained and not spec["abstain_expected"] and not goal_reached)

    # Permanent stall: rover stops far below goal without expected outage abstention and without colliding
    stall_thresh = min(0.18, 0.30 * spec["goal_x"])
    perm_stall = bool((not collided) and (not goal_reached) and (not app_abstain) and (x_m < stall_thresh))

    col_free_timeout = bool((not collided) and (not goal_reached) and (not app_abstain) and (not perm_stall))

    switches = sum(1 for i in range(1, len(actions_seq)) if actions_seq[i] != actions_seq[i - 1])
    path_eff = float(min(1.5, x_m / max(spec["ref_dist"], 1e-6))) if x_m > 0 else 0.0

    # Recovery latency: steps spent braking/turning in a stall before first resumed forward step after an interrupt
    recov_latency = 0
    seen_interrupt = False
    for a in actions_seq:
        if a in ("brake", "left", "right"):
            seen_interrupt = True
            recov_latency += 1
        elif seen_interrupt and a in ("forward", "slow"):
            break

    info_recov = float(u_task_start - u_task_end)
    return {
        "scenario": scenario,
        "x_m": float(x_m),
        "y_m": float(y_m),
        "dist_m": float(dist_traveled),
        "body_collisions": int(body_collisions),
        "whisker_contacts": int(whisker_events),
        "goal_completion": goal_reached,
        "collision_free_goal_completion": col_free_goal,
        "collision_free_timeout": col_free_timeout,
        "appropriate_abstention": app_abstain,
        "unnecessary_abstention": unn_abstain,
        "permanent_stall": perm_stall,
        "info_recovery_delta_u": info_recov,
        "path_efficiency": path_eff,
        "action_switches": int(switches),
        "recovery_latency_steps": int(recov_latency),
        "outage_steps": int(outage_steps),
    }


def aggregate_episodes(rows: List[dict]) -> dict:
    """Aggregate disaggregated episode counts, rates, Clopper-Pearson CIs, and clustered progress."""
    n = len(rows)
    col_eps = sum(1 for r in rows if r["body_collisions"] > 0)
    by_sc: Dict[str, List[float]] = {}
    by_sc_goal: Dict[str, List[float]] = {}
    for r in rows:
        by_sc.setdefault(r["scenario"], []).append(float(r["x_m"]))
        by_sc_goal.setdefault(r["scenario"], []).append(1.0 if r["collision_free_goal_completion"] else 0.0)

    cp_lo, cp_hi = clopper_pearson_ci(col_eps, n, alpha=0.05)
    return {
        "episodes": n,
        "collision_episodes": col_eps,
        "total_body_collision_events": sum(r["body_collisions"] for r in rows),
        "total_whisker_contacts": sum(r["whisker_contacts"] for r in rows),
        "collision_rate_clopper_pearson_95ci": [cp_lo, cp_hi],
        "goal_completion_rate": float(np.mean([r["goal_completion"] for r in rows])) if n else 0.0,
        "collision_free_goal_completion_rate": float(np.mean([r["collision_free_goal_completion"] for r in rows])) if n else 0.0,
        "collision_free_timeout_rate": float(np.mean([r["collision_free_timeout"] for r in rows])) if n else 0.0,
        "appropriate_abstention_count": sum(1 for r in rows if r["appropriate_abstention"]),
        "unnecessary_abstention_count": sum(1 for r in rows if r["unnecessary_abstention"]),
        "permanent_stall_rate": float(np.mean([r["permanent_stall"] for r in rows])) if n else 0.0,
        "mean_progress_m": float(np.mean([r["x_m"] for r in rows])) if n else 0.0,
        "world_clustered_progress_m": world_clustered_mean_ci(by_sc),
        "world_clustered_goal_rate": world_clustered_mean_ci(by_sc_goal),
        "mean_path_efficiency": float(np.mean([r["path_efficiency"] for r in rows])) if n else 0.0,
        "mean_action_switches": float(np.mean([r["action_switches"] for r in rows])) if n else 0.0,
        "mean_info_recovery_delta_u": float(np.mean([r["info_recovery_delta_u"] for r in rows])) if n else 0.0,
    }
