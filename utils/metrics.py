"""Rollout evaluation metric helpers, shared by both benchmarks' policy trainers.

Both `_get_episode_score` and `compute_metrics` branch on the env id / name and are otherwise
env-agnostic, so they live here rather than being duplicated in each train_policy.py.
"""


def _get_episode_score(env_id: str, total_reward: float, max_reward: float, info: dict) -> float:
    """Compute the scalar episode score used for aggregation across all env types."""
    is_mimicgen = env_id not in ["pusht", "blockpush", "libero_goal", "kitchen-v0"]
    if env_id == "pusht":
        return info.get("final_coverage", 0.0)
    elif is_mimicgen:
        return float(max_reward)
    else:
        return total_reward


def compute_metrics(env_name, max_coverage, final_coverage):
    metric_final = "final coverage" if env_name == "pusht" else "entered"
    metric_max   = "max coverage"   if env_name == "pusht" else "moved"
    return {
        f"{metric_final} mean": sum(final_coverage) / len(final_coverage),
        f"{metric_final} max":  max(final_coverage),
        f"{metric_final} min":  min(final_coverage),
        f"{metric_max} mean":   sum(max_coverage) / len(max_coverage),
        f"{metric_max} max":    max(max_coverage),
        f"{metric_max} min":    min(max_coverage),
    }
