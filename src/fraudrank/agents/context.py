"""Context builders that trim what each agent sees.

Each builder returns the minimal dict an agent needs — no raw column names,
no data the agent shouldn't reason about.
"""
from __future__ import annotations


def experiment_context(families: dict[str, list[str]], agent_cols: list[str],
                       history: list[dict]) -> dict:
    return {
        "feature_sets": {
            "all": sum(len(v) for v in families.values()),
            "no_size": sum(len(v) for k, v in families.items() if k != "size"),
            "agent": len(agent_cols),
        },
        "models_available": ["logistic", "random_forest", "lightgbm"],
        "imbalance_options": ["none", "class_weight", "smote"],
        "history": [
            {"run_id": h["run_id"], "model": h["model"], "feature_set": h["feature_set"],
             "imbalance": h["imbalance"], "pr_auc_mean": h["pr_auc_mean"],
             "pr_auc_sd": h["pr_auc_sd"]}
            for h in history
        ],
    }


def critic_context(history: list[dict], rounds_left: int, selection: dict) -> dict:
    return {
        "history": history,
        "rounds_left": rounds_left,
        "n_configs_tried": len(history),
        "selection": selection,
    }
