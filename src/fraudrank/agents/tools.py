"""Read-only tools for agents (least privilege).

Each tool is (params: dict, ctx: dict) -> dict. The ctx dict is injected by the
harness and holds the data each tool needs; agents never access it directly.
"""
from __future__ import annotations

import statistics

from .case_file import FEATURE_TEXT, peer_percentiles


def lookup_provider(params: dict, ctx: dict) -> dict:
    pid = params["provider_id"]
    df = ctx["df"]
    if pid not in df.index:
        return {"error": f"provider {pid} not found"}
    row = df.loc[pid]
    ptype = {(1, 1): "inpatient and outpatient", (1, 0): "inpatient only",
             (0, 1): "outpatient only"}.get((int(row.get("has_ip", 0)), int(row.get("has_op", 0))), "unknown")
    cols = [c for c in df.columns if c != "PotentialFraud"]
    values = {c: round(float(row[c]), 4) for c in cols[:30] if not str(row[c]) == "nan"}
    return {"provider": pid, "type": ptype, "n_features": len(cols), "values": values}


def get_peer_stats(params: dict, ctx: dict) -> dict:
    pid = params["provider_id"]
    cols = params.get("features", None)
    df = ctx["df"]
    dev = ctx["dev"]
    if pid not in df.index:
        return {"error": f"provider {pid} not found"}
    if cols is None:
        shap_df = ctx.get("shap")
        if shap_df is not None and pid in shap_df.index:
            s = shap_df.loc[pid]
            cols = list(s.reindex(s.abs().sort_values(ascending=False).index)[:5].index)
        else:
            cols = list(FEATURE_TEXT.keys())[:5]
    pct = peer_percentiles(df, dev, pid, [c for c in cols if c in df.columns])
    return {"provider": pid, "percentiles": {k: round(v, 1) for k, v in pct.items()}}


def search_playbook(params: dict, ctx: dict) -> dict:
    retriever = ctx.get("retriever")
    if retriever is None:
        return {"error": "no playbook retriever available"}
    query = params["query"]
    k = min(params.get("k", 3), 5)
    results = retriever.search(query, k=k)
    return {"query": query, "results": [
        {"chunk_id": c["id"], "title": c["title"], "text": c["text"][:300],
         "signals": c.get("signals", [])}
        for c in results
    ]}


def get_fold_detail(params: dict, ctx: dict) -> dict:
    """Return per-fold PR-AUC for a completed experiment run."""
    rid = params["run_id"]
    fold_details = ctx.get("fold_details", {})
    if rid not in fold_details:
        return {"error": f"run {rid} not found"}
    scores = fold_details[rid]
    return {
        "run_id": rid,
        "pr_auc_per_fold": [round(s, 4) for s in scores],
        "mean": round(statistics.mean(scores), 4),
        "sd": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "n_folds": len(scores),
    }


TOOL_REGISTRY = {
    "lookup_provider": lookup_provider,
    "get_peer_stats": get_peer_stats,
    "search_playbook": search_playbook,
    "get_fold_detail": get_fold_detail,
}

CASE_FILE_TOOLS = [
    {
        "name": "lookup_provider",
        "description": "Look up a provider's feature values and type. Returns up to 30 features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "provider_id": {"type": "string", "description": "The provider ID to look up"}
            },
            "required": ["provider_id"],
        },
    },
    {
        "name": "get_peer_stats",
        "description": "Get a provider's percentile rank among same-type peers for given features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "provider_id": {"type": "string", "description": "The provider ID"},
                "features": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Feature names to compare (default: top 5 by SHAP importance)",
                },
            },
            "required": ["provider_id"],
        },
    },
    {
        "name": "search_playbook",
        "description": "Search the fraud typology playbook for patterns matching a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query describing the fraud pattern"},
                "k": {"type": "integer", "description": "Number of results (max 5, default 3)"},
            },
            "required": ["query"],
        },
    },
]


CRITIC_TOOLS = [
    {
        "name": "get_fold_detail",
        "description": (
            "Get the per-fold PR-AUC breakdown for a completed experiment run. "
            "Use this to compare fold-level consistency between runs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "integer", "description": "The run ID to inspect"},
            },
            "required": ["run_id"],
        },
    },
]


def execute_tool(name: str, params: dict, ctx: dict) -> dict:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(params, ctx)
    except Exception as e:
        return {"error": f"{name} failed: {e}"}
