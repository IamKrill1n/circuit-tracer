from __future__ import annotations

from typing import Any

import numpy as np

from circuit_tracer.subgraph.cluster import _is_embedding, _is_logit, _parse_layer
from circuit_tracer.subgraph.prune import PruneGraph


def _classify_node(node_id: str, attr: dict[str, dict[str, Any]]) -> str:
    if _is_embedding(attr, node_id):
        return "emb"
    if _is_logit(attr, node_id):
        return "logit"
    return "middle"


def supernodes_to_mapping(
    prune_graph: PruneGraph,
    supernodes: list[list[str]],
    middle_prefix: str = "SN",
) -> dict[str, list[str]]:
    """Convert `cluster_graph` output into a named supernode mapping."""
    attr = prune_graph.attr
    middle: list[list[str]] = []
    emb: list[list[str]] = []
    logit: list[list[str]] = []

    for sn in supernodes:
        if not sn:
            continue
        first = sn[0]
        kind = _classify_node(first, attr)
        if kind == "emb":
            emb.append(sn)
        elif kind == "logit":
            logit.append(sn)
        else:
            middle.append(sn)

    middle = sorted(middle, key=lambda m: min(_parse_layer(attr, n) for n in m))
    named: dict[str, list[str]] = {f"{middle_prefix}_{i}": sn for i, sn in enumerate(middle)}
    named.update({f"SN_EMB_{i}": sn for i, sn in enumerate(emb)})
    named.update({f"SN_LOGIT_{i}": sn for i, sn in enumerate(logit)})
    return named


def build_supernode_graph(
    prune_graph: PruneGraph,
    final_supernodes: dict[str, list[str]] | list[list[str]],
) -> dict[str, Any]:
    """
    Build flow quantities on supernodes from a pruned node-level graph.

    Returns sn-level adjacency and influence metrics used by auto-grouping and
    flow-faithfulness analysis.
    """
    if isinstance(final_supernodes, list):
        final_supernodes = supernodes_to_mapping(prune_graph, final_supernodes)

    kept_ids = prune_graph.kept_ids
    attr = prune_graph.attr
    adj = prune_graph.pruned_adj.clone().float().T  # sender-indexed

    node_to_idx = {nid: i for i, nid in enumerate(kept_ids)}
    sn_names = list(final_supernodes.keys())
    sn_members_idx: list[list[int]] = []
    for sn in sn_names:
        members = [node_to_idx[n] for n in final_supernodes[sn] if n in node_to_idx]
        if not members:
            continue
        sn_members_idx.append(members)

    sn_names = [sn for sn, members in zip(sn_names, sn_members_idx) if members]
    k = len(sn_names)
    sn_adj = np.zeros((k, k), dtype=np.float64)

    for i, src in enumerate(sn_members_idx):
        for j, dst in enumerate(sn_members_idx):
            if i == j:
                continue
            block = adj[np.ix_(src, dst)].detach().cpu().numpy()
            w = float(block.sum())
            if w != 0.0:
                sn_adj[i, j] = w

    logit_idx = [i for i, nid in enumerate(kept_ids) if _is_logit(attr, nid)]
    sn_inf = np.zeros(k, dtype=np.float64)
    if logit_idx:
        for i, src in enumerate(sn_members_idx):
            block = adj[np.ix_(src, logit_idx)].detach().cpu().numpy()
            sn_inf[i] = float(block.sum())

    # Conservative aliases for compatibility with existing demo output schema.
    f_sn = np.maximum(sn_adj, 0.0)
    sn_reach = np.maximum(f_sn.sum(axis=1), 0.0)
    sn_act_norm = sn_reach / (sn_reach.max() + 1e-12) if k else sn_reach

    total_node_inf = float(adj[:, logit_idx].sum().item()) if logit_idx else 0.0
    total_sn_inf = float(sn_inf.sum())
    inf_conservation = (
        total_sn_inf / (total_node_inf + 1e-12) if abs(total_node_inf) > 1e-12 else 1.0
    )

    node_nonzero = float((adj != 0).sum().item())
    sn_nonzero = float(np.count_nonzero(sn_adj))
    edge_conservation = sn_nonzero / (node_nonzero + 1e-12) if node_nonzero > 0 else 1.0

    dominant_paths = [
        {"src": sn_names[i], "tgt": sn_names[j], "weight": float(sn_adj[i, j])}
        for i in range(k)
        for j in range(k)
        if i != j and sn_adj[i, j] > 0
    ]
    dominant_paths.sort(key=lambda x: -x["weight"])
    dominant_paths = dominant_paths[:20]

    bottleneck_sns = [
        {"sn": sn_names[i], "in_minus_out": float(sn_adj[:, i].sum() - sn_adj[i, :].sum())}
        for i in range(k)
    ]
    bottleneck_sns.sort(key=lambda x: -abs(x["in_minus_out"]))

    return {
        "sn_names": sn_names,
        "sn_adj": sn_adj,
        "F_sn": f_sn,
        "sn_reach": sn_reach,
        "sn_act_norm": sn_act_norm,
        "sn_inf": sn_inf,
        "preservation": {"inf_conservation": inf_conservation, "edge_conservation": edge_conservation},
        "orig_reach_total": float(node_nonzero),
        "surr_reach_total": float(sn_nonzero),
        "inf_conservation": inf_conservation,
        "edge_conservation": edge_conservation,
        "dominant_paths": dominant_paths,
        "bottleneck_sns": bottleneck_sns,
    }


def path_attribution_decomposition(
    sng: dict[str, Any],
    final_supernodes: dict[str, list[str]] | list[list[str]],
    top_k: int = 10,
) -> dict[str, Any]:
    del final_supernodes  # retained for API compatibility
    sn_names = sng["sn_names"]
    sn_adj = sng["sn_adj"]
    sn_inf = sng["sn_inf"]
    k = len(sn_names)

    totals = []
    for i in range(k):
        out_pos = float(np.maximum(sn_adj[i], 0.0).sum())
        inf_pos = max(0.0, float(sn_inf[i]))
        totals.append({"path": [sn_names[i]], "flow": out_pos + inf_pos})

    totals.sort(key=lambda x: -x["flow"])
    total_flow = sum(t["flow"] for t in totals) + 1e-12
    paths = []
    running = 0.0
    concentration: dict[str, float] = {}
    for rank, t in enumerate(totals[: max(top_k, 1)], 1):
        frac = t["flow"] / total_flow
        running += frac
        concentration[str(rank)] = running
        paths.append({"path": t["path"], "flow": t["flow"], "frac": frac, "length": 1})

    top_k_frac = running
    d_phi = 1.0 - top_k_frac
    return {
        "paths": paths,
        "total_flow": float(total_flow),
        "n_paths": len(totals),
        "concentration": concentration,
        "D_phi": float(d_phi),
        "top_k": top_k,
        "top_k_frac": float(top_k_frac),
    }


def local_flow_residuals(
    sng: dict[str, Any],
    final_supernodes: dict[str, list[str]] | list[list[str]],
) -> dict[str, Any]:
    del final_supernodes  # retained for API compatibility
    sn_names = sng["sn_names"]
    sn_adj = sng["sn_adj"]
    sn_inf = sng["sn_inf"]

    per_sn: dict[str, Any] = {}
    residuals = []
    for i, sn in enumerate(sn_names):
        if "EMB" in sn or "LOGIT" in sn:
            continue
        in_flow = float(sn_adj[:, i].sum())
        out_flow = float(sn_adj[i, :].sum() + sn_inf[i])
        residual = abs(in_flow - out_flow) / (abs(in_flow) + 1e-12)
        residuals.append(residual)
        per_sn[sn] = {
            "in_flow_net": in_flow,
            "total_out": out_flow,
            "inf_exit": float(sn_inf[i]),
            "residual_rel": residual,
            "balance": in_flow - out_flow,
            "is_suppressive": float(sn_inf[i]) < 0,
            "suppression_rel": abs(float(sn_inf[i])) / (abs(in_flow) + 1e-12),
        }

    r_phi = float(np.mean(residuals)) if residuals else 0.0
    r_phi_max = float(np.max(residuals)) if residuals else 0.0
    n_suppressive = sum(1 for v in per_sn.values() if v["is_suppressive"])
    n_middle = len(per_sn)
    n_balanced = n_middle - n_suppressive
    return {
        "per_sn": per_sn,
        "R_phi_balance": r_phi,
        "R_phi_suppressive": 0.0,
        "R_phi_max": r_phi_max,
        "R_phi": r_phi,
        "n_middle": n_middle,
        "n_suppressive": n_suppressive,
        "n_balanced": n_balanced,
    }


def shortcut_analysis(
    sng: dict[str, Any],
    final_supernodes: dict[str, list[str]] | list[list[str]],
) -> dict[str, Any]:
    del final_supernodes  # retained for API compatibility
    sn_names = sng["sn_names"]
    sn_adj = sng["sn_adj"]
    k = len(sn_names)
    edges = []
    tot = 0.0
    shortcut_tot = 0.0

    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            w = float(sn_adj[i, j])
            if w <= 0:
                continue
            best_med = 0.0
            med_name = None
            for b in range(k):
                if b in (i, j):
                    continue
                mediation = min(float(sn_adj[i, b]), float(sn_adj[b, j]))
                if mediation > best_med:
                    best_med = mediation
                    med_name = sn_names[b]
            ratio = w / (w + best_med + 1e-12)
            is_shortcut = ratio < 0.5
            tot += w
            if is_shortcut:
                shortcut_tot += w
            edges.append(
                {
                    "src": sn_names[i],
                    "tgt": sn_names[j],
                    "weight": w,
                    "shortcut_ratio": ratio,
                    "is_shortcut": is_shortcut,
                    "best_mediator": med_name,
                    "mediation_strength": best_med,
                }
            )
    edges.sort(key=lambda x: -x["weight"])
    return {
        "edges": edges,
        "n_shortcuts": sum(1 for e in edges if e["is_shortcut"]),
        "n_direct": sum(1 for e in edges if not e["is_shortcut"]),
        "n_total": len(edges),
        "global_shortcut_frac": float(shortcut_tot / (tot + 1e-12)),
    }


def flow_faithfulness_score(
    path_result: dict[str, Any],
    residual_result: dict[str, Any],
    shortcut_result: dict[str, Any],
    w_concentration: float = 0.40,
    w_residual: float = 0.30,
    w_shortcut: float = 0.30,
) -> dict[str, Any]:
    path_score = 1.0 - float(path_result["D_phi"])
    residual_score = 1.0 - min(1.0, float(residual_result["R_phi_balance"]))
    shortcut_score = 1.0 - float(shortcut_result["global_shortcut_frac"])
    f_phi = w_concentration * path_score + w_residual * residual_score + w_shortcut * shortcut_score
    return {
        "F_phi": f_phi,
        "path_score": path_score,
        "residual_score": residual_score,
        "shortcut_score": shortcut_score,
        "D_phi": float(path_result["D_phi"]),
        "R_phi": float(residual_result["R_phi_balance"]),
        "R_phi_balance": float(residual_result["R_phi_balance"]),
        "R_phi_suppressive": float(residual_result["R_phi_suppressive"]),
        "R_phi_max": float(residual_result["R_phi_max"]),
        "shortcut_frac": float(shortcut_result["global_shortcut_frac"]),
        "n_paths": int(path_result["n_paths"]),
        "top_k_frac": float(path_result["top_k_frac"]),
        "n_shortcuts": int(shortcut_result["n_shortcuts"]),
        "n_direct": int(shortcut_result["n_direct"]),
        "n_suppressive": int(residual_result["n_suppressive"]),
        "n_balanced": int(residual_result["n_balanced"]),
    }


def flow_faithfulness_report(
    sng: dict[str, Any],
    final_supernodes: dict[str, list[str]] | list[list[str]],
    top_k: int = 10,
) -> dict[str, Any]:
    path_result = path_attribution_decomposition(sng, final_supernodes, top_k=top_k)
    residual_result = local_flow_residuals(sng, final_supernodes)
    shortcut_result = shortcut_analysis(sng, final_supernodes)
    combined = flow_faithfulness_score(path_result, residual_result, shortcut_result)
    return {
        "path_decomposition": path_result,
        "local_residuals": residual_result,
        "shortcut_analysis": shortcut_result,
        "combined": combined,
    }
