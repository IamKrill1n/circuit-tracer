from __future__ import annotations

from typing import Any

import numpy as np

from summarization.prune import PruneGraph
from summarization.utils import _is_embedding, _is_logit, _parse_layer


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
    enforce_dag: bool = False,
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
            nz = block[block != 0.0]
            if nz.size == 0:
                continue
            w = float(nz.mean())
            if w != 0.0:
                sn_adj[i, j] = w

    if enforce_dag:
        for i in range(k):
            for j in range(i + 1, k):
                w_ij = float(sn_adj[i, j])
                w_ji = float(sn_adj[j, i])
                if w_ij == 0.0 or w_ji == 0.0:
                    continue
                if abs(w_ij) >= abs(w_ji):
                    sn_adj[j, i] = 0.0
                else:
                    sn_adj[i, j] = 0.0

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


def _classify_sn(sn_name: str) -> str:
    if "EMB" in sn_name:
        return "emb"
    if "LOGIT" in sn_name:
        return "logit"
    return "middle"


def _build_sn_dag_order(sn_names: list[str], final_supernodes: dict[str, list[str]]) -> list[str]:
    def min_layer(sn: str) -> int:
        members = final_supernodes.get(sn, [])
        return min(_parse_layer({}, n) for n in members) if members else 0

    return sorted(sn_names, key=min_layer)


def path_attribution_decomposition(
    sng: dict[str, Any],
    final_supernodes: dict[str, list[str]] | list[list[str]],
    top_k: int = 10,
    min_flow_frac: float = 1e-4,
) -> dict[str, Any]:
    if isinstance(final_supernodes, list):
        raise ValueError("path_attribution_decomposition requires named supernodes mapping.")

    sn_names = sng["sn_names"]
    sn_adj = sng["sn_adj"]
    sn_inf = sng["sn_inf"]
    k = len(sn_names)
    name2idx = {sn: i for i, sn in enumerate(sn_names)}

    emb_sns = [sn for sn in sn_names if _classify_sn(sn) == "emb"]
    logit_sns = [sn for sn in sn_names if _classify_sn(sn) == "logit"]
    topo_order = _build_sn_dag_order(sn_names, final_supernodes)

    path_flows: dict[tuple[str, ...], float] = {}
    for sn_e in emb_sns:
        i = name2idx[sn_e]
        out_total = sum(max(0.0, float(sn_adj[i, j])) for j in range(k) if j != i)
        if out_total > 0:
            path_flows[(sn_e,)] = out_total

    for sn_e in emb_sns:
        i = name2idx[sn_e]
        direct_inf = float(sn_inf[i])
        if direct_inf > 0:
            for sn_l in logit_sns:
                key = (sn_e, sn_l)
                path_flows[key] = path_flows.get(key, 0.0) + direct_inf

    total_emb_out = sum(path_flows.get((sn_e,), 0.0) for sn_e in emb_sns)
    prune_threshold = total_emb_out * min_flow_frac if total_emb_out > 0 else 0.0
    completed_paths: dict[tuple[str, ...], float] = {}

    for sn in topo_order:
        i = name2idx[sn]
        kind = _classify_sn(sn)
        if kind == "logit":
            for path, flow in list(path_flows.items()):
                if path[-1] == sn:
                    completed_paths[path] = completed_paths.get(path, 0.0) + flow
            continue

        if kind == "emb":
            prefixes = [(p, f) for p, f in path_flows.items() if p == (sn,)]
        else:
            prefixes = [(p, f) for p, f in path_flows.items() if p[-1] == sn and _classify_sn(p[-1]) != "logit"]
        if not prefixes:
            continue

        out_weights: dict[str, float] = {}
        out_total = 0.0
        for j in range(k):
            if j == i:
                continue
            w = float(sn_adj[i, j])
            if w > 0:
                out_weights[sn_names[j]] = w
                out_total += w
        direct_inf = float(sn_inf[i]) if kind == "middle" else 0.0
        exit_total = out_total + max(0.0, direct_inf)
        if exit_total <= 0:
            continue

        for prefix, flow in prefixes:
            if prefix in path_flows:
                del path_flows[prefix]
            for sn_next, w in out_weights.items():
                distributed = flow * (w / exit_total)
                if distributed < prune_threshold:
                    continue
                new_path = prefix + (sn_next,)
                path_flows[new_path] = path_flows.get(new_path, 0.0) + distributed
            if direct_inf > 0:
                inf_flow = flow * (direct_inf / exit_total)
                if inf_flow >= prune_threshold:
                    for sn_l in logit_sns:
                        exit_path = prefix + (sn_l,)
                        completed_paths[exit_path] = completed_paths.get(exit_path, 0.0) + inf_flow

    for path, flow in path_flows.items():
        if _classify_sn(path[-1]) == "logit":
            completed_paths[path] = completed_paths.get(path, 0.0) + flow

    sorted_paths = sorted(completed_paths.items(), key=lambda x: -x[1])
    total_flow = sum(f for _, f in sorted_paths) if sorted_paths else 0.0

    concentration: dict[str, float] = {}
    cumulative = 0.0
    for rank, (_, flow) in enumerate(sorted_paths, 1):
        cumulative += flow
        frac = cumulative / (total_flow + 1e-12)
        concentration[str(rank)] = frac
        if rank >= top_k and frac > 0.99:
            break

    top_k_frac = float(concentration.get(str(min(top_k, len(sorted_paths))), 0.0))
    d_phi = 1.0 - top_k_frac
    paths = [
        {
            "path": list(path),
            "flow": float(flow),
            "frac": float(flow / (total_flow + 1e-12)),
            "length": len(path),
        }
        for path, flow in sorted_paths[:top_k]
    ]
    return {
        "paths": paths,
        "total_flow": float(total_flow),
        "n_paths": len(sorted_paths),
        "concentration": concentration,
        "D_phi": float(d_phi),
        "top_k": top_k,
        "top_k_frac": float(top_k_frac),
    }


def local_flow_residuals(
    sng: dict[str, Any],
    final_supernodes: dict[str, list[str]] | list[list[str]],
) -> dict[str, Any]:
    del final_supernodes
    sn_names = sng["sn_names"]
    sn_adj = sng["sn_adj"]
    sn_inf = sng["sn_inf"]
    k = len(sn_names)

    per_sn: dict[str, Any] = {}
    balance_residuals: list[float] = []
    suppressive_ratios: list[float] = []
    for i, sn in enumerate(sn_names):
        if "EMB" in sn or "LOGIT" in sn:
            continue

        in_flow_pos = sum(max(0.0, float(sn_adj[j, i])) for j in range(k) if j != i)
        in_flow_neg = sum(min(0.0, float(sn_adj[j, i])) for j in range(k) if j != i)
        out_flow_pos = sum(max(0.0, float(sn_adj[i, j])) for j in range(k) if j != i)
        out_flow_neg = sum(min(0.0, float(sn_adj[i, j])) for j in range(k) if j != i)
        inf_exit = float(sn_inf[i])
        in_flow_net = in_flow_pos + in_flow_neg
        total_out = out_flow_pos + out_flow_neg + inf_exit
        residual_abs = abs(in_flow_net - total_out)
        residual_rel = residual_abs / (abs(in_flow_net) + 1e-12)
        is_suppressive = inf_exit < 0
        suppression_rel = abs(inf_exit) / (abs(in_flow_net) + 1e-12)

        if is_suppressive:
            suppressive_ratios.append(suppression_rel)
        else:
            balance_residuals.append(residual_rel)

        per_sn[sn] = {
            "in_flow_pos": in_flow_pos,
            "in_flow_neg": in_flow_neg,
            "in_flow_net": in_flow_net,
            "out_flow_pos": out_flow_pos,
            "out_flow_neg": out_flow_neg,
            "inf_exit": inf_exit,
            "total_out": total_out,
            "residual_abs": residual_abs,
            "residual_rel": residual_rel,
            "balance": in_flow_net - total_out,
            "is_suppressive": is_suppressive,
            "suppression_rel": suppression_rel if is_suppressive else None,
        }

    r_phi_balance = float(np.mean(balance_residuals)) if balance_residuals else 0.0
    r_phi_suppressive = float(np.mean(suppressive_ratios)) if suppressive_ratios else 0.0
    r_phi_max = float(np.max(balance_residuals)) if balance_residuals else 0.0
    n_suppressive = sum(1 for v in per_sn.values() if v["is_suppressive"])
    n_middle = len(per_sn)
    n_balanced = n_middle - n_suppressive
    return {
        "per_sn": per_sn,
        "R_phi_balance": r_phi_balance,
        "R_phi_suppressive": r_phi_suppressive,
        "R_phi_max": r_phi_max,
        "R_phi": r_phi_balance,
        "n_middle": n_middle,
        "n_suppressive": n_suppressive,
        "n_balanced": n_balanced,
    }


def shortcut_analysis(
    sng: dict[str, Any],
    final_supernodes: dict[str, list[str]] | list[list[str]],
    min_edge_weight: float = 1e-6,
) -> dict[str, Any]:
    del final_supernodes
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
            if w < min_edge_weight:
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
    sigma_phi = float(shortcut_result["global_shortcut_frac"])
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
        "shortcut_frac": sigma_phi,
        "Rbal_phi": float(residual_result["R_phi_balance"]),
        "Rsup_phi": float(residual_result["R_phi_suppressive"]),
        "sigma_phi": sigma_phi,
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
