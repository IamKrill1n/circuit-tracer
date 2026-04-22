from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.linalg import eigvalsh

from summarization.cluster import (
    _is_fixed,
    cluster_graph,
    compute_similarity,
)
from summarization.flow_analysis import (
    build_supernode_graph,
    flow_faithfulness_report,
    supernodes_to_mapping,
)
from summarization.prune import PruneGraph


def _middle_indices(prune_graph: PruneGraph) -> list[int]:
    return [i for i, nid in enumerate(prune_graph.kept_ids) if not _is_fixed(prune_graph.attr, nid)]


def eigengap_analysis(
    similarity: Any,
    prune_graph: PruneGraph,
    max_k: int = 20,
) -> dict[str, Any]:
    """Estimate a plausible k range via normalized-Laplacian eigengap."""
    s = np.asarray(similarity.detach().cpu().numpy() if hasattr(similarity, "detach") else similarity)
    mid = _middle_indices(prune_graph)
    m = len(mid)
    if m < 3:
        return {"eigengap_k": 2, "eigenvalues": np.array([0.0, 1.0]), "gaps": np.array([1.0]), "search_range": (2, 2)}

    s_mid = ((s[np.ix_(mid, mid)] + s[np.ix_(mid, mid)].T) / 2.0).clip(0.0, 1.0)
    deg = s_mid.sum(axis=1)
    deg_safe = np.where(deg > 1e-8, deg, 1e-8)
    d_inv = np.diag(1.0 / np.sqrt(deg_safe))
    l_norm = d_inv @ (np.diag(deg) - s_mid) @ d_inv

    n_eig = min(max_k + 1, m)
    evals = np.sort(eigvalsh(l_norm))[:n_eig]
    gaps = np.diff(evals)

    search_end = min(len(gaps), max_k)
    if search_end < 2:
        k_hat = 2
    else:
        k_hat = int(np.argmax(gaps[1:search_end])) + 2

    k_min = max(2, k_hat - 2)
    k_max = min(m - 1, k_hat + 2)
    if k_max - k_min < 2:
        k_max = min(m - 1, k_min + 4)

    return {"eigengap_k": k_hat, "eigenvalues": evals, "gaps": gaps, "search_range": (k_min, k_max)}


def _check_dag_safety(final_supernodes: dict[str, list[str]]) -> list[tuple[str, str]]:
    names = list(final_supernodes.keys())
    layer_ranges = {}
    for sn, members in final_supernodes.items():
        layers = [int(m.split("_")[0]) for m in members if "_" in m and m.split("_")[0].isdigit()]
        if not layers:
            continue
        layer_ranges[sn] = (min(layers), max(layers))
    warns = []
    for i, a in enumerate(names):
        if a not in layer_ranges:
            continue
        a_lo, a_hi = layer_ranges[a]
        for b in names[i + 1 :]:
            if b not in layer_ranges:
                continue
            b_lo, b_hi = layer_ranges[b]
            interleave = (a_lo < b_lo < a_hi < b_hi) or (b_lo < a_lo < b_hi < a_hi)
            if interleave:
                warns.append((a, b))
    return warns


def _evaluate_grouping(
    final_supernodes: dict[str, list[str]],
    prune_graph: PruneGraph,
    similarity: Any,
) -> dict[str, dict[str, float]]:
    s = similarity
    ids = prune_graph.kept_ids
    idx = {nid: i for i, nid in enumerate(ids)}
    out: dict[str, dict[str, float]] = {}
    for sn, members in final_supernodes.items():
        member_idx = [idx[n] for n in members if n in idx]
        if not member_idx:
            continue
        if len(member_idx) == 1:
            intra_vals = [1.0]
        else:
            block = s[np.ix_(member_idx, member_idx)] if isinstance(s, np.ndarray) else s[member_idx][:, member_idx].detach().cpu().numpy()
            upper = block[np.triu_indices(len(member_idx), k=1)]
            intra_vals = upper.tolist() if upper.size else [1.0]
        layers = [int(n.split("_")[0]) for n in members if "_" in n and n.split("_")[0].isdigit()]
        out[sn] = {
            "n": float(len(members)),
            "intra_sim_mean": float(np.mean(intra_vals)),
            "intra_sim_min": float(np.min(intra_vals)),
            "layer_lo": float(min(layers) if layers else -1),
            "layer_hi": float(max(layers) if layers else -1),
        }
    return out


def score_k(
    final_supernodes: dict[str, list[str]] | list[list[str]],
    prune_graph: PruneGraph,
    similarity: Any,
    target_n_middle: int,
    w_intra: float = 0.30,
    w_dag: float = 0.25,
    w_attr: float = 0.25,
    w_size: float = 0.20,
    use_flow_faithfulness: bool = False,
    w_flow: float = 0.40,
    enforce_dag: bool = False,
) -> dict[str, Any]:
    if isinstance(final_supernodes, list):
        final_supernodes = supernodes_to_mapping(prune_graph, final_supernodes)

    stats = _evaluate_grouping(final_supernodes, prune_graph, similarity)
    dag_warnings = _check_dag_safety(final_supernodes)
    sng = build_supernode_graph(prune_graph, final_supernodes, enforce_dag=enforce_dag)
    middle = {sn: st for sn, st in stats.items() if "EMB" not in sn and "LOGIT" not in sn}
    n_middle = len(middle)
    if n_middle == 0:
        return {
            "total": 0.0,
            "intra_sim": 0.0,
            "dag_safety": 0.0,
            "attr_balance": 0.0,
            "size_score": 0.0,
            "n_middle": 0,
            "n_warnings": 0,
            "inf_conservation": 0.0,
            "edge_conservation": 0.0,
            "details": {},
        }

    sizes = [int(st["n"]) for st in middle.values()]
    total_n = max(1, sum(sizes))
    intra_raw = sum(st["intra_sim_mean"] * int(st["n"]) / total_n for st in middle.values())
    s = np.asarray(similarity.detach().cpu().numpy() if hasattr(similarity, "detach") else similarity)
    mid_idx = _middle_indices(prune_graph)
    s_mid = s[np.ix_(mid_idx, mid_idx)]
    upper = s_mid[np.triu_indices(len(mid_idx), k=1)]
    global_mean = float(upper.mean()) + 1e-8 if upper.size else 1.0
    intra = min(1.0, intra_raw / (2.0 * global_mean))
    weak = sum(1 for st in middle.values() if st["intra_sim_min"] < 0.3)
    intra *= 1.0 - 0.5 * weak / max(n_middle, 1)

    pairs = n_middle * (n_middle - 1) / 2
    dag_safety = 1.0 - len(dag_warnings) / max(pairs, 1)

    sn_names = sng["sn_names"]
    sn_inf = sng["sn_inf"]
    attr_vals = np.array(
        [max(float(sn_inf[sn_names.index(sn)]), 0.0) for sn in middle if sn in sn_names],
        dtype=np.float64,
    )
    probs = attr_vals / (attr_vals.sum() + 1e-8)
    probs = probs[probs > 1e-10]
    if len(probs) > 1:
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        attr_balance = float(entropy / np.log(len(probs)))
    else:
        attr_balance = 0.0

    ideal_k = max(2, int(np.sqrt(target_n_middle)))
    dev = abs(n_middle - ideal_k) / max(target_n_middle, 1)
    size_score = max(0.0, 1.0 - dev)

    total_base = w_intra * intra + w_dag * dag_safety + w_attr * attr_balance + w_size * size_score
    flow_report: dict[str, Any] | None = None
    f_phi = 0.0
    if use_flow_faithfulness:
        flow_report = flow_faithfulness_report(sng, final_supernodes, top_k=10)
        f_phi = float(flow_report["combined"]["F_phi"])
        total = (1.0 - w_flow) * total_base + w_flow * f_phi
    else:
        total = total_base

    out = {
        "total": float(total),
        "total_base": float(total_base),
        "intra_sim": float(intra),
        "dag_safety": float(dag_safety),
        "flow_balance": float(attr_balance),
        "attr_balance": float(attr_balance),
        "size_score": float(size_score),
        "n_middle": int(n_middle),
        "n_warnings": int(len(dag_warnings)),
        "inf_conservation": float(sng["inf_conservation"]),
        "edge_conservation": float(sng["edge_conservation"]),
        "details": {sn: {"intra_sim": st["intra_sim_mean"], "n": int(st["n"])} for sn, st in middle.items()},
    }
    if use_flow_faithfulness and flow_report is not None:
        combined = flow_report["combined"]
        out.update(
            {
                "F_phi": f_phi,
                "D_phi": float(combined["D_phi"]),
                "R_phi": float(combined["R_phi"]),
                "R_phi_balance": float(combined["R_phi_balance"]),
                "R_phi_suppressive": float(combined["R_phi_suppressive"]),
                "sigma_phi": float(combined.get("sigma_phi", combined.get("shortcut_frac", 0.0))),
                "shortcut_frac": float(combined["shortcut_frac"]),
                "flow_report": flow_report,
            }
        )
    return out


def find_best_k(
    prune_graph: PruneGraph,
    similarity: Any | None = None,
    max_layer_span: int = 4,
    k_min_override: int | None = None,
    k_max_override: int | None = None,
    weights: dict[str, float] | None = None,
    max_sn: int | None = None,
    gamma: float = 1,
    mediation_penalty: float = 0.1,
    similarity_mode: Literal["edge", "node"] = "edge",
    use_flow_faithfulness: bool = True,
    w_flow: float = 0.40,
    enforce_dag: bool = False,
) -> tuple[int, dict[int, dict[str, Any]]]:
    """
    Auto-select k for `cluster_graph` and return sweep metrics.

    Returns `(best_k, results)` where each results[k] includes `final_supernodes`.
    """
    sim = similarity
    if sim is None:
        sim = compute_similarity(
            prune_graph,
            gamma=gamma,
            mediation_penalty=mediation_penalty,
            similarity_mode=similarity_mode,
        )
    s_np = np.asarray(sim.detach().cpu().numpy() if hasattr(sim, "detach") else sim)
    n_middle = len(_middle_indices(prune_graph))
    if n_middle < 3:
        return 2, {}

    eg = eigengap_analysis(s_np, prune_graph, max_k=min(20, n_middle - 1))
    k_min = k_min_override if k_min_override is not None else int(eg["search_range"][0])
    k_max = k_max_override if k_max_override is not None else int(eg["search_range"][1])
    k_min = max(2, k_min)
    k_max = min(n_middle - 1, k_max)
    if k_min > k_max:
        k_min = k_max

    w = weights or {}
    results: dict[int, dict[str, Any]] = {}
    for k in range(k_min, k_max + 1):
        supernodes = cluster_graph(
            prune_graph,
            target_k=k,
            max_layer_span=max_layer_span,
            max_sn=max_sn,
            gamma=gamma,
            mediation_penalty=mediation_penalty,
            similarity_mode=similarity_mode,
            enforce_dag=enforce_dag,
        )
        final_supernodes = supernodes_to_mapping(prune_graph, supernodes)
        sc = score_k(
            final_supernodes,
            prune_graph,
            s_np,
            target_n_middle=n_middle,
            w_intra=w.get("w_intra", 0.30),
            w_dag=w.get("w_dag", 0.25),
            w_attr=w.get("w_attr", 0.25),
            w_size=w.get("w_size", 0.20),
            use_flow_faithfulness=use_flow_faithfulness,
            w_flow=w_flow,
            enforce_dag=enforce_dag,
        )
        sc["final_supernodes"] = final_supernodes
        results[k] = sc

    if not results:
        return int(eg["eigengap_k"]), {}
    best_k = max(results, key=lambda x: float(results[x]["total"]))
    return best_k, results
