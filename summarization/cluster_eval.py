"""
Clustering evaluation helpers: baselines and JSON-safe score export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering

from summarization.auto_grouping import find_best_k, score_k
from summarization.cluster import _edge_channels_sender_indexed, compute_similarity
from summarization.flow_analysis import supernodes_to_mapping
from summarization.prune import PruneGraph
from summarization.utils import _is_embedding, _is_fixed, _is_logit


def _middle_indices(prune_graph: PruneGraph) -> list[int]:
    return [i for i, nid in enumerate(prune_graph.kept_ids) if not _is_fixed(prune_graph.attr, nid)]


def _assemble_supernode_lists(
    prune_graph: PruneGraph,
    middle_clusters: list[list[str]],
) -> list[list[str]]:
    """Same tail as `cluster_graph`: named middle buckets + emb + logit singletons."""
    from summarization.cluster import _name_middle_supernodes

    attr = prune_graph.attr
    kept_ids = prune_graph.kept_ids
    named_middle = _name_middle_supernodes(middle_clusters, attr)
    emb_singletons = [[nid] for nid in kept_ids if _is_embedding(attr, nid)]
    logit_singletons = [[nid] for nid in kept_ids if _is_logit(attr, nid)]
    return list(named_middle.values()) + emb_singletons + logit_singletons


def baseline_kmeans_edge_profile(
    prune_graph: PruneGraph,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 20,
) -> list[list[str]]:
    """K-means on outgoing relevance×influence×adj weight profiles for middle nodes."""
    mid_idx = _middle_indices(prune_graph)
    middle_ids = [prune_graph.kept_ids[i] for i in mid_idx]
    if not middle_ids:
        return []
    adj = prune_graph.pruned_adj.float().T
    rel_sender, inf_sender = _edge_channels_sender_indexed(prune_graph, adj)
    w = (adj * rel_sender * inf_sender).detach().cpu().numpy()
    x = w[mid_idx]
    k = max(1, min(int(n_clusters), len(middle_ids)))
    if k == 1:
        labels = np.zeros(len(middle_ids), dtype=np.int64)
    elif k == len(middle_ids):
        labels = np.arange(len(middle_ids), dtype=np.int64)
    else:
        labels = KMeans(
            n_clusters=k,
            random_state=int(random_state),
            n_init=int(n_init),
        ).fit_predict(x)
    grouped: dict[int, list[str]] = {}
    for nid, lbl in zip(middle_ids, labels):
        grouped.setdefault(int(lbl), []).append(nid)
    middle_clusters = list(grouped.values())
    return _assemble_supernode_lists(prune_graph, middle_clusters)


def baseline_spectral_adjacency(
    prune_graph: PruneGraph,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 20,
) -> list[list[str]]:
    """Spectral clustering on absolute symmetrized pruned adjacency (middle nodes only)."""
    mid_idx = _middle_indices(prune_graph)
    middle_ids = [prune_graph.kept_ids[i] for i in mid_idx]
    if not middle_ids:
        return []
    adj_r = prune_graph.pruned_adj.float().detach().cpu().numpy()
    sub = adj_r[np.ix_(mid_idx, mid_idx)]
    aff = (np.abs(sub) + np.abs(sub.T)) / 2.0
    k = max(1, min(int(n_clusters), len(middle_ids)))
    if k == 1:
        labels = np.zeros(len(middle_ids), dtype=np.int64)
    elif k == len(middle_ids):
        labels = np.arange(len(middle_ids), dtype=np.int64)
    else:
        labels = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=int(random_state),
            n_init=int(n_init),
        ).fit_predict(aff)
    grouped: dict[int, list[str]] = {}
    for nid, lbl in zip(middle_ids, labels):
        grouped.setdefault(int(lbl), []).append(nid)
    middle_clusters = list(grouped.values())
    return _assemble_supernode_lists(prune_graph, middle_clusters)


def score_to_jsonable(obj: Any) -> Any:
    """Recursively convert torch / numpy / Path for JSON."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): score_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [score_to_jsonable(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def full_score_bundle(
    prune_graph: PruneGraph,
    supernodes: list[list[str]],
    similarity: Any,
    target_n_middle: int,
    use_flow_in_total: bool = False,
    w_flow: float = 0.40,
    enforce_dag: bool = False,
) -> dict[str, Any]:
    """
    `score_k` with flow metrics attached (flow_report) for evaluation tables.

    When ``use_flow_in_total`` is False, ``total`` matches the non-flow objective
    while flow diagnostics are still present.
    """
    mapping = supernodes_to_mapping(prune_graph, supernodes)
    s_np = np.asarray(similarity.detach().cpu().numpy() if hasattr(similarity, "detach") else similarity)
    base = score_k(
        mapping,
        prune_graph,
        s_np,
        target_n_middle=target_n_middle,
        use_flow_faithfulness=False,
        enforce_dag=enforce_dag,
    )
    with_flow = score_k(
        mapping,
        prune_graph,
        s_np,
        target_n_middle=target_n_middle,
        use_flow_faithfulness=True,
        w_flow=w_flow,
        enforce_dag=enforce_dag,
    )
    out = dict(base)
    out["total_with_flow"] = float(with_flow["total"])
    for key in (
        "F_phi",
        "D_phi",
        "R_phi",
        "R_phi_balance",
        "R_phi_suppressive",
        "sigma_phi",
        "shortcut_frac",
    ):
        if key in with_flow:
            out[key] = with_flow[key]
    if "flow_report" in with_flow:
        out["flow_report"] = with_flow["flow_report"]
    return out


MeanMethod = Literal["harm", "arith"]
NormMode = Literal["cos", "cos_relu"]
SimMode = Literal["node", "edge"]


def run_hyperparam_grid(
    prune_graph: PruneGraph,
    mean_methods: tuple[MeanMethod, ...] = ("harm", "arith"),
    normalizations: tuple[NormMode, ...] = ("cos", "cos_relu"),
    similarity_modes: tuple[SimMode, ...] = ("node", "edge"),
    max_layer_span: int = 4,
    max_sn: int | None = None,
    mediation_penalty: float = 0.1,
    enforce_dag: bool = False,
    random_state: int = 42,
    n_init: int = 20,
) -> dict[str, Any]:
    """Sweep clustering hyperparameters with auto-k (no flow in k selection)."""
    n_middle = len(_middle_indices(prune_graph))
    rows: list[dict[str, Any]] = []

    for mean_method in mean_methods:
        for normalization in normalizations:
            for similarity_mode in similarity_modes:
                sim = compute_similarity(
                    prune_graph,
                    mean_method=mean_method,
                    mediation_penalty=mediation_penalty,
                    similarity_mode=similarity_mode,
                    normalization=normalization,
                )
                best_k, sweep = find_best_k(
                    prune_graph,
                    similarity=sim,
                    max_layer_span=max_layer_span,
                    max_sn=max_sn,
                    mean_method=mean_method,
                    mediation_penalty=mediation_penalty,
                    similarity_mode=similarity_mode,
                    normalization=normalization,
                    use_flow_faithfulness=False,
                    w_flow=0.40,
                    enforce_dag=enforce_dag,
                    random_state=random_state,
                    n_init=n_init,
                )
                if not sweep or best_k not in sweep:
                    rows.append(
                        {
                            "mean_method": mean_method,
                            "normalization": normalization,
                            "similarity_mode": similarity_mode,
                            "best_k": int(best_k) if best_k else None,
                            "error": "empty_sweep",
                        }
                    )
                    continue
                final_map = sweep[best_k]["final_supernodes"]
                supernode_lists = list(final_map.values())
                bundle = full_score_bundle(
                    prune_graph,
                    supernode_lists,
                    sim,
                    target_n_middle=n_middle,
                    use_flow_in_total=False,
                    enforce_dag=enforce_dag,
                )
                row = {
                    "method": "cluster_graph_auto_k",
                    "mean_method": mean_method,
                    "normalization": normalization,
                    "similarity_mode": similarity_mode,
                    "best_k": int(best_k),
                    "k_sweep": {str(k): {kk: vv for kk, vv in v.items() if kk != "final_supernodes"} for k, v in sweep.items()},
                    "scores": bundle,
                    "final_supernodes": final_map,
                }
                rows.append(row)

    # Baselines use k from the default main cell (harm + cos_relu + node) if present
    ref_k: int | None = None
    for r in rows:
        if r.get("mean_method") == "harm" and r.get("normalization") == "cos_relu" and r.get("similarity_mode") == "node":
            ref_k = r.get("best_k")
            break
    if ref_k is None and rows and rows[0].get("best_k") is not None:
        ref_k = int(rows[0]["best_k"])

    if ref_k is not None and n_middle >= 2:
        for name, fn in (
            ("kmeans_edge_profile", baseline_kmeans_edge_profile),
            ("spectral_abs_adj", baseline_spectral_adjacency),
        ):
            supernodes = fn(prune_graph, n_clusters=ref_k, random_state=random_state, n_init=n_init)
            sim_ref = compute_similarity(
                prune_graph,
                mean_method="harm",
                mediation_penalty=mediation_penalty,
                similarity_mode="node",
                normalization="cos_relu",
            )
            bundle = full_score_bundle(
                prune_graph,
                supernodes,
                sim_ref,
                target_n_middle=n_middle,
                enforce_dag=enforce_dag,
            )
            rows.append(
                {
                    "method": name,
                    "k": int(ref_k),
                    "scores": bundle,
                    "final_supernodes": supernodes_to_mapping(prune_graph, supernodes),
                }
            )

    return {
        "n_middle": int(n_middle),
        "reference_k_for_baselines": ref_k,
        "rows": rows,
    }


def save_eval_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(score_to_jsonable(payload), f, indent=2)
