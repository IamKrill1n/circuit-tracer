from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.linalg import eigvalsh

from summarization.cluster import (
    cluster_graph,
    clusters_to_supernodes,
    compute_similarity,
)
from summarization.cluster_scoring import (
    _middle_indices,
    score_clusters,
)
from summarization.prune import PruneGraph


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


def find_best_k(
    prune_graph: PruneGraph,
    similarity: Any | None = None,
    max_layer_span: int = 4,
    k_min_override: int | None = None,
    k_max_override: int | None = None,
    weights: dict[str, float] | None = None,
    max_sn: int | None = None,
    mean_method: Literal["geo", "harm", "arith"] = "arith",
    decay_rate: float | None = None,
    enforce_dag: bool = False,
    random_state: int = 42,
    n_init: int = 20,
) -> tuple[int, dict[int, dict[str, Any]]]:
    """
    Auto-select k for `cluster_graph` and return sweep metrics.

    Returns `(best_k, results)` where each results[k] includes `final_supernodes`.
    """
    sim = similarity
    if sim is None:
        sim = compute_similarity(
            prune_graph,
            mean_method=mean_method,
            decay_rate=decay_rate,
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

    del weights  # legacy weight kwargs are no longer used by score_k
    results: dict[int, dict[str, Any]] = {}
    for k in range(k_min, k_max + 1):
        supernodes = cluster_graph(
            prune_graph,
            target_k=k,
            max_layer_span=max_layer_span,
            max_sn=max_sn,
            mean_method=mean_method,
            enforce_dag=enforce_dag,
            random_state=random_state,
            n_init=n_init,
        )
        rows = clusters_to_supernodes(prune_graph, supernodes)
        sc = score_clusters(
            rows,
            prune_graph,
            s_np,
            enforce_dag=enforce_dag,
        )
        sc["final_supernodes"] = {s.name: s.member_node_ids() for s in rows}
        results[k] = sc

    if not results:
        return int(eg["eigengap_k"]), {}
    best_k = max(results, key=lambda x: float(results[x]["score_arith"]))
    return best_k, results


def find_best_k_for_clusterer(
    *,
    prune_graph: PruneGraph,
    similarity: Any,
    clusterer: Any,
    k_min_override: int | None = None,
    k_max_override: int | None = None,
    weights: dict[str, float] | None = None,
    enforce_dag: bool = False,
) -> tuple[int, dict[int, dict[str, Any]]]:
    """
    Auto-select k for an arbitrary clusterer using the same scoring objective
    as `find_best_k`.
    """
    del weights  # legacy weight kwargs are no longer used by score_k
    s_np = np.asarray(similarity.detach().cpu().numpy() if hasattr(similarity, "detach") else similarity)
    n_middle = len(_middle_indices(prune_graph))
    if n_middle < 3:
        fallback_k = max(0, n_middle)
        clusters = clusterer(fallback_k)
        rows = clusters_to_supernodes(prune_graph, clusters)
        result = score_clusters(
            rows,
            prune_graph,
            s_np,
            enforce_dag=enforce_dag,
        )
        result["final_supernodes"] = {s.name: s.member_node_ids() for s in rows}
        return fallback_k, {fallback_k: result}

    eigengap = eigengap_analysis(s_np, prune_graph, max_k=min(20, n_middle - 1))
    k_min = k_min_override if k_min_override is not None else int(eigengap["search_range"][0])
    k_max = k_max_override if k_max_override is not None else int(eigengap["search_range"][1])
    k_min = max(2, min(k_min, n_middle))
    k_max = max(k_min, min(k_max, n_middle))

    results: dict[int, dict[str, Any]] = {}
    for target_k in range(k_min, k_max + 1):
        clusters = clusterer(target_k)
        rows = clusters_to_supernodes(prune_graph, clusters)
        result = score_clusters(
            rows,
            prune_graph,
            s_np,
            enforce_dag=enforce_dag,
        )
        result["final_supernodes"] = {s.name: s.member_node_ids() for s in rows}
        results[target_k] = result

    best_k = max(results, key=lambda k: float(results[k]["score_arith"]))
    return best_k, results
