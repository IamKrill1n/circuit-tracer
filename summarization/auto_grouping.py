from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.linalg import eigvalsh
from sklearn.metrics import silhouette_score

from summarization.cluster import (
    build_supernode_graph,
    cluster_graph,
    compute_similarity,
    supernodes_to_mapping,
)
from summarization.prune import PruneGraph
from summarization.utils import _is_fixed


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


def _layer_range_from_members(members: list[str]) -> tuple[int, int] | None:
    layers: list[int] = []
    for m in members:
        if "_" not in m:
            continue
        head = m.split("_")[0]
        if head.isdigit():
            layers.append(int(head))
    if not layers:
        return None
    return min(layers), max(layers)


def _interleaving_pairs(layer_ranges: dict[str, tuple[int, int]]) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    items = list(layer_ranges.items())
    for i, (a, (a_lo, a_hi)) in enumerate(items):
        for b, (b_lo, b_hi) in items[i + 1 :]:
            if (a_lo < b_lo < a_hi < b_hi) or (b_lo < a_lo < b_hi < a_hi):
                pairs.add(frozenset((a, b)))
    return pairs


def _silhouette_over_middle(
    similarity: np.ndarray,
    prune_graph: PruneGraph,
    final_supernodes: dict[str, list[str]],
) -> tuple[float, float]:
    """
    Mean silhouette score over middle nodes, plus its [0, 1]-normalized form.

    Returns (silhouette_raw, silhouette_norm) where silhouette_norm = (sil + 1) / 2.
    Returns (0.0, 0.5) when silhouette is undefined (single cluster, all singletons,
    or no middle nodes assigned).
    """
    ids = prune_graph.kept_ids
    id_to_idx = {nid: i for i, nid in enumerate(ids)}

    nid_to_label: dict[str, int] = {}
    label_idx = 0
    for sn, members in final_supernodes.items():
        if "EMB" in sn or "LOGIT" in sn:
            continue
        assigned = False
        for nid in members:
            if nid in id_to_idx:
                nid_to_label[nid] = label_idx
                assigned = True
        if assigned:
            label_idx += 1

    if not nid_to_label:
        return 0.0, 0.5

    node_indices = [id_to_idx[nid] for nid in nid_to_label]
    labels_arr = np.fromiter(
        (nid_to_label[ids[i]] for i in node_indices),
        dtype=np.int64,
        count=len(node_indices),
    )
    n_distinct = int(len(set(labels_arr.tolist())))
    if n_distinct < 2 or n_distinct >= len(labels_arr):
        return 0.0, 0.5

    s_block = similarity[np.ix_(node_indices, node_indices)]
    s_block = (s_block + s_block.T) / 2.0
    s_block = np.clip(s_block, 0.0, 1.0)
    distance = 1.0 - s_block
    np.fill_diagonal(distance, 0.0)
    sil = float(silhouette_score(distance, labels_arr, metric="precomputed"))
    return sil, float((sil + 1.0) / 2.0)


def _dag_interleave_edge_fraction(
    sn_adj: np.ndarray,
    sn_names: list[str],
    final_supernodes: dict[str, list[str]],
) -> float:
    """
    Edge-weighted DAG-safety score in [0, 1]:

        1 - (sum of |sn_adj| weight between layer-interleaving SN pairs)
            / (sum of all |sn_adj| weight)

    Higher is better; 1.0 means no flow occurs between supernode pairs whose
    layer ranges interleave.
    """
    layer_ranges: dict[str, tuple[int, int]] = {}
    for sn, members in final_supernodes.items():
        if "EMB" in sn or "LOGIT" in sn:
            continue
        rng = _layer_range_from_members(members)
        if rng is not None:
            layer_ranges[sn] = rng

    abs_adj = np.abs(sn_adj)
    total_w = float(abs_adj.sum())
    if total_w <= 1e-12:
        return 1.0

    pairs = _interleaving_pairs(layer_ranges)
    if not pairs:
        return 1.0

    name_to_idx = {name: idx for idx, name in enumerate(sn_names)}
    violation_w = 0.0
    for pair in pairs:
        a, b = tuple(pair)
        if a not in name_to_idx or b not in name_to_idx:
            continue
        i, j = name_to_idx[a], name_to_idx[b]
        violation_w += float(abs_adj[i, j] + abs_adj[j, i])

    return float(max(0.0, 1.0 - violation_w / (total_w + 1e-12)))


def score_k(
    final_supernodes: dict[str, list[str]] | list[list[str]],
    prune_graph: PruneGraph,
    similarity: Any,
    enforce_dag: bool = False
) -> dict[str, Any]:
    """
    Score a clustering using two complementary metrics:

      total = silhouette_norm * dag_score

    where:
      - silhouette_norm = (mean silhouette over middle nodes + 1) / 2, in [0, 1].
      - dag_score = 1 - (interleaving-edge weight / total SN edge weight), in [0, 1].

    The legacy components (intra_sim, attr_balance, size_score, dag_safety) are no
    longer computed. Legacy weight kwargs (`w_intra`, `w_dag`, `w_attr`, `w_size`)
    are accepted for backward compatibility but ignored.
    """

    if isinstance(final_supernodes, list):
        final_supernodes = supernodes_to_mapping(prune_graph, final_supernodes)

    sng = build_supernode_graph(prune_graph, final_supernodes, enforce_dag=enforce_dag)
    middle_keys = [sn for sn in final_supernodes if "EMB" not in sn and "LOGIT" not in sn]
    n_middle = len(middle_keys)

    if n_middle == 0:
        return {
            "score_arith": 0.0,
            "score_harm": 0.0,
            "score_geo": 0.0,
            "sil_raw": 0.0,
            "sil_norm": 0.0,
            "dag_score": 1.0,
            "n_middle": 0,
        }

    s = np.asarray(
        similarity.detach().cpu().numpy() if hasattr(similarity, "detach") else similarity,
        dtype=np.float64,
    )
    sil_raw, sil_norm = _silhouette_over_middle(s, prune_graph, final_supernodes)

    sn_names = list(sng["sn_names"])
    sn_adj = np.asarray(sng["sn_adj"], dtype=np.float64)
    dag_score = _dag_interleave_edge_fraction(sn_adj, sn_names, final_supernodes)

    print(f"sil_norm: {sil_norm}, dag_score: {dag_score}")
    score_arith = (sil_norm + dag_score) / 2.0
    score_harm = 2 / ((1 / (sil_norm + 1e-12)) + (1 / (dag_score + 1e-12)))
    score_geo = np.sqrt(sil_norm * dag_score)

    return {
        "score_arith": float(score_arith),
        "score_harm": float(score_harm),
        "score_geo": float(score_geo),
        "sil_raw": float(sil_raw),
        "sil_norm": float(sil_norm),
        "dag_score": float(dag_score),
        "n_middle": int(n_middle),
    }


def find_best_k(
    prune_graph: PruneGraph,
    similarity: Any | None = None,
    max_layer_span: int = 4,
    k_min_override: int | None = None,
    k_max_override: int | None = None,
    weights: dict[str, float] | None = None,
    max_sn: int | None = None,
    mean_method: Literal["geo", "harm", "arith"] = "arith",
    mediation_penalty: float = 0.1,
    similarity_mode: Literal["edge", "node"] = "node",
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

    del weights  # legacy weight kwargs are no longer used by score_k
    results: dict[int, dict[str, Any]] = {}
    for k in range(k_min, k_max + 1):
        supernodes = cluster_graph(
            prune_graph,
            target_k=k,
            max_layer_span=max_layer_span,
            max_sn=max_sn,
            mean_method=mean_method,
            mediation_penalty=mediation_penalty,
            similarity_mode=similarity_mode,
            enforce_dag=enforce_dag,
            random_state=random_state,
            n_init=n_init,
        )
        final_supernodes = supernodes_to_mapping(prune_graph, supernodes)
        sc = score_k(
            final_supernodes,
            prune_graph,
            s_np,
            enforce_dag=enforce_dag,
        )
        sc["final_supernodes"] = final_supernodes
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
        mapping = supernodes_to_mapping(prune_graph, clusters)
        result = score_k(
            mapping,
            prune_graph,
            s_np,
            enforce_dag=enforce_dag,
        )
        result["final_supernodes"] = mapping
        return fallback_k, {fallback_k: result}

    eigengap = eigengap_analysis(s_np, prune_graph, max_k=min(20, n_middle - 1))
    k_min = k_min_override if k_min_override is not None else int(eigengap["search_range"][0])
    k_max = k_max_override if k_max_override is not None else int(eigengap["search_range"][1])
    k_min = max(2, min(k_min, n_middle))
    k_max = max(k_min, min(k_max, n_middle))

    results: dict[int, dict[str, Any]] = {}
    for target_k in range(k_min, k_max + 1):
        clusters = clusterer(target_k)
        mapping = supernodes_to_mapping(prune_graph, clusters)
        result = score_k(
            mapping,
            prune_graph,
            s_np,
            enforce_dag=enforce_dag,
        )
        result["final_supernodes"] = mapping
        results[target_k] = result

    best_k = max(results, key=lambda k: float(results[k]["score_arith"]))
    return best_k, results
