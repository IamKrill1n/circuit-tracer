from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.linalg import eigvalsh
from sklearn.metrics import silhouette_score

from summarization.cluster import (
    build_supernode_graph,
    cluster_graph,
    clusters_to_supernodes,
    compute_similarity,
    mapping_dict_to_supernodes,
)
from summarization.prune import PruneGraph
from summarization.supernode_graph import Supernode
from summarization.utils import _is_fixed


def _middle_indices(prune_graph: PruneGraph) -> list[int]:
    return [i for i, nid in enumerate(prune_graph.kept_ids) if not _is_fixed(prune_graph.attr, nid)]


def _as_supernode_rows(
    prune_graph: PruneGraph,
    final_supernodes: dict[str, list[str]] | list[list[str]] | list[Supernode],
) -> list[Supernode]:
    if isinstance(final_supernodes, list) and (
        not final_supernodes or isinstance(final_supernodes[0], Supernode)
    ):
        return final_supernodes
    if isinstance(final_supernodes, list):
        return clusters_to_supernodes(prune_graph, final_supernodes)
    return mapping_dict_to_supernodes(prune_graph, final_supernodes)


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


def _silhouette_over_middle(
    similarity: np.ndarray,
    prune_graph: PruneGraph,
    rows: list[Supernode],
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
    for row in rows:
        if row.type != "features":
            continue
        assigned = False
        for nid in row.member_node_ids():
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
    rows: list[Supernode],
) -> float:
    """
    Edge-weighted DAG-safety score in [0, 1] using backward-edge mass ratio:

        1 - (sum of |sn_adj[i, j]| for SN_i -> SN_j with layer_j <= layer_i)
            / (sum of all off-diagonal |sn_adj| among middle supernodes)

    Higher is better; 1.0 means no backward flow by layer ordering.
    """
    layer_centers: dict[str, float] = {}
    for row in rows:
        if row.type != "features":
            continue
        layer_centers[row.name] = float(row.layer_min + row.layer_max) / 2.0

    name_to_idx = {name: idx for idx, name in enumerate(sn_names)}
    valid_names = [name for name in sn_names if name in layer_centers]
    if len(valid_names) < 2:
        return 1.0

    abs_adj = np.abs(sn_adj)
    total_w = 0.0
    backward_w = 0.0
    for src_name in valid_names:
        i = name_to_idx[src_name]
        src_layer = layer_centers[src_name]
        for dst_name in valid_names:
            if src_name == dst_name:
                continue
            j = name_to_idx[dst_name]
            w = float(abs_adj[i, j])
            if w <= 0.0:
                continue
            total_w += w
            dst_layer = layer_centers[dst_name]
            if dst_layer <= src_layer:
                backward_w += w

    if total_w <= 1e-12:
        return 1.0

    return float(max(0.0, 1.0 - backward_w / (total_w + 1e-12)))


def score_clusters(
    final_supernodes: dict[str, list[str]] | list[list[str]] | list[Supernode],
    prune_graph: PruneGraph,
    similarity: Any,
    enforce_dag: bool = False,
) -> dict[str, Any]:
    """
    Score a clustering using two complementary metrics:

      total = silhouette_norm * dag_score

    where:
      - silhouette_norm = (mean silhouette over middle nodes + 1) / 2, in [0, 1].
      - dag_score = 1 - (backward-edge mass / total middle SN edge mass), in [0, 1].

    The legacy components (intra_sim, attr_balance, size_score, dag_safety) are no
    longer computed. Legacy weight kwargs (`w_intra`, `w_dag`, `w_attr`, `w_size`)
    are accepted for backward compatibility but ignored.
    """

    rows = _as_supernode_rows(prune_graph, final_supernodes)
    sng = build_supernode_graph(prune_graph, rows, enforce_dag=enforce_dag)
    n_middle = sum(1 for r in rows if r.type == "features")

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
    sil_raw, sil_norm = _silhouette_over_middle(s, prune_graph, rows)

    sn_names = list(sng.sn_names)
    sn_adj = np.asarray(sng.sn_adj, dtype=np.float64)
    dag_score = _dag_interleave_edge_fraction(sn_adj, sn_names, rows)
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
    similarity_mode: Literal["edge", "node"] = "node",
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
            similarity_mode=similarity_mode,
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
            similarity_mode=similarity_mode,
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
