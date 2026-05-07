from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from sklearn.cluster import SpectralClustering

from summarization.prune import PruneGraph
from summarization.supernode_graph import Node, Supernode, cluster_kind_to_supernode_type, node_from_prune_graph
from summarization.utils import (
    layer_index_from_node,
    layer_index_from_node_id,
    node_is_embedding,
    node_is_fixed,
    node_is_logit,
)


def _nodes_by_id(prune_graph: PruneGraph) -> dict[str, Node]:
    return {n.node_id: n for n in prune_graph.nodes}


def _classify_node(node_id: str, nodes_by_id: dict[str, Node]) -> str:
    n = nodes_by_id.get(node_id)
    if n is None:
        return "middle"
    if node_is_embedding(n):
        return "emb"
    if node_is_logit(n):
        return "logit"
    return "middle"


def _layer_numeric(node_id: str, nodes_by_id: dict[str, Node]) -> int:
    n = nodes_by_id.get(node_id)
    return layer_index_from_node(n) if n is not None else layer_index_from_node_id(node_id)


def _cosine_norm(matrix: torch.Tensor) -> torch.Tensor:
    diag = torch.sqrt(torch.diag(matrix).clamp(min=1e-8))
    return matrix / diag.unsqueeze(1) / diag.unsqueeze(0)


def _weighted_row_cosine(features: torch.Tensor) -> torch.Tensor:
    gram = features @ features.T
    return _cosine_norm(gram)


def _normalize_edge_weights(weights: torch.Tensor) -> torch.Tensor:
    # Preserve sign by normalizing with max absolute magnitude.
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    w_abs_max = float(weights.abs().max().item()) if weights.numel() else 0.0
    if w_abs_max <= 0.0:
        return torch.zeros_like(weights)
    return weights / (w_abs_max + 1e-8)


def _normalize_node_weights(scores: torch.Tensor | None, n_nodes: int, device: torch.device) -> torch.Tensor:
    # Missing tensors can happen for older serialized PruneGraph payloads.
    if scores is None:
        return torch.ones(n_nodes, dtype=torch.float32, device=device)

    values = scores.detach().float().to(device).reshape(-1)
    if values.numel() != n_nodes:
        return torch.ones(n_nodes, dtype=torch.float32, device=device)

    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    v_min = float(values.min().item()) if values.numel() else 0.0
    v_max = float(values.max().item()) if values.numel() else 0.0
    if v_max - v_min <= 1e-8:
        return torch.ones_like(values)

    return ((values - v_min) / (v_max - v_min + 1e-8)).clamp(0.0, 1.0)


def _edge_channels_sender_indexed(prune_graph: PruneGraph, adj_sender: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edge_rel = prune_graph.edge_relevance
    edge_inf = prune_graph.edge_influence
    if edge_rel is None or edge_inf is None:
        # Backward-compatible fallback for older payloads.
        fallback = _normalize_edge_weights(adj_sender)
        return fallback, fallback

    rel_sender = _normalize_edge_weights(edge_rel.float().T)
    inf_sender = _normalize_edge_weights(edge_inf.float().T)
    return rel_sender, inf_sender

def sign_aware_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0) / (tensor.abs().max() + 1e-8)

def compute_similarity(
    prune_graph: PruneGraph,
    mean_method: Literal["geo", "harm", "arith"] = "arith",
    similarity_mode: Literal["edge", "node"] = "node",
    decay_rate: float | None = None,
    max_layer_span: int | None = None,
) -> torch.Tensor:
    """
    Compute node similarity from weighted shared out/in structure.

    - `edge` mode uses edge influence/relevance channels (current behavior).
    - `node` mode applies node influence/relevance pairwise weights.

    Output/input cosine similarities are always clamped to ``[0, 1]`` before
    being combined.
    """
    adj = prune_graph.pruned_adj.clone().float().T
    if similarity_mode == "edge":
        rel_sender, inf_sender = _edge_channels_sender_indexed(prune_graph, adj)
        adj_sign = adj.sign()
        weighted_out = adj_sign * inf_sender
        weighted_in = (adj_sign * rel_sender).T
        s_out = weighted_out @ weighted_out.T
        s_in = weighted_in @ weighted_in.T
    elif similarity_mode == "node":
        n_nodes = adj.shape[0]
        node_inf = _normalize_node_weights(prune_graph.node_influence, n_nodes, adj.device)
        node_rel = _normalize_node_weights(prune_graph.node_relevance, n_nodes, adj.device)
        s_out = adj @ torch.diag(node_inf) @ adj.T
        s_in = adj.T @ torch.diag(node_rel) @ adj
    else:
        raise ValueError("Unsupported similarity_mode. Expected 'edge' or 'node'.")

    s_out_cos = _cosine_norm(s_out).clamp(0.0, 1.0)
    s_in_cos = _cosine_norm(s_in).clamp(0.0, 1.0)

    if mean_method == "geo":
        s = (s_out_cos * s_in_cos).sqrt()
    elif mean_method == "harm":
        # Equivalent to 2 / (1/a + 1/b) but safe when a or b is exactly 0
        # (cosine values are clamped to [0, 1] above, so zeros are common).
        s = (2.0 * s_out_cos * s_in_cos) / (s_out_cos + s_in_cos + 1e-12)
    elif mean_method == "arith":
        s = (s_out_cos + s_in_cos) / 2.0
    else:
        raise ValueError(f"Unsupported mean_method={mean_method!r}.")

    if decay_rate is not None and decay_rate > 0.0:
        layer_indices = []
        for n in prune_graph.nodes:
            try:
                layer_indices.append(layer_index_from_node(n))
            except Exception:
                layer_indices.append(0)

        layers_t = torch.tensor(layer_indices, dtype=torch.float32, device=s.device)
        layer_diffs = torch.abs(layers_t.unsqueeze(1) - layers_t.unsqueeze(0))
        
        penalty = torch.exp(-decay_rate * layer_diffs)
        if max_layer_span is not None:
            penalty[layer_diffs > max_layer_span] = 0.0
            
        s = s * penalty

    return s


def _layer_span_nodes(cluster: list[str], nodes_by_id: dict[str, Node]) -> int:
    lv = [_layer_numeric(n, nodes_by_id) for n in cluster]
    return max(lv) - min(lv)


def _split_cluster_by_span(
    cluster: list[str], nodes_by_id: dict[str, Node], max_layer_span: int
) -> list[list[str]]:
    work = [cluster]
    out: list[list[str]] = []
    while work:
        current = work.pop()
        span = _layer_span_nodes(current, nodes_by_id)
        if span <= max_layer_span or len(current) <= 1:
            out.append(current)
            continue
        current_sorted = sorted(current, key=lambda n: _layer_numeric(n, nodes_by_id))
        lo = _layer_numeric(current_sorted[0], nodes_by_id)
        hi = _layer_numeric(current_sorted[-1], nodes_by_id)
        cut = (lo + hi) // 2
        left = [n for n in current_sorted if _layer_numeric(n, nodes_by_id) <= cut]
        right = [n for n in current_sorted if _layer_numeric(n, nodes_by_id) > cut]
        if not left or not right:
            half = max(1, len(current_sorted) // 2)
            left, right = current_sorted[:half], current_sorted[half:]
        work.extend([left, right])
    return out


def _split_cluster_by_boundary(
    cluster: list[str], boundary_layer: int, nodes_by_id: dict[str, Node]
) -> list[list[str]]:
    left = [n for n in cluster if _layer_numeric(n, nodes_by_id) < boundary_layer]
    right = [n for n in cluster if _layer_numeric(n, nodes_by_id) >= boundary_layer]
    out: list[list[str]] = []
    if left:
        out.append(left)
    if right:
        out.append(right)
    return out


def _resolve_layer_interleaving(
    clusters: list[list[str]], nodes_by_id: dict[str, Node]
) -> list[list[str]]:
    """
    Split interleaving/containment ranges until no layer-range conflicts remain.
    """
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            if changed:
                break
            a = clusters[i]
            a_layers = [_layer_numeric(n, nodes_by_id) for n in a]
            a_lo, a_hi = min(a_layers), max(a_layers)

            for j in range(i + 1, len(clusters)):
                b = clusters[j]
                b_layers = [_layer_numeric(n, nodes_by_id) for n in b]
                b_lo, b_hi = min(b_layers), max(b_layers)

                # interleaving: a_lo < b_lo < a_hi < b_hi (or symmetric)
                a_wraps_b_boundary = a_lo < b_lo < a_hi < b_hi
                b_wraps_a_boundary = b_lo < a_lo < b_hi < a_hi
                # containment
                a_contains_b = a_lo < b_lo and b_hi < a_hi
                b_contains_a = b_lo < a_lo and a_hi < b_hi

                if a_wraps_b_boundary or a_contains_b:
                    replacement = _split_cluster_by_boundary(a, b_lo, nodes_by_id)
                    clusters = clusters[:i] + replacement + clusters[i + 1 :]
                    changed = True
                    break

                if b_wraps_a_boundary or b_contains_a:
                    replacement = _split_cluster_by_boundary(b, a_lo, nodes_by_id)
                    clusters = clusters[:j] + replacement + clusters[j + 1 :]
                    changed = True
                    break

    clusters.sort(key=lambda c: min(_layer_numeric(n, nodes_by_id) for n in c))
    return clusters


def _merge_to_budget(
    clusters: list[list[str]], nodes_by_id: dict[str, Node], max_sn: int
) -> list[list[str]]:
    """Greedily merge layer-adjacent clusters until budget is met."""
    while len(clusters) > max_sn:
        best_i = -1
        best_gap = float("inf")
        for i in range(len(clusters) - 1):
            hi_i = max(_layer_numeric(n, nodes_by_id) for n in clusters[i])
            lo_j = min(_layer_numeric(n, nodes_by_id) for n in clusters[i + 1])
            gap = abs(lo_j - hi_i)
            if gap < best_gap:
                best_gap = gap
                best_i = i

        if best_i < 0:
            break

        merged = clusters[best_i] + clusters[best_i + 1]
        clusters = clusters[:best_i] + [merged] + clusters[best_i + 2 :]

    return clusters


def _name_middle_supernodes(
    clusters: list[list[str]], nodes_by_id: dict[str, Node]
) -> dict[str, list[str]]:
    clusters = sorted(clusters, key=lambda c: min(_layer_numeric(n, nodes_by_id) for n in c))
    return {f"SN_{i}": members for i, members in enumerate(clusters)}


def _supernode_from_member_ids(
    prune_graph: PruneGraph,
    name: str,
    member_ids: list[str],
    kind: str,
    id_to_idx: dict[str, int] | None = None,
) -> Supernode:
    nodes = [node_from_prune_graph(prune_graph, node_id, id_to_idx=id_to_idx) for node_id in member_ids]
    nodes_map = _nodes_by_id(prune_graph)
    layers = [_layer_numeric(node_id, nodes_map) for node_id in member_ids]
    if kind == "emb":
        sn_type = cluster_kind_to_supernode_type("emb")
    elif kind == "logit":
        sn_type = cluster_kind_to_supernode_type("logit")
    else:
        sn_type = cluster_kind_to_supernode_type("middle")
    return Supernode(
        name=name,
        features=nodes,
        type=sn_type,
        layer_min=min(layers) if layers else 0,
        layer_max=max(layers) if layers else 0,
    )


def labels_to_supernodes(
    prune_graph: PruneGraph,
    middle_ids: list[str],
    labels: np.ndarray,
) -> list[list[str]]:
    grouped: dict[int, list[str]] = {}
    for node_id, label in zip(middle_ids, labels):
        grouped.setdefault(int(label), []).append(node_id)

    middle_clusters = [grouped[label] for label in sorted(grouped)]
    emb_singletons = [[n.node_id] for n in prune_graph.nodes if node_is_embedding(n)]
    logit_singletons = [[n.node_id] for n in prune_graph.nodes if node_is_logit(n)]
    return middle_clusters + emb_singletons + logit_singletons


def cluster_graph(
    prune_graph: PruneGraph,
    target_k: int = 7,
    max_layer_span: int = 4,
    max_sn: int | None = None,
    mean_method: Literal["geo", "harm", "arith"] = "arith",
    similarity_mode: Literal["edge", "node"] = "node",
    decay_rate: float | None = 1.0,
    enforce_dag: bool = True,
    random_state: int = 42,
    n_init: int = 20,
) -> list[list[str]]:
    """
    Cluster a pruned attribution graph into supernodes.

    Args:
        prune_graph: Output of `prune_graph_pipeline`.
        target_k: Target number of middle supernodes.
        max_layer_span: Maximum allowed layer span within a middle supernode.
        max_sn: Optional hard cap on number of middle supernodes.
        mean_method: Mean used to combine output/input cosine similarities.
        similarity_mode: `edge` or `node` similarity construction.
        random_state: Random seed for spectral clustering k-means init.
        n_init: Number of k-means runs for `SpectralClustering(assign_labels="kmeans")`.

    Returns:
        List of supernodes where each supernode is a list of node ids.
        Embedding/logit nodes are returned as singleton supernodes.
    """
    kept_ids = prune_graph.node_ids
    nodes_by_id = _nodes_by_id(prune_graph)

    if not kept_ids:
        return []

    sim = compute_similarity(
        prune_graph,
        mean_method=mean_method,
        similarity_mode=similarity_mode,
        decay_rate=decay_rate,
        max_layer_span=max_layer_span,
    )

    middle_idx = [i for i, nid in enumerate(kept_ids) if not node_is_fixed(nodes_by_id[nid])]
    middle_ids = [kept_ids[i] for i in middle_idx]

    if not middle_ids:
        fixed_only = [[nid] for nid in kept_ids]
        return fixed_only

    mid_sim = sim[middle_idx][:, middle_idx].detach().cpu().numpy().clip(0.0, 1.0)

    # assert symmetry of mid_sim
    assert np.allclose(mid_sim, mid_sim.T)
    # mid_sim = ((mid_sim + mid_sim.T) / 2.0).clip(0.0, 1.0)
    target_k = max(1, min(target_k, len(middle_ids)))

    if target_k == 1:
        labels = np.zeros(len(middle_ids), dtype=np.int64)
    elif target_k == len(middle_ids):
        labels = np.arange(len(middle_ids), dtype=np.int64)
    else:
        labels = SpectralClustering(
            n_clusters=target_k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=int(random_state),
            n_init=int(n_init),
        ).fit_predict(mid_sim)

    grouped: dict[int, list[str]] = {}
    for nid, lbl in zip(middle_ids, labels):
        grouped.setdefault(int(lbl), []).append(nid)
    middle_clusters = list(grouped.values())

    if enforce_dag:
        span_safe: list[list[str]] = []
        for cluster in middle_clusters:
            span_safe.extend(_split_cluster_by_span(cluster, nodes_by_id, max_layer_span=max_layer_span))
        middle_clusters = span_safe

    if enforce_dag:
        middle_clusters = _resolve_layer_interleaving(middle_clusters, nodes_by_id)

    if max_sn is not None and enforce_dag:
        middle_clusters = _merge_to_budget(middle_clusters, nodes_by_id, max_sn=max_sn)

    # Keep deterministic naming order for middle SNs, but return member lists only.
    named_middle = _name_middle_supernodes(middle_clusters, nodes_by_id)

    emb_singletons = [[nid] for nid in kept_ids if node_is_embedding(nodes_by_id[nid])]
    logit_singletons = [[nid] for nid in kept_ids if node_is_logit(nodes_by_id[nid])]

    supernodes = list(named_middle.values()) + emb_singletons + logit_singletons
    return supernodes


def cluster_graph_with_labels(
    prune_graph: PruneGraph,
    **kwargs,
) -> list[list[str]]:
    """
    Convenience wrapper for Neuronpedia-style format:
    [[label, node_id, ...], ...]
    """
    raw = cluster_graph(prune_graph, **kwargs)
    out: list[list[str]] = []
    for i, members in enumerate(raw):
        if len(members) == 1:
            continue
        out.append([f"cluster_{i}", *members])
    return out


def clusters_to_supernodes(
    prune_graph: PruneGraph,
    supernodes: list[list[str]],
    middle_prefix: str = "SN",
) -> list[Supernode]:
    """Convert `cluster_graph` member lists into named `Supernode` rows (middle + emb + logit)."""
    nodes_by_id = _nodes_by_id(prune_graph)
    middle: list[list[str]] = []
    emb: list[list[str]] = []
    logit: list[list[str]] = []

    for sn in supernodes:
        if not sn:
            continue
        first = sn[0]
        kind = _classify_node(first, nodes_by_id)
        if kind == "emb":
            emb.append(sn)
        elif kind == "logit":
            logit.append(sn)
        else:
            middle.append(sn)

    middle = sorted(middle, key=lambda m: min(_layer_numeric(n, nodes_by_id) for n in m))
    out: list[Supernode] = []
    id_to_idx = {n.node_id: i for i, n in enumerate(prune_graph.nodes)}
    for i, sn in enumerate(middle):
        k = _classify_node(sn[0], nodes_by_id)
        out.append(_supernode_from_member_ids(prune_graph, f"{middle_prefix}_{i}", list(sn), k, id_to_idx=id_to_idx))
    emb_logit: list[Supernode] = []
    for i, sn in enumerate(emb):
        k = _classify_node(sn[0], nodes_by_id)
        emb_logit.append(_supernode_from_member_ids(prune_graph, f"SN_EMB_{i}", list(sn), k, id_to_idx=id_to_idx))
    for i, sn in enumerate(logit):
        k = _classify_node(sn[0], nodes_by_id)
        emb_logit.append(_supernode_from_member_ids(prune_graph, f"SN_LOGIT_{i}", list(sn), k, id_to_idx=id_to_idx))
    return out + emb_logit


def supernodes_to_mapping(
    prune_graph: PruneGraph,
    supernodes: list[list[str]],
    middle_prefix: str = "SN",
) -> dict[str, list[str]]:
    """Convert `cluster_graph` output into a named supernode mapping (dict shim)."""
    rows = clusters_to_supernodes(prune_graph, supernodes, middle_prefix=middle_prefix)
    return {s.name: s.member_node_ids() for s in rows}


def mapping_dict_to_supernodes(prune_graph: PruneGraph, mapping: dict[str, list[str]]) -> list[Supernode]:
    """Preserve dict insertion order; one `Supernode` per key (features may be filtered later in graph build)."""
    nodes_by_id = _nodes_by_id(prune_graph)
    out: list[Supernode] = []
    id_to_idx = {n.node_id: i for i, n in enumerate(prune_graph.nodes)}
    for name, feats in mapping.items():
        if not feats:
            continue
        k = _classify_node(feats[0], nodes_by_id)
        out.append(_supernode_from_member_ids(prune_graph, name, list(feats), k, id_to_idx=id_to_idx))
    return out

