from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from sklearn.cluster import SpectralClustering

from summarization.prune import PruneGraph
from summarization.utils import _is_embedding, _is_fixed, _is_logit, _parse_layer


def _classify_node(node_id: str, attr: dict[str, dict[str, Any]]) -> str:
    if _is_embedding(attr, node_id):
        return "emb"
    if _is_logit(attr, node_id):
        return "logit"
    return "middle"


def _cosine_norm(matrix: torch.Tensor) -> torch.Tensor:
    diag = torch.sqrt(torch.diag(matrix).clamp(min=1e-8))
    return matrix / diag.unsqueeze(1) / diag.unsqueeze(0)


def _compute_mediation_penalty(
    adj: torch.Tensor,
    layers: list[int],
    mediation_penalty: float,
) -> torch.Tensor:
    """
    Penalize similarity(i, j) when there exists k with layer_i < layer_k < layer_j
    and i -> k -> j path, which tends to create cycle-prone merges.
    """
    n = adj.shape[0]
    if mediation_penalty >= 1.0:
        return torch.ones((n, n), dtype=adj.dtype, device=adj.device)

    layer_t = torch.tensor(layers, dtype=torch.float32, device=adj.device)
    # Treat any non-zero edge (positive or negative) as a mediating connection.
    a = (adj != 0).float()
    penalty = torch.ones((n, n), dtype=adj.dtype, device=adj.device)

    unique_layers = sorted(set(layers))
    for lk in unique_layers:
        k_mask = (layer_t == float(lk)).float()
        if k_mask.sum() == 0:
            continue

        # has_ik[i,k] and has_kj[k,j]
        has_ik = a * k_mask.unsqueeze(0)
        has_kj = k_mask.unsqueeze(1) * a

        # exists k with i->k and k->j for each (i,j)
        mediated = (has_ik @ has_kj) > 0

        li_lt_lk = layer_t.unsqueeze(1) < float(lk)
        lj_gt_lk = layer_t.unsqueeze(0) > float(lk)
        between = li_lt_lk & lj_gt_lk
        mark = mediated & between
        mark = mark | mark.T
        penalty[mark] = mediation_penalty

    penalty.fill_diagonal_(1.0)
    return penalty


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
    mediation_penalty: float = 0.1,
    similarity_mode: Literal["edge", "node"] = "node",
) -> torch.Tensor:
    """
    Compute node similarity from weighted shared out/in structure.

    - `edge` mode uses edge influence/relevance channels (current behavior).
    - `node` mode applies node influence/relevance pairwise weights.

    Output/input cosine similarities are always clamped to ``[0, 1]`` before
    being combined.
    """
    kept_ids = prune_graph.kept_ids
    attr = prune_graph.attr

    adj = prune_graph.pruned_adj.clone().float().T
    if similarity_mode == "edge":
        rel_sender, inf_sender = _edge_channels_sender_indexed(prune_graph, adj)
        weighted_out = adj * inf_sender
        weighted_in = (adj * rel_sender).T
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
        
    layers = [_parse_layer(attr, n) for n in kept_ids]
    if mediation_penalty < 1.0:
        p = _compute_mediation_penalty(adj=adj, layers=layers, mediation_penalty=mediation_penalty)
        s = (s * p).clamp(0.0, 1.0)

    return s


def _layer_span_nodes(cluster: list[str], attr: dict[str, dict]) -> int:
    lv = [_parse_layer(attr, n) for n in cluster]
    return max(lv) - min(lv)


def _split_cluster_by_span(cluster: list[str], attr: dict[str, dict], max_layer_span: int) -> list[list[str]]:
    work = [cluster]
    out: list[list[str]] = []
    while work:
        current = work.pop()
        span = _layer_span_nodes(current, attr)
        if span <= max_layer_span or len(current) <= 1:
            out.append(current)
            continue
        current_sorted = sorted(current, key=lambda n: _parse_layer(attr, n))
        lo = _parse_layer(attr, current_sorted[0])
        hi = _parse_layer(attr, current_sorted[-1])
        cut = (lo + hi) // 2
        left = [n for n in current_sorted if _parse_layer(attr, n) <= cut]
        right = [n for n in current_sorted if _parse_layer(attr, n) > cut]
        if not left or not right:
            half = max(1, len(current_sorted) // 2)
            left, right = current_sorted[:half], current_sorted[half:]
        work.extend([left, right])
    return out


def _split_cluster_by_boundary(cluster: list[str], boundary_layer: int, attr: dict[str, dict]) -> list[list[str]]:
    left = [n for n in cluster if _parse_layer(attr, n) < boundary_layer]
    right = [n for n in cluster if _parse_layer(attr, n) >= boundary_layer]
    out: list[list[str]] = []
    if left:
        out.append(left)
    if right:
        out.append(right)
    return out


def _resolve_layer_interleaving(clusters: list[list[str]], attr: dict[str, dict]) -> list[list[str]]:
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
            a_layers = [_parse_layer(attr, n) for n in a]
            a_lo, a_hi = min(a_layers), max(a_layers)

            for j in range(i + 1, len(clusters)):
                b = clusters[j]
                b_layers = [_parse_layer(attr, n) for n in b]
                b_lo, b_hi = min(b_layers), max(b_layers)

                # interleaving: a_lo < b_lo < a_hi < b_hi (or symmetric)
                a_wraps_b_boundary = a_lo < b_lo < a_hi < b_hi
                b_wraps_a_boundary = b_lo < a_lo < b_hi < a_hi
                # containment
                a_contains_b = a_lo < b_lo and b_hi < a_hi
                b_contains_a = b_lo < a_lo and a_hi < b_hi

                if a_wraps_b_boundary or a_contains_b:
                    replacement = _split_cluster_by_boundary(a, b_lo, attr)
                    clusters = clusters[:i] + replacement + clusters[i + 1 :]
                    changed = True
                    break

                if b_wraps_a_boundary or b_contains_a:
                    replacement = _split_cluster_by_boundary(b, a_lo, attr)
                    clusters = clusters[:j] + replacement + clusters[j + 1 :]
                    changed = True
                    break

    clusters.sort(key=lambda c: min(_parse_layer(attr, n) for n in c))
    return clusters


def _merge_to_budget(clusters: list[list[str]], attr: dict[str, dict], max_sn: int) -> list[list[str]]:
    """Greedily merge layer-adjacent clusters until budget is met."""
    while len(clusters) > max_sn:
        best_i = -1
        best_gap = float("inf")
        for i in range(len(clusters) - 1):
            hi_i = max(_parse_layer(attr, n) for n in clusters[i])
            lo_j = min(_parse_layer(attr, n) for n in clusters[i + 1])
            gap = abs(lo_j - hi_i)
            if gap < best_gap:
                best_gap = gap
                best_i = i

        if best_i < 0:
            break

        merged = clusters[best_i] + clusters[best_i + 1]
        clusters = clusters[:best_i] + [merged] + clusters[best_i + 2 :]

    return clusters


def _name_middle_supernodes(clusters: list[list[str]], attr: dict[str, dict]) -> dict[str, list[str]]:
    clusters = sorted(clusters, key=lambda c: min(_parse_layer(attr, n) for n in c))
    return {f"SN_{i}": members for i, members in enumerate(clusters)}


def labels_to_supernodes(
    prune_graph: PruneGraph,
    middle_ids: list[str],
    labels: np.ndarray,
) -> list[list[str]]:
    grouped: dict[int, list[str]] = {}
    for node_id, label in zip(middle_ids, labels):
        grouped.setdefault(int(label), []).append(node_id)

    middle_clusters = [grouped[label] for label in sorted(grouped)]
    emb_singletons = [[nid] for nid in prune_graph.kept_ids if _is_embedding(prune_graph.attr, nid)]
    logit_singletons = [[nid] for nid in prune_graph.kept_ids if _is_logit(prune_graph.attr, nid)]
    return middle_clusters + emb_singletons + logit_singletons


def cluster_graph(
    prune_graph: PruneGraph,
    target_k: int = 7,
    max_layer_span: int = 4,
    max_sn: int | None = None,
    mean_method: Literal["geo", "harm", "arith"] = "arith",
    mediation_penalty: float = 0.1,
    similarity_mode: Literal["edge", "node"] = "edge",
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
        mediation_penalty: Penalty factor for mediated non-adjacent pairs.
        similarity_mode: `edge` or `node` similarity construction.
        random_state: Random seed for spectral clustering k-means init.
        n_init: Number of k-means runs for `SpectralClustering(assign_labels="kmeans")`.

    Returns:
        List of supernodes where each supernode is a list of node ids.
        Embedding/logit nodes are returned as singleton supernodes.
    """
    kept_ids = prune_graph.kept_ids
    attr = prune_graph.attr

    if not kept_ids:
        return []

    sim = compute_similarity(
        prune_graph,
        mean_method=mean_method,
        mediation_penalty=mediation_penalty,
        similarity_mode=similarity_mode,
    )

    middle_idx = [i for i, nid in enumerate(kept_ids) if not _is_fixed(attr, nid)]
    middle_ids = [kept_ids[i] for i in middle_idx]

    if not middle_ids:
        fixed_only = [[nid] for nid in kept_ids]
        return fixed_only

    mid_sim = sim[middle_idx][:, middle_idx].detach().cpu().numpy().clip(0.0, 1.0)
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
            span_safe.extend(_split_cluster_by_span(cluster, attr, max_layer_span=max_layer_span))
        middle_clusters = span_safe

    if enforce_dag:
        middle_clusters = _resolve_layer_interleaving(middle_clusters, attr)

    if max_sn is not None and enforce_dag:
        middle_clusters = _merge_to_budget(middle_clusters, attr, max_sn=max_sn)

    # Keep deterministic naming order for middle SNs, but return member lists only.
    named_middle = _name_middle_supernodes(middle_clusters, attr)

    emb_singletons = [[nid] for nid in kept_ids if _is_embedding(attr, nid)]
    logit_singletons = [[nid] for nid in kept_ids if _is_logit(attr, nid)]

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
    Build a clustered supernode graph from a pruned node-level graph.

    Returns sn-level adjacency and influence metrics that downstream consumers
    can use for scoring, reporting, and visualization.
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