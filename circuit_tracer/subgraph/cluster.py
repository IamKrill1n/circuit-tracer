from __future__ import annotations

from collections import defaultdict

import torch

from circuit_tracer.subgraph.prune import PruneGraph
from circuit_tracer.subgraph.utils import _is_embedding, _is_fixed, _is_logit, _parse_layer


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
    a = (adj > 0).float()
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
        penalty[mark] = mediation_penalty

    penalty.fill_diagonal_(1.0)
    return penalty


def compute_similarity(
    prune_graph: PruneGraph,
    alpha: float = 0.5,
    beta: float = 0.5,
    mediation_penalty: float = 0.1,
) -> torch.Tensor:
    """Compute node similarity from shared in/out neighbors, weighted by act/inf."""
    kept_ids = prune_graph.kept_ids
    attr = prune_graph.attr

    # Match structure_grouping convention: sender-indexed adjacency.
    adj = prune_graph.pruned_adj.clone().float().T

    act_t = torch.tensor(
        [float(attr.get(n, {}).get("activation", 0.0) or 0.0) for n in kept_ids],
        dtype=torch.float32,
        device=adj.device,
    )
    inf_t = torch.tensor(
        [float(attr.get(n, {}).get("influence", 0.0) or 0.0) for n in kept_ids],
        dtype=torch.float32,
        device=adj.device,
    )

    act_norm = act_t / (act_t.max() + 1e-8)
    inf_norm = inf_t / (inf_t.max() + 1e-8)
    w = torch.diag(alpha * act_norm + beta * inf_norm)

    s_out_cos = _cosine_norm(adj @ w @ adj.T)
    s_in_cos = _cosine_norm(adj.T @ w @ adj)
    s = (0.5 * s_out_cos + 0.5 * s_in_cos).clamp(0.0, 1.0)

    layers = [_parse_layer(attr, n) for n in kept_ids]
    if mediation_penalty < 1.0:
        p = _compute_mediation_penalty(adj=adj, layers=layers, mediation_penalty=mediation_penalty)
        s = (s * p).clamp(0.0, 1.0)

    return s


def _cluster_similarity(cl_a: list[int], cl_b: list[int], sim: torch.Tensor) -> float:
    vals = sim[cl_a][:, cl_b]
    return float(vals.mean().item()) if vals.numel() else -1.0


def _layer_span(cluster_idx: list[int], layers: list[int]) -> int:
    lv = [layers[i] for i in cluster_idx]
    return max(lv) - min(lv)


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


def cluster_graph(
    prune_graph: PruneGraph,
    target_k: int = 7,
    max_layer_span: int = 4,
    max_sn: int | None = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    mediation_penalty: float = 0.1,
) -> list[list[str]]:
    """
    Cluster a pruned attribution graph into supernodes.

    Args:
        prune_graph: Output of `prune_graph_pipeline`.
        target_k: Target number of middle supernodes.
        max_layer_span: Maximum allowed layer span within a middle supernode.
        max_sn: Optional hard cap on number of middle supernodes.
        alpha: Similarity weighting coefficient for activation.
        beta: Similarity weighting coefficient for influence.
        mediation_penalty: Penalty factor for mediated non-adjacent pairs.

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
        alpha=alpha,
        beta=beta,
        mediation_penalty=mediation_penalty,
    )

    middle_idx = [i for i, nid in enumerate(kept_ids) if not _is_fixed(attr, nid)]
    middle_ids = [kept_ids[i] for i in middle_idx]
    layers = [_parse_layer(attr, n) for n in middle_ids]

    if not middle_ids:
        fixed_only = [[nid] for nid in kept_ids]
        return fixed_only

    # Start with singleton clusters in middle-node local index space.
    clusters: list[list[int]] = [[i] for i in range(len(middle_ids))]
    target_k = max(1, min(target_k, len(clusters)))

    # Greedy agglomerative merge under layer-span constraint.
    while len(clusters) > target_k:
        best: tuple[float, int, int] | None = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                merged_idx = clusters[i] + clusters[j]
                if _layer_span(merged_idx, layers) > max_layer_span:
                    continue

                sim_score = _cluster_similarity(
                    [middle_idx[x] for x in clusters[i]],
                    [middle_idx[x] for x in clusters[j]],
                    sim,
                )

                if best is None or sim_score > best[0]:
                    best = (sim_score, i, j)

        if best is None:
            # No further legal merge under constraints.
            break

        _, i_best, j_best = best
        merged = clusters[i_best] + clusters[j_best]
        clusters = [c for k, c in enumerate(clusters) if k not in (i_best, j_best)] + [merged]

    middle_clusters = [[middle_ids[i] for i in c] for c in clusters]
    middle_clusters = _resolve_layer_interleaving(middle_clusters, attr)

    if max_sn is not None:
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