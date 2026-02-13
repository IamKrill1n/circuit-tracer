import math
import torch
from typing import Any, Dict, List, Tuple, Optional

from circuit_tracer.subgraph.utils import get_data_from_json
import networkx as nx  # type: ignore


# ---------------------------------------------------------------------------
# Structural cleanup
# ---------------------------------------------------------------------------

def fix(G: nx.DiGraph, attr: Dict[str, Any]) -> nx.DiGraph:
    """Keep only valid boundary nodes: sources=embedding, sinks=logit."""
    while True:
        rm = set()
        for n in list(G.nodes):
            t = attr.get(n, {}).get("feature_type")
            if G.in_degree(n) == 0 and t != "embedding":
                rm.add(n)
            if G.out_degree(n) == 0 and t != "logit":
                rm.add(n)
        if not rm:
            break
        G.remove_nodes_from(rm)
    return G

def remove_error_nodes(
    adj: torch.Tensor,
    node_ids: List[str],
    attr: Dict[str, Any],
) -> Tuple[torch.Tensor, List[str], Dict[str, Any]]:
    
    keep_idx: List[int] = []
    for i, nid in enumerate(node_ids):
        a = attr.get(nid, {})
        if a.get("feature_type") == "mlp reconstruction error":
            continue
        keep_idx.append(i)

    idx_t = torch.tensor(keep_idx, dtype=torch.long)
    adj = adj[idx_t][:, idx_t]
    node_ids = [node_ids[i] for i in keep_idx]
    attr = {nid: attr[nid] for nid in node_ids if nid in attr}
    return adj, node_ids, attr


# ---------------------------------------------------------------------------
# 2. Normalize adjacency matrix
# ---------------------------------------------------------------------------

def normalize_graph(
    adj: torch.Tensor,
    preprocess: str = "abs",
    norm_method: str = "sum",
    max_edges: int = 100,
) -> torch.Tensor:
    """
    Preprocess + row-normalize + optional top-k sparsification.
    Used only for relevance scoring; original adj is kept for final graph.
    """
    adj = adj.clone()

    # preprocess
    if preprocess == "abs":
        adj = adj.abs()
    elif preprocess == "relu":
        adj = torch.relu(adj)
    elif preprocess == "none":
        pass
    else:
        raise ValueError(f"Unknown preprocess={preprocess}")

    # normalize
    if norm_method == "sum":
        adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-10)
    elif norm_method == "none":
        pass
    else:
        raise ValueError(f"Unknown norm_method={norm_method}")

    # limit to max_edges per row
    if max_edges is not None and max_edges > 0 and max_edges < adj.size(1):
        _, topk_idx = torch.topk(adj, k=max_edges, dim=1)
        m = torch.zeros_like(adj, dtype=torch.bool)
        m.scatter_(1, topk_idx, True)
        adj = adj.masked_fill(~m, 0.0)
        if norm_method == "sum":
            adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-10)

    return adj


# ---------------------------------------------------------------------------
# 3. Build sparse nx.DiGraph from adjacency
# ---------------------------------------------------------------------------

def graph_from_adj(
    adj: torch.Tensor,
    node_ids: List[str],
) -> nx.DiGraph:
    """
    Build a DiGraph from adj where adj[target, source] = weight.
    Edge direction: source -> target.
    Only adds edges with weight > 0.
    """
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    n = len(node_ids)
    rows, cols = (adj > 0).nonzero(as_tuple=True)
    for r, c in zip(rows.tolist(), cols.tolist()):
        G.add_edge(node_ids[c], node_ids[r], weight=float(adj[r, c].item()))
    return G


# ---------------------------------------------------------------------------
# 4. Compute relevance in log-space
# ---------------------------------------------------------------------------

def _logsumexp(vals: List[float]) -> float:
    if not vals:
        return float("-inf")
    m = max(vals)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


def compute_relevance(
    G: nx.DiGraph,
    token_weights: Dict[str, float],
    logit_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute per-node relevance R[v] = F[v] * B[v]

    F: accumulated weight of all source->v paths
    B: accumulated weight of all v->target paths

    Args:
        G: normalized DiGraph (edges carry normalized weights)
        token_weights: initial weights for tokens nodes
        logit_weights: initial weights for logits nodes

    Returns:
        R: dict  node_id -> relevance score (in normal space)
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph is not a DAG")
    
    order = list(nx.topological_sort(G))

    # Initialize F and B with token/logit weights

    F = token_weights.copy()
    B = logit_weights.copy()

    # -- Forward pass --
    for v in order:
        for u in G.predecessors(v):
            w = float(G[u][v].get("weight", 0.0))
            F[v] += w * F.get(u, 0.0)

    # -- Backward pass --
    for v in reversed(order):
        for u in G.successors(v):
            w = float(G[v][u].get("weight", 0.0))
            B[v] += w * B.get(u, 0.0)

    # -- Relevance --
    R: Dict[str, float] = {}
    for v in G.nodes():
        f = F.get(v, 0.0)
        b = B.get(v, 0.0)
        R[v] = f * b
    
    return R


# ---------------------------------------------------------------------------
# 5. Find relevance threshold that keeps x% of total relevance
# ---------------------------------------------------------------------------

def find_threshold(
    R: Dict[str, float],
    keep_frac: float = 0.8,
) -> float:
    """
    Return the minimum relevance value such that keeping all nodes with
    relevance >= that value accounts for at least keep_frac of total relevance.

    Args:
        R: node_id -> relevance
        keep_frac: fraction of total relevance to keep (e.g. 0.8 = 80%)

    Returns:
        cutoff: relevance threshold value
    """
    items = sorted(R.values(), reverse=True)
    total = sum(items)
    if total <= 0:
        return 0.0
    cumsum = 0.0
    cutoff = 0.0
    for v in items:
        cumsum += v
        cutoff = v
        if cumsum / total >= keep_frac:
            break
    return cutoff


# ---------------------------------------------------------------------------
# 6. Prune graph  (returns nx.DiGraph with ORIGINAL edge weights)
# ---------------------------------------------------------------------------

def prune_graph(
    G_original: nx.DiGraph,
    attr: Dict[str, Any],
    R: Dict[str, float],
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Prune nodes/edges by relevance, return graph with original weights.

    Args:
        G_original: graph built from the raw (un-normalized) adjacency
        attr: node attributes
        R: per-node relevance scores (from compute_relevance on normalized graph)
        node_threshold: keep nodes that cover this fraction of total relevance
        edge_threshold: keep edges that cover this fraction of total edge relevance

    Returns:
        Pruned nx.DiGraph with original edge weights, and filtered attr
    """
    # -- Node pruning --
    cutoff = find_threshold(R, keep_frac=node_threshold)
    keep_nodes = {n for n, r in R.items() if r >= cutoff}

    # always keep boundary nodes
    for n in G_original.nodes():
        t = attr.get(n, {}).get("feature_type", "")
        if t in ("embedding", "logit") or attr.get(n, {}).get("is_target_logit", False):
            keep_nodes.add(n)

    H = G_original.subgraph(keep_nodes).copy()

    # -- Edge pruning by edge relevance = R_src * |w| * R_tgt --
    edge_scores: List[Tuple[Tuple[str, str], float]] = []
    for u, v, d in H.edges(data=True):
        w = abs(float(d.get("weight", 0.0)))
        score = R.get(u, 0.0) * w * R.get(v, 0.0)
        edge_scores.append(((u, v), score))

    if edge_scores:
        edge_scores.sort(key=lambda x: x[1], reverse=True)
        total_e = sum(s for _, s in edge_scores)
        if total_e > 0:
            cumsum = 0.0
            keep_edges = set()
            for edge, s in edge_scores:
                keep_edges.add(edge)
                cumsum += s
                if cumsum / total_e >= edge_threshold:
                    break
            drop = [e for e, _ in edge_scores if e not in keep_edges]
            H.remove_edges_from(drop)

    # structural cleanup
    fix(H, attr)
    out_attr = {n: attr[n] for n in H.nodes() if n in attr}
    return H, out_attr


# ---------------------------------------------------------------------------
# 7. Full pipeline
# ---------------------------------------------------------------------------

def prune_graph_pipeline(
    json_path: str,
    token_weights:  
    logit_weights:
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    preprocess: str = "relu",
    norm_method: str = "sum",
    max_edges: int = 100,
) -> Tuple[nx.DiGraph, Dict[str, Any], Dict[str, float]]:
    """
    Full pipeline:
      1. get_data_from_json   -> raw adj, node_ids, attr, metadata
      2. mask_tokens           -> remove unwanted embedding nodes
      3. normalize_graph       -> normalized adj  (for scoring only)
      4. graph_from_adj        -> nx.DiGraph from normalized adj
      5. compute_relevance     -> per-node relevance R
      6. graph_from_adj        -> nx.DiGraph from ORIGINAL adj
      7. prune_graph           -> pruned graph with original weights

    Args:
        json_path: path to the graph JSON file
        mask: boolean list per token position; None keeps all tokens
        node_threshold: fraction of total relevance to keep (e.g. 0.8)
        edge_threshold: fraction of total edge relevance to keep
        preprocess: 'relu', 'abs', or 'none'
        norm_method: 'sum' or 'none'
        max_edges: top-k incoming edges per node for normalization

    Returns:
        G_pruned: pruned nx.DiGraph with original edge weights
        attr: pruned node attributes
        R: full relevance dict (before pruning)
    """
    # 1. Load
    adj, node_ids, attr, metadata = get_data_from_json(json_path)
    print(f"Loaded graph: {len(node_ids)} nodes, {int((adj != 0).sum())} edges")

    # 2. Mask tokens
    if mask is not None:
        adj, node_ids, attr = mask_tokens(adj, node_ids, attr, mask)
        print(f"After masking: {len(node_ids)} nodes, {int((adj != 0).sum())} edges")

    # 3. Normalize (used only for relevance scoring)
    adj_norm = normalize_graph(adj, preprocess=preprocess, norm_method=norm_method, max_edges=max_edges)

    # 4. Build normalized graph -> compute relevance
    G_norm = graph_from_adj(adj_norm, node_ids)
    fix(G_norm, attr)
    R = compute_relevance(G_norm, attr)

    cutoff = find_threshold(R, keep_frac=node_threshold)
    n_kept = sum(1 for r in R.values() if r >= cutoff)
    print(f"Relevance cutoff={cutoff:.6g}, keeping {n_kept}/{len(R)} nodes at {node_threshold:.0%}")

    # 5. Build graph from ORIGINAL adjacency (only positive edges)
    adj_orig = torch.relu(adj)  # keep sign convention consistent
    G_original = graph_from_adj(adj_orig, node_ids)
    fix(G_original, attr)

    # 6. Prune using relevance scores, keep original weights
    G_pruned, attr_pruned = prune_graph(
        G_original, attr, R,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    print(f"Pruned graph: {G_pruned.number_of_nodes()} nodes, {G_pruned.number_of_edges()} edges")
    return G_pruned, attr_pruned, R

if __name__ == "__main__":
    mask = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]

    G, attr, R = prune_graph_pipeline(
        json_path="demos/graph_files/puppy-clt.json",
        mask=mask,
        node_threshold=0.7,
        edge_threshold=0.8,
        preprocess="abs",
        norm_method="sum",
        max_edges=100,
    )

    
