import math
import torch
from typing import Any, Dict, List, Tuple, Optional
from circuit_tracer.graph import compute_node_influence, compute_edge_influence, compute_node_relevance, compute_edge_relevance, find_threshold
from circuit_tracer.subgraph.utils import get_data_from_json
import networkx as nx  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_type(attr: Dict[str, Any], node: str) -> str:
    return attr.get(node, {}).get("feature_type", "")


def _is_target_logit(attr: Dict[str, Any], node: str) -> bool:
    return attr.get(node, {}).get("is_target_logit", False)


def _is_feature(attr: Dict[str, Any], node: str) -> bool:
    t = _node_type(attr, node)
    return t not in ("embedding", "logit", "mlp reconstruction error", "")


def _is_error(attr: Dict[str, Any], node: str) -> bool:
    return _node_type(attr, node) == "mlp reconstruction error"


def _is_embedding(attr: Dict[str, Any], node: str) -> bool:
    return _node_type(attr, node) == "embedding"


def _is_logit(attr: Dict[str, Any], node: str) -> bool:
    return _node_type(attr, node) == "logit"


# ---------------------------------------------------------------------------
# Structural cleanup
# ---------------------------------------------------------------------------

def fix(G: nx.DiGraph, attr: Dict[str, Any]) -> nx.DiGraph:
    """
    Iteratively remove invalid nodes:
      - sources must be embedding nodes
      - only the target_logit node is a valid sink
    """
    target_logit = None
    for n in list(G.nodes):
        if _is_target_logit(attr, n):
            target_logit = n
            break

    while True:
        rm = set()
        for n in list(G.nodes):
            if G.in_degree(n) == 0 and not _is_embedding(attr, n):
                rm.add(n)
            if G.out_degree(n) == 0 and n != target_logit:
                rm.add(n)
        if not rm:
            break
        G.remove_nodes_from(rm)
    return G


# ---------------------------------------------------------------------------
# 1. Mask token nodes
# ---------------------------------------------------------------------------

# def mask_tokens(
#     adj: torch.Tensor,
#     node_ids: List[str],
#     attr: Dict[str, Any],
#     mask: List[bool],
# ) -> Tuple[torch.Tensor, List[str], Dict[str, Any]]:
#     """
#     Remove embedding nodes whose token position is masked out.

#     Args:
#         adj:      (n, n) adjacency, adj[target, source] = weight
#         node_ids: node id strings aligned with adj rows/cols
#         attr:     per-node attribute dicts
#         mask:     mask[i]=True => keep embedding at position i

#     Returns:
#         adj, node_ids, attr with masked rows/cols removed
#     """
#     keep_idx: List[int] = []
#     for i, nid in enumerate(node_ids):
#         if _is_embedding(attr, nid):
#             ctx = attr.get(nid, {}).get("ctx_idx", 0)
#             if ctx < len(mask) and not mask[ctx]:
#                 continue
#         keep_idx.append(i)

#     idx_t = torch.tensor(keep_idx, dtype=torch.long)
#     adj = adj[idx_t][:, idx_t]
#     node_ids = [node_ids[i] for i in keep_idx]
#     attr = {nid: attr[nid] for nid in node_ids if nid in attr}
#     return adj, node_ids, attr

def _build_index_sets(
    node_ids: List[str],
    attr: Dict[str, Any],
) -> Dict[str, List[int]]:
    """Return index sets for each node type."""
    sets: Dict[str, List[int]] = {
        "feature": [],
        "error": [],
        "embedding": [],
        "logit": [],
        "target_logit": [],
    }
    for i, nid in enumerate(node_ids):
        if _is_target_logit(attr, nid):
            sets["target_logit"].append(i)
        if _is_logit(attr, nid):
            sets["logit"].append(i)
        elif _is_embedding(attr, nid):
            sets["embedding"].append(i)
        elif _is_error(attr, nid):
            sets["error"].append(i)
        elif _is_feature(attr, nid):
            sets["feature"].append(i)
    return sets

def prune_by_influence(
    adj: torch.Tensor,
    node_ids: List[str],
    attr: Dict[str, Any],
    logit_weights: str,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    keep_all_tokens_and_logits: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prune nodes/edges by backward influence (logit -> input).

    Args:
        adj:              raw adjacency matrix (n, n)
        node_ids:         node id list
        attr:             node attribute dict
        logit_weights:    seed for influence; "probs" => use logit probs from attr;
        node_threshold:   cumulative fraction of influence to keep
        edge_threshold:   cumulative fraction of edge influence to keep
        keep_all_tokens_and_logits: if True, keep all embeddings and all logits;
                                     if False, keep all embeddings and only target logit

    Returns:
        node_mask, edge_mask  (both boolean tensors)
    """
    n = adj.shape[0]
    idx = _build_index_sets(node_ids, attr)

    if logit_weights == "probs":
        # default with logit probs
        logit_weights = torch.zeros(n, device=adj.device)
        for i in idx["logit"]:
            nid = node_ids[i]
            prob = attr.get(nid, {}).get("token_prob", 0.0)
            logit_weights[i] = prob
    elif logit_weights == 'target':
        logit_weights = torch.zeros(n, device=adj.device)
        for i in idx["target_logit"]:
            logit_weights[i] = 1.0

    # node influence
    node_inf = compute_node_influence(adj, logit_weights)
    node_mask = node_inf >= find_threshold(node_inf, node_threshold)

    # always keep embeddings
    for i in idx["embedding"]:
        node_mask[i] = True

    if keep_all_tokens_and_logits:
        # keep all logits
        for i in idx["logit"]:
            node_mask[i] = True
    else:
        # keep only target logit
        for i in idx["target_logit"]:
            node_mask[i] = True

    # prune matrix
    pruned = adj.clone()
    pruned[~node_mask] = 0
    pruned[:, ~node_mask] = 0

    # edge influence
    edge_scores = compute_edge_influence(pruned, logit_weights)
    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)

    # iterative structural cleanup
    feature_idx = torch.tensor(idx["feature"], dtype=torch.long, device=adj.device)
    non_boundary = torch.tensor(idx["feature"] + idx["error"], dtype=torch.long, device=adj.device)

    old = node_mask.clone()
    # non-boundary must have outgoing edges
    if len(non_boundary):
        node_mask[non_boundary] &= edge_mask[:, non_boundary].any(0)
    # features must have incoming edges
    if len(feature_idx):
        node_mask[feature_idx] &= edge_mask[feature_idx].any(1)

    while not torch.all(node_mask == old):
        old[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        if len(non_boundary):
            node_mask[non_boundary] &= edge_mask[:, non_boundary].any(0)
        if len(feature_idx):
            node_mask[feature_idx] &= edge_mask[feature_idx].any(1)

    return node_mask, edge_mask

def prune_by_relevance(
    adj: torch.Tensor,
    node_ids: List[str],
    attr: Dict[str, Any],
    token_weights: Optional[List[float]] = None,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    keep_all_tokens_and_logits: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prune nodes/edges by forward relevance (input -> logit).

    Args:
        adj:              raw adjacency matrix (n, n)
        node_ids:         node id list
        attr:             node attribute dict
        token_weights:    relevance init weights for each token
        node_threshold:   cumulative fraction of relevance to keep
        edge_threshold:   cumulative fraction of edge relevance to keep
        keep_all_tokens_and_logits: if True, keep all embeddings and all logits;
                                     if False, keep all embeddings and only target logit

    Returns:
        node_mask, edge_mask  (both boolean tensors)
    """
    n = adj.shape[0]
    idx = _build_index_sets(node_ids, attr)

    if token_weights is None:
        emb_weights = torch.zeros(n, device=adj.device)
        n_emb = len(idx["embedding"])
        for i in idx["embedding"]:
            emb_weights[i] = 1.0 / max(n_emb, 1)
    else:
        emb_weights = torch.zeros(n, device=adj.device)
        for i, id in enumerate(idx["embedding"]):
            emb_weights[id] = token_weights[i]
    # node relevance
    node_rel = compute_node_relevance(adj, emb_weights)
    node_mask = node_rel >= find_threshold(node_rel, node_threshold)

    if keep_all_tokens_and_logits:
        # keep all logits
        for i in idx["logit"]:
            node_mask[i] = True
    else:
        # keep only target logit
        for i in idx["target_logit"]:
            node_mask[i] = True

    # prune matrix
    pruned = adj.clone()
    pruned[~node_mask] = 0
    pruned[:, ~node_mask] = 0

    # edge relevance
    edge_scores = compute_edge_relevance(pruned, emb_weights)
    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)

    # iterative structural cleanup
    feature_idx = torch.tensor(idx["feature"], dtype=torch.long, device=adj.device)
    non_boundary = torch.tensor(idx["feature"] + idx["error"], dtype=torch.long, device=adj.device)

    old = node_mask.clone()
    if len(non_boundary):
        node_mask[non_boundary] &= edge_mask[:, non_boundary].any(0)
    if len(feature_idx):
        node_mask[feature_idx] &= edge_mask[feature_idx].any(1)

    while not torch.all(node_mask == old):
        old[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        if len(non_boundary):
            node_mask[non_boundary] &= edge_mask[:, non_boundary].any(0)
        if len(feature_idx):
            node_mask[feature_idx] &= edge_mask[feature_idx].any(1)

    return node_mask, edge_mask

def prune_combined(
    adj: torch.Tensor,
    node_ids: List[str],
    attr: Dict[str, Any],
    logit_weights: str,
    token_weights: Optional[List[float]] = None,
    node_influence_threshold: float = 0.8,
    edge_influence_threshold: float = 0.98,
    node_relevance_threshold: float = 0.8,
    edge_relevance_threshold: float = 0.98,
    keep_all_tokens_and_logits: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-pass pruning on the raw adjacency matrix:
      1. influence  (backward: logit -> sources)
      2. relevance  (forward:  sources -> logit)
      3. AND the two masks
      4. iterative structural cleanup

    Args:
        keep_all_tokens_and_logits: if True, keep all embeddings and all logits;
                                     if False, keep all embeddings and only target logit

    Returns:
        node_mask, edge_mask  (boolean tensors on the original adj indexing)
    """
    # Pass 1 – influence
    inf_node, inf_edge = prune_by_influence(
        adj, node_ids, attr,
        logit_weights=logit_weights,
        node_threshold=node_influence_threshold,
        edge_threshold=edge_influence_threshold,
        keep_all_tokens_and_logits=keep_all_tokens_and_logits,
    )

    # Pass 2 – relevance
    rel_node, rel_edge = prune_by_relevance(
        adj, node_ids, attr,
        token_weights=token_weights,
        node_threshold=node_relevance_threshold,
        edge_threshold=edge_relevance_threshold,
        keep_all_tokens_and_logits=keep_all_tokens_and_logits,
    )

    # AND combine
    node_mask = inf_node & rel_node
    edge_mask = inf_edge & rel_edge

    # iterative cleanup on merged result
    idx = _build_index_sets(node_ids, attr)
    feature_idx = torch.tensor(idx["feature"], dtype=torch.long, device=adj.device)
    non_boundary = torch.tensor(idx["feature"] + idx["error"], dtype=torch.long, device=adj.device)

    old = node_mask.clone()
    if len(non_boundary):
        node_mask[non_boundary] &= edge_mask[:, non_boundary].any(0)
    if len(feature_idx):
        node_mask[feature_idx] &= edge_mask[feature_idx].any(1)

    while not torch.all(node_mask == old):
        old[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        if len(non_boundary):
            node_mask[non_boundary] &= edge_mask[:, non_boundary].any(0)
        if len(feature_idx):
            node_mask[feature_idx] &= edge_mask[feature_idx].any(1)

    return node_mask, edge_mask

def masks_to_digraph(
    adj: torch.Tensor,
    node_ids: List[str],
    attr: Dict[str, Any],
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Build a sparse nx.DiGraph from the original adjacency and boolean masks.
    Edges carry the **original** (un-normalized) weights.
    """
    G = nx.DiGraph()
    kept = [i for i in range(len(node_ids)) if node_mask[i]]
    for i in kept:
        G.add_node(node_ids[i])

    for i in kept:
        for j in kept:
            if edge_mask[i, j]:
                w = float(adj[i, j].item())
                if w != 0.0:
                    # adj[target=i, source=j] -> edge j -> i
                    G.add_edge(node_ids[j], node_ids[i], weight=w)

    out_attr = {node_ids[i]: attr[node_ids[i]] for i in kept if node_ids[i] in attr}
    return G, out_attr

def prune_graph_pipeline(
    json_path: str,
    logit_weights: str,
    token_weights: Optional[List] = None,
    node_influence_threshold: float = 0.8,
    edge_influence_threshold: float = 0.98,
    node_relevance_threshold: float = 0.8,
    edge_relevance_threshold: float = 0.98,
    keep_all_tokens_and_logits: bool = True,
) -> Tuple[nx.DiGraph, Dict[str, Any], Dict[str, Any]]:
    """
    Args:
        json_path:                   path to the graph JSON
        logit_weights:               (n,) seed for influence; None => logit probs;
                                     'target' => target_logit=1
        token_weights:               (n,) seed for relevance; None => uniform embeddings
        node_influence_threshold:    cumulative influence fraction to keep
        edge_influence_threshold:    cumulative edge-influence fraction to keep
        node_relevance_threshold:    cumulative relevance fraction to keep
        edge_relevance_threshold:    cumulative edge-relevance fraction to keep
        keep_all_tokens_and_logits:  if True, keep all embeddings and all logits;
                                     if False, keep all embeddings and only target logit

    Returns:
        G:        pruned nx.DiGraph with original edge weights
        attr:     filtered node attributes
        metadata: original JSON metadata
    """
    # 1. Load
    adj, node_ids, attr, metadata = get_data_from_json(json_path)
    print(f"[1] Loaded: {len(node_ids)} nodes, {int((adj != 0).sum())} edges")

    # 3+4+5. Combined influence + relevance pruning
    node_mask, edge_mask = prune_combined(
        adj, node_ids, attr,
        logit_weights=logit_weights,
        token_weights=token_weights,
        node_influence_threshold=node_influence_threshold,
        edge_influence_threshold=edge_influence_threshold,
        node_relevance_threshold=node_relevance_threshold,
        edge_relevance_threshold=edge_relevance_threshold,
        keep_all_tokens_and_logits=keep_all_tokens_and_logits,
    )

    n_nodes_kept = int(node_mask.sum().item())
    n_edges_kept = int(edge_mask.sum().item())
    print(f"[3] After influence+relevance: {n_nodes_kept}/{len(node_ids)} nodes, "
          f"{n_edges_kept}/{int((adj != 0).sum())} edges")

    # 6. Build nx.DiGraph with original weights
    G, out_attr = masks_to_digraph(adj, node_ids, attr, node_mask, edge_mask)
    # fix(G, out_attr)
    print(f"[4] Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G, out_attr, metadata

if __name__ == "__main__":

    G, attr, metadata = prune_graph_pipeline(
        json_path="demos/temp_graph_files/austin.json",
        logit_weights='target',
        token_weights=[0, 0, 0, 0, 1/3, 0, 0, 1/3, 0, 1/3, 0],
        node_influence_threshold=0.5,
        edge_influence_threshold=0.7,
        node_relevance_threshold=0.5,
        edge_relevance_threshold=0.8,
        keep_all_tokens_and_logits=False,
    )

    print("Subgraph has {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))
    for n in G.nodes(data=True):
        print(n)
