from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from circuit_tracer.subgraph.prune import PruneGraph


def _default_candidates(pg: PruneGraph) -> List[str]:
    out = []
    for nid in pg.kept_ids:
        t = pg.attr.get(nid, {}).get("feature_type", "")
        if t == "cross layer transcoder":
            out.append(nid)
    return out


def _edge_signature(adj: torch.Tensor, idx: int, cand_idx: torch.Tensor) -> torch.Tensor:
    out_vec = adj[idx, cand_idx]
    in_vec = adj[cand_idx, idx]
    sig = torch.cat([out_vec, in_vec], dim=0)
    return sig


def _build_similarity_matrix(adj_sub: torch.Tensor) -> torch.Tensor:
    x = F.normalize(adj_sub, p=2, dim=1, eps=1e-12)
    sim = x @ x.T
    sim.fill_diagonal_(1.0)
    return sim


class _DSU:
    def __init__(self, n: int, layers: List[Optional[int]]):
        self.parent = list(range(n))
        self.size = [1] * n
        self.min_layer = [l if l is not None else 10**9 for l in layers]
        self.max_layer = [l if l is not None else -(10**9) for l in layers]
        self.has_known_layer = [l is not None for l in layers]

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def can_union(self, a: int, b: int, max_diff: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        has_layer = self.has_known_layer[ra] and self.has_known_layer[rb]
        if not has_layer:
            return True
        new_min = min(self.min_layer[ra], self.min_layer[rb])
        new_max = max(self.max_layer[ra], self.max_layer[rb])
        return (new_max - new_min) <= max_diff

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.min_layer[ra] = min(self.min_layer[ra], self.min_layer[rb])
        self.max_layer[ra] = max(self.max_layer[ra], self.max_layer[rb])
        self.has_known_layer[ra] = self.has_known_layer[ra] or self.has_known_layer[rb]
        return True


def _cluster_label(cluster_nodes: List[str], attr: Dict[str, Any], i: int) -> str:
    clerps = [attr.get(n, {}).get("clerp", "").strip() for n in cluster_nodes]
    clerps = [c for c in clerps if c]
    if not clerps:
        return f"cluster_{i+1}"
    first = sorted(clerps, key=len)[0]
    return first[:80]


def cluster_supernodes_by_edge_similarity(
    pruned_graph: PruneGraph,
    candidate_node_ids: Optional[List[str]] = None,
    similarity_threshold: float = 0.85,
    max_layer_diff: Optional[int] = None,
    include_singletons: bool = False,
) -> List[List[str]]:
    """
    Cluster nodes by edge-weight similarity with iterative layer-diff sweep.

    Returns:
        supernodes in Neuronpedia format:
        [
          ["label", "node_id_a", "node_id_b", ...],
          ...
        ]
    """
    if candidate_node_ids is None:
        candidate_node_ids = _default_candidates(pruned_graph)

    if not candidate_node_ids:
        return []

    id_to_idx = {nid: i for i, nid in enumerate(pruned_graph.kept_ids)}
    valid_nodes = [nid for nid in candidate_node_ids if nid in id_to_idx]
    if len(valid_nodes) <= 1:
        if include_singletons and valid_nodes:
            return [[_cluster_label(valid_nodes, pruned_graph.attr, 0), valid_nodes[0]]]
        return []

    cand_idx = torch.tensor([id_to_idx[n] for n in valid_nodes], dtype=torch.long, device=pruned_graph.pruned_adj.device)
    m = len(valid_nodes)

    # Build per-node edge signatures against candidate set: [out_edges || in_edges]
    sigs = []
    for i in range(m):
        sig = _edge_signature(pruned_graph.pruned_adj, int(cand_idx[i].item()), cand_idx)
        sigs.append(sig)
    sig_mat = torch.stack(sigs, dim=0)  # [m, 2m]

    sim = _build_similarity_matrix(sig_mat).detach().cpu()

    layers = [int(pruned_graph.attr[nid].get('layer')) for nid in valid_nodes]
    known_layers = [l for l in layers if l is not None]
    if max_layer_diff is None:
        max_layer_diff = max(0, (max(known_layers) - min(known_layers))) if known_layers else 0

    # Candidate pairs sorted by similarity desc
    pairs: List[Tuple[float, int, int]] = []
    for i in range(m):
        for j in range(i + 1, m):
            s = float(sim[i, j].item())
            if s >= similarity_threshold:
                pairs.append((s, i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)

    dsu = _DSU(m, layers)

    # Sweep allowed layer difference: 0..max_layer_diff
    for diff in range(max_layer_diff + 1):
        changed = True
        while changed:
            changed = False
            for s, i, j in pairs:
                if dsu.can_union(i, j, max_diff=diff):
                    if dsu.union(i, j):
                        changed = True

    groups: Dict[int, List[str]] = defaultdict(list)
    for i, nid in enumerate(valid_nodes):
        groups[dsu.find(i)].append(nid)

    clusters = list(groups.values())
    clusters.sort(key=lambda g: (-len(g), g[0]))

    supernodes: List[List[str]] = []
    for i, g in enumerate(clusters):
        if not include_singletons and len(g) == 1:
            continue
        label = _cluster_label(g, pruned_graph.attr, i)
        supernodes.append([label, *g])

    return supernodes

if __name__ == "__main__":
    prune_graph = torch.load("demos/subgraph/austin_clt.pt")
    print(type(prune_graph))
    prune_graph = PruneGraph(
        kept_ids=prune_graph["kept_ids"],
        pruned_adj=prune_graph["pruned_adj"],
        node_influence=prune_graph["node_inf"],
        node_relevance=prune_graph["node_rel"],
        attr=prune_graph["attr"],
        metadata=prune_graph["metadata"],
    )
    # print(prune_graph)
    supernodes = cluster_supernodes_by_edge_similarity(
        pruned_graph=prune_graph, # your result from prune_graph_pipeline
        similarity_threshold=0.82,
        max_layer_diff=3,
        include_singletons=False
    )
    print(supernodes)