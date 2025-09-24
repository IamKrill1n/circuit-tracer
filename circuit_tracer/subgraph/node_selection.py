import torch

from circuit_tracer.graph import Graph, PruneResult, prune_graph, normalize_matrix, compute_influence
from typing import Optional
import json
import networkx as nx  # type: ignore

def get_adj_from_json(nodes, links):
    node_ids = [n["node_id"] for n in nodes]
    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    n = len(node_ids)
    adj_matrix = torch.zeros((n, n), dtype=torch.float32)

    for link in links:
        src = link["source"]
        tgt = link["target"]
        weight = link.get("weight", 0.0)
        if src in id_to_index and tgt in id_to_index:
            src_idx = id_to_index[src]
            tgt_idx = id_to_index[tgt]
            adj_matrix[tgt_idx, src_idx] = weight  # Note: row=tgt, col=src for incoming edges

    return adj_matrix, node_ids

def select_nodes_from_json(json_path: str, crit: str, top_k: int = 3, edge_weight_threshold: float = 3):
    """Create a NetworkX DiGraph from a graph JSON file produced by create_graph_files*.

    This operates purely on the saved JSON (no original `.pt` graph needed). It keeps the
    existing node set and then, for each target node, retains up to `top_k` strongest
    incoming edges by absolute weight.

    Args:
        json_path: Path to the JSON file (e.g. output/<slug>.json).
        crit: Criterion for pruning. Topk or edge weight.
        top_k: Keep at most this many strongest incoming edges per target.
        edge_weight_threshold: If crit is 'edge_weight', only keep edges with weight above this threshold.
        include_attributes: If True, copy node attributes from JSON into the NetworkX graph.

    Returns:
        nx.DiGraph with pruned edges.
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    links = data.get("links", [])

    attr = {node["node_id"]: node for node in nodes}
    adj, node_ids = get_adj_from_json(nodes, links)

    n = adj.size(0)
    visisted = torch.zeros(n, dtype=torch.bool)
    target_logit = next((i for i, node in enumerate(nodes) if node.get("is_target_logit") is True), None)

    G = nx.DiGraph()

    def dfs(node_idx):
        visisted[node_idx] = True
        G.add_node(tuple([node_ids[node_idx]]))
        row = adj[node_idx]
        if torch.sum(row) == 0:
            return
        nonzero_idx = (row != 0).nonzero(as_tuple=True)[0]
        if len(nonzero_idx) == 0:
            return
        
        if crit == "topk":
            topk_idx = torch.topk(row[nonzero_idx], min(top_k, len(nonzero_idx))).indices
        elif crit == "edge_weight":
            topk_idx = (row[nonzero_idx] >= edge_weight_threshold).nonzero(as_tuple=True)[0]
        
        for src in nonzero_idx[topk_idx].tolist():
            if attr[node_ids[src]]['feature_type'] == 'mlp reconstruction error':
                continue
            G.add_edge(tuple([node_ids[src]]), tuple([node_ids[node_idx]]), weight=adj[node_idx, src].item())
            if not visisted[src]:
                dfs(src)

    dfs(target_logit)

    assert nx.is_directed_acyclic_graph(G), "The resulting graph G is not a DAG."
    return G, attr

if __name__ == "__main__":
    graph_path = "demos/graph_files/dallas-austin.json"
    G, attr = select_nodes_from_json(graph_path, crit="topk", top_k = 3)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print("Nodes:", list(G.nodes(data=False))[:5])
