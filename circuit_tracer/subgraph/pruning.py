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

def trim_graph(json_path: str, top_k: int = 3, edge_threshold: float = 0.85):
    """Create a NetworkX DiGraph from a graph JSON file produced by create_graph_files*.

    This operates purely on the saved JSON (no original `.pt` graph needed). It keeps the
    existing node set and then, for each target node, retains up to `top_k` strongest
    incoming edges by absolute weight.

    Args:
        json_path: Path to the JSON file (e.g. output/<slug>.json).
        top_k: Keep at most this many strongest incoming edges per target.
        edge_threshold: Keep edges with weight above this percentage after top_k pruning

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
    edges = []
    def dfs(node_idx):
        visisted[node_idx] = True
        G.add_node(node_ids[node_idx])
        row = adj[node_idx]
        if torch.sum(row) == 0:
            return
        nonzero_idx = (row != 0).nonzero(as_tuple=True)[0]
        if len(nonzero_idx) == 0:
            return

        topk_idx = torch.topk(row[nonzero_idx], min(top_k, len(nonzero_idx))).indices
        
        for src in nonzero_idx[topk_idx].tolist():
            if attr[node_ids[src]]['feature_type'] == 'mlp reconstruction error':
                continue
            G.add_edge(node_ids[src], node_ids[node_idx], weight=adj[node_idx, src].item())
            edges.append((node_ids[src], node_ids[node_idx], adj[node_idx, src].item()))
            if not visisted[src]:
                dfs(src)

    dfs(target_logit)

    # Remove low-weight edges
    sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
    keep_num = int(len(sorted_edges) * edge_threshold)
    edges_to_remove = sorted_edges[keep_num:]
    G.remove_edges_from([(src, tgt) for src, tgt, _ in edges_to_remove])

    # Remove nodes with no incoming or outgoing edges
    while True:
        nodes_to_remove  = set()

        for node in list(G.nodes):
            if G.in_degree(node) == 0 and attr[node]['feature_type'] != 'embedding':
                nodes_to_remove.add(node)
            if G.out_degree(node) == 0 and attr[node]['feature_type'] != 'logit':
                nodes_to_remove.add(node)

        if not nodes_to_remove:
            break

        G.remove_nodes_from(nodes_to_remove)

    attr = {node: attr[node] for node in G.nodes}

    assert nx.is_directed_acyclic_graph(G), "The resulting graph G is not a DAG."
    return G, attr

if __name__ == "__main__":
    graph_path = "demos/graph_files/factthelargestco-1755767633671_2025-09-26T13-34-02-113Z.json"
    G, attr = trim_graph(graph_path, top_k = 5, edge_threshold=0.85)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print("Nodes:", list(G.nodes(data=False))[:5])
