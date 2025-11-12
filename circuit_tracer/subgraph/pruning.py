import torch
import numpy as np
from circuit_tracer.graph import Graph, PruneResult, prune_graph, normalize_matrix, compute_influence
from typing import List, Optional
from circuit_tracer.subgraph.utils import get_data_from_json, get_clerp
import networkx as nx  # type: ignore


def get_graph_from_json(json_path: str):
    """Load a nx.DiGraph object from a JSON file."""
    adj, node_ids, attr, metadata = get_data_from_json(json_path)

    G = nx.from_numpy_array(adj.cpu().numpy().T, create_using=nx.DiGraph)

    mapping = {i: node_id for i, node_id in enumerate(node_ids)}
    G = nx.relabel_nodes(G, mapping)

    return G, attr

def remove_edges(edges_to_remove, G: nx.DiGraph):
    new_G = G.copy()
    new_G.remove_edges_from([(src, tgt) for src, tgt, _ in edges_to_remove])
    while True:
        nodes_to_remove = set()

        for node in list(new_G.nodes):
            if new_G.in_degree(node) == 0 and attr[node]['feature_type'] != 'embedding':
                nodes_to_remove.add(node)
            if new_G.out_degree(node) == 0 and attr[node]['feature_type'] != 'logit':
                nodes_to_remove.add(node)

        if not nodes_to_remove:
            break

        new_G.remove_nodes_from(nodes_to_remove)
    
    return new_G

def trim_graph(adj, node_ids, attr, top_k: int = 10, edge_threshold: float = 0.3, debug : bool = False):
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
    
    n = adj.size(0)
    visisted = torch.zeros(n, dtype=torch.bool)
    target_logit = next((i for i, node in enumerate(node_ids) if attr[node].get("is_target_logit") is True), None)

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

    if debug == True:
        import matplotlib.pyplot as plt
        import os
        # Plot edge weight distribution
        sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
        weights = [w for _, _, w in sorted_edges]
        plt.figure(figsize=(8, 5))
        plt.hist(weights, bins=50)
        plt.title("Edge Weight Distribution")
        plt.xlabel("Weight")
        plt.ylabel("Frequency")
        # Save plot
        out = "demos/plots/edge_weight_hist.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
        plt.close()

        num_nodes = []
        num_edges = []

        for threshold in np.linspace(0.5, 0.1, 11):
            keep_num = int(len(sorted_edges) * threshold)
            print(f"Threshold: {threshold:.2f}, Keeping top {keep_num} edges out of {len(sorted_edges)}")
            new_G = remove_edges(sorted_edges[keep_num:], G)
            print(f"Trimmed graph has {new_G.number_of_nodes()} nodes and {new_G.number_of_edges()} edges.")
            num_nodes.append(new_G.number_of_nodes())
            num_edges.append(new_G.number_of_edges())
            if new_G.number_of_nodes() == 0:
                break
        plt.figure(figsize=(8, 5))
        plt.plot(np.linspace(0.5, 0.1, len(num_nodes)),
                    num_nodes, label="Number of Nodes")
        plt.plot(np.linspace(0.5, 0.1, len(num_edges)),
                    num_edges, label="Number of Edges")
        plt.xlabel("Edge Retention Threshold")
        plt.ylabel("Count")
        plt.title("Graph Size vs Edge Retention Threshold")
        plt.legend()
        out = "demos/plots/graph_size_vs_threshold.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
        plt.close()
    # print(f"Subgraph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    # Remove low-weight edges
    sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
    keep_num = int(len(sorted_edges) * edge_threshold)
    edges_to_remove = sorted_edges[keep_num:]
    G = remove_edges(edges_to_remove, G)

    # print(f"Trimmed graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    attr = {node: attr[node] for node in G.nodes}

    assert nx.is_directed_acyclic_graph(G), "The resulting graph G is not a DAG."

    return G, attr

def mask_token(graph: nx.DiGraph, attr: dict, mask: List):
    """Create a masked version of the input graph by removing embedding nodes specified in the mask.

    Args:
        graph: The input graph.
        attr: Node attributes.
        mask: A boolean tensor indicating which nodes to remove.

    Returns:
        A new graph with the specified nodes removed.
    """

    G = graph.copy()
    nodes_to_remove = [node for node in G.nodes if attr[node]['feature_type'] == 'embedding' and not mask[attr[node]['ctx_idx']]]
    G.remove_nodes_from(nodes_to_remove)

    attr = {node: attr[node] for node in G.nodes}

    assert nx.is_directed_acyclic_graph(G), "The resulting graph G is not a DAG."
    print(f"Masked graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, attr


if __name__ == "__main__":
    graph_path = "graph_files/cold-clt-hp.json"
    adj, node_ids, attr, metadata = get_data_from_json(graph_path)
    G, attr = trim_graph(adj, node_ids, attr, top_k=10, edge_threshold=0.3, debug=True)
    # for node in list(G.nodes)[:5]:
    #     print(f"Node: {node}, clerp: {attr[node].get('clerp', '')}")
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    # print("Nodes:", list(G.nodes(data=False))[:5])
