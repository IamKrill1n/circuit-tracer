import torch
import networkx as nx
import numpy as np
from typing import Optional, List, Set, Tuple, Dict

from circuit_tracer.graph import Graph
from circuit_tracer.subgraph.prune import prune_graph_topk, prune_graph_edge_weights
# from circuit_tracer.demos.graph_visualization import create_graph_visualization, Supernode, InterventionGraph

def _get_nx_graph(adj_matrix: np.ndarray, directed: bool) -> nx.Graph:
    """Helper to create a networkx graph from a numpy adjacency matrix."""
    if directed:
        return nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    else:
        return nx.from_numpy_array(adj_matrix)


def compute_invalid_merge_pairs(
    graph: Graph,
    node_mask: Optional[torch.Tensor] = None,
    edge_mask: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute a reduced invalid-merge boolean matrix for nodes selected by node_mask.

    Returns:
      - invalid_merge_pairs : (m x m) boolean numpy array where True means the
                              pair cannot be merged (either path between them OR
                              one of the nodes is not a 'Feature').
      - valid_nodes         : list of original node indices corresponding to rows/cols.

    Rules:
      - If there exists a directed path a->b or b->a in the directed reduced graph,
        then (a,b) is invalid to merge.
      - Only nodes labeled "Feature" in node_type may be merged with other "Feature"
        nodes. Any pair where one or both nodes are not "Feature" will be marked invalid.
    """
    adj = graph.adjacency_matrix.cpu().numpy()
    n_nodes = adj.shape[0]
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_tokens)
    n_features = len(graph.selected_features)
    layers = graph.cfg.n_layers
    error_end_idx = n_features + graph.n_pos * layers

    # Resolve node_mask -> list of original node indices
    if node_mask is None:
        valid_nodes = list(range(n_nodes))
    else:
        valid_nodes = torch.where(node_mask.view(-1))[0].tolist()

    m = len(valid_nodes)
    if m == 0:
        return np.zeros((0, 0), dtype=bool), valid_nodes

    # Reduced directed adjacency
    reduced_adj_directed = adj[np.ix_(valid_nodes, valid_nodes)].copy()

    # apply edge_mask if provided
    if edge_mask is not None:
        edge_mask_np = (
            edge_mask.cpu().numpy() if isinstance(edge_mask, torch.Tensor) else np.asarray(edge_mask)
        )
        reduced_edge_mask = edge_mask_np[np.ix_(valid_nodes, valid_nodes)]
        reduced_adj_directed[~reduced_edge_mask] = 0.0

    node_type = {}
    
    for nodes in valid_nodes:
        if nodes in range(n_features):
            node_type[nodes] = "Feature"
        elif nodes in range(n_features, error_end_idx):
            node_type[nodes] = "Error"
        elif nodes in range(error_end_idx, error_end_idx + n_tokens):
            node_type[nodes] = "Token"
        else:
            node_type[nodes] = "Logit"

    # Build directed graph for reachability (use binary adjacency)
    directed_binary = (reduced_adj_directed != 0).astype(float)
    try:
        dag = _get_nx_graph(directed_binary, directed=True)
        tc = nx.transitive_closure_dag(dag)
        reachability = nx.to_numpy_array(tc, nodelist=list(range(m)), dtype=bool)
    except Exception:
        dag = _get_nx_graph(directed_binary, directed=True)
        tc = nx.transitive_closure(dag)
        reachability = nx.to_numpy_array(tc, nodelist=list(range(m)), dtype=bool)

    # invalid if either direction has a path
    invalid = reachability | reachability.T

    # enforce only "Feature" nodes can be merged: any pair involving a non-feature is invalid
    if node_type is not None:
        # node_type keyed by original indices; build reduced list of types
        reduced_types = [node_type.get(orig, "") for orig in valid_nodes]
        for i in range(m):
            for j in range(m):
                if reduced_types[i] != "Feature" or reduced_types[j] != "Feature":
                    invalid[i, j] = True

    # diagonal should be True (can't merge node with itself)
    np.fill_diagonal(invalid, True)

    return invalid, valid_nodes


def hierarchical_cluster_from_invalid(
    graph: Graph,
    invalid_merge_pairs: np.ndarray,
    valid_nodes: List[int],
    edge_mask: Optional[torch.Tensor] = None,
    distance_graph: Optional[Graph] = None,
    linkage: str = "single",
    num_clusters: Optional[int] = None,
) -> List[Set[int]]:
    """
    Perform hierarchical clustering over the nodes in valid_nodes, using the
    provided invalid_merge_pairs matrix to enforce merge constraints.

    Args:
      - invalid_merge_pairs : m x m boolean matrix for the valid_nodes ordering.
      - valid_nodes         : list of original node indices corresponding to rows/cols.
      - edge_mask           : optional full-size edge mask (applied when computing distances).
      - distance_graph      : optional Graph used for distances (defaults to graph).
      - linkage             : only 'single' supported.
      - num_clusters        : stop when this many clusters remain (default 1).

    Returns:
      - list of sets containing original node indices (clusters).
    """
    if linkage != "single":
        raise NotImplementedError("Only 'single' linkage is currently supported.")

    m = len(valid_nodes)
    if m == 0:
        return []

    # Prepare mappings
    orig_to_pos = {orig: pos for pos, orig in enumerate(valid_nodes)}
    pos_to_orig = {pos: orig for pos, orig in enumerate(valid_nodes)}

    # Build distance adjacency for reduced nodes
    full_adj = (distance_graph.adjacency_matrix.cpu().numpy() if distance_graph
                else graph.adjacency_matrix.cpu().numpy()).copy()

    if edge_mask is not None:
        edge_mask_np = (
            edge_mask.cpu().numpy() if isinstance(edge_mask, torch.Tensor) else np.asarray(edge_mask)
        )
        full_adj[~edge_mask_np] = 0.0

    reduced_dist_adj = full_adj[np.ix_(valid_nodes, valid_nodes)].copy()
    reduced_dist_binary = (reduced_dist_adj != 0).astype(int)
    undirected_graph = _get_nx_graph(reduced_dist_binary, directed=False)

    # all-pairs shortest paths (reduced indices)
    node_dist_matrix = np.full((m, m), np.inf)
    path_lengths = dict(nx.all_pairs_shortest_path_length(undirected_graph))
    for i in range(m):
        if i in path_lengths:
            for j, length in path_lengths[i].items():
                node_dist_matrix[i, j] = length

    # initialize clusters: representative original idx -> set(orig indices)
    cluster_map = {orig: {orig} for orig in valid_nodes}
    node_to_cluster_rep = {orig: orig for orig in valid_nodes}

    # potential merges: (dist, orig_i, orig_j) using reduced positions
    potential_merges = []
    for i_pos in range(m):
        for j_pos in range(i_pos + 1, m):
            dist = node_dist_matrix[i_pos, j_pos]
            if not np.isinf(dist):
                potential_merges.append((dist, pos_to_orig[i_pos], pos_to_orig[j_pos]))
    potential_merges.sort(key=lambda x: x[0])

    num_active_clusters = len(cluster_map)
    target_clusters = 1 if num_clusters is None else max(1, num_clusters)

    for dist, orig1, orig2 in potential_merges:
        if num_active_clusters <= target_clusters:
            break

        rep1 = node_to_cluster_rep[orig1]
        rep2 = node_to_cluster_rep[orig2]
        if rep1 == rep2:
            continue

        cluster1_nodes = cluster_map[rep1]
        cluster2_nodes = cluster_map[rep2]

        # Validate merge using invalid_merge_pairs on reduced positions
        is_valid = True
        for a in cluster1_nodes:
            for b in cluster2_nodes:
                a_pos = orig_to_pos[a]
                b_pos = orig_to_pos[b]
                if invalid_merge_pairs[a_pos, b_pos]:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            continue

        # Merge smaller into larger
        if len(cluster1_nodes) < len(cluster2_nodes):
            rep1, rep2 = rep2, rep1
            cluster1_nodes, cluster2_nodes = cluster2_nodes, cluster1_nodes

        cluster_map[rep1].update(cluster2_nodes)
        for node in cluster2_nodes:
            node_to_cluster_rep[node] = rep1

        del cluster_map[rep2]
        num_active_clusters -= 1

    final_clusters = [set(nodes) for nodes in cluster_map.values()]
    return final_clusters

# if __name__ == "__main__":
#     from circuit_tracer.graph import Graph
#     graph_path = "demos/graphs/example_graph.pt"
#     graph = Graph.from_pt(graph_path)
#     node_mask, edge_mask, cumulative_scores= prune_graph_topk(graph, top_k = 3)
#     print(edge_mask.sum(), node_mask.sum())
#     clusters = hierarchical_cluster_from_invalid(graph, *compute_invalid_merge_pairs(graph, node_mask, edge_mask), edge_mask=edge_mask, num_clusters=10)
#     supernodes = []
#     for i, cluster in enumerate(clusters):
#         name = f"Cluster {i}"
#         # Optionally, you could store additional info from the cluster in 'features'
#         sn = Supernode(name=name, features=[])
#         supernodes.append(sn)

#     # For visualization purposes we group the clusters in one row.
#     ordered_nodes = [supernodes]

#     # Create an InterventionGraph object with a prompt describing the clustering
#     prompt = "Hierarchical Clustering Visualization of the Attribution Graph"
#     intervention_graph = InterventionGraph(ordered_nodes=ordered_nodes, prompt=prompt)

#     # For demonstration, define dummy top outputs (you can replace these with model outputs)
#     top_outputs = [("Dummy Output", 1.0)]

#     # Visualize the intervention graph (this uses the SVG-based visualization)
#     create_graph_visualization(intervention_graph, top_outputs)