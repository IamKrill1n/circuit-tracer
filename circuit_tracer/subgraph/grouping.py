from circuit_tracer.subgraph.pruning import trim_graph
import torch
import numpy as np
import networkx as nx  # type: ignore
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import namedtuple

Feature = namedtuple("Feature", ["layer", "pos", "feature_idx"])
Intervention = namedtuple('Intervention', ['supernode', 'scaling_factor'])

def get_path_invalid_merge(
    graph: nx.DiGraph,
    attr: Dict[Any, Dict[str, Any]],
    conditions: Optional[Callable[[Any, Any, nx.DiGraph], bool]] = None,
) -> np.ndarray:
    """Compute invalid merge pairs for a DAG represented as a NetworkX DiGraph.

    Assumes that graph.nodes are tuples of strings.
    A pair (u, v) is invalid to merge if:
      - There exists a directed path u -> v or v -> u in `graph` (DAG reachability)
      - If `attr` is provided, nodes missing in attr or with `feature_type` not equal to
        'cross layer transcoder' are considered invalid to merge with any other node.
      - If `conditions` is provided, pairs for which conditions(u, v, graph) returns False
        are marked invalid.

    Args:
      - graph: nx.DiGraph assumed to be a DAG.
      - attr: Optional dict mapping node keys (string) to attribute dicts.
      - conditions: Optional callable (u, v, graph) -> bool.

    Returns:
      - invalid_merge_pairs: m x m boolean numpy array where True means the pair cannot be merged.
      - node_list: The ordering of nodes corresponding to the rows/columns of the matrix.
    """
    node_list = list(graph.nodes())
    n = len(node_list)
    if n == 0:
        return np.zeros((0, 0), dtype=bool)

    # Helper: extract key from node tuple
    def node_key(node: Any) -> Any:
        return node[0] if isinstance(node, tuple) and len(node) == 1 else node

    node_index = {node: i for i, node in enumerate(node_list)}

    # Compute transitive closure
    tc = nx.transitive_closure(graph)
    reachability = np.zeros((n, n), dtype=bool)
    for u in node_list:
        i = node_index[u]
        for v in tc[u]:
            j = node_index.get(v)
            if j is not None:
                reachability[i, j] = True

    # A pair is invalid if there is a path in either direction
    invalid = reachability | reachability.T

    # Apply attr-based filtering using the node key from the tuple
    for i, node in enumerate(node_list):
        node_attr = attr.get(node_key(node), {})
        if node_attr.get('feature_type') != 'cross layer transcoder':
            invalid[i, :] = True
            invalid[:, i] = True

    # Apply conditions if provided
    if conditions is not None:
        for i in range(n):
            for j in range(n):
                if not invalid[i, j]:
                    if not conditions(node_list[i], node_list[j], graph):
                        invalid[i, j] = True

    # Ensure diagonal is True
    np.fill_diagonal(invalid, True)

    return invalid

def group_distance(
    group1: Tuple[Any],
    group2: Tuple[Any],
    distance_graph: np.ndarray,
    node_index: Dict[Any, int],
):
    distances = []
    for node1 in group1:
        for node2 in group2:
            i = node_index[node1]
            j = node_index[node2]
            distances.append(distance_graph[i, j])
    return np.mean(distances) if distances else np.inf

def merge_nodes(   
    group1: Tuple[Any],
    group2: Tuple[Any],
    graph: nx.DiGraph,
): 
    # May be try nx.contracted_nodes in the future
    """Merge two clusters of nodes in the graph.

    Creates a new node representing the merged cluster, transfers edges,
    and removes the original nodes.

    Args:
      - cluster1: Tuple of nodes to merge (first cluster).
      - cluster2: Tuple of nodes to merge (second cluster).
      - graph: The NetworkX DiGraph to modify.
    """
    new_node = tuple(group1 + group2)
    graph.add_node(new_node)

    for pred in graph.predecessors(group1):
        graph.add_edge(pred, new_node)  

    for succ in graph.successors(group1):
        graph.add_edge(new_node, succ)

    for pred in graph.predecessors(group2):
        graph.add_edge(pred, new_node)
    
    for succ in graph.successors(group2):
        graph.add_edge(new_node, succ)

    # Remove original nodes
    graph.remove_node(group1)
    graph.remove_node(group2)


def greedy_grouping(
    graph: nx.DiGraph,
    distance_graph: np.ndarray,
    attr: Dict[Any, Dict[str, Any]],
    conditions: Optional[Callable[[Any, Any, nx.DiGraph], bool]] = None,
    num_groups: Optional[int] = None,
):
    """Greedily group nodes in a DAG represented as a NetworkX DiGraph.

    Merges are performed consecutively (one pair at a time).
    After each merge the merged_graph is updated and the group-level invalid matrix
    is recomputed
    Returns:
      - groups: list of tuples (each tuple contains original node ids)
      - merged_graph: the NetworkX DiGraph after performing the merges
    """
    # Compute original per-node invalid merge matrix (operates on original node ids)
    invalid_pairs = get_path_invalid_merge(graph, attr, conditions)
    node_index = {node: i for i, node in enumerate(graph.nodes())}

    n = len(node_index)
    if n == 0:
        return [], graph.copy()

    mapping = {node: (node,) for node in graph.nodes()}
    merged_graph = nx.relabel_nodes(graph, mapping, copy=True)

    groups = list(merged_graph.nodes())
 
    while True:
        m = len(groups)
        # Stop if target number of groups reached
        if num_groups is not None and m <= num_groups:
            break

        best_dist = np.inf
        best_pair = (-1, -1)
        # find best valid pair (i < j)
        for i in range(m):
            for j in range(i + 1, m):
                if invalid_pairs[i, j]:
                    continue  # invalid to merge
                # Only consider groups where all members are eligible by attr (init_invalid_merge already enforced attr-based invalids)
                d = group_distance(groups[i], groups[j], distance_graph, node_index)
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)

        # No valid merges left
        if best_pair == (-1, -1) or not np.isfinite(best_dist):
            break

        i, j = best_pair
        # Merge j into i (keep ordering stable)
        c1 = groups[i]
        c2 = groups[j]
        merge_nodes(c1, c2, merged_graph)

        # Update groups list and invalid_pairs matrix
        groups = list(merged_graph.nodes())
        invalid_pairs = get_path_invalid_merge(merged_graph, attr, conditions)

    # Recompute edge weight between groups using the original paper definition
    for u, v in merged_graph.edges():
        weights = []
        for nt in u:
            for ns in v:
                if graph.has_edge(ns, nt):
                    weights.append(graph[ns][nt]['weight'])  
        merged_graph[u][v]['weight'] = np.mean(weights) if weights else 0

    # Final check: merged_graph must remain a DAG
    assert nx.is_directed_acyclic_graph(merged_graph), "Resulting merged_graph is not a DAG"

    return groups, merged_graph


if __name__ == "__main__":
    graph_path = "demos/graph_files/dallas-austin.json"
    G, attr = trim_graph(graph_path, crit="edge_weight", edge_weight_threshold=3)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print("Nodes:", list(G.nodes(data=False))[:5])    

    # print("Nodes:", list(G.nodes(data=True))[:5])
    distance_graph = np.random.rand(G.number_of_nodes(), G.number_of_nodes())
    groups, merged_G = greedy_grouping(G, distance_graph, attr, num_groups=15)
    print(f"Formed {len(groups)} clusters.")
    for i, c in enumerate(groups):
        print(f"Cluster {i}: {c}")