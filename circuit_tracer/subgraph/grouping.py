from circuit_tracer.subgraph.node_selection import select_nodes_from_json
import torch
import numpy as np
import networkx as nx  # type: ignore
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import namedtuple

Feature = namedtuple("Feature", ["layer", "pos", "feature_idx"])
Intervention = namedtuple('Intervention', ['supernode', 'scaling_factor'])

def init_invalid_merge(
    graph: nx.DiGraph,
    attr: Dict[Any, Dict[str, Any]],
    conditions: Optional[Callable[[Any, Any, nx.DiGraph], bool]] = None,
) -> Tuple[np.ndarray, List[Any]]:
    """Compute invalid merge pairs for a DAG represented as a NetworkX DiGraph.

    Assumes that graph.nodes are tuples of one string.
    A pair (u, v) is invalid to merge if:
      - There exists a directed path u -> v or v -> u in `graph` (DAG reachability)
      - If `attr` is provided, nodes missing in attr or with `feature_type` not equal to
        'cross layer transcoder' are considered invalid to merge with any other node.
      - If `conditions` is provided, pairs for which conditions(u, v, graph) returns False
        are marked invalid.

    Args:
      - graph: nx.DiGraph assumed to be a DAG (from select_nodes_from_json).
      - attr: Optional dict mapping node keys (string) to attribute dicts.
      - conditions: Optional callable (u, v, graph) -> bool.

    Returns:
      - invalid_merge_pairs: m x m boolean numpy array where True means the pair cannot be merged.
      - node_list: The ordering of nodes corresponding to the rows/columns of the matrix.
    """
    node_list = list(graph.nodes())
    n = len(node_list)
    if n == 0:
        return np.zeros((0, 0), dtype=bool), node_list

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

    return invalid, node_list

def check_valid_merge(cluster1: Tuple[Any], cluster2: Tuple[Any], invalid_merge: np.ndarray, node_list: List[Any]) -> bool:
    for node1 in cluster1:
        for node2 in cluster2:
            i = node_list.index(node1)
            j = node_list.index(node2)
            if invalid_merge[i, j]:
                return False
    return True

def merge_nodes(cluster1: Tuple[Any, ...], cluster2: Tuple[Any, ...], graph: nx.DiGraph):
    """Merge two clusters (tuples of member node-ids) into a new supernode.

    - Each member node is itself a graph node-id (a tuple of one string).
    - The new supernode id is the concatenation of the two clusters: cluster1 + cluster2
      (a tuple of member node-ids).
    - Incoming edges from outside the union are redirected to the supernode.
    - Outgoing edges to outside the union are redirected from the supernode.
    - Edges internal to the union are dropped (no self-loops created).
    - For multiple edges from the same source (or to the same target) across members,
      weights are summed.
    """
    members: Set[Any] = set(cluster1) | set(cluster2)
    new_node = tuple(cluster1 + cluster2)

    # Create the new supernode
    if not graph.has_node(new_node):
        graph.add_node(new_node)

    # Collect incoming and outgoing edges to/from the union (excluding internal edges)
    in_accum: Dict[Any, float] = {}
    out_accum: Dict[Any, float] = {}

    for u in list(members):
        # Redirect incoming edges src -> u (src not in union)
        for src, _, data in list(graph.in_edges(u, data=True)):
            if src in members:
                continue
            w = float(data.get("weight", 1.0))
            in_accum[src] = in_accum.get(src, 0.0) + w

        # Redirect outgoing edges u -> dst (dst not in union)
        for _, dst, data in list(graph.out_edges(u, data=True)):
            if dst in members:
                continue
            w = float(data.get("weight", 1.0))
            out_accum[dst] = out_accum.get(dst, 0.0) + w

    # Add redirected edges to/from the new supernode
    for src, w in in_accum.items():
        graph.add_edge(src, new_node, weight=w)
    for dst, w in out_accum.items():
        graph.add_edge(new_node, dst, weight=w)

    # Remove member nodes (and their incident edges)
    for u in list(members):
        if graph.has_node(u):
            graph.remove_node(u)
    
def cluster_distance(
    cluster1: Tuple[Any],
    cluster2: Tuple[Any],
    distance_graph: np.ndarray,
    node_index: Dict[Any, int],
):
    distances = []
    for node1 in cluster1:
        for node2 in cluster2:
            i = node_index[node1]
            j = node_index[node2]
            distances.append(distance_graph[i, j])
    return np.mean(distances) if distances else np.inf

def greedy_clustering(
    graph: nx.DiGraph,
    distance_graph: Optional[np.ndarray],
    attr: Dict[Any, Dict[str, Any]],
    conditions: Optional[Callable[[Any, Any, nx.DiGraph], bool]] = None,
    num_clusters: Optional[int] = None,
):
    """Greedily cluster nodes in a DAG represented as a NetworkX DiGraph.

    Merges are performed consecutively (one pair at a time). A pair of clusters is
    eligible for merging only if no original-node pair between them was marked
    invalid by init_invalid_merge (this enforces the "no path between them" rule).
    After each merge the merged_graph is updated and the cluster-level invalid matrix
    is recomputed (from the original per-node invalid matrix).

    Returns:
      - clusters: list of tuples (each tuple contains original node ids)
      - merged_graph: the NetworkX DiGraph after performing the merges
    """
    # Compute original per-node invalid merge matrix (operates on original node ids)
    invalid_orig, node_list = init_invalid_merge(graph, attr, conditions)
    node_index = {node: i for i, node in enumerate(node_list)}

    n = len(node_list)
    if n == 0:
        return [], graph.copy()

    # Default distance graph: all ones except diagonal zeros
    if distance_graph is None:
        distance_graph = np.ones((n, n), dtype=float)
        np.fill_diagonal(distance_graph, 0.0)
    elif distance_graph.shape != (n, n):
        raise ValueError("distance_graph must be None or an n x n array matching number of nodes")

    # Start with singleton clusters (tuples of original nodes)
    clusters: List[Tuple[Any, ...]] = [ (node,) for node in node_list ]

    # Helper: cluster-level invalid matrix derived from invalid_orig
    def build_cluster_invalid(clusters: List[Tuple[Any, ...]]) -> np.ndarray:
        m = len(clusters)
        cm = np.zeros((m, m), dtype=bool)
        for i in range(m):
            for j in range(m):
                if i == j:
                    cm[i, j] = True
                    continue
                # cluster invalid if any pair of original nodes between clusters is invalid
                any_invalid = False
                for a in clusters[i]:
                    for b in clusters[j]:
                        ia = node_index[a]
                        ib = node_index[b]
                        if invalid_orig[ia, ib]:
                            any_invalid = True
                            break
                    if any_invalid:
                        break
                cm[i, j] = any_invalid
        return cm

    # Helper: compute mean pairwise distance between two clusters (uses original node indices)
    def cluster_dist_mean(c1: Tuple[Any, ...], c2: Tuple[Any, ...]) -> float:
        vals = []
        for a in c1:
            for b in c2:
                ia = node_index[a]
                ib = node_index[b]
                vals.append(float(distance_graph[ia, ib]))
        return float(np.mean(vals)) if vals else np.inf

    merged_graph = graph.copy()

    # Greedy merge loop
    cluster_invalid = build_cluster_invalid(clusters)
    while True:
        m = len(clusters)
        # Stop if target number of clusters reached
        if num_clusters is not None and m <= num_clusters:
            break

        best_dist = np.inf
        best_pair = (-1, -1)
        # find best valid pair (i < j)
        for i in range(m):
            for j in range(i + 1, m):
                if cluster_invalid[i, j]:
                    continue  # invalid to merge
                # Only consider clusters where all members are eligible by attr (init_invalid_merge already enforced attr-based invalids)
                d = cluster_dist_mean(clusters[i], clusters[j])
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)

        # No valid merges left
        if best_pair == (-1, -1) or not np.isfinite(best_dist):
            break

        i, j = best_pair
        # Merge j into i (keep ordering stable)
        c1 = clusters[i]
        c2 = clusters[j]
        new_cluster = tuple(c1 + c2)

        # Update merged_graph by creating merged node and transferring edges
        merge_nodes(c1, c2, merged_graph)

        # Replace cluster entries
        # ensure we replace the smaller index with new_cluster and remove the larger
        if i < j:
            clusters[i] = new_cluster
            del clusters[j]
        else:
            clusters[j] = new_cluster
            del clusters[i]

        # Recompute cluster-level invalid matrix (from original invalid matrix)
        cluster_invalid = build_cluster_invalid(clusters)

    # Final check: merged_graph must remain a DAG
    assert nx.is_directed_acyclic_graph(merged_graph), "Resulting merged_graph is not a DAG"

    return clusters, merged_graph


if __name__ == "__main__":
    graph_path = "demos/graph_files/dallas-austin.json"
    G, attr = select_nodes_from_json(graph_path, crit="topk", top_k=3)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print("Nodes:", list(G.nodes(data=True))[:5])
    clusters, merged_G = greedy_clustering(G, None, attr, num_clusters=5)
    print(f"Formed {len(clusters)} clusters.")
    for i, c in enumerate(clusters):
        print(f"Cluster {i}: {c}")