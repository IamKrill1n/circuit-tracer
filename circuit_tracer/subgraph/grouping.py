from circuit_tracer.subgraph.node_selection import select_nodes_from_json
import torch
import numpy as np
import networkx as nx  # type: ignore
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# =====================
# NetworkX-based API
# =====================
def compute_invalid_merge(
    graph: nx.DiGraph,
    attr: Dict[Any, Dict[str, Any]],
    conditions: Optional[Callable[[Any, Any, nx.DiGraph], bool]] = None,
) -> Tuple[np.ndarray, List[Any]]:
    """Compute invalid merge pairs for a DAG represented as a NetworkX DiGraph.

    A pair (u, v) is invalid to merge if:
      - There exists a directed path u -> v or v -> u in `graph` (DAG reachability)
      - If `attr` is provided, nodes missing in attr or with `feature_type` not equal to
        'cross layer transcoder'
        are considered invalid to merge with any other node.
      - If `conditions` is provided, pairs for which conditions(u, v, graph) returns False
        are marked invalid.

    Args:
      - graph: nx.DiGraph assumed to be a DAG (from select_nodes_from_json).
      - attr: Optional dict mapping node IDs to attribute dicts.
      - nodes: Optional ordering of nodes. Defaults to list(graph.nodes()).
      - conditions: Optional callable (u, v, graph) -> bool.

    Returns:
      - invalid_merge_pairs: m x m boolean numpy array where True means the pair cannot be merged.
      - node_list: The ordering of nodes corresponding to the rows/columns of the matrix.
    """
    node_list = list(graph.nodes())
    n = len(node_list)
    if n == 0:
        return np.zeros((0, 0), dtype=bool), node_list

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

    # Apply attr-based filtering
    for i, node in enumerate(node_list):
        node_attr = attr.get(node, {})
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

def hierarchical_cluster_nx(
    graph: nx.DiGraph,
    attr: Dict[Any, Dict[str, Any]],
    conditions: Optional[Callable[[Any, Any, nx.DiGraph], bool]] = None,
    distance_graph: Optional[nx.Graph] = None,
    num_clusters: int = 10,
) -> List[Set[Any]]:
    """Hierarchical clustering over DAG nodes with merge constraints.

    Performs single-linkage style agglomerative clustering over the provided `nodes` using
    shortest-path distance on the undirected version of `distance_graph` (defaults to `graph`).
    Clusters are merged from closest to farthest, subject to the constraint that no merged
    cluster contains any pair of nodes with a directed path between them, and any additional
    `conditions(u, v, graph)` predicate provided.

    Args:
      - graph: nx.DiGraph (DAG) with nodes to cluster.
      - conditions: Optional callable (u, v, graph) -> bool. Pairs for which this returns False
                    are considered invalid to merge.
      - distance_graph: Optional graph to define distances. If None, uses `graph`.
      - num_clusters: Target number of clusters. Defaults to 1 if None.

    Returns:
      - List of sets of node IDs (clusters).
    """
    invalid_merge_pairs, node_list = compute_invalid_merge(
        graph, attr=attr, conditions=conditions
    )
    n = len(attr)
    if n == 0:
        return []

    # Prepare target cluster count
    target_clusters = 1 if num_clusters is None else max(1, int(num_clusters))

    # Distance graph (undirected, unweighted shortest path length)
    dist_g = distance_graph if distance_graph is not None else graph
    undirected = dist_g.to_undirected(as_view=False)

    # Compute all-pairs shortest path lengths among node_list
    node_pos = {n: i for i, n in enumerate(node_list)}
    node_dist_matrix = np.full((n, n), np.inf)
    spl = dict(nx.all_pairs_shortest_path_length(undirected))
    for u, dists in spl.items():
        if u not in node_pos:
            continue
        i = node_pos[u]
        for v, L in dists.items():
            j = node_pos.get(v)
            if j is not None:
                node_dist_matrix[i, j] = L

    # Initialize clusters
    clusters: Dict[Any, Set[Any]] = {n: {n} for n in node_list}
    rep_of: Dict[Any, Any] = {n: n for n in node_list}

    # Build potential merges sorted by distance
    potential_merges: List[Tuple[float, Any, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = node_dist_matrix[i, j]
            if not np.isinf(d):
                potential_merges.append((float(d), node_list[i], node_list[j]))
    potential_merges.sort(key=lambda x: x[0])

    # Helper to test cluster-level validity
    def clusters_can_merge(rep1: Any, rep2: Any) -> bool:
        c1 = clusters[rep1]
        c2 = clusters[rep2]
        for a in c1:
            ia = node_pos[a]
            for b in c2:
                ib = node_pos[b]
                if invalid_merge_pairs[ia, ib]:
                    return False
                if conditions is not None and not conditions(a, b, graph):
                    return False
        return True

    active = len(clusters)
    for dist, u, v in potential_merges:
        if active <= target_clusters:
            break
        ru = rep_of[u]
        rv = rep_of[v]
        if ru == rv:
            continue
        if not clusters_can_merge(ru, rv):
            continue

        # Merge smaller into larger
        if len(clusters[ru]) < len(clusters[rv]):
            ru, rv = rv, ru
        # Move nodes from rv into ru
        for x in clusters[rv]:
            rep_of[x] = ru
        clusters[ru].update(clusters[rv])
        del clusters[rv]
        active -= 1

    return [set(s) for s in clusters.values()]

if __name__ == "__main__":
    graph_path = "demos/graph_files/dallas-austin.json"
    G, attr = select_nodes_from_json(graph_path, crit="topk", top_k=3)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print("Nodes:", list(G.nodes(data=True))[:5])
    clusters = hierarchical_cluster_nx(G, attr, num_clusters=5)
    print(f"Formed {len(clusters)} clusters.")
    for i, c in enumerate(clusters):
        print(f"Cluster {i}: {c}")