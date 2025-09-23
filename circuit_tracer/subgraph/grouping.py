from circuit_tracer.subgraph.node_selection import select_nodes_from_json
import torch
import numpy as np
import networkx as nx  # type: ignore
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import namedtuple

Feature = namedtuple("Feature", ["layer", "pos", "feature_idx"])
Intervention = namedtuple('Intervention', ['supernode', 'scaling_factor'])

class Supernode:
    name: str
    activation: float | None
    default_activations: torch.Tensor | None
    children: List["Supernode"]
    intervention: Intervention | None
    replacement_node: Optional["Supernode"]

    def __init__(
        self,
        name: str,
        features: List[Feature],
        children: List["Supernode"] = [],
        intervention: Optional[str] = None,
        replacement_node: Optional["Supernode"] = None,
    ):
        self.name = name
        self.features = features
        self.activation = None
        self.default_activations = None
        self.children = children
        self.intervention = intervention
        self.replacement_node = replacement_node

    def __repr__(self):
        return f"Node(name={self.name}, activation={self.activation}, children={self.children}, intervention={self.intervention}, replacement_node={self.replacement_node})"

def init_invalid_merge(
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

# def greedy_clustering(
#     graph: nx.DiGraph,
#     distance_graph: np.ndarray,
#     attr: Dict[Any, Dict[str, Any]],
#     conditions: Optional[Callable[[Any, Any, nx.DiGraph], bool]] = None,
# ):
#     """Greedily cluster nodes in a DAG represented as a NetworkX DiGraph.

#     Args:
#       - graph: nx.DiGraph assumed to be a DAG (from select_nodes_from_json).
#       - attr: Optional dict mapping node IDs to attribute dicts.
#       - conditions: Optional callable (u, v, graph) -> bool.
#       - distance_graph: n x n numpy array of distances between nodes.
#         If None, defaults to all ones except diagonal zeros.
#     """

#     initial_invalid, node_list = init_invalid_merge(graph, attr, conditions)



#     def merge_nodes()

if __name__ == "__main__":
    graph_path = "demos/graph_files/dallas-austin.json"
    G, attr = select_nodes_from_json(graph_path, crit="topk", top_k=3)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print("Nodes:", list(G.nodes(data=True))[:5])
    
    print(f"Formed {len(clusters)} clusters.")
    for i, c in enumerate(clusters):
        print(f"Cluster {i}: {c}")