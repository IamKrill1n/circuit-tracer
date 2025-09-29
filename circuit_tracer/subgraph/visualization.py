from platform import node
from circuit_tracer.subgraph.pruning import trim_graph
from circuit_tracer.subgraph.grouping import greedy_grouping
from circuit_tracer import ReplacementModel
import networkx as nx  # type: ignore
from typing import Any, Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt  # type: ignore
import torch
import numpy as np

def topological_layers(graph: nx.DiGraph) -> List[List[Any]]:
    """Return topological layers: each layer is a list of nodes that have no edges between them.

    Uses Kahn's algorithm: repeatedly remove nodes with indegree 0. Raises ValueError if graph has a cycle.
    """
    indeg = {n: graph.in_degree(n) for n in graph.nodes()}
    layers: List[List[Any]] = []
    # Work on a copy of indegree dict only
    while indeg:
        zero = [n for n, d in indeg.items() if d == 0]
        if not zero:
            raise ValueError("Graph contains a cycle; cannot produce topological layers.")
        layers.append(zero)
        for u in zero:
            # decrement indegree of successors
            for _, v in graph.out_edges(u):
                if v in indeg:
                    indeg[v] -= 1
            # remove u from consideration
            del indeg[u]
    return layers

def visualize_clusters(
    graph: nx.DiGraph,
    draw: bool = True,
    filename: Optional[str] = None,
    label_fn: Optional[Callable[[Any], str]] = None,
) -> List[List[Any]]:
    """Print topological layers and optionally draw the graph laid out by layers.

    - Each printed line is a layer where nodes have no edges between them.
    - If draw=True, produces a matplotlib figure where layers are horizontal ranks.
    - filename: if provided, save the figure to this path instead of showing.
    - label_fn: optional function to convert node id -> label string.
    """
    if label_fn is None:
        def label_fn(n: Any) -> str:
            return n[0] if isinstance(n, tuple) and len(n) == 1 else str(n)

    layers = topological_layers(graph)

    # Print layers
    for i, layer in enumerate(layers):
        labels = [label_fn(n) for n in layer]
        print(f"Layer {i}: " + ", ".join(labels))

    if draw:
        # build positions: nodes in same layer are spread horizontally, layers stacked vertically
        pos: Dict[Any, Tuple[float, float]] = {}
        max_width = max(len(l) for l in layers) if layers else 1
        for i, layer in enumerate(layers):
            y = -i  # top layer at y=0, subsequent layers below
            count = len(layer)
            if count == 0:
                continue
            for idx, node in enumerate(layer):
                x = idx - (count - 1) / 2.0
                pos[node] = (x, y)

        plt.figure(figsize=(max(6, max_width * 0.8), max(4, len(layers) * 0.8)))
        # nx.draw_networkx_nodes(graph, pos, node_size=350, node_color="blue", linewidths=0.5)
        labels = {n: label_fn(n) for layer in layers for n in layer}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
        edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
        nx.draw(graph, pos, node_color='lightblue', edgelist=edges, edge_color=weights, width=2.0, arrows = False, edge_cmap=plt.cm.Greens, with_labels=False)
        # print(weights)
        # nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=list(weights), width=10.0, edge_cmap='Reds')
        plt.axis("off")
        plt.tight_layout()
        if filename:
            plt.savefig(filename, bbox_inches="tight")
            print(f"Saved visualization to {filename}")
            plt.close()
        else:
            plt.show()

    return layers

if __name__ == "__main__":
    prompt = "Fact: the capital of the state containing Dallas is"
    graph_path = "demos/graph_files/factthelargestco-1755767633671_2025-09-26T13-34-02-113Z.json"
    G, attr = trim_graph(graph_path, crit="topk", top_k=3)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print("Nodes:", list(G.nodes(data=True))[:5])
    distance_graph = np.random.rand(G.number_of_nodes(), G.number_of_nodes())
    groups, merged_G = greedy_grouping(G, distance_graph=distance_graph, attr=attr, num_groups=15)
    # model = ReplacementModel.from_pretrained("google/gemma-2-2b", 'gemma', dtype=torch.bfloat16)
    # visualize_intervention_graph(G, prompt, attr, model = model)

    visualize_clusters(
        G,
        draw=True,
        filename='demos/subgraphs/subgraph.png',
        label_fn=lambda node: attr[node].get('clerp') if attr[node].get('clerp') != "" else str(node)
    )
    visualize_clusters(
        merged_G,
        draw=True,
        filename='demos/subgraphs/merged_subgraph.png',
        label_fn=lambda tuple_node: " + ".join(attr[node].get('clerp') if attr[node].get('clerp') != "" else str(node) for node in tuple_node)
    )