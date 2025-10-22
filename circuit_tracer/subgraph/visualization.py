from platform import node
from circuit_tracer.subgraph.pruning import trim_graph
from circuit_tracer.subgraph.grouping import greedy_grouping
from circuit_tracer.subgraph.distance import build_distance_graph_from_clerp
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

        plt.figure(figsize=(max(20, max_width * 0.8), max(16, len(layers) * 0.8)))
        nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue', linewidths=0.5)
        labels = {n: label_fn(n) for layer in layers for n in layer}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=15)

        # draw edges with color mapped to weight and small curvature for perfectly vertical edges
        from matplotlib.patches import FancyArrowPatch
        import matplotlib as mpl

        edge_list = list(graph.edges())
        if edge_list:
            edge_weights = [graph[u][v].get("weight", 1.0) for u, v in edge_list]
            norm = mpl.colors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)) if len(edge_weights) > 1 else mpl.colors.Normalize(vmin=0, vmax=1)
            cmap = plt.cm.Greens
            ax = plt.gca()

            for idx, (u, v) in enumerate(edge_list):
                w = edge_weights[idx]
                color = cmap(norm(w))
                xyA = pos[u]
                xyB = pos[v]

                # If the edge is (nearly) vertical, give it curvature so it is visible
                dx = xyB[0] - xyA[0]
                # small threshold to detect vertical alignment
                if abs(dx) < 1e-6:
                    # alternate sign to avoid overlapping multiple edges
                    rad = 0.18 * (1 if (idx % 2 == 0) else -1)
                else:
                    # slight curvature for non-vertical edges can be zero
                    rad = 0.0

                arrow = FancyArrowPatch(
                    xyA, xyB,
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle='-|>',
                    mutation_scale=12,
                    linewidth=2.0,
                    color=color,
                    zorder=0,
                )
                ax.add_patch(arrow)
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
    prompt = "Before is to after as past is to"
    graph_path = "demos/graph_files/puppy-clt.json"
    name = graph_path.split('/')[-1].split('.')[0]
    top_k = 7
    edge_threshold = 0.3
    G, attr = trim_graph(graph_path, top_k=top_k, edge_threshold=edge_threshold)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    for node in G.nodes():
        print(node, attr[node].get('clerp', ''))
    

    # Russian
    # attr['18_868_11']['clerp'] = 'say a country'
    # attr['20_15360_11']['clerp'] = 'say a country'
    # attr['17_14546_11']['clerp'] = 'abstract: country'
    # attr['18_8149_11']['clerp'] = 'abstract: area and size'
    # attr['15_16322_11']['clerp'] = 'abstract: sequencing/comparison'
    # attr['19_11557_11']['clerp'] = 'say Russian'

    # Dallas

    # distance_graph = np.random.rand(G.number_of_nodes(), G.number_of_nodes())
    distance_graph = build_distance_graph_from_clerp(G, attr, progress=True, normalize=True)
    groups, merged_G = greedy_grouping(G, distance_graph=distance_graph, attr=attr, num_groups=15)
    # model = ReplacementModel.from_pretrained("google/gemma-2-2b", 'gemma', dtype=torch.bfloat16)
    # visualize_intervention_graph(G, prompt, attr, model = model)
    print(f"Formed {len(groups)} clusters.")
    visualize_clusters(
        G,
        draw=True,
        filename=f'demos/subgraphs/{name}_k_{top_k}_e_{edge_threshold}.png',
        label_fn=lambda node: attr[node].get('clerp') if attr[node].get('clerp') != "" else str(node)
    )
    visualize_clusters(
        merged_G,
        draw=True,
        filename=f'demos/subgraphs/merged_{name}_k_{top_k}_e_{edge_threshold}.png',
        label_fn=lambda tuple_node: " + ".join(attr[node].get('clerp') if attr[node].get('clerp') != "" else str(node) for node in tuple_node)
    )