from platform import node
from circuit_tracer.subgraph.prune import prune_graph_pipeline
from circuit_tracer.subgraph.utils import get_data_from_json, get_clerp
from circuit_tracer.subgraph.grouping import greedy_grouping
from circuit_tracer.subgraph.distance import build_distance_graph_from_clerp
from circuit_tracer import ReplacementModel
import networkx as nx  # type: ignore
from typing import Any, Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt  # type: ignore
import torch
import numpy as np

def build_nx_graph(adj_matrix: torch.Tensor, node_ids: List[Any]) -> nx.DiGraph:
    """Convert adjacency matrix and node ids into a NetworkX directed graph."""
    G = nx.DiGraph()
    n = adj_matrix.shape[0]
    G.add_nodes_from(node_ids)
    for i in range(n):
        for j in range(n):
            weight = adj_matrix[i, j].item()
            if weight != 0:
                G.add_edge(node_ids[j], node_ids[i], weight=weight)  # edge from src=j to tgt=i
    return G

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

    # Build a safe string label for every node using label_fn, coercing non-str results.
    label_map: Dict[Any, str] = {}
    for layer in layers:
        for n in layer:
            raw = label_fn(n)
            if isinstance(raw, list):
                # join list entries into a single string
                label_map[n] = " + ".join(str(x) for x in raw)
            else:
                label_map[n] = "" if raw is None else str(raw)

    # Print layers
    for i, layer in enumerate(layers):
        labels = [label_map[n] for n in layer]
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
        labels = {n: label_map[n] for layer in layers for n in layer}
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
    prompt = "<start_of_turn>user⏎What is the capital of the state containing Dallas? Answer immediately.<end_of_turn>⏎<start_of_turn>model⏎Austin"
    prompts_tokens = ["<start_of_turn>","user","⏎","What"," is"," the"," capital"," of"," the"," state"," containing"," Dallas","?"," Answer"," immediately",".","<end_of_turn>","⏎","<start_of_turn>","model","⏎"]
    # graph_path = "demos/temp_graph_files/dallas-austin-gs2-27b-it_2026-03-05T07-24-51-963Z.json"
    graph_path = 'demos/temp_graph_files/dallas-austin_gemma3.json'
    adj, node_ids, attr, metadata = get_data_from_json(graph_path)
    name = graph_path.split('/')[-1].split('.')[0]
    token_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/3, 0, 0, 1/3, 0, 1/3, 0, 0, 0, 0, 0, 0]
    # token_weights = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     token_weights = [0.00198786, 0.03153391, 0.00083086, 0.01473883, 0.22338926, 0.00649094,
#   0.00222269, 0.01996207, 0.0052309, 0.67869559, 0.01491708]

    kept_ids, pruned_adj, attr, metadata = prune_graph_pipeline(
        json_path=graph_path,
        logit_weights="target",
        token_weights=token_weights,
        node_threshold=0.6,
        edge_threshold=0.7,
        keep_all_tokens_and_logits=False,
    )

    print(f"Kept {len(kept_ids)} nodes and {(pruned_adj != 0).sum().item()} edges after pruning.")
    for node in kept_ids:
        clerp = attr.get(node, {}).get("clerp", "")
        print(f"  {node}: {clerp[:60]}")

    # Sweep relevance thresholds to see how the graph changes
    # num_nodes = []
    # for node_relevance_threshold in np.linspace(0.1, 0.9, 5):
    #     # for edge_relevance_threshold in np.linspace(0.1, 0.9, 5):
    #     print(f"Node relevance threshold: {node_relevance_threshold:.2f}, Edge relevance threshold: 0.95")
    #     kept_ids, pruned_adj, attr, metadata = prune_graph_pipeline(
    #         json_path=graph_path,
    #         logit_weights="target",
    #         token_weights=None,
    #         node_threshold=0.8,
    #         edge_threshold=0.95,
    #         keep_all_tokens_and_logits=False,
    #     )

    #     num_nodes_i = len(kept_ids)
    #     num_edges = (pruned_adj != 0).sum().item()
    #     print(f"Kept {num_nodes_i} nodes and {num_edges} edges after pruning.")
    #     num_nodes.append(num_nodes_i)

    # plt.figure(figsize=(8, 5))
    # plt.plot(np.linspace(0.1, 0.9, 5), num_nodes, marker='o')
    # plt.title("Number of Nodes Kept vs Node Relevance Threshold")
    # plt.xlabel("Node Relevance Threshold")
    # plt.ylabel("Number of Nodes Kept")
    # plt.grid()
    # plt.savefig(f'demos/plots/{name}_node_relevance_sweep.png', bbox_inches="tight")
    # print(f"Saved node relevance sweep plot to demos/plots/{name}_node_relevance_sweep.png")
    # G = build_nx_graph(pruned_adj, kept_ids)

    # # Print top nodes
    # for node in list(G.nodes):
    #     clerp = attr.get(node, {}).get("clerp", "")
    #     print(f"  {node}: {clerp[:60]}")
    
    # visualize_clusters(
    #     G,
    #     draw=True,
    #     filename=f'demos/subgraphs/{name}.png',
    #     label_fn=lambda node: attr[node].get('clerp') if attr[node].get('clerp') != "" else str(node)
    # )
    # visualize_clusters(
    #     merged_G,
    #     draw=True,
    #     filename=f'subgraphs/merged_{name}_n_{node_threshold}_e_{edge_threshold}.png',
    #     label_fn=lambda tuple_node: " + ".join(attr[node].get('clerp') if attr[node].get('clerp') != "" else str(node) for node in tuple_node)
    # )