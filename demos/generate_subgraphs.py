import os
import json
import networkx as nx

from circuit_tracer.subgraph.pruning import trim_graph, mask_token
from circuit_tracer.subgraph.visualization import visualize_clusters
from circuit_tracer.subgraph.grouping import greedy_grouping, merge_nodes
from circuit_tracer.subgraph.distance import build_distance_graph_from_clerp

def generate_subgraphs(out_path : str, graph_path: str, top_k: int, edge_threshold: float, mask = None):
    # Find subgraph
    G, attr = trim_graph(graph_path, top_k=top_k, edge_threshold=edge_threshold)
    if mask is not None:
        G, attr = mask_token(G, attr, mask=mask)

    visualize_clusters(
        G,
        draw=True,
        filename=f'subgraphs/{out_path}_k_{top_k}_e_{edge_threshold}.png',
        label_fn=lambda node: attr[node].get('clerp') if attr[node].get('clerp') != "" else str(node)
    )

    distance_graph = build_distance_graph_from_clerp(G, attr, progress=True, normalize=True)
    groups, merged_G = greedy_grouping(G, distance_graph=distance_graph, attr=attr, num_groups=15)
    print(f"Formed {len(groups)} clusters.")
    visualize_clusters(
        merged_G,
        draw=True,
        filename=f'subgraphs/merged_{out_path}_k_{top_k}_e_{edge_threshold}.png',
        label_fn=lambda tuple_node: " + ".join(attr[node].get('clerp') if attr[node].get('clerp') != "" else str(node) for node in tuple_node)
    )

if __name__ == "__main__":
    graph_dir = "graph_files"
    out_path = "subgraphs"
    os.makedirs(out_path, exist_ok=True)
    top_k = 10
    for graph_file in os.listdir(graph_dir):
        print(f"Processing {graph_file}...")
        graph_path = os.path.join(graph_dir, graph_file)
        name = graph_file.split('.')[0]
        edge_threshold = 0.3
        mask = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
        generate_subgraphs(out_path=out_path, graph_path=graph_path, top_k=top_k, edge_threshold=edge_threshold, mask=mask)