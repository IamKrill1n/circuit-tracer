import os
import json
import networkx as nx

from circuit_tracer.subgraph.pruning import trim_graph, mask_token
from circuit_tracer.subgraph.visualization import visualize_clusters
from circuit_tracer.subgraph.grouping import greedy_grouping, merge_nodes
from circuit_tracer.subgraph.distance import build_distance_graph_from_clerp
from circuit_tracer.subgraph.utils import get_data_from_json, get_clerp

def generate_subgraphs(out_path : str, graph_path: str, top_k: int, edge_threshold: float, include_clerp : bool, mask = None):
    # Find subgraph
    adj, node_ids, attr, metadata = get_data_from_json(graph_path)

    G, attr = trim_graph(adj, node_ids, attr, top_k=top_k, edge_threshold=edge_threshold)
    if mask is not None:
        G, attr = mask_token(G, attr, mask=mask)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    if include_clerp:
        get_clerp(metadata, attr)

    visualize_clusters(
        G,
        draw=True,
        filename=f'{out_path}_k_{top_k}_e_{edge_threshold}.png',
        label_fn=lambda node: attr[node].get('clerp') if attr[node].get('clerp') != "" else str(node)
    )

    # distance_graph = build_distance_graph_from_clerp(G, attr, progress=True, normalize=True)
    # groups, merged_G = greedy_grouping(G, distance_graph=distance_graph, attr=attr, num_groups=15)
    # print(f"Formed {len(groups)} clusters.")
    # visualize_clusters(
    #     merged_G,
    #     draw=True,
    #     filename=f'subgraphs/merged_{out_path}_k_{top_k}_e_{edge_threshold}.png',
    #     label_fn=lambda tuple_node: " + ".join(attr[node].get('clerp') if attr[node].get('clerp') != "" else str(node) for node in tuple_node)
    # )

if __name__ == "__main__":
    prompts = "<bos> <start_of_turn> user ⏎ Answer immediately: what's the capital of the state containing Dallas? <end_of_turn> ⏎ <start_of_turn> model ⏎"
    mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    generate_subgraphs(graph_path = 'demos/graph_files/dallas-austin_gemma3.json', out_path = 'demos/subgraphs/dallas-austin_gemma3', top_k=5, edge_threshold=0.6, include_clerp = False, mask=None)
    # graph_dir = "graph_files"
    # out_dir = "subgraphs"
    # os.makedirs(out_dir, exist_ok=True)
    # top_k = 10
    # for graph_file in os.listdir(graph_dir):
    #     print(f"Processing {graph_file}...")
    #     graph_path = os.path.join(graph_dir, graph_file)
    #     name = graph_file.split('.')[0]
    #     out_path=os.path.join(out_dir, name)
    #     edge_threshold = 0.3
    #     if os.path.exists(out_path + f"_k_{top_k}_e_{edge_threshold}.png"):
    #         print(f"Subgraph for {graph_file} already exists, skipping...")
    #         continue
    #     mask = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    #     generate_subgraphs(out_path=out_path, graph_path=graph_path, top_k=top_k, edge_threshold=edge_threshold, mask=mask)