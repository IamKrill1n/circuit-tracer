import json
import torch
from circuit_tracer.subgraph.api import get_feature
import networkx as nx


def get_data_from_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    nodes = data.get("nodes", [])
    links = data.get("links", [])
    node_ids = [n["node_id"] for n in nodes]
    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    n = len(node_ids)
    adj_matrix = torch.zeros((n, n), dtype=torch.float32)

    for link in links:
        src = link["source"]
        tgt = link["target"]
        weight = link.get("weight", 0.0)
        if src in id_to_index and tgt in id_to_index:
            src_idx = id_to_index[src]
            tgt_idx = id_to_index[tgt]
            adj_matrix[tgt_idx, src_idx] = weight  # Note: row=tgt, col=src for incoming edges

    
    attr = {node["node_id"]: node for node in nodes}

    return adj_matrix, node_ids, attr, metadata

def get_clerp(metadata: dict, attr: dict):
    '''
    Get clerp of all the nodes in the graph.
    Update attr dict in place.
    '''

    source_url = metadata.get("info", {}).get("source_urls", "")[0]
    modelId = source_url.split("/")[-2]
    source_set = source_url.split("/")[-1]
    tokens = metadata.get("prompt_tokens", [])

    node_list = attr.keys()
    for node in node_list:
        clerp = attr[node].get("clerp", "")
        if clerp == "":
            tmp = node.split("_")
            layer = tmp[0]
            index = tmp[1]
            ctx_idx = tmp[2]
            if modelId is not None and layer is not None and index is not None:
                if layer == 'E':
                    attr[node]["clerp"] = "Embedding: " + tokens[int(ctx_idx)]
                    continue
                layer = layer + "-" + source_set
                status, data = get_feature(modelId, layer, index)
                if status == 200:
                    try:
                        feature_info = json.loads(data)
                        clerp = feature_info.get("explanations", "")
                        if clerp != []:
                            clerp = str(clerp[0].get("description", ""))
                        print(clerp)
                        attr[node]["clerp"] = clerp
                    except Exception as e:
                        print(f"Failed to parse feature info for node {node}: {e}")
                else:
                    print(f"Failed to get feature for node {node}: status {status}")
