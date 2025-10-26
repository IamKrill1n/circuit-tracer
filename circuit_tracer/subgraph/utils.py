import json
import torch

def get_adj_from_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

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
    
    return adj_matrix, node_ids, attr
