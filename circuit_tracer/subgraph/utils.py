import json
import time
import torch
from circuit_tracer.subgraph.api import get_feature, generate_autointerp
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


def get_clerp(metadata: dict, attr: dict, generate_missing: bool = True, retry_delay: float = 1.0):
    '''
    Get clerp of all the nodes in the graph.
    Update attr dict in place.
    
    Args:
        metadata: Graph metadata containing model info and tokens
        attr: Node attribute dictionary to update
        generate_missing: If True, generate auto-interp for features without explanations
        retry_delay: Delay in seconds between API calls to avoid rate limiting
    '''

    source_url = metadata.get("info", {}).get("source_urls", [""])[0]
    if not source_url:
        print("Warning: No source URL found in metadata")
        return
    
    modelId = source_url.split("/")[-2]
    source_set = source_url.split("/")[-1]
    tokens = metadata.get("prompt_tokens", [])

    node_list = list(attr.keys())
    for node in node_list:
        clerp = attr[node].get("clerp", "")
        if clerp != "":
            continue  # Already has a clerp, skip
            
        tmp = node.split("_")
        if len(tmp) < 3:
            print(f"Skipping node {node}: unexpected format")
            continue
            
        layer = tmp[0]
        index = tmp[1]
        ctx_idx = tmp[2]
        
        # Handle embedding nodes
        if layer == 'E':
            if int(ctx_idx) < len(tokens):
                attr[node]["clerp"] = "Embedding: " + tokens[int(ctx_idx)]
            else:
                attr[node]["clerp"] = f"Embedding: [idx={ctx_idx}]"
            continue
        
        # Handle feature nodes - try to get existing explanation
        layer_key = layer + "-" + source_set
        status, data = get_feature(modelId, layer_key, int(index))
        
        if status == 200:
            try:
                feature_info = json.loads(data)
                explanations = feature_info.get("explanations", [])
                if explanations and len(explanations) > 0:
                    clerp = str(explanations[0].get("description", ""))
                    if clerp:
                        print(f"Found existing clerp for {node}: {clerp[:50]}...")
                        attr[node]["clerp"] = clerp
                        continue
            except Exception as e:
                print(f"Failed to parse feature info for node {node}: {e}")
        
        # No existing explanation found - try to generate one
        if generate_missing:
            print(f"No explanation found for {node}, generating auto-interp...")
            time.sleep(retry_delay)  # Rate limiting
            
            gen_status, gen_data = generate_autointerp(
                modelId=modelId,
                layer=layer_key,
                index=int(index),
                explanationModelName="gemini-2.5-flash",
                explanationType="oai_token-act-pair"
            )
            
            if gen_status == 200:
                try:
                    gen_info = json.loads(gen_data)
                    # The response may contain the generated explanation directly
                    # or we may need to fetch the feature again
                    generated_clerp = gen_info.get("description", "")
                    if not generated_clerp:
                        generated_clerp = gen_info.get("explanation", "")
                    
                    if generated_clerp:
                        print(f"Generated clerp for {node}: {generated_clerp[:50]}...")
                        attr[node]["clerp"] = generated_clerp
                        continue
                    
                    # If not in response, try fetching feature again
                    time.sleep(retry_delay)
                    status2, data2 = get_feature(modelId, layer_key, int(index))
                    if status2 == 200:
                        feature_info2 = json.loads(data2)
                        explanations2 = feature_info2.get("explanations", [])
                        if explanations2 and len(explanations2) > 0:
                            clerp2 = str(explanations2[0].get("description", ""))
                            if clerp2:
                                print(f"Fetched newly generated clerp for {node}: {clerp2[:50]}...")
                                attr[node]["clerp"] = clerp2
                                continue
                except Exception as e:
                    print(f"Failed to parse auto-interp response for node {node}: {e}")
            else:
                print(f"Failed to generate auto-interp for {node}: status {gen_status}")
        
        # Fallback: use node ID as clerp
        attr[node]["clerp"] = f"Feature {layer}_{index}"
        print(f"Using fallback clerp for {node}")


