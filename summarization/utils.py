import json
import time
import torch
from api import get_feature, generate_autointerp
import networkx as nx
from typing import Any, Literal, NamedTuple

from summarization.supernode_graph import Node


def _node_from_json_dict(raw: dict) -> Node:
    nid = str(raw.get("node_id", ""))
    node_idx = int(raw.get("node_idx", -1))
    act_raw = raw.get("activation")
    activation = float(act_raw) if isinstance(act_raw, (int, float)) else None
    inf_raw = raw.get("influence")
    influence = float(inf_raw) if isinstance(inf_raw, (int, float)) else None
    rel_raw = raw.get("relevance")
    relevance = float(rel_raw) if isinstance(rel_raw, (int, float)) else None
    return Node(
        node_id=nid,
        node_idx=node_idx,
        feature=int(raw.get("feature", 0)),
        layer=str(raw.get("layer", "")),
        ctx_idx=int(raw.get("ctx_idx", 0)),
        feature_type=str(raw.get("feature_type", "")),
        token_prob=float(raw.get("token_prob", 0.0)),
        is_target_logit=bool(raw.get("is_target_logit", False)),
        run_idx=int(raw.get("run_idx", 0)),
        reverse_ctx_idx=int(raw.get("reverse_ctx_idx", 0)),
        jsNodeId=str(raw.get("jsNodeId", "") or nid),
        clerp=str(raw.get("clerp", "")),
        influence=influence,
        activation=activation,
        relevance=relevance,
    )


def get_data_from_json(json_path: str):
    # Explicit UTF-8 avoids UnicodeDecodeError on Windows (default locale is often cp1252).
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    raw_nodes = data.get("nodes", [])
    links = data.get("links", [])
    node_ids = [n["node_id"] for n in raw_nodes]
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

    nodes = [_node_from_json_dict(n) for n in raw_nodes]
    for idx, node in enumerate(nodes):
        if node.node_idx < 0:
            node.set_node_idx(idx)

    return adj_matrix, nodes, metadata


def node_is_embedding(node: Node) -> bool:
    if node.node_id.startswith("E"):
        return True
    return "embedding" in str(node.feature_type).lower()


def node_is_logit(node: Node) -> bool:
    if node.is_target_logit:
        return True
    return "logit" in str(node.feature_type).lower()


def node_is_fixed(node: Node) -> bool:
    return node_is_embedding(node) or node_is_logit(node)


def layer_index_from_node(node: Node) -> int:
    """Layer index from a typed ``Node`` (same fallbacks as historical JSON attr parsing)."""
    layer_val = node.layer
    if isinstance(layer_val, int):
        return layer_val
    if isinstance(layer_val, str) and layer_val.isdigit():
        return int(layer_val)
    nid = node.node_id
    if nid.startswith("E"):
        return -1
    try:
        return int(nid.split("_")[0])
    except (ValueError, IndexError):
        return 10_000


def layer_index_from_node_id(node_id: str, *, layer: str | int | None = None) -> int:
    """Layer index when only an id string is known (optional ``layer`` from metadata)."""
    if isinstance(layer, int):
        return layer
    if isinstance(layer, str) and layer.isdigit():
        return int(layer)
    if node_id.startswith("E"):
        return -1
    try:
        return int(node_id.split("_")[0])
    except (ValueError, IndexError):
        return 10_000


def _build_index_sets(nodes: list[Node]) -> dict[str, list[int]]:
    sets: dict[str, list[int]] = {
        "feature": [],
        "error": [],
        "embedding": [],
        "logit": [],
        "target_logit": [],
    }
    for i, node in enumerate(nodes):
        ft = node.feature_type
        if node.is_target_logit:
            sets["target_logit"].append(i)
        if node_is_logit(node):
            sets["logit"].append(i)
        elif node_is_embedding(node):
            sets["embedding"].append(i)
        elif ft == "mlp reconstruction error":
            sets["error"].append(i)
        elif ft not in ("embedding", "logit", "mlp reconstruction error", ""):
            sets["feature"].append(i)
    return sets


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


if '__main__' == __name__:
    adj_matrix, nodes, metadata = get_data_from_json("demos/temp_graph_files/austin_clt.json")
    print(metadata)