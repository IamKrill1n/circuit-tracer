from circuit_tracer.subgraph.api import get_feature
from typing import List, Dict, Any
import json
import re

def normalize_text(text):
    """Normalize text by lowercasting and remove special characters.
    Args:
        text: Input text string
    Returns:
        Normalized text string
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def heuristic_classify(node_id: str, clerp, top_tokens, top_next_tokens, top_logits, act_density):
    """Classify a single feature to 3 types: input, abstract, output
    Input features represent tokens or other low-level properties of the input, usually act as detectors
    Output features promote certain next tokens when activated eg say something
    Abstract features repressent high level concepts

    Args:
        node_id: Node ID of the feature
        clerp: autointerp explanation for the feature
        top_tokens: List of top tokens activating the feature
        top_next_tokens: List of top next tokens following the activations
        top_logits: List of top logits promoted by the feature
        act_density: Activation density of the feature
    Returns:
        A string representing the classified type of the feature
    """
    

    # Normalize all text
    clerp = normalize_text(clerp)
    top_tokens = [normalize_text(token) for token in top_tokens]
    top_next_tokens = [normalize_text(token) for token in top_next_tokens]
    top_logits = [normalize_text(logit) for logit in top_logits]
    
    # Heuristic rules for classification
    # If clerp is the same as top tokens, it's likely an input feature
    # If clerp starts with "say ", it's an output feature
    # If top next tokens are the same then it's likely an output feature
    # If top logits are similar then it's likely and output feature
    # Otherwise classify it as an abstract feature

    if clerp.startswith("say "):
        return "output"

    # Count the number of clerp in top tokens
    clerp_in_top_tokens = sum([(clerp == token) for token in top_tokens])
    if clerp_in_top_tokens > 0.9 * len(top_tokens):
        return "input"

    # Count the number of same tokens in top next tokens
    next_token_counts = {}
    for token in top_next_tokens:
        if token not in next_token_counts:
            next_token_counts[token] = 0
        next_token_counts[token] += 1
    most_common_next_token, count = max(next_token_counts.items(), key=lambda x: x[1])
    if count > 0.5 * len(top_next_tokens):
        return "output"

    # Count the number of same logits in top logits
    logit_counts = {}
    for logit in top_logits:
        if logit not in logit_counts:
            logit_counts[logit] = 0
        logit_counts[logit] += 1
    most_common_logit, count = max(logit_counts.items(), key=lambda x: x[1])
    if count >= 3:
        return "output"

    return "abstract"



def classify_features(
    node_ids: List[str],
    modelId: str,
    source_set: str
):
    """Classify features into different types
    Args:
        node_ids: List of node IDs to classify
        modelId: Model identifier (e.g., "gemma-2-2b")
        attr: Additional attributes
    """

    feature_type = {}

    # Get info for each node and classify
    for node in node_ids:
        # node_ids are in the format {layer}_{node_id}_{ctx_id}
        layer, index = node.split("_")[:2]
        index = int(index)
        if modelId == "gemma-2-2b":
            layer = layer + "-" + source_set
        else:
            raise ValueError(f"Unsupported modelId: {modelId}")
        
        status, data = get_feature(modelId = modelId, layer=layer, index=index)
        if status != 200:
            print(f"Failed to get feature for node {node}: {data}")
            continue

        json_data = json.loads(data)
        clerp = json_data.get("explanations", [])[0].get("description", "") 
        act_density = json_data.get("frac_nonzero", 0)
        # remove too frequently activated features
        if act_density > 0.1:
            continue
        top_activations = json_data.get("activations")[:10]
        top_tokens = [prompt['tokens'][prompt['maxValueTokenIndex']] for prompt in top_activations]
        top_next_tokens = [prompt['tokens'][prompt['maxValueTokenIndex'] + 1] for prompt in top_activations]
        top_logits = json_data.get("pos_str", [])

        feature_type[node] = heuristic_classify(node, clerp, top_tokens, top_next_tokens, top_logits, act_density)

        # print("Clerp: ", clerp)
        # print("Top tokens: ", top_tokens)
        # print("Top next tokens: ", top_next_tokens)
        # print("Activation density: ", act_density)

    return feature_type

    


if __name__ == "__main__":
    # print(normalize_text("_Dallas"))
    node_ids = ["25_9974_1"]
    modelId = "gemma-2-2b"
    # source_set = "clt-hp"
    source_set = "gemmascope-transcoder-16k"
    feature_types = classify_features(node_ids, modelId, source_set)
    print(feature_types)