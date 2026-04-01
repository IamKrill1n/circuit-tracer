from circuit_tracer.subgraph.api import get_feature
from circuit_tracer.subgraph.config import get_env
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

def heuristic_classify(clerp, top_tokens, top_next_tokens, top_logits):
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
        if token not in next_token_counts and token != "":
            next_token_counts[token] = 0
        next_token_counts[token] += 1
    most_common_next_token, count = max(next_token_counts.items(), key=lambda x: x[1])
    if count > 0.7 * len(top_next_tokens):
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
    attr: Dict[str, Any],
    metadata: Dict[str, Any]
):
    """Classify features into different types
    Args:
        node_ids: List of node IDs to classify
        attr: Dictionary of node attributes
        metadata: Dictionary of graph metadata
    """

    feature_type = {}
    modelId = metadata.get("scan", "")
    source_set = metadata['info'].get("neuronpedia_source_set", "")
    # Get info for each node and classify
    for node in node_ids:
        if attr[node].get("feature_type") == 'embedding':
            feature_type[node] = "embedding"
            continue
        elif attr[node].get("feature_type") == 'logit':
            feature_type[node] = "logit"
            continue

        # node_ids are in the format {layer}_{node_id}_{ctx_id}
        layer, index = node.split("_")[:2]
        index = int(index)
        layer = layer + "-" + source_set
        # print(layer)
        status, data = get_feature(modelId=modelId, layer=layer, index=index)
        if status != 200:
            print(f"Failed node={node} modelId={modelId} layer={layer} status={status} body={data[:200]}")
            continue

        json_data = json.loads(data)
        clerp = json_data.get("explanations", [])[0].get("description", "") 
        act_density = json_data.get("frac_nonzero", 0)
        # remove too frequently activated features
        if act_density > 0.1:
            feature_type[node] = "trash"
            continue
        top_activations = json_data.get("activations", [])[:10]
        top_tokens = [prompt['tokens'][prompt['maxValueTokenIndex']] for prompt in top_activations]
        top_next_tokens = [prompt['tokens'][prompt['maxValueTokenIndex'] + 1] if prompt['maxValueTokenIndex'] + 1 < len(prompt['tokens']) else "" for prompt in top_activations]
        
        top_logits = json_data.get("pos_str", [])

        feature_type[node] = heuristic_classify(clerp, top_tokens, top_next_tokens, top_logits)

        # print("Clerp: ", clerp)
        # print("Top tokens: ", top_tokens)
        # print("Top next tokens: ", top_next_tokens)
        # print("Activation density: ", act_density)

    return feature_type


# ---------------------------------------------------------------------------
# LLM-based classification using OpenAI
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = """\
You are an expert AI interpretability researcher classifying features from a \
language model's attribution graph.

## Task
Given a list of features, classify each one into exactly one category, or mark \
it as "trash" if it should be discarded.

## Categories
- **Input**: Detects specific input tokens, character patterns, or low-level \
syntactic properties. Acts as a surface-level detector (e.g. "the word Dallas", \
"uppercase letter", "token at position 3").
- **Abstract**: Represents a high-level semantic concept, topic, or abstract \
relationship that helps the model reason (e.g. "Texas geography", \
"capital cities", "US states").
- **Output**: Promotes or steers specific next tokens when activated \
(e.g. "say Austin", "predict city name").
- **trash**: Discard this feature. Use for:
  - Pure syntactic/positional artifacts unrelated to meaning
  - Features with no clear semantic content
  - Irrelevant or noise features that do not contribute to the reasoning chain

## Input format
```json
{
  "features": [
    {"id": "<node_id>", "label": "<feature description>", "hint": "<raw feature type>"}
  ]
}
```

## Output format (strict JSON only — no markdown, no explanation)
```json
{"<node_id>": "Input" | "Abstract" | "Output" | "trash", ...}
```

Rules:
1. Every feature id must appear in the output.
2. Output must be a single valid JSON object mapping id -> category.
3. Be conservative with "trash" — only discard clearly syntactic or irrelevant features.
"""

_VALID_CLASSES = {"Input", "Abstract", "Output", "trash"}


def _get_openai_api_key() -> str:
    return (get_env("OPENAI_API_KEY") or "").strip()


def classify_features_with_llm(
    node_ids: List[str],
    labels: Dict[str, str],
    feature_types: Dict[str, str],
    model: str = "gpt-4o",
    temperature: float = 0.1,
) -> Dict[str, str]:
    """Classify features into Input / Abstract / Output using an OpenAI LLM.

    This replaces or supplements the heuristic ``classify_features`` step.
    The ``frac_nonzero < 10%`` density filter should already have been applied
    upstream (inside ``classify_features``); the LLM additionally discards
    syntactic or irrelevant features.

    Args:
        node_ids: Ordered list of node IDs to classify (only CLT / feature
            nodes; embedding and logit nodes should be excluded).
        labels: Mapping from node_id to its human-readable clerp description.
        feature_types: Mapping from node_id to its raw feature-type hint
            (e.g. "cross layer transcoder", "input", "abstract", …).  Used as
            a soft hint for the LLM.
        model: OpenAI model name (default "gpt-4o").
        temperature: Sampling temperature (low values give more deterministic
            results; default 0.1).

    Returns:
        Dict mapping node_id -> "Input" | "Abstract" | "Output".
        Nodes classified as "trash" are **omitted** from the result.

    Raises:
        ValueError: If the OpenAI API key is not set or the response cannot be
            parsed.
    """
    try:
        from openai import OpenAI  # lazy import to keep the module loadable
    except ImportError as exc:
        raise ImportError(
            "openai package is required for classify_features_with_llm. "
            "Install it with: pip install openai"
        ) from exc

    api_key = _get_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )

    if not node_ids:
        return {}

    # Build the feature list for the LLM
    features = [
        {
            "id": nid,
            "label": labels.get(nid, ""),
            "hint": feature_types.get(nid, ""),
        }
        for nid in node_ids
    ]
    user_message = json.dumps({"features": features}, ensure_ascii=False, indent=2)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content or ""
    result = _parse_classify_response(raw, node_ids)
    return result


def _parse_classify_response(
    response_text: str,
    node_ids: List[str],
) -> Dict[str, str]:
    """Parse the LLM JSON response and return only the valid, non-trash entries."""
    # Strip optional markdown fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
    if match:
        response_text = match.group(1).strip()

    # Extract the first JSON object
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        response_text = match.group(0)

    data = json.loads(response_text)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object from the LLM, got: {type(data).__name__}"
        )

    node_id_set = set(node_ids)
    out: Dict[str, str] = {}
    for node_id, label in data.items():
        if node_id not in node_id_set:
            continue
        label = str(label).strip()
        if label not in _VALID_CLASSES:
            # Try case-insensitive normalisation
            normalised = label.capitalize()
            if normalised not in _VALID_CLASSES:
                continue
            label = normalised
        if label == "trash":
            continue
        out[node_id] = label
    return out


# if __name__ == "__main__":
#     # print(normalize_text("_Dallas"))
#     node_ids = ["25_9974_1"]
#     modelId = "gemma-2-2b"
#     source_set = "clt-hp"
#     # source_set = "gemmascope-transcoder-16k"
#     feature_types = classify_features(node_ids, attr={"25_9974_1": {"feature_type": "cross layer transcoder"}}, metadata={"scan": modelId, "neuronpedia_source_set": source_set})
#     print(feature_types)