import os
import json
import re
from typing import Any, Dict, List, Tuple, Optional

from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a meticulous AI researcher. You are experimenting on a language model, which you find several features activating inside the model for a given prompt. Your job is to group similar activating features together.

You are given a list of features activating on a prompt. These features are 
classify into these types:
- Input: features that activate on a single narrow concept
- Abstract: features that activate on higher level concept
- Output: features that promote certain tokens
- Trash: syntactic features or not relevant features

Cluster these features into groups with similar meaning within the context of the prompt. Do not group features of different types together. Label each group a short name in the format {Features' type}: {name}.

Write the result as a JSON list of objects, where each object has:
- "label": the group label string in format "{type}: {name}"
- "node_ids": list of node_id strings belonging to this group

Return ONLY the JSON list, no other text."""


def _get_gemini_key() -> str:
    return str(os.getenv("GEMINI_API_KEY", ""))


def _format_features_for_prompt(
    node_ids: List[str],
    attr: Dict[str, Any],
    prompt: str = "",
) -> str:
    """Format feature list into a readable string for the LLM."""
    lines = []
    if prompt:
        lines.append(f"Prompt: {prompt!r}\n")
    lines.append("Features:")
    lines.append("-" * 60)

    for nid in node_ids:
        a = attr.get(nid, {})
        ftype = a.get("feature_type", "unknown")

        # Skip embeddings and logits — they are not features to cluster
        if ftype in ("embedding", "logit"):
            continue

        clerp = a.get("clerp", "")
        layer = a.get("layer", "?")
        ctx_idx = a.get("ctx_idx", "?")
        activation = a.get("activation", 0.0)

        line = f"  node_id: {nid}"
        line += f"  | layer: {layer}"
        line += f"  | pos: {ctx_idx}"
        if activation:
            line += f"  | act: {activation:.3f}"
        if clerp:
            line += f"  | description: {clerp}"
        lines.append(line)

    return "\n".join(lines)


def _parse_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse the JSON list from Gemini's response."""
    # Try to extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
    if match:
        response_text = match.group(1).strip()

    # Try to find a JSON array
    match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if match:
        response_text = match.group(0)

    try:
        clusters = json.loads(response_text)
        if not isinstance(clusters, list):
            raise ValueError("Expected a JSON list")
        return clusters
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Gemini response as JSON: {e}\nResponse: {response_text}")


def group_features(
    node_ids: List[str],
    attr: Dict[str, Any],
    prompt: str = "",
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Use Gemini to cluster feature nodes into semantic groups.

    Args:
        node_ids:    list of node_id strings (features to cluster)
        attr:        per-node attribute dicts (must contain clerp for meaningful results)
        prompt:      the original model prompt (for context)
        model_name:  Gemini model to use
        temperature: sampling temperature

    Returns:
        List of dicts, each with:
          - "label": str, e.g. "Input: dog-related concepts"
          - "node_ids": List[str], node IDs in this group
    """
    api_key = _get_gemini_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")

    client = genai.Client(api_key=api_key)

    user_message = _format_features_for_prompt(node_ids, attr, prompt)

    response = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
        ),
        contents=user_message,
    )

    clusters = _parse_response(response.text)
    return clusters


def clusters_to_supernodes(clusters: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Convert cluster dicts from group_features into the supernode format
    expected by save_subgraph: [[label, node_id1, node_id2, ...], ...]

    Args:
        clusters: output of group_features

    Returns:
        List of supernodes, each is [label, node_id, node_id, ...]
    """
    supernodes = []
    for cluster in clusters:
        label = cluster.get("label", "unknown")
        nids = cluster.get("node_ids", [])
        if len(nids) >= 2:  # only create supernodes with 2+ members
            supernodes.append([label] + nids)
    return supernodes


if __name__ == "__main__":
    from circuit_tracer.subgraph.utils import get_data_from_json, get_clerp

    json_path = "demos/graph_files/dallas-austin_gemma3.json"
    _, node_ids, attr, metadata = get_data_from_json(json_path)

    # Fetch clerp descriptions for meaningful grouping
    get_clerp(metadata, attr)

    prompt = metadata.get("prompt", "")

    # Filter to feature nodes only
    feature_ids = [
        nid for nid in node_ids
        if attr.get(nid, {}).get("feature_type", "")
        not in ("embedding", "logit", "mlp reconstruction error", "")
    ]

    print(f"Clustering {len(feature_ids)} features...")
    clusters = group_features(feature_ids, attr, prompt=prompt)

    for c in clusters:
        print(f"\n{c['label']}:")
        for nid in c["node_ids"]:
            clerp = attr.get(nid, {}).get("clerp", "")
            print(f"  {nid}: {clerp[:70]}")

    # Convert to supernodes format for save_subgraph
    supernodes = clusters_to_supernodes(clusters)
    print(f"\n{len(supernodes)} supernodes ready for save_subgraph")