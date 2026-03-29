import os
import json
import re
from typing import Any, Dict, List, Sequence, Tuple, Union

from google import genai
from google.genai import types
from circuit_tracer.subgraph.config import get_env

# - Input: features that activate on a single narrow concept
# - Abstract: features that activate on higher level concept
# - Output: features that promote certain tokens
# - Trash: syntactic features or not relevant features

SYSTEM_PROMPT = """You are a meticulous AI researcher. You are experimenting on a language model, which you find several features activating inside the model for a given prompt. Your job is to group similar activating features together.

You are given a list of features activating on a prompt. These features are 
classify into four types: Input feature, Abstract feature, Output feature

They will be written in the format {node_id}-{label}-{type}, where node_id is a unique identifier for the feature, label is a short description of the feature, and type is one of the types above.

Cluster these features into groups with similar meaning within the context of the prompt. Do not group features of different types together.
Label each group with a short description, based on the labels of the features in that group.


Write the result as a JSON list of objects, where each object has:
- "label": the group label in string format
- "type": the feature type Input feature, Abstract feature, Output feature
- "node_ids": list of node_id strings belonging to this group

Return ONLY the JSON list, no other text."""


def _get_gemini_api_key() -> str:
    # Runtime lookup (not cached at import time)
    return (get_env("GEMINI_API_KEY") or get_env("GENAI_API_KEY") or "").strip()


# def _normalize_prompts(prompts: Union[str, Sequence[str], None]) -> str:
#     if prompts is None:
#         return ""
#     if isinstance(prompts, str):
#         return prompts.strip()
#     cleaned = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
#     return "\n".join(cleaned)


def _format_features_for_prompt(
    node_ids: List[str],
    labels: List[str],
    feature_types: List[str],
    prompt: str,
) -> str:
    """Format feature list into a readable string for the LLM."""
    lines: List[str] = []
    lines.append(f"Prompt: {prompt}")
    lines.append("Features:")
    for i, nid in enumerate(node_ids):
        lines.append(f"{nid}-{labels[i]}-{feature_types[i]}")
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
    labels: List[str],
    feature_types: List[str],
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 1,
) -> List[Dict[str, Any]]:
    """
    Group features into semantic clusters.

    Args:
        node_ids: list of feature node ids
        labels: mapping node_id -> "label (type)"
        prompts: prompt string or list of prompt strings for context
        model_name: Gemini model name
        temperature: generation temperature

    Returns:
        List[{"label": str, "node_ids": List[str]}]
    """

    api_key = _get_gemini_api_key()
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY (or GENAI_API_KEY) in environment")

    client = genai.Client(api_key=api_key)
    user_message = _format_features_for_prompt(node_ids, labels, feature_types, prompt)

    response = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
        ),
        contents=user_message,
    )
    return _parse_response(response.text)


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


def grouping_pipeline(
    node_ids: List[str],
    labels: List[str],
    feature_types: List[str],
    prompt: str,
) -> List[Dict[str, Any]]:
    """Pipeline entrypoint: input node_ids + labels + prompts -> clusters."""
    return group_features(node_ids=node_ids, labels=labels, feature_types=feature_types, prompt=prompt)


if __name__ == "__main__":
    # print(bool(_get_gemini_api_key()))
    node_ids = ["16_89970_9", "20_44686_9", "20_44686_10", "20_20300_10", "22_11998_10"]
    labels = [
        "Texas",
        "Texas",
        "capital",
        "Texas related",
        "say Austin",
    ]
    feature_types = [
        "input",
        "abstract",
        "abstract",
        "abstract",
        "output",
    ]
    prompt = "Fact: The capital of the state containing Texas is"
    print(grouping_pipeline(node_ids, labels, feature_types, prompt))