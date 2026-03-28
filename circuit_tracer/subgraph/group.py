import os
import json
import re
from typing import Any, Dict, List, Sequence, Union

from google import genai
from google.genai import types
from circuit_tracer.subgraph.config import GENAI_API_KEY

# - Input: features that activate on a single narrow concept
# - Abstract: features that activate on higher level concept
# - Output: features that promote certain tokens
# - Trash: syntactic features or not relevant features

SYSTEM_PROMPT = """You are a meticulous AI researcher. You are experimenting on a language model, which you find several features activating inside the model for a given prompt. Your job is to group similar activating features together.

You are given a list of features activating on a prompt. These features are 
classify into four types: Detector, Abstractor, Concluder

They will be written in the format {node_id}: {label} ({type}), where node_id is a unique identifier for the feature, label is a short description of the feature, and type is one of the types above.

Cluster these features into groups with similar meaning within the context of the prompt. Do not group features of different types together.
Label each group with a descriptive name that captures the common theme of the features in that group.
For examples, if you have features that all activate on concepts related to cities, you might group them together and label the group "city-related concepts". If you have features that seem to promote certain output tokens, you might label them "say something".

Write the result as a JSON list of objects, where each object has:
- "label": the group label string in format "{type}: {label}"
- "node_ids": list of node_id strings belonging to this group

Return ONLY the JSON list, no other text."""


def _get_gemini_api_key() -> str:
    # runtime lookup (works for both names)
    return GENAI_API_KEY

def _normalize_prompts(prompts: Union[str, Sequence[str], None]) -> str:
    if prompts is None:
        return ""
    if isinstance(prompts, str):
        return prompts.strip()
    cleaned = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
    return "\n".join(cleaned)


def _format_features_for_prompt(
    node_ids: List[str],
    labels: Dict[str, Any],
    prompts: Union[str, Sequence[str], None] = None,
) -> str:
    """Format feature list into a readable string for the LLM."""
    lines: List[str] = []
    prompt_text = _normalize_prompts(prompts)
    if prompt_text:
        lines.append(f"Prompt:\n{prompt_text}\n")
    lines.append("Features:")
    lines.append("-" * 60)
    for nid in node_ids:
        if nid not in labels:
            continue
        lines.append(f"{nid}: {labels[nid]}")
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
    labels: Dict[str, Any],
    prompts: Union[str, Sequence[str], None] = None,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2,
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
    if not node_ids:
        return []
    if not labels:
        raise ValueError("labels cannot be empty")
    missing = [nid for nid in node_ids if nid not in labels]
    if missing:
        raise ValueError(f"labels missing node_ids: {missing[:5]}")

    api_key = _get_gemini_api_key()
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY (or GENAI_API_KEY) in environment")

    client = genai.Client(api_key=api_key)
    user_message = _format_features_for_prompt(node_ids, labels, prompts)

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
    labels: Dict[str, Any],
    prompts: Union[str, Sequence[str], None] = None,
) -> List[Dict[str, Any]]:
    """Pipeline entrypoint: input node_ids + labels + prompts -> clusters."""
    return group_features(node_ids=node_ids, labels=labels, prompts=prompts)


if __name__ == "__main__":
    node_ids = ["16_89970_9", "20_44686_9", "20_44686_10", "22_11998_10"]
    labels = {
        "16_89970_9": "Texas (detector)",
        "20_44686_9": "Texas (abstractor)",
        "20_44686_10": "capital (abstractor)",
        "22_11998_10": "say Austin (concluder)",
    }
    prompts = ["What is the capital of Texas?"]
    print(grouping_pipeline(node_ids, labels, prompts))