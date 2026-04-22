from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from summarization.utils import get_data_from_json

NormalizeMethod = str


def format_prompt(prompt: str) -> str:
    """Normalize prompts before attribution."""
    return prompt.replace("<bos>", "").strip()


def _normalize_scores(values: torch.Tensor, method: NormalizeMethod) -> torch.Tensor:
    if method == "softmax":
        return torch.softmax(values, dim=0)
    if method == "relu_softmax":
        return torch.softmax(torch.relu(values), dim=0)
    raise ValueError(f"Invalid normalize method: {method}")


@lru_cache(maxsize=8)
def _cached_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=4)
def _cached_model(model_name: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model


@lru_cache(maxsize=4)
def _cached_explainer(model_name: str):
    import shap

    model = _cached_model(model_name)
    tokenizer = _cached_tokenizer(model_name)
    return shap.Explainer(model, tokenizer)


@lru_cache(maxsize=64)
def _cached_prompt_from_graph(graph_path: str) -> str:
    _adj, _node_ids, _attr, metadata = get_data_from_json(graph_path)
    prompt = metadata.get("prompt", "")
    if not prompt:
        raise ValueError(f"Graph metadata does not include prompt: {graph_path}")
    return format_prompt(prompt)


def _extract_shap_values(raw_explanation: Any) -> torch.Tensor:
    """Handle SHAP outputs that may be a single Explanation or a list."""
    explanation = raw_explanation
    if isinstance(explanation, list):
        if not explanation:
            raise ValueError("SHAP explainer returned an empty explanation list.")
        explanation = explanation[0]
    values = getattr(explanation, "values", None)
    if values is None:
        raise ValueError("SHAP explanation does not include values.")
    tensor_values = torch.as_tensor(values, dtype=torch.float32).squeeze()
    if tensor_values.ndim != 1:
        tensor_values = tensor_values.reshape(-1)
    return tensor_values


def get_token_attribution(
    prompt: str,
    model_name: str,
    normalize_method: NormalizeMethod = "softmax",
) -> torch.Tensor:
    """Compute normalized SHAP token attributions for a prompt."""
    explainer = _cached_explainer(model_name)
    shap_values = explainer(format_prompt(prompt))
    values = _extract_shap_values(shap_values)
    return _normalize_scores(values, normalize_method)


def get_token_attribution_from_graph(
    graph_path: str | Path,
    model_name: str,
    normalize_method: NormalizeMethod = "softmax",
) -> torch.Tensor:
    """Compute normalized SHAP token attributions from graph metadata prompt."""
    prompt = _cached_prompt_from_graph(str(graph_path))
    return get_token_attribution(prompt, model_name, normalize_method=normalize_method)