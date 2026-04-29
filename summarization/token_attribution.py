from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from entmax import sparsemax  # type: ignore[import-not-found]

from summarization.utils import get_data_from_json

NormalizeMethod = Literal["softmax", "sparsemax"]
SPECIAL_TOKEN_RE = re.compile(r"<[^>]+>")
SPARSEMAX_MASK_VALUE = -1e9


def format_prompt(prompt: str) -> str:
    """Deprecated compatibility shim: keep prompt text unchanged."""
    return prompt


def _special_token_mask(prompt_tokens: list[str]) -> torch.Tensor:
    return torch.tensor(
        [bool(SPECIAL_TOKEN_RE.fullmatch(token)) for token in prompt_tokens],
        dtype=torch.bool,
    )


def _normalize_scores(
    values: torch.Tensor,
    method: NormalizeMethod,
    special_mask: torch.Tensor,
) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"Expected 1D token scores, got shape={tuple(values.shape)}")
    if special_mask.ndim != 1 or special_mask.shape[0] != values.shape[0]:
        raise ValueError(
            f"special_mask shape {tuple(special_mask.shape)} must match values shape {tuple(values.shape)}"
        )

    non_special_mask = ~special_mask
    if not non_special_mask.any():
        raise ValueError("All prompt tokens are special tokens; cannot normalize attribution scores.")

    masked_scores = values.to(torch.float32).clone()
    if method == "softmax":
        masked_scores[special_mask] = float("-inf")
        normalized = torch.softmax(masked_scores, dim=0)
    elif method == "sparsemax":
        masked_scores[special_mask] = SPARSEMAX_MASK_VALUE
        normalized = sparsemax(masked_scores, dim=0)
    else:
        raise ValueError(f"Invalid normalize method: {method}")

    normalized = normalized.to(torch.float32)
    normalized[special_mask] = 0.0

    mass = normalized[non_special_mask].sum()
    if not torch.isfinite(mass) or mass <= 0:
        fallback = torch.zeros_like(normalized)
        fallback[non_special_mask] = 1.0 / float(non_special_mask.sum().item())
        return fallback

    normalized = normalized / mass
    normalized[special_mask] = 0.0
    return normalized


@lru_cache(maxsize=8)
def _cached_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=4)
def _cached_model(model_name: str, device: str):
    from transformers import AutoModelForCausalLM

    model: Any = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model


@lru_cache(maxsize=32)
def _cached_logit_target_explainer(model_name: str, target_token_id: int, device: str):
    import shap
    from shap.maskers import Text

    model = _cached_model(model_name, device)
    tokenizer = _cached_tokenizer(model_name)

    def f_logit_target(prompts: list[str] | np.ndarray) -> np.ndarray:
        if isinstance(prompts, np.ndarray):
            prompts_list = [str(item) for item in prompts.tolist()]
        elif isinstance(prompts, list):
            prompts_list = [str(item) for item in prompts]
        else:
            prompts_list = [str(prompts)]
        if not prompts_list:
            return np.array([], dtype=np.float32)

        encoded = tokenizer(
            prompts_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            last_positions = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)
            target_logits = logits[batch_idx, last_positions, int(target_token_id)]
        return target_logits.detach().cpu().numpy().astype(np.float32)

    return shap.Explainer(f_logit_target, masker=Text(tokenizer))


@lru_cache(maxsize=64)
def _cached_prompt_payload_from_graph(graph_path: str) -> tuple[str, tuple[str, ...], int]:
    _adj, _node_ids, attr, metadata = get_data_from_json(graph_path)
    prompt = str(metadata.get("prompt", ""))
    if not prompt:
        raise ValueError(f"Graph metadata does not include prompt: {graph_path}")

    prompt_tokens_raw = metadata.get("prompt_tokens", [])
    if not isinstance(prompt_tokens_raw, list) or not prompt_tokens_raw:
        raise ValueError(f"Graph metadata does not include non-empty prompt_tokens: {graph_path}")
    prompt_tokens = tuple(str(token) for token in prompt_tokens_raw)

    target_token_id = None
    for node_attr in attr.values():
        if node_attr.get("is_target_logit"):
            feature = node_attr.get("feature")
            if feature is None:
                continue
            target_token_id = int(feature)
            break
    if target_token_id is None:
        raise ValueError(f"No is_target_logit node with feature id found in graph: {graph_path}")

    return format_prompt(prompt), prompt_tokens, target_token_id


def _extract_shap_values(raw_explanation: Any) -> torch.Tensor:
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
    prompt_tokens: list[str],
    model_name: str,
    target_token_id: int,
    normalize_method: NormalizeMethod = "sparsemax",
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Compute normalized token attributions using SHAP over target-token logits."""
    device_str = str(device)
    explainer = _cached_logit_target_explainer(model_name, int(target_token_id), device_str)
    shap_values = explainer([format_prompt(prompt)])
    values = _extract_shap_values(shap_values)
    if values.shape[0] != len(prompt_tokens):
        raise ValueError(
            f"SHAP token length mismatch: got {values.shape[0]}, expected {len(prompt_tokens)} from prompt_tokens."
        )
    special_mask = _special_token_mask(prompt_tokens)
    return _normalize_scores(values, normalize_method, special_mask)


def get_token_attribution_from_graph(
    graph_path: str | Path,
    model_name: str,
    normalize_method: NormalizeMethod = "sparsemax",
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Compute normalized SHAP token attributions from graph metadata prompt and target logit."""
    prompt, prompt_tokens, target_token_id = _cached_prompt_payload_from_graph(str(graph_path))
    return get_token_attribution(
        prompt=prompt,
        prompt_tokens=list(prompt_tokens),
        model_name=model_name,
        target_token_id=target_token_id,
        normalize_method=normalize_method,
        device=device,
    )