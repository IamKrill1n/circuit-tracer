from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import torch
from entmax import sparsemax, entmax15  # type: ignore[import-not-found]

from summarization.utils import get_data_from_json

NormalizeMethod = Literal["softmax", "sparsemax", "entmax15", "relu_l1"]
SPECIAL_TOKEN_RE = re.compile(r"<[^>]+>")
SPARSEMAX_MASK_VALUE = -1e9


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
    elif method == "entmax15":
        masked_scores[special_mask] = SPARSEMAX_MASK_VALUE
        normalized = entmax15(masked_scores, dim=0)
    elif method == "relu_l1":
        pos = torch.clamp(masked_scores, min=0)
        pos[special_mask] = 0
        s = pos.sum()
        normalized = pos / s if s > 0 else torch.ones_like(pos) / pos.shape[0]
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


def _strip_leading_bos_for_shap(
    prompt: str,
    prompt_tokens: list[str],
    tokenizer,
) -> tuple[str, list[str], int]:
    """Drop a leading BOS token from SHAP input while tracking removed prefix length."""
    bos_tok = getattr(tokenizer, "bos_token", None) or ""
    if prompt.startswith("<bos>") and prompt_tokens and prompt_tokens[0] == "<bos>":
        rest = prompt[len("<bos>") :].lstrip()
        return rest, prompt_tokens[1:], 1
    if bos_tok and prompt.startswith(bos_tok) and prompt_tokens and prompt_tokens[0] == bos_tok:
        rest = prompt[len(bos_tok) :].lstrip()
        return rest, prompt_tokens[1:], 1
    return prompt, prompt_tokens, 0


def _apply_shap_notebook_generation_defaults(model: Any) -> None:
    """Align `model.generate` with `token_attribution_compare/shap.ipynb` when unset.

    SHAP's TeacherForcing calls `generate` with `task_specific_params['text-generation']`.
    If that dict is missing or empty, use the same defaults as the comparison notebook
    so the generated target sentence Y matches `shap.Explainer(model, tokenizer)`.
    """
    if getattr(model.config, "is_decoder", None) is not True:
        model.config.is_decoder = True
    if getattr(model.config, "task_specific_params", None) is None:
        model.config.task_specific_params = {}
    tg = model.config.task_specific_params.get("text-generation")
    if not tg:
        model.config.task_specific_params["text-generation"] = {
            "do_sample": True,
            "max_new_tokens": 1,
            "temperature": 1,
            "top_k": 50,
            "no_repeat_ngram_size": 2,
        }


def _build_shap_lm_explainer(
    model_name: str,
    device: str,
    keep_prefix: int | None = None,
):
    """Same construction as ``shap.Explainer(hf_causal_lm, hf_tokenizer)`` for LMs.

    Do not pre-wrap ``TeacherForcing`` — :class:`shap.explainers.Explainer` already
    replaces the HF model with :class:`shap.models.TeacherForcing` and wraps the
    ``Text`` masker in :class:`shap.maskers.OutputComposite` with ``TextGeneration``
    so the explained target ``Y`` matches ``model.generate`` on the **original**
    (unmasked) prompt.

    This matches the notebook's scoring path: **log-odds** of producing the generated
    continuation, not a fixed vocab-id logit from circuit-tracer graphs.
    """
    import shap
    from shap.maskers import Text

    model = _cached_model(model_name, device)
    _apply_shap_notebook_generation_defaults(model)
    tokenizer = _cached_tokenizer(model_name)

    masker = Text(tokenizer, mask_token="...", collapse_mask_token=True)
    if keep_prefix is not None:
        if keep_prefix < 0:
            raise ValueError(f"keep_prefix must be >= 0, got {keep_prefix}")
        masker.keep_prefix = int(keep_prefix)

    return shap.Explainer(model, masker=masker)


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

    return prompt, prompt_tokens, target_token_id


def _extract_shap_values(raw_explanation: Any) -> torch.Tensor:
    """Extract per-input-token SHAP contributions, dropping the leading '' segment.

    The high-level Text + TeacherForcing pipeline returns values of shape
    (n_input_segments, n_output_tokens). We sum across output tokens so each
    input segment gets a single scalar log-odds contribution toward the full
    generated target sentence. The leading segment is the empty '' slot that
    SHAP's Text masker prepends; we drop it to align with `prompt_tokens`.
    """
    explanation = raw_explanation
    # print("explanation:", explanation)
    if isinstance(explanation, list):
        if not explanation:
            raise ValueError("SHAP explainer returned an empty explanation list.")
        explanation = explanation[0]
    values = getattr(explanation, "values", None)
    if values is None:
        raise ValueError("SHAP explanation does not include values.")
    tensor_values = torch.as_tensor(values, dtype=torch.float32).squeeze()
    if tensor_values.ndim == 2:
        tensor_values = tensor_values.sum(dim=-1)
    elif tensor_values.ndim != 1:
        tensor_values = tensor_values.reshape(-1)
    return tensor_values[1:]


def get_token_attribution(
    prompt: str,
    prompt_tokens: list[str],
    model_name: str,
    normalize_method: NormalizeMethod = "sparsemax",
    device: str | torch.device = "cpu",
    *,
    masker_keep_prefix: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same pipeline as ``shap.Explainer(model, tokenizer)`` on an HF causal LM.

    Uses Teacher forcing + log-odds of **generated** ``Y`` (see ``shap.models.TeacherForcing``),
    identical to the high-level SHAP constructor when given ``mask_token='...'`` and
    ``collapse_mask_token=True``.

    Parameters
    ----------
    masker_keep_prefix
        Optional SHAP masker setting to keep the first *k* token segments fixed
        (unmasked) during masking.
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Raw SHAP values and normalized weights, both length ``len(prompt_tokens)``.
    """
    device_str = str(device)
    tokenizer = _cached_tokenizer(model_name)
    work_prompt, work_tokens, n_prefix_tokens_dropped = _strip_leading_bos_for_shap(
        prompt, list(prompt_tokens), tokenizer
    )

    explainer = _build_shap_lm_explainer(
        model_name=model_name,
        device=device_str,
        keep_prefix=masker_keep_prefix,
    )
    try:
        shap_values = explainer([work_prompt], batch_size=1)
    except RuntimeError as exc:
        # SHAP TeacherForcing may fail for keep_prefix > 1 due to batch-shape mismatch.
        # Fall back to standard masking and emulate prefix pinning post-hoc.
        msg = str(exc)
        if masker_keep_prefix is None or "Sizes of tensors must match" not in msg:
            raise
        explainer = _build_shap_lm_explainer(
            model_name=model_name,
            device=device_str,
            keep_prefix=None,
        )
        shap_values = explainer([work_prompt], batch_size=1)
    values = _extract_shap_values(shap_values)
    if values.shape[0] != len(work_tokens):
        raise ValueError(
            f"SHAP token length mismatch: got {values.shape[0]}, expected {len(work_tokens)} from prompt_tokens."
        )

    special_mask = _special_token_mask(work_tokens)
    if masker_keep_prefix is not None and masker_keep_prefix > 0:
        k = min(int(masker_keep_prefix), special_mask.shape[0])
        special_mask = special_mask.clone()
        special_mask[:k] = True
    normalized = _normalize_scores(values, normalize_method, special_mask)
    if n_prefix_tokens_dropped > 0:
        prefix = torch.zeros(
            n_prefix_tokens_dropped,
            dtype=normalized.dtype,
            device=normalized.device,
        )
        normalized = torch.cat([prefix, normalized], dim=0)
        values = torch.cat(
            [prefix.to(dtype=values.dtype, device=values.device), values], dim=0
        )
    return values, normalized


def get_token_attribution_from_graph(
    graph_path: str | Path,
    model_name: str,
    normalize_method: NormalizeMethod = "sparsemax",
    device: str | torch.device = "cpu",
    masker_keep_prefix: int | None = None,
) -> torch.Tensor:
    """Compute normalized SHAP token weights from a graph's prompt metadata.

    Returns the **normalized** weight vector (same length as ``metadata['prompt_tokens']``).
    """
    prompt, prompt_tokens, _target_token_id = _cached_prompt_payload_from_graph(str(graph_path))
    _raw, normalized = get_token_attribution(
        prompt=prompt,
        prompt_tokens=list(prompt_tokens),
        model_name=model_name,
        normalize_method=normalize_method,
        device=device,
        masker_keep_prefix=masker_keep_prefix,
    )
    return normalized

if __name__ == "__main__":
    _gp = "demos/temp_graph_files/clt-hp/clt-hp-p05-fact-the-language-spoken-20260424-171104-257980.json"
    _prompt, _ptok, _ = _cached_prompt_payload_from_graph(_gp)
    raw, normalized = get_token_attribution(
        prompt=_prompt,
        prompt_tokens=list(_ptok),
        model_name="google/gemma-2-2b",
        normalize_method="relu_l1",
        device="cuda",
        masker_keep_prefix=2,
    )
    print("raw:", raw)
    print("normalized:", normalized)