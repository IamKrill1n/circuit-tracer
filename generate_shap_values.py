from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import torch

from summarization.token_attribution import get_token_attribution

DEFAULT_PROMPTS_FILE = Path("demos") / "prompts2.txt"
DEFAULT_OUTPUT_FILE = Path("demos") / "shap_values.json"
DEFAULT_MODEL_NAME = "google/gemma-2-2b"


def _resolve_device(device_flag: str) -> str:
    if device_flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_flag == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested --device cuda, but CUDA is not available.")
    return device_flag


def _load_prompts(path: Path, limit: int | None = None) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as in_f:
        for line in in_f:
            text = line.strip()
            if not text:
                continue
            words = text.split()
            # Match generate_new_graphs.py behavior: drop target word.
            truncated = " ".join(words[:-1]).strip() if len(words) > 1 else ""
            if truncated:
                prompts.append(truncated)
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


@lru_cache(maxsize=8)
def _tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer


def _prompt_tokens(prompt: str, model_name: str) -> list[str]:
    tokenizer = _tokenizer(model_name)
    # Keep tokenization aligned with SHAP Text masker segmentation.
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return [str(tok) for tok in tokenizer.convert_ids_to_tokens(token_ids)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SHAP token attributions for prompts in demos/prompts2.txt "
            "after removing each prompt's final word."
        )
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=DEFAULT_PROMPTS_FILE,
        help="Path to newline-delimited prompts.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON file for SHAP results.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HF model used for SHAP attribution.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for SHAP/model execution.",
    )
    parser.add_argument(
        "--masker-keep-prefix",
        type=int,
        default=None,
        help="Optional SHAP masker keep_prefix value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of prompts to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    prompts = _load_prompts(args.prompts_file, args.limit)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompts_file}")

    output_rows: list[dict] = []
    total = len(prompts)
    for idx, prompt in enumerate(prompts, start=1):
        prompt_tokens = _prompt_tokens(prompt, args.model_name)
        raw, _normalized = get_token_attribution(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            model_name=args.model_name,
            device=device,
            masker_keep_prefix=args.masker_keep_prefix,
        )
        output_rows.append(
            {
                "index": idx,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "raw_shap": raw.detach().cpu().to(torch.float32).tolist(),
            }
        )
        print(f"[{idx}/{total}] computed shap for prompt")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": args.model_name,
        "device": device,
        "masker_keep_prefix": args.masker_keep_prefix,
        "n_prompts": len(output_rows),
        "results": output_rows,
    }
    with args.output_file.open("w", encoding="utf-8") as out_f:
        json.dump(payload, out_f, ensure_ascii=False, indent=2)

    print(f"[done] wrote {len(output_rows)} prompt attributions to {args.output_file}")


if __name__ == "__main__":
    main()
