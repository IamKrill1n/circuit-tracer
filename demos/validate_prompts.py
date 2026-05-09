"""
Validate prompt files for gemma-2-2b compatibility.

Keeps only prompts where:
  - model top-1 prediction matches the expected target (first subtoken)
  - model confidence is in (LOWER, UPPER) — avoids trivial and impossible completions

Usage:
  conda run -n circuit python demos/validate_prompts.py demos/prompts.txt
  conda run -n circuit python demos/validate_prompts.py demos/prompts2.txt
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOWER, UPPER = 0.05, 0.30
DEFAULT_MODEL = "google/gemma-2-2b"


def load_prompts(path: Path) -> list[tuple[str, str]]:
    """Split each line into (prefix, target_word) by stripping the last whitespace-delimited token."""
    pairs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        prefix, _, target = line.rpartition(" ")
        pairs.append((prefix, target))
    return pairs


def first_token_id(tokenizer, word: str) -> int:
    """Token id of `word` as it appears mid-sentence (leading space encodes word boundary)."""
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0]


@torch.inference_mode()
def validate(prompt_file: Path, model_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    pairs = load_prompts(prompt_file)
    kept = []

    for prefix, target in pairs:
        target_tid = first_token_id(tokenizer, target)
        inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
        logits = model(**inputs).logits[0, -1]  # (vocab,)
        probs = logits.softmax(-1)

        top1_id = logits.argmax().item()
        top1_str = tokenizer.decode([top1_id]).strip()
        p = probs[target_tid].item()

        top1_matches = top1_id == target_tid
        in_range = LOWER < p < UPPER
        keep = top1_matches and in_range

        tag = "KEEP" if keep else "DROP"
        reasons = []
        if not top1_matches:
            reasons.append(f"top1={top1_str!r}")
        if not in_range:
            reasons.append(f"p={p:.3f}")
        note = f"  [{'; '.join(reasons)}]" if reasons else ""
        print(f"[{tag}] {prefix!r} → {target!r}{note}")

        if keep:
            kept.append(f"{prefix} {target}")

    out = prompt_file.with_stem(prompt_file.stem + "_validated")
    out.write_text("\n".join(kept) + "\n")
    print(f"\n{len(kept)}/{len(pairs)} kept → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_file", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    validate(args.prompt_file, args.model)
