"""
Generate attribution graphs and SHAP values for a prompt file, then upload to HuggingFace.

Prompts are first validated against gemma-2-2b: only those where the model's top-1
prediction matches the target word and confidence is in (LOWER, UPPER) are processed.

Usage:
  conda run -n circuit python generate_and_upload.py \
    --prompts-file demos/prompts.txt \
    --hf-repo <owner/repo-name>
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from api import generate_graph
from config import HUGGINGFACE_API_KEY

try:
    from summarization.token_attribution import get_token_attribution
except ImportError:
    get_token_attribution = None  # type: ignore[assignment]

DEFAULT_MODEL_ID = "gemma-2-2b"
DEFAULT_MODEL_NAME = "google/gemma-2-2b"
DEFAULT_SOURCE_SET = "clt-hp"
REQUEST_DELAY = 1.0
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 30.0
# Confidence bounds from validate_prompts.py
LOWER, UPPER = 0.3, 1


# ---------------------------------------------------------------------------
# Prompt loading (from demos/validate_prompts.py)
# ---------------------------------------------------------------------------

def load_prompts(path: Path) -> list[tuple[str, str]]:
    """Return (prefix, target_word) pairs by stripping the last token from each line."""
    pairs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        prefix, _, target = line.rpartition(" ")
        pairs.append((prefix, target))
    return pairs


def _first_token_id(tokenizer, word: str) -> int:
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0]


@torch.inference_mode()
def _passes_validation(
    prefix: str,
    target: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> bool:
    target_tid = _first_token_id(tokenizer, target)
    inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1]
    probs = logits.softmax(-1)
    top1_id = int(logits.argmax().item())
    p = probs[target_tid].item()
    return top1_id == target_tid and LOWER < p < UPPER


# ---------------------------------------------------------------------------
# Graph generation (from generate_new_graphs.py)
# ---------------------------------------------------------------------------

def slugify_prompt(prompt: str, index: int, source_set: str) -> str:
    trimmed = re.sub(r"\s+", " ", prompt.strip()).lower()
    core = re.sub(r"[^a-z0-9]+", "-", trimmed).strip("-")
    words = [w for w in core.split("-") if w] if core else []
    core = "-".join(words[:4]) if words else f"prompt-{index + 1:02d}"
    timestamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}"
    return f"{source_set}-p{index + 1:02d}-{core}-{timestamp}"


def _fetch_json(url: str) -> dict:
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == DOWNLOAD_RETRIES:
                raise
            time.sleep(1.0)
    raise RuntimeError("unreachable")


def _download_graph(
    prompt: str,
    slug: str,
    args: argparse.Namespace,
) -> dict | None:
    """Call Neuronpedia API, download graph JSON into memory. Returns None on failure."""
    print(f"  requesting graph slug='{slug}'")
    try:
        status, body = generate_graph(
            args.model_id,
            prompt,
            slug,
            args.source_set,
            desiredLogitProb=0.99,
            edgeThreshold=0.98,
            maxFeatureNodes=10000,
            maxNLogits=15,
            nodeThreshold=0.95,
        )
    except Exception as exc:
        print(f"  API request failed for '{slug}': {exc}")
        return None

    try:
        info = json.loads(body)
    except Exception:
        print(f"  failed to parse API response for '{slug}', status={status}")
        return None

    s3url = info.get("s3url") or info.get("s3Url") or info.get("url")
    if not s3url:
        print(f"  no s3url for '{slug}': {info}")
        return None

    try:
        data = _fetch_json(s3url)
    except Exception as exc:
        print(f"  failed to download {s3url}: {exc}")
        return None

    print(f"  graph downloaded (nodes={len(data.get('nodes', []))})")
    return data


# ---------------------------------------------------------------------------
# SHAP helpers (from generate_shap_values.py)
# ---------------------------------------------------------------------------

def _prompt_tokens(prompt: str, tokenizer: AutoTokenizer) -> list[str]:
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return [str(tok) for tok in tokenizer.convert_ids_to_tokens(token_ids)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-file", type=Path, required=True,
                        help="Newline-delimited prompt file (last token is target word).")
    parser.add_argument("--hf-repo", required=True,
                        help="HuggingFace dataset repo, e.g. 'username/my-dataset'.")
    parser.add_argument("--source-set", default=DEFAULT_SOURCE_SET)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help="Neuronpedia model ID.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="HuggingFace model name for validation and SHAP.")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--private", action="store_true",
                        help="Create a private HF dataset repo.")
    parser.add_argument("--skip-graphs", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument(
        "--existing-graphs-dir", type=Path, default=None,
        help="Upload pre-existing graph JSONs from this directory instead of generating.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1. Load prompts and validate against the model
    print(f"Loading prompts from {args.prompts_file}")
    pairs = load_prompts(args.prompts_file)
    if args.max_prompts:
        pairs = pairs[: args.max_prompts]

    print(f"Validating {len(pairs)} prompts with {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    valid_pairs: list[tuple[str, str]] = []
    for prefix, target in pairs:
        keep = _passes_validation(prefix, target, model, tokenizer)
        tag = "KEEP" if keep else "DROP"
        print(f"  [{tag}] {prefix!r} → {target!r}")
        if keep:
            valid_pairs.append((prefix, target))

    prompts = [prefix for prefix, _ in valid_pairs]
    print(f"{len(prompts)}/{len(pairs)} prompts passed validation")

    # 2. Set up HF repo
    hf = HfApi(token=HUGGINGFACE_API_KEY)
    hf.create_repo(repo_id=args.hf_repo, repo_type="dataset",
                   exist_ok=True, private=args.private)
    print(f"HF repo ready: {args.hf_repo}")

    # 3. Upload original prompts file
    hf.upload_file(
        path_or_fileobj=args.prompts_file.read_bytes(),
        path_in_repo="prompts.txt",
        repo_id=args.hf_repo,
        repo_type="dataset",
    )

    manifest: list[dict] = []

    # 4. Upload graphs — from existing directory or by generating fresh ones
    if not args.skip_graphs:
        if args.existing_graphs_dir:
            graph_files = sorted(args.existing_graphs_dir.glob("*.json"))
            print(f"\nUploading {len(graph_files)} existing graphs from {args.existing_graphs_dir} ...")
            for idx, path in enumerate(graph_files, start=1):
                slug = path.stem
                hf.upload_file(
                    path_or_fileobj=path.read_bytes(),
                    path_in_repo=f"graphs/{path.name}",
                    repo_id=args.hf_repo,
                    repo_type="dataset",
                )
                manifest.append({"index": idx, "slug": slug})
                print(f"  [{idx}/{len(graph_files)}] uploaded graphs/{path.name}")
        else:
            print(f"\nGenerating {len(prompts)} graphs ...")
            for idx, prompt in enumerate(prompts):
                slug = slugify_prompt(prompt, idx, args.source_set)
                graph_data = _download_graph(prompt, slug, args)
                if graph_data is None:
                    time.sleep(REQUEST_DELAY)
                    continue
                buf = json.dumps(graph_data, indent=2, ensure_ascii=False).encode()
                hf.upload_file(
                    path_or_fileobj=buf,
                    path_in_repo=f"graphs/{slug}.json",
                    repo_id=args.hf_repo,
                    repo_type="dataset",
                )
                manifest.append({"index": idx + 1, "slug": slug, "prompt": prompt})
                print(f"  uploaded graphs/{slug}.json")
                time.sleep(REQUEST_DELAY)

    # 5. Generate SHAP values and upload in-memory
    if not args.skip_shap:
        if get_token_attribution is None:
            print("WARNING: summarization package not available, skipping SHAP.")
        else:
            print(f"\nGenerating SHAP values for {len(prompts)} prompts ...")
            device = str(next(model.parameters()).device)
            shap_rows: list[dict] = []
            for idx, prompt in enumerate(prompts, start=1):
                tokens = _prompt_tokens(prompt, tokenizer)
                raw, _ = get_token_attribution(
                    prompt=prompt,
                    prompt_tokens=tokens,
                    model_name=args.model_name,
                    device=device,
                )
                shap_rows.append({
                    "index": idx,
                    "prompt": prompt,
                    "prompt_tokens": tokens,
                    "raw_shap": raw.detach().cpu().to(torch.float32).tolist(),
                })
                print(f"  [{idx}/{len(prompts)}] computed shap")

            payload = {
                "model_name": args.model_name,
                "n_prompts": len(shap_rows),
                "results": shap_rows,
            }
            buf = json.dumps(payload, ensure_ascii=False, indent=2).encode()
            hf.upload_file(
                path_or_fileobj=buf,
                path_in_repo="shap_values.json",
                repo_id=args.hf_repo,
                repo_type="dataset",
            )
            print("  uploaded shap_values.json")

    # 6. Upload manifest
    buf = json.dumps(manifest, indent=2, ensure_ascii=False).encode()
    hf.upload_file(
        path_or_fileobj=buf,
        path_in_repo="manifest.json",
        repo_id=args.hf_repo,
        repo_type="dataset",
    )
    print(f"\nDone. {len(manifest)} graphs uploaded to {args.hf_repo}")


if __name__ == "__main__":
    main()
