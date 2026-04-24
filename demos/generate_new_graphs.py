from api import generate_graph
import os
import time
import json
import re
from pathlib import Path
from typing import Optional
import requests

# configure defaults used for API calls
DEFAULT_MODEL = "gemma-2-2b"
DEFAULT_SOURCE_SET = "clt-hp"
# pause between requests to avoid rate limits
REQUEST_DELAY = 1.0
# number of times to retry downloading the saved s3 json
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 30.0
PROMPTS_FILE = Path(__file__).with_name("prompts2.txt")
OUTPUT_DIR = Path(__file__).with_name("temp_graph_files")
SOURCE_SETS = ("clt-hp", "gemmascope-transcoder-16k")
MAX_PROMPTS = 10


def fetch_json_url(url: str, retries: int = DOWNLOAD_RETRIES) -> dict:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == retries:
                raise
            time.sleep(1.0)
    raise RuntimeError("unreachable")

def download_graph(
    prompt: Optional[str] = None,
    slug: Optional[str] = None,
    out_path: Optional[str] = None,
    model_id: str = DEFAULT_MODEL,
    source_set: str = DEFAULT_SOURCE_SET,
    desired_logit_prob: float = 0.99,
    edge_threshold: float = 0.98,
    max_feature_nodes: int = 10000,
    max_n_logits: int = 15,
    node_threshold: float = 0.95,
):
    if not prompt or not slug or not out_path:
        raise ValueError("Missing prompt, slug, or output path.")

    print(f"requesting graph for slug='{slug}' prompt='{prompt[:64]}...'")
    status, body = generate_graph(
        model_id,
        prompt,
        slug,
        source_set,
        desiredLogitProb=desired_logit_prob,
        edgeThreshold=edge_threshold,
        maxFeatureNodes=max_feature_nodes,
        maxNLogits=max_n_logits,
        nodeThreshold=node_threshold,
    )

    # API returned non-200? still try to parse body for s3url
    try:
        info = json.loads(body)
    except Exception:
        print(f"failed to parse API response for '{slug}', status={status}")
        print(body[:200])
        time.sleep(REQUEST_DELAY)
        return

    s3url = info.get("s3url") or info.get("s3Url") or info.get("url")
    if not s3url:
        print(f"No s3url returned for '{slug}': {info}")
        time.sleep(REQUEST_DELAY)
        return

    print(f"downloading graph json from {s3url}")
    try:
        data = fetch_json_url(s3url)
    except Exception as exc:
        print(f"failed to download {s3url}: {exc}")
        time.sleep(REQUEST_DELAY)
        return

    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(data, out_f, indent=2, ensure_ascii=False)

    print(f"saved {slug} -> {out_path} (nodes={len(data.get('nodes', []))})")
    time.sleep(REQUEST_DELAY)
    # break


def slugify_prompt(prompt: str, index: int, source_set: str) -> str:
    trimmed = re.sub(r"\s+", " ", prompt.strip()).lower()
    core = re.sub(r"[^a-z0-9]+", "-", trimmed).strip("-")
    if core:
        words = [w for w in core.split("-") if w]
        core = "-".join(words[:4])
    else:
        core = f"prompt-{index + 1:02d}"
    timestamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}"
    return f"{source_set}-p{index + 1:02d}-{core}-{timestamp}"


def load_prompts(path: Path, limit: int) -> list[str]:
    with path.open("r", encoding="utf-8") as in_f:
        prompts = []
        for line in in_f:
            text = line.strip()
            if not text:
                continue
            words = text.split()
            # Remove the final token (the target word) before generation.
            truncated = " ".join(words[:-1]).strip() if len(words) > 1 else ""
            if truncated:
                prompts.append(truncated)
    return prompts[:limit]


if __name__ == "__main__":
    prompts = load_prompts(PROMPTS_FILE, MAX_PROMPTS)
    if not prompts:
        raise ValueError(f"No prompts found in {PROMPTS_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for source_set in SOURCE_SETS:
        source_out_dir = OUTPUT_DIR / source_set
        source_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating {len(prompts)} graphs for source_set='{source_set}'")
        for idx, prompt in enumerate(prompts):
            slug = slugify_prompt(prompt, idx, source_set)
            out_path = source_out_dir / f"{slug}.json"
            download_graph(
                prompt=prompt,
                slug=slug,
                out_path=str(out_path),
                source_set=source_set,
                node_threshold=0.95,
                edge_threshold=0.98,
                desired_logit_prob=0.99,
                max_feature_nodes=10000,
                max_n_logits=15,
            )

