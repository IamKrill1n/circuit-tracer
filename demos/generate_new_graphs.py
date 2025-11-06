from circuit_tracer.api import generate_graph
import os
import time
import json
from typing import Tuple
import requests

GRAPH_DIR = os.path.join(os.path.dirname(__file__), "..", "graph_files")
os.makedirs(GRAPH_DIR, exist_ok=True)

# configure defaults used for API calls
DEFAULT_MODEL = "gemma-2-2b"
DEFAULT_SOURCE_SET = "clt-hp"
# pause between requests to avoid rate limits
REQUEST_DELAY = 1.0
# number of times to retry downloading the saved s3 json
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 30.0


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


def process_prompt_line(prompt: str) -> Tuple[str, str]:
    """Return (prompt, slug). slug is last whitespace-separated token of prompt."""
    p = prompt.strip()
    if not p:
        return "", ""
    parts = p.split()
    slug = parts[-1]
    p = " ".join(parts[:-1])
    return p, slug


def main(
    prompts_path: str = "demos/prompts.txt",
    model_id: str = DEFAULT_MODEL,
    source_set: str = DEFAULT_SOURCE_SET,
    desired_logit_prob: float = 0.95,
    edge_threshold: float = 0.85,
    max_feature_nodes: int = 5000,
    max_n_logits: int = 10,
    node_threshold: float = 0.8,
):
    if not os.path.exists(prompts_path):
        print(f"prompts file not found: {prompts_path}")
        return

    with open(prompts_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for raw in lines:
        prompt, slug = process_prompt_line(raw)
        if not prompt or not slug:
            continue
        slug += f"-{DEFAULT_SOURCE_SET}"
        out_path = os.path.join(GRAPH_DIR, f"{slug}.json")
        if os.path.exists(out_path):
            print(f"skipping {slug}: already exists at {out_path}")
            continue
        
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
            continue

        s3url = info.get("s3url") or info.get("s3Url") or info.get("url")
        if not s3url:
            print(f"No s3url returned for '{slug}': {info}")
            time.sleep(REQUEST_DELAY)
            continue

        print(f"downloading graph json from {s3url}")
        try:
            data = fetch_json_url(s3url)
        except Exception as exc:
            print(f"failed to download {s3url}: {exc}")
            time.sleep(REQUEST_DELAY)
            continue

        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(data, out_f, indent=2, ensure_ascii=False)

        print(f"saved {slug} -> {out_path} (nodes={len(data.get('nodes', []))})")
        time.sleep(REQUEST_DELAY)
        # break


if __name__ == "__main__":
    main()

