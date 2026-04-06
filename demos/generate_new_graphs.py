from circuit_tracer.subgraph.api import generate_graph
import os
import time
import json
from typing import Tuple
import requests

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

def download_graph(
    prompt: str = None,
    slug: str = None,
    out_path: str = None,
    model_id: str = DEFAULT_MODEL,
    source_set: str = DEFAULT_SOURCE_SET,
    desired_logit_prob: float = 0.99,
    edge_threshold: float = 1,
    max_feature_nodes: int = 10000,
    max_n_logits: int = 15,
    node_threshold: float = 1,
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


if __name__ == "__main__":
    prompt = "abc"
    out_path = "temp_graph_files/abc.json"
    download_graph(prompt=prompt, slug="test-graph-abc", out_path=out_path, node_threshold=0.5, edge_threshold=0.8, desired_logit_prob=0.95, max_feature_nodes=5000, max_n_logits=10)

