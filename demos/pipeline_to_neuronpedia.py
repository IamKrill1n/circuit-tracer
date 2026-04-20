from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests
from api import generate_graph, save_subgraph
from summarization.auto_grouping import find_best_k
from summarization.cluster import cluster_graph_with_labels
from summarization.prune import prune_graph_pipeline


def _download_graph_json(
    *,
    prompt: str,
    slug: str,
    out_path: str,
    model_id: str,
    source_set: str,
    desired_logit_prob: float,
    edge_threshold: float,
    max_feature_nodes: int,
    max_n_logits: int,
    node_threshold: float,
) -> str:
    status, body = generate_graph(
        modelId=model_id,
        prompt=prompt,
        slug=slug,
        sourceSetName=source_set,
        desiredLogitProb=desired_logit_prob,
        edgeThreshold=edge_threshold,
        maxFeatureNodes=max_feature_nodes,
        maxNLogits=max_n_logits,
        nodeThreshold=node_threshold,
    )
    if status != 200:
        raise RuntimeError(f"generate_graph failed with status={status}: {body[:300]}")

    payload = json.loads(body)
    s3url = payload.get("s3url") or payload.get("s3Url") or payload.get("url")
    if not s3url:
        raise RuntimeError(f"generate_graph response has no s3url: {payload}")

    resp = requests.get(s3url, timeout=60)
    resp.raise_for_status()
    graph_json = resp.json()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph_json, indent=2), encoding="utf-8")
    return str(out)


def _parse_token_weights(raw: str | None) -> list[float] | None:
    if raw is None or raw.strip() == "":
        return None
    values = json.loads(raw)
    if not isinstance(values, list):
        raise ValueError("--token-weights must be a JSON list, e.g. '[0,0.5,0.5]'")
    return [float(x) for x in values]


def _save_supernodes(supernodes: list[list[str]], out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(supernodes, indent=2), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    graph_json_path = args.graph_json
    if not args.use_existing_graph:
        graph_json_path = _download_graph_json(
            prompt=args.prompt,
            slug=args.slug,
            out_path=args.graph_json,
            model_id=args.model_id,
            source_set=args.source_set,
            desired_logit_prob=args.desired_logit_prob,
            edge_threshold=args.graph_edge_threshold,
            max_feature_nodes=args.max_feature_nodes,
            max_n_logits=args.max_n_logits,
            node_threshold=args.graph_node_threshold,
        )

    token_weights = _parse_token_weights(args.token_weights)

    prune_graph = prune_graph_pipeline(
        json_path=graph_json_path,
        logit_weights=args.logit_weights,
        token_weights=token_weights,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        keep_all_tokens_and_logits=args.keep_all_tokens_and_logits,
        filter_act_density=args.filter_act_density,
        act_density_lb=args.act_density_lb,
        act_density_ub=args.act_density_ub,
    )

    best_k, sweep = find_best_k(
        prune_graph,
        max_layer_span=args.max_layer_span,
        k_min_override=args.k_min,
        k_max_override=args.k_max,
        max_sn=args.max_sn,
    )
    if args.target_k is not None:
        best_k = args.target_k

    supernodes = cluster_graph_with_labels(
        prune_graph,
        target_k=best_k,
        max_layer_span=args.max_layer_span,
        max_sn=args.max_sn,
        gamma=args.gamma,
        mediation_penalty=args.mediation_penalty,
    )

    _save_supernodes(supernodes, args.supernodes_out)

    status, body = save_subgraph(
        modelId=args.model_id,
        slug=args.slug,
        displayName=args.display_name,
        pinnedIds=prune_graph.kept_ids,
        supernodes=supernodes,
        pruningThreshold=args.upload_pruning_threshold,
        densityThreshold=args.upload_density_threshold,
    )

    return {
        "graph_json_path": graph_json_path,
        "best_k": best_k,
        "sweep": sweep,
        "num_pruned_nodes": prune_graph.num_nodes,
        "num_pruned_edges": prune_graph.num_edges,
        "supernodes": supernodes,
        "upload_status": status,
        "upload_body": body,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end prompt -> prune -> group -> upload supernodes pipeline."
    )

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--slug", type=str, required=True, help="Parent graph slug on Neuronpedia")
    parser.add_argument("--display-name", type=str, required=True, help="Name for uploaded subgraph")
    parser.add_argument("--model-id", type=str, default="gemma-2-2b")
    parser.add_argument("--source-set", type=str, default="gemmascope-transcoder-16k")
    parser.add_argument("--graph-json", type=str, default="temp_graph_files/pipeline_graph.json")
    parser.add_argument("--use-existing-graph", action="store_true", help="Skip graph generation and use --graph-json")
    parser.add_argument("--desired-logit-prob", type=float, default=0.99)
    parser.add_argument("--graph-edge-threshold", type=float, default=0.98)
    parser.add_argument("--graph-node-threshold", type=float, default=0.95)
    parser.add_argument("--max-feature-nodes", type=int, default=10000)
    parser.add_argument("--max-n-logits", type=int, default=15)

    parser.add_argument("--logit-weights", type=str, choices=["probs", "target"], default="target")
    parser.add_argument("--token-weights", type=str, default=None, help="JSON list string, e.g. '[0,0,0.5,0.5]'")
    parser.add_argument("--node-threshold", type=float, default=0.5)
    parser.add_argument("--edge-threshold", type=float, default=0.98)
    parser.add_argument("--alpha", type=float, default=0.5, help="Deprecated: use --gamma")
    parser.add_argument("--keep-all-tokens-and-logits", action="store_true")
    parser.add_argument("--filter-act-density", action="store_true")
    parser.add_argument("--act-density-lb", type=float, default=2e-5)
    parser.add_argument("--act-density-ub", type=float, default=0.1)

    parser.add_argument("--target-k", type=int, default=None, help="Override auto-k if set")
    parser.add_argument("--k-min", type=int, default=None)
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--max-layer-span", type=int, default=4)
    parser.add_argument("--max-sn", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.5, help="Deprecated")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--mediation-penalty", type=float, default=0.1)
    parser.add_argument("--supernodes-out", type=str, default="temp_graph_files/supernodes.json")

    parser.add_argument("--upload-pruning-threshold", type=float, default=0.7)
    parser.add_argument("--upload-density-threshold", type=float, default=0.99)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(args)

    print("\n=== Pipeline Summary ===")
    print(f"graph_json_path: {result['graph_json_path']}")
    print(f"pruned nodes: {result['num_pruned_nodes']}")
    print(f"pruned edges: {result['num_pruned_edges']}")
    print(f"best_k: {result['best_k']}")
    print(f"supernodes: {len(result['supernodes'])}")
    print(f"upload status: {result['upload_status']}")
    try:
        print(json.dumps(json.loads(result["upload_body"]), indent=2))
    except Exception:
        print(result["upload_body"])

    print("\n=== Supernodes (for Neuronpedia) ===")
    for sn in result["supernodes"]:
        print(sn)


if __name__ == "__main__":
    main()
