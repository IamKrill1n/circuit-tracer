from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

from api import generate_graph, save_subgraph
from summarization.auto_grouping import find_best_k
from summarization.cluster import build_supernode_graph, cluster_graph, supernodes_to_mapping
from summarization.cluster_viz import supernode_graph_figure
from summarization.flow_analysis import flow_faithfulness_report
from summarization.prune import prune_graph_pipeline


def _download_graph_json(
    *,
    prompt: str,
    slug: str,
    out_path: str,
    model_id: str,
    source_set: str,
    desired_logit_prob: float,
    graph_edge_threshold: float,
    graph_node_threshold: float,
    max_feature_nodes: int,
    max_n_logits: int,
) -> str:
    status, body = generate_graph(
        modelId=model_id,
        prompt=prompt,
        slug=slug,
        sourceSetName=source_set,
        desiredLogitProb=desired_logit_prob,
        edgeThreshold=graph_edge_threshold,
        maxFeatureNodes=max_feature_nodes,
        maxNLogits=max_n_logits,
        nodeThreshold=graph_node_threshold,
    )
    if status != 200:
        raise RuntimeError(f"generate_graph failed with status={status}: {body[:300]}")

    payload = json.loads(body)
    s3_url = payload.get("s3url") or payload.get("s3Url") or payload.get("url")
    if not s3_url:
        raise RuntimeError(f"generate_graph response has no s3url: {payload}")

    response = requests.get(s3_url, timeout=60)
    response.raise_for_status()
    graph_json = response.json()

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
    return [float(value) for value in values]


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(value) for value in obj]
    if hasattr(obj, "detach"):
        return obj.detach().cpu().tolist()
    return obj


def _save_json(path: str | None, payload: Any) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def _clustered_supernodes_for_upload(clusters: list[list[str]]) -> list[list[str]]:
    """Convert clusters into Neuronpedia upload format [label, member_1, ...]."""
    return [
        [f"cluster_{idx}", *members]
        for idx, members in enumerate(clusters)
        if len(members) > 1
    ]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    graph_json_path = args.graph_json
    if not args.use_existing_graph:
        if not args.prompt:
            raise ValueError("--prompt is required unless --use-existing-graph is set.")
        if not args.slug:
            raise ValueError("--slug is required unless --use-existing-graph is set.")
        graph_json_path = _download_graph_json(
            prompt=args.prompt,
            slug=args.slug,
            out_path=args.graph_json,
            model_id=args.model_id,
            source_set=args.source_set,
            desired_logit_prob=args.desired_logit_prob,
            graph_edge_threshold=args.graph_edge_threshold,
            graph_node_threshold=args.graph_node_threshold,
            max_feature_nodes=args.max_feature_nodes,
            max_n_logits=args.max_n_logits,
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

    sweep: dict[int, dict[str, Any]] = {}
    resolved_k = args.target_k
    if args.auto_k:
        resolved_k, sweep = find_best_k(
            prune_graph=prune_graph,
            max_layer_span=args.max_layer_span,
            k_min_override=args.k_min,
            k_max_override=args.k_max,
            weights={
                "w_intra": args.w_intra,
                "w_dag": args.w_dag,
                "w_attr": args.w_attr,
                "w_size": args.w_size,
            },
            max_sn=args.max_sn,
            mean_method=args.mean_method,
            mediation_penalty=args.mediation_penalty,
            similarity_mode=args.similarity_mode,
            normalization=args.normalization,
            enforce_dag=args.enforce_dag,
            random_state=args.random_state,
            n_init=args.n_init,
        )
    if resolved_k is None:
        resolved_k = 7

    clusters = cluster_graph(
        prune_graph,
        target_k=resolved_k,
        max_layer_span=args.max_layer_span,
        max_sn=args.max_sn,
        mean_method=args.mean_method,
        mediation_penalty=args.mediation_penalty,
        similarity_mode=args.similarity_mode,
        normalization=args.normalization,
        enforce_dag=args.enforce_dag,
        random_state=args.random_state,
        n_init=args.n_init,
    )
    supernode_map = supernodes_to_mapping(prune_graph, clusters)
    sng = build_supernode_graph(prune_graph, supernode_map, enforce_dag=args.enforce_dag)
    flow_report = flow_faithfulness_report(sng, supernode_map)

    labelled_supernodes = _clustered_supernodes_for_upload(clusters)

    _save_json(args.supernodes_out, labelled_supernodes)
    _save_json(args.supernode_map_out, supernode_map)
    _save_json(args.supernode_flow_out, sng)
    _save_json(
        args.auto_k_sweep_out,
        {
            str(k): {
                key: value for key, value in score.items() if key not in ("final_supernodes", "flow_report")
            }
            for k, score in sweep.items()
        },
    )

    figure_path = None
    if args.figure_html_out:
        fig = supernode_graph_figure(
            sng=sng,
            final_supernodes=supernode_map,
            attr=prune_graph.attr,
            title="Summarization supernode graph",
        )
        out = Path(args.figure_html_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), include_plotlyjs="cdn")
        figure_path = str(out)

    upload_status = None
    upload_body = None
    if args.upload:
        if not args.slug or not args.display_name:
            raise ValueError("--upload requires --slug and --display-name.")
        if not 0.0 <= float(args.upload_pruning_threshold) <= 1.0:
            raise ValueError("--upload-pruning-threshold must be between 0 and 1.")
        if not 0.0 <= float(args.upload_density_threshold) <= 1.0:
            raise ValueError("--upload-density-threshold must be between 0 and 1.")
        upload_status, upload_body = save_subgraph(
            modelId=args.model_id,
            slug=args.slug,
            displayName=args.display_name,
            pinnedIds=prune_graph.kept_ids,
            supernodes=labelled_supernodes,
            pruningThreshold=args.upload_pruning_threshold,
            densityThreshold=args.upload_density_threshold,
        )

    return {
        "graph_json_path": graph_json_path,
        "pruned_nodes": prune_graph.num_nodes,
        "pruned_edges": prune_graph.num_edges,
        "resolved_k": resolved_k,
        "auto_k_candidates": len(sweep),
        "supernodes": labelled_supernodes,
        "supernode_map": supernode_map,
        "flow_report": flow_report,
        "figure_html_out": figure_path,
        "upload_status": upload_status,
        "upload_body": upload_body,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate/load graph JSON, prune, cluster, and visualize grouped supernodes."
    )
    parser.add_argument("--prompt", type=str, default=None, help="Prompt used when generating a graph.")
    parser.add_argument("--slug", type=str, default=None, help="Parent graph slug on Neuronpedia.")
    parser.add_argument("--display-name", type=str, default=None, help="Display name used for upload.")
    parser.add_argument("--model-id", type=str, default="gemma-2-2b")
    parser.add_argument("--source-set", type=str, default="gemmascope-transcoder-16k")
    parser.add_argument("--graph-json", type=str, default="temp_graph_files/pipeline_graph.json")
    parser.add_argument("--use-existing-graph", action="store_true")
    parser.add_argument("--desired-logit-prob", type=float, default=0.99)
    parser.add_argument("--graph-edge-threshold", type=float, default=0.98)
    parser.add_argument("--graph-node-threshold", type=float, default=0.95)
    parser.add_argument("--max-feature-nodes", type=int, default=10000)
    parser.add_argument("--max-n-logits", type=int, default=15)

    parser.add_argument("--logit-weights", type=str, choices=["probs", "target"], default="target")
    parser.add_argument("--token-weights", type=str, default=None, help="JSON list string, e.g. '[0,0,0.5,0.5]'.")
    parser.add_argument("--node-threshold", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.98)
    parser.add_argument("--keep-all-tokens-and-logits", action="store_true")
    parser.add_argument("--filter-act-density", action="store_true")
    parser.add_argument("--act-density-lb", type=float, default=2e-5)
    parser.add_argument("--act-density-ub", type=float, default=0.1)

    parser.add_argument("--target-k", type=int, default=7)
    parser.add_argument("--auto-k", action="store_true")
    parser.add_argument("--k-min", type=int, default=None)
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--max-layer-span", type=int, default=4)
    parser.add_argument("--max-sn", type=int, default=None)
    parser.add_argument(
        "--mean-method",
        type=str,
        choices=["geo", "harm", "arith"],
        default="arith",
        help=(
            "How to combine output/input cosine similarities when building clustering affinity."
        ),
    )
    parser.add_argument("--mediation-penalty", type=float, default=0.1)
    parser.add_argument(
        "--similarity-mode",
        type=str,
        choices=["edge", "node"],
        default="edge",
        help=(
            "Similarity construction mode for clustering: "
            "'edge' uses edge influence/relevance channels; "
            "'node' uses node influence/relevance pairwise weighting."
        ),
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["cos", "cos_relu"],
        default="cos",
        help="Similarity normalization method for output/input structure matrices.",
    )
    parser.add_argument("--enforce-dag", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)

    parser.add_argument("--w-intra", type=float, default=0.30)
    parser.add_argument("--w-dag", type=float, default=0.25)
    parser.add_argument("--w-attr", type=float, default=0.25)
    parser.add_argument("--w-size", type=float, default=0.20)

    parser.add_argument("--supernodes-out", type=str, default="temp_graph_files/supernodes.json")
    parser.add_argument("--supernode-map-out", type=str, default="temp_graph_files/supernode_map.json")
    parser.add_argument("--supernode-flow-out", type=str, default=None)
    parser.add_argument("--auto-k-sweep-out", type=str, default=None)
    parser.add_argument("--figure-html-out", type=str, default=None, help="Optional HTML visualization output path.")

    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--upload-pruning-threshold", type=float, default=0.8)
    parser.add_argument("--upload-density-threshold", type=float, default=0.99)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(args)

    print("\n=== Summarization Pipeline ===")
    print(f"graph_json_path: {result['graph_json_path']}")
    print(f"pruned_nodes: {result['pruned_nodes']}")
    print(f"pruned_edges: {result['pruned_edges']}")
    print(f"resolved_k: {result['resolved_k']}")
    print(f"auto_k_candidates: {result['auto_k_candidates']}")
    print(f"supernodes: {len(result['supernodes'])}")
    print("flow_report:")
    print(json.dumps(_to_jsonable(result["flow_report"]), indent=2))
    if result["figure_html_out"]:
        print(f"figure_html_out: {result['figure_html_out']}")
    if result["upload_status"] is not None:
        print(f"upload_status: {result['upload_status']}")
        if result["upload_body"]:
            print(result["upload_body"])


if __name__ == "__main__":
    main()
