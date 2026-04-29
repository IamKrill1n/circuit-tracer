from __future__ import annotations

import argparse
from pathlib import Path

import torch

from summarization.prune import LogitWeightMode
from summarization.prune import prune_graph_pipeline, save_prune_graph
from summarization.token_attribution import get_token_attribution_from_graph

DEFAULT_SOURCE_SETS = ("clt-hp", "gemmascope-transcoder-16k")

def _resolve_device(device_flag: str) -> str:
    if device_flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_flag == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested --device cuda, but CUDA is not available.")
    return device_flag


def _discover_graph_files(graphs_root: Path, source_sets: tuple[str, ...]) -> dict[str, list[Path]]:
    discovered: dict[str, list[Path]] = {}
    for source_set in source_sets:
        src_dir = graphs_root / source_set
        files = sorted(src_dir.glob("*.json"))
        discovered[source_set] = files
    return discovered


def run(args: argparse.Namespace) -> None:
    graphs_root = Path(args.graphs_root)
    output_root = Path(args.output_root)
    source_sets = tuple(args.source_sets)
    device = _resolve_device(args.device)
    discovered = _discover_graph_files(graphs_root, source_sets)

    totals = {}
    for source_set, paths in discovered.items():
        total_paths = len(paths) if args.limit is None else min(len(paths), args.limit)
        totals[source_set] = {"total": total_paths, "ok": 0, "failed": 0}
    failures: list[str] = []

    for source_set, graph_paths in discovered.items():
        total_for_source = totals[source_set]["total"]
        print(f"\n=== Source set: {source_set} ({total_for_source} files) ===")
        if args.limit is not None:
            graph_paths = graph_paths[: args.limit]

        for graph_path in graph_paths:
            stem = graph_path.stem
            try:
                token_weights_tensor = get_token_attribution_from_graph(
                    graph_path=graph_path,
                    model_name=args.model_name,
                    normalize_method=args.normalize_method,
                    device=device,
                ).detach().cpu().to(torch.float32)
                token_weights = token_weights_tensor.tolist()

                prune_graph = prune_graph_pipeline(
                    json_path=str(graph_path),
                    logit_weights=args.logit_weights,
                    token_weights=token_weights,
                    node_threshold=args.node_threshold,
                    edge_threshold=args.edge_threshold,
                    keep_all_tokens_and_logits=args.keep_all_tokens_and_logits,
                    filter_act_density=args.filter_act_density,
                    act_density_lb=args.act_density_lb,
                    act_density_ub=args.act_density_ub,
                )

                out_dir = output_root / source_set
                out_dir.mkdir(parents=True, exist_ok=True)
                token_weights_path = out_dir / f"{stem}_token_weights.pt"
                prune_graph_path = out_dir / f"{stem}_prune_graph.pt"

                torch.save(token_weights_tensor, token_weights_path)
                save_prune_graph(prune_graph, str(prune_graph_path))

                totals[source_set]["ok"] += 1
                print(
                    f"[ok] {graph_path.name} -> nodes={prune_graph.num_nodes}, edges={prune_graph.num_edges}"
                )
            except Exception as exc:
                totals[source_set]["failed"] += 1
                failure = f"{source_set}/{graph_path.name}: {exc}"
                failures.append(failure)
                print(f"[failed] {failure}")

    total_files = sum(v["total"] for v in totals.values())
    total_ok = sum(v["ok"] for v in totals.values())
    total_failed = sum(v["failed"] for v in totals.values())

    print("\n=== Batch Summary ===")
    print(f"model_name: {args.model_name}")
    print(f"device: {device}")
    print(f"normalize_method: {args.normalize_method}")
    print(f"graphs_root: {graphs_root}")
    print(f"output_root: {output_root}")
    for source_set in source_sets:
        stats = totals[source_set]
        print(
            f"{source_set}: total={stats['total']} ok={stats['ok']} failed={stats['failed']}"
        )
    print(f"overall: total={total_files} ok={total_ok} failed={total_failed}")
    if failures:
        print("\nFailures:")
        for item in failures:
            print(f"- {item}")

    if total_ok == 0:
        raise RuntimeError("No graphs were successfully processed.")


def _parse_logit_weights(value: str) -> LogitWeightMode:
    if value not in ("probs", "target"):
        raise argparse.ArgumentTypeError("--logit-weights must be 'probs' or 'target'.")
    return value  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run prune_graph_pipeline with SHAP token weights for graph JSON files "
            "in demos/temp_graph_files/<source_set>."
        )
    )
    parser.add_argument(
        "--graphs-root",
        default="demos/temp_graph_files",
        help="Root folder containing source-set graph subdirectories.",
    )
    parser.add_argument(
        "--source-sets",
        nargs="+",
        default=list(DEFAULT_SOURCE_SETS),
        help="Source-set subdirectories under --graphs-root to process.",
    )
    parser.add_argument(
        "--output-root",
        default="demos/subgraph",
        help="Output root for .pt artifacts (saved under <output-root>/<source_set>/).",
    )
    parser.add_argument(
        "--model-name",
        default="google/gemma-2-2b",
        help="HF model name for SHAP attribution.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
        help="Device used for model/explainer initialization.",
    )
    parser.add_argument(
        "--normalize-method",
        choices=["softmax", "sparsemax"],
        default="sparsemax",
        help="Normalization applied to SHAP attribution scores.",
    )
    parser.add_argument(
        "--logit-weights",
        type=_parse_logit_weights,
        default="target",
    )
    parser.add_argument("--node-threshold", type=float, default=0.7)
    parser.add_argument("--edge-threshold", type=float, default=0.98)
    parser.add_argument("--keep-all-tokens-and-logits", action="store_true")
    parser.add_argument("--filter-act-density", action="store_true")
    parser.add_argument("--act-density-lb", type=float, default=2e-5)
    parser.add_argument("--act-density-ub", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Optional max files per source set.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
