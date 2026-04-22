from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure repo root is importable when running as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from summarization.prune import prune_combined
from summarization.token_attribution import get_token_attribution_from_graph
from summarization.utils import _build_index_sets, get_data_from_json
from circuit_tracer.graph import find_threshold


DEFAULT_INPUT_ROOT = Path("demos") / "temp_graph_files"
DEFAULT_OUTPUT_ROOT = Path("demos") / "plots"
DEFAULT_SOURCE_SETS = ("clt-hp", "gemmascope-transcoder-16k")
DEFAULT_TOKEN_ATTRIBUTION_MODEL = "google/gemma-2-2b"


@dataclass
class ExperimentResult:
    source_set: str
    input_json: Path
    output_plot: Path
    n_feature_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot feature-node influence vs relevance for prune experiments."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root folder containing source-set subfolders with graph JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where generated plots are written.",
    )
    parser.add_argument(
        "--source-set",
        action="append",
        choices=list(DEFAULT_SOURCE_SETS),
        help="Source set to process. Repeat to run multiple source sets.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of JSON files per source set.",
    )
    parser.add_argument(
        "--logit-weights",
        choices=("target", "probs"),
        default="target",
        help="Logit weighting mode used for influence calculation.",
    )
    parser.add_argument(
        "--node-threshold",
        type=float,
        default=0.8,
        help="Node threshold passed to prune_combined (does not affect raw scores).",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.98,
        help="Edge threshold passed to prune_combined (does not affect raw scores).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Influence/relevance blend alpha used by prune_combined.",
    )
    parser.add_argument(
        "--combined-scores-method",
        choices=("geometric", "arithmetic", "harmonic"),
        default="geometric",
        help="Combined node/edge score method used by prune_combined.",
    )
    parser.add_argument(
        "--normalization",
        choices=("min_max", "rank"),
        default="min_max",
        help="Score normalization passed to prune_combined.",
    )
    parser.add_argument(
        "--token-model",
        default=DEFAULT_TOKEN_ATTRIBUTION_MODEL,
        help=(
            "Hugging Face model name used by summarization/token_attribution.py "
            "to compute token weights."
        ),
    )
    return parser.parse_args()


def _iter_graphs(source_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(source_dir.glob("*.json"))
    if limit is not None:
        return files[:limit]
    return files


def _should_use_log_scale(values: np.ndarray) -> bool:
    positive = values[values > 0]
    if positive.size == 0:
        return False
    spread = positive.max() / max(positive.min(), 1e-20)
    return spread > 1e4


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _compute_token_weights_or_none(graph_path: Path, token_model: str, n_embeddings: int) -> list[float] | None:
    try:
        token_weights_tensor = get_token_attribution_from_graph(
            graph_path=graph_path,
            model_name=token_model,
        )
        token_weights = _to_numpy(token_weights_tensor).tolist()
    except Exception as exc:
        print(
            f"[warn] token attribution failed for {graph_path.name} "
            f"with model '{token_model}': {exc}. Falling back to uniform token weights."
        )
        return None

    if len(token_weights) != n_embeddings:
        print(
            f"[warn] token attribution length mismatch for {graph_path.name}: "
            f"got {len(token_weights)}, expected {n_embeddings}. "
            "Falling back to uniform token weights."
        )
        return None
    return token_weights


def run_single_graph(
    source_set: str,
    graph_path: Path,
    output_root: Path,
    logit_weights: str,
    node_threshold: float,
    edge_threshold: float,
    alpha: float,
    combined_scores_method: str,
    normalization: str,
    token_model: str,
) -> ExperimentResult:
    adj, node_ids, attr, _metadata = get_data_from_json(str(graph_path))
    idx = _build_index_sets(node_ids, attr)
    feature_indices = idx["feature"]
    embedding_indices = idx["embedding"]
    if not feature_indices:
        raise ValueError(f"No feature nodes found in {graph_path}")

    token_weights = _compute_token_weights_or_none(
        graph_path=graph_path,
        token_model=token_model,
        n_embeddings=len(embedding_indices),
    )

    (
        _node_mask,
        _edge_mask,
        node_influence,
        node_relevance,
        _edge_influence,
        _edge_relevance,
    ) = prune_combined(
        adj=adj,
        node_ids=node_ids,
        attr=attr,
        logit_weights=logit_weights,  # type: ignore[arg-type]
        token_weights=token_weights,
        node_influence_threshold=node_threshold,
        node_relevance_threshold=node_threshold,
        edge_influence_threshold=edge_threshold,
        edge_relevance_threshold=edge_threshold,
        keep_all_tokens_and_logits=True,
    )

    feature_influence = _to_numpy(node_influence[feature_indices])
    feature_relevance = _to_numpy(node_relevance[feature_indices])
    relevance_cutoff = float(find_threshold(node_relevance, node_threshold).item())
    influence_cutoff = float(find_threshold(node_influence, node_threshold).item())

    output_dir = output_root / source_set
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{graph_path.stem}_influence_vs_relevance.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        feature_relevance,
        feature_influence,
        s=8,
        alpha=0.35,
        linewidths=0,
    )
    ax.axvline(
        relevance_cutoff,
        color="tab:blue",
        linestyle="--",
        linewidth=1.2,
        label=f"relevance threshold={node_threshold:.2f}",
    )
    ax.axhline(
        influence_cutoff,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=f"influence threshold={node_threshold:.2f}",
    )

    if _should_use_log_scale(feature_relevance):
        ax.set_xscale("log")
    if _should_use_log_scale(feature_influence):
        ax.set_yscale("log")

    ax.set_xlabel("Feature node relevance")
    ax.set_ylabel("Feature node influence")
    ax.set_title(f"{source_set}: {graph_path.stem}")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return ExperimentResult(
        source_set=source_set,
        input_json=graph_path,
        output_plot=output_path,
        n_feature_points=len(feature_indices),
    )


def main() -> None:
    args = parse_args()
    source_sets = args.source_set or list(DEFAULT_SOURCE_SETS)
    output_root = args.output_root

    all_results: list[ExperimentResult] = []
    for source_set in source_sets:
        source_dir = args.input_root / source_set
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source directory: {source_dir}")

        graph_files = _iter_graphs(source_dir, args.limit)
        if not graph_files:
            print(f"[skip] {source_set}: no json files in {source_dir}")
            continue

        print(f"[run] {source_set}: {len(graph_files)} graph files")
        for graph_path in graph_files:
            result = run_single_graph(
                source_set=source_set,
                graph_path=graph_path,
                output_root=output_root,
                logit_weights=args.logit_weights,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
                alpha=args.alpha,
                combined_scores_method=args.combined_scores_method,
                normalization=args.normalization,
                token_model=args.token_model,
            )
            all_results.append(result)
            print(
                f"[ok] {result.input_json.name} -> {result.output_plot} "
                f"({result.n_feature_points} feature points)"
            )

    print(f"[done] wrote {len(all_results)} plot(s) to {output_root}")


if __name__ == "__main__":
    main()
