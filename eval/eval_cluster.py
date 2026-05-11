from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast

import networkx as nx
import numpy as np
import torch
from sklearn.cluster import SpectralClustering

from summarization.cluster_scoring import score_k
from summarization.cluster import (
    cluster_graph,
    compute_similarity,
    labels_to_supernodes,
    supernodes_to_mapping,
)
from summarization.prune import PruneGraph, load_prune_graph
from summarization.utils import layer_index_from_node, node_is_fixed

METHOD_GRID: list[dict[str, str | float]] = [
    {
        "mean_method": mean_method,
        "decay_rate": decay_rate,
    }
    for mean_method in ("arith", "harm", "geo")
    for decay_rate in (i / 10.0 for i in range(11))
]

SUMMARY_COLUMNS = [
    "graph_name",
    "dataset",
    "graph_path",
    "num_nodes",
    "k_selection",
    "method",
    "method_family",
    "mean_method",
    "decay_rate",
    "best_k",
    "auto_k_candidates",
    "n_supernodes",
    "score_geo",
    "sil_raw",
    "sil_norm",
    "internal_independence",
    "dag_score",
    "cv_cluster_sizes",
    "opposing_sign_frac",
    "n_middle",
    "result_path",
    "supernode_map_path",
    "auto_k_sweep_path",
]


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def _safe_slug(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")
    return slug or "graph"


_RE_NODE_INF_REL = re.compile(
    r"^node_inf_(?P<inf>\d+(?:\.\d+)?)_rel_(?P<rel>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
_RE_NODE_SIMPLE = re.compile(
    r"^node_(?P<v>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)


def _path_matches_node_threshold(
    graph_path: Path, threshold: float, *, eps: float = 1e-5
) -> bool:
    """True if a parent folder encodes this threshold (SHAP or simple layouts)."""

    def close(x: float) -> bool:
        return abs(x - threshold) <= eps

    for part in graph_path.parts:
        m = _RE_NODE_INF_REL.match(part)
        if m:
            inf = float(m.group("inf"))
            rel = float(m.group("rel"))
            if close(inf) and close(rel):
                return True
            continue
        m = _RE_NODE_SIMPLE.match(part)
        if m and close(float(m.group("v"))):
            return True
    return False


def _discover_prune_graphs(
    input_paths: Sequence[str],
    *,
    node_threshold: float | None,
) -> list[Path]:
    discovered: list[Path] = []
    for raw_path in input_paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            if path.suffix == ".pt" and path.name.endswith("_prune_graph.pt"):
                discovered.append(path)
            continue
        if path.is_dir():
            discovered.extend(
                sorted(
                    p.resolve()
                    for p in path.rglob("*_prune_graph.pt")
                    if p.is_file()
                )
            )
    unique = sorted(dict.fromkeys(discovered))
    if not unique:
        raise FileNotFoundError(
            "No prune-graph .pt files were found in the provided --input-path locations."
        )
    if node_threshold is None:
        return unique
    filtered = [p for p in unique if _path_matches_node_threshold(p, node_threshold)]
    if not filtered:
        raise FileNotFoundError(
            f"No prune-graph .pt files matched --node-threshold {node_threshold:g}. "
            "Expected a path directory like 'node_0.7' or 'node_inf_0.7_rel_0.7' "
            "under --input-path."
        )
    return filtered


def _default_input_paths() -> list[str]:
    candidates = [
        "demos/eval_shap_prune",
        "demos/subgraph/clt-hp",
        "demos/subgraph/clt",
        "demos/subgraph/gemma-scope-16k",
        "demos/subgraph/gemmascope-transcoder-16k",
    ]
    existing = [path for path in candidates if Path(path).exists()]
    return existing or ["demos/eval_shap_prune"]


def _graph_identity(graph_path: Path, input_paths: Sequence[str]) -> tuple[str, str]:
    resolved_inputs = []
    for raw_path in input_paths:
        path = Path(raw_path).expanduser()
        if path.exists():
            resolved_inputs.append(path.resolve())

    for root in resolved_inputs:
        if root.is_file() and root == graph_path:
            return graph_path.stem, root.parent.name or "graphs"
        if root.is_dir():
            try:
                rel = graph_path.relative_to(root)
            except ValueError:
                continue
            dataset = root.name or "graphs"
            stem = _safe_slug(str(rel.with_suffix("")).replace("/", "__"))
            return stem, dataset
    return _safe_slug(graph_path.stem), graph_path.parent.name or "graphs"


def _middle_indices(prune_graph: PruneGraph) -> list[int]:
    return [i for i, n in enumerate(prune_graph.nodes) if not node_is_fixed(n)]


def _fixed_k_from_num_nodes(n_middle: int, divisor: int) -> int:
    """Integer k from floor(num_nodes / divisor), clamped to [1, n_middle]."""
    if n_middle <= 0:
        return 1
    return max(1, min(n_middle // divisor, n_middle))


def _fixed_k_schedule(n_middle: int) -> list[tuple[str, int]]:
    """Two policies: k ≈ n/2 and k ≈ n/3 (middle-node count)."""
    return [
        ("n_over_2", _fixed_k_from_num_nodes(n_middle, 2)),
        ("n_over_3", _fixed_k_from_num_nodes(n_middle, 3)),
    ]


def _cosine_similarity(features: np.ndarray, nonnegative: bool = False) -> np.ndarray:
    safe = np.asarray(features, dtype=np.float64)
    if safe.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    norms = np.linalg.norm(safe, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    normalized = safe / norms
    similarity = np.nan_to_num(normalized @ normalized.T, nan=0.0, posinf=0.0, neginf=0.0)
    if nonnegative:
        similarity = np.clip((similarity + 1.0) / 2.0, 0.0, 1.0)
    return similarity


def _adjacency_affinity(prune_graph: PruneGraph) -> np.ndarray:
    adj_sender = prune_graph.pruned_adj.clone().float().T.detach().cpu().numpy()
    affinity = np.abs(adj_sender)
    affinity = (affinity + affinity.T) / 2.0
    max_val = float(affinity.max()) if affinity.size else 0.0
    if max_val > 0.0:
        affinity = affinity / max_val
    np.fill_diagonal(affinity, 1.0)
    return affinity


def _random_middle_labels(n: int, target_k: int, random_state: int) -> np.ndarray:
    if n == 0:
        return np.array([], dtype=np.int64)
    k = max(1, min(target_k, n))
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    if k == n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed=random_state)
    labels = np.arange(n, dtype=np.int64) % k
    rng.shuffle(labels)
    return labels


def _layer_cluster_labels(prune_graph: PruneGraph, middle_ids: list[str]) -> np.ndarray:
    if not middle_ids:
        return np.array([], dtype=np.int64)
    node_by_id = {node.node_id: node for node in prune_graph.nodes}
    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, node_id in enumerate(middle_ids):
        node = node_by_id.get(node_id)
        layer = layer_index_from_node(node) if node is not None else 0
        grouped[int(layer)].append(idx)
    labels = np.zeros(len(middle_ids), dtype=np.int64)
    for label, layer in enumerate(sorted(grouped)):
        for idx in grouped[layer]:
            labels[idx] = label
    return labels


def _louvain_middle_labels(adjacency_mid: np.ndarray, random_state: int) -> np.ndarray:
    n = adjacency_mid.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    graph = nx.from_numpy_array(adjacency_mid)
    communities = nx.algorithms.community.louvain_communities(
        graph,
        weight="weight",
        seed=random_state,
    )
    labels = np.zeros(n, dtype=np.int64)
    for label, community in enumerate(communities):
        for idx in community:
            labels[int(idx)] = int(label)
    return labels


def _flatten_metrics(
    *,
    graph_name: str,
    dataset: str,
    graph_path: Path,
    num_nodes: int,
    k_selection: str,
    method: str,
    method_family: str,
    mean_method: str | None,
    decay_rate: float | None,
    best_k: int,
    auto_k_candidates: int,
    final_supernodes: dict[str, list[str]],
    base_score: dict[str, Any],
    result_path: Path,
    supernode_map_path: Path,
    auto_k_sweep_path: Path | None,
) -> dict[str, Any]:
    row = {
        "graph_name": graph_name,
        "dataset": dataset,
        "graph_path": str(graph_path),
        "num_nodes": num_nodes,
        "k_selection": k_selection,
        "method": method,
        "method_family": method_family,
        "mean_method": mean_method,
        "decay_rate": decay_rate,
        "best_k": best_k,
        "auto_k_candidates": auto_k_candidates,
        "n_supernodes": len(final_supernodes),
        "score_geo": base_score.get("score_geo"),
        "sil_raw": base_score.get("sil_raw"),
        "sil_norm": base_score.get("sil_norm"),
        "internal_independence": base_score.get("internal_independence"),
        "dag_score": base_score.get("dag_score"),
        "cv_cluster_sizes": base_score.get("cv_cluster_sizes"),
        "opposing_sign_frac": base_score.get("opposing_sign_frac"),
        "n_middle": base_score.get("n_middle"),
        "result_path": str(result_path),
        "supernode_map_path": str(supernode_map_path),
        "auto_k_sweep_path": str(auto_k_sweep_path) if auto_k_sweep_path else "",
    }
    return row


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def _evaluate_ours_fixed_k(
    *,
    prune_graph: PruneGraph,
    graph_path: Path,
    graph_name: str,
    dataset: str,
    output_dir: Path,
    method_config: dict[str, str | float],
    target_k: int,
    k_selection: str,
    num_nodes: int,
    max_layer_span: int,
    random_state: int,
    n_init: int,
    eval_similarity: np.ndarray,
) -> dict[str, Any]:
    mean_method = cast(Literal["geo", "harm", "arith"], method_config["mean_method"])
    decay_rate = float(cast(float, method_config["decay_rate"]))
    # method-specific similarity used only for clustering, not for scoring
    similarity = compute_similarity(
        prune_graph,
        mean_method=mean_method,
        decay_rate=decay_rate,
        max_layer_span=max_layer_span,
    )
    clusters = cluster_graph(
        prune_graph,
        target_k=target_k,
        max_layer_span=max_layer_span,
        mean_method=mean_method,
        decay_rate=decay_rate,
        enforce_dag=True,
        random_state=random_state,
        n_init=n_init,
    )
    final_supernodes = supernodes_to_mapping(prune_graph, clusters)
    base_score = score_k(
        final_supernodes,
        prune_graph,
        eval_similarity,
        enforce_dag=True,
    )

    method_slug = f"ours-{method_config['mean_method']}-decay-{decay_rate:.1f}"
    run_dir = output_dir / "runs" / graph_name / method_slug / k_selection
    supernode_map_path = run_dir / "supernode_map.json"
    auto_k_sweep_path = run_dir / "fixed_k_metrics.json"
    result_path = run_dir / "result.json"

    _write_json(supernode_map_path, final_supernodes)
    _write_json(
        auto_k_sweep_path,
        {
            "k_selection": k_selection,
            "target_k": target_k,
            "num_nodes": num_nodes,
            "metrics": {key: value for key, value in base_score.items()},
        },
    )

    summary_row = _flatten_metrics(
        graph_name=graph_name,
        dataset=dataset,
        graph_path=graph_path,
        num_nodes=num_nodes,
        k_selection=k_selection,
        method=method_slug,
        method_family="ours",
        mean_method=str(method_config["mean_method"]),
        decay_rate=decay_rate,
        best_k=target_k,
        auto_k_candidates=0,
        final_supernodes=final_supernodes,
        base_score=base_score,
        result_path=result_path,
        supernode_map_path=supernode_map_path,
        auto_k_sweep_path=auto_k_sweep_path,
    )
    result_payload = {
        **summary_row,
        "final_supernodes": final_supernodes,
        "score_details": base_score.get("details", {}),
    }
    _write_json(result_path, result_payload)
    return result_payload


def _evaluate_baseline_fixed_k(
    *,
    prune_graph: PruneGraph,
    graph_path: Path,
    graph_name: str,
    dataset: str,
    output_dir: Path,
    method: str,
    eval_similarity: np.ndarray,
    clusterer: Callable[[int], list[list[str]]],
    target_k: int,
    k_selection: str,
    num_nodes: int,
    enforce_dag: bool,
) -> dict[str, Any]:
    clusters = clusterer(target_k)
    final_supernodes = supernodes_to_mapping(prune_graph, clusters)
    base_score = score_k(
        final_supernodes,
        prune_graph,
        eval_similarity,
        enforce_dag=enforce_dag,
    )

    run_dir = output_dir / "runs" / graph_name / method / k_selection
    supernode_map_path = run_dir / "supernode_map.json"
    auto_k_sweep_path = run_dir / "fixed_k_metrics.json"
    result_path = run_dir / "result.json"

    _write_json(supernode_map_path, final_supernodes)
    _write_json(
        auto_k_sweep_path,
        {
            "k_selection": k_selection,
            "target_k": target_k,
            "num_nodes": num_nodes,
            "metrics": {key: value for key, value in base_score.items()},
        },
    )

    summary_row = _flatten_metrics(
        graph_name=graph_name,
        dataset=dataset,
        graph_path=graph_path,
        num_nodes=num_nodes,
        k_selection=k_selection,
        method=method,
        method_family="baseline",
        mean_method=None,
        decay_rate=None,
        best_k=target_k,
        auto_k_candidates=0,
        final_supernodes=final_supernodes,
        base_score=base_score,
        result_path=result_path,
        supernode_map_path=supernode_map_path,
        auto_k_sweep_path=auto_k_sweep_path,
    )
    result_payload = {
        **summary_row,
        "final_supernodes": final_supernodes,
        "score_details": base_score.get("details", {}),
    }
    _write_json(result_path, result_payload)
    return result_payload


def evaluate_prune_graph(
    *,
    graph_path: Path,
    input_paths: Sequence[str],
    output_dir: Path,
    map_location: str,
    max_layer_span: int,
    random_state: int,
    n_init: int,
) -> list[dict[str, Any]]:
    prune_graph = load_prune_graph(str(graph_path), map_location=map_location)
    graph_name, dataset = _graph_identity(graph_path, input_paths)
    rows: list[dict[str, Any]] = []

    # fixed evaluation similarity space shared by all methods (default arith, no decay)
    eval_similarity = compute_similarity(prune_graph).detach().cpu().numpy()

    num_nodes = len(_middle_indices(prune_graph))
    k_schedule = _fixed_k_schedule(num_nodes)

    for method_config in METHOD_GRID:
        for k_selection, target_k in k_schedule:
            rows.append(
                _evaluate_ours_fixed_k(
                    prune_graph=prune_graph,
                    graph_path=graph_path,
                    graph_name=graph_name,
                    dataset=dataset,
                    output_dir=output_dir,
                    method_config=method_config,
                    target_k=target_k,
                    k_selection=k_selection,
                    num_nodes=num_nodes,
                    max_layer_span=max_layer_span,
                    random_state=random_state,
                    n_init=n_init,
                    eval_similarity=eval_similarity,
                )
            )

    mid_idx = _middle_indices(prune_graph)
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    adjacency_affinity = _adjacency_affinity(prune_graph)
    adjacency_mid = adjacency_affinity[np.ix_(mid_idx, mid_idx)]

    def random_clusterer(target_k: int) -> list[list[str]]:
        if len(middle_ids) == 0:
            return labels_to_supernodes(prune_graph, [], np.array([], dtype=np.int64))
        labels = _random_middle_labels(
            len(middle_ids),
            target_k=target_k,
            random_state=random_state + target_k,
        )
        return labels_to_supernodes(prune_graph, middle_ids, labels)

    def louvain_clusterer(target_k: int) -> list[list[str]]:
        del target_k
        if len(middle_ids) == 0:
            return labels_to_supernodes(prune_graph, [], np.array([], dtype=np.int64))
        labels = _louvain_middle_labels(adjacency_mid, random_state=random_state)
        n_clusters = int(labels.max()) + 1 if labels.size else 0
        if n_clusters >= len(middle_ids):
            labels = np.arange(len(middle_ids), dtype=np.int64)
        elif n_clusters <= 1 and len(middle_ids) > 1:
            labels = SpectralClustering(
                n_clusters=min(max(2, len(middle_ids) // 2), len(middle_ids)),
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=random_state,
                n_init=n_init,  # type: ignore[arg-type]
            ).fit_predict(adjacency_mid)
        return labels_to_supernodes(prune_graph, middle_ids, labels)

    by_layer_labels = _layer_cluster_labels(prune_graph, middle_ids)
    by_layer_clusters = labels_to_supernodes(prune_graph, middle_ids, by_layer_labels)
    by_layer_k = int(by_layer_labels.max()) + 1 if by_layer_labels.size else 0
    rows.append(
        _evaluate_baseline_fixed_k(
            prune_graph=prune_graph,
            graph_path=graph_path,
            graph_name=graph_name,
            dataset=dataset,
            output_dir=output_dir,
            method="baseline-by-layer",
            eval_similarity=eval_similarity,
            clusterer=lambda _target_k: by_layer_clusters,
            target_k=by_layer_k,
            k_selection="by_layer",
            num_nodes=num_nodes,
            enforce_dag=True,
        )
    )

    for k_selection, target_k in k_schedule:
        rows.append(
            _evaluate_baseline_fixed_k(
                prune_graph=prune_graph,
                graph_path=graph_path,
                graph_name=graph_name,
                dataset=dataset,
                output_dir=output_dir,
                method="baseline-random",
                eval_similarity=eval_similarity,
                clusterer=random_clusterer,
                target_k=target_k,
                k_selection=k_selection,
                num_nodes=num_nodes,
                enforce_dag=True,
            )
        )
        rows.append(
            _evaluate_baseline_fixed_k(
                prune_graph=prune_graph,
                graph_path=graph_path,
                graph_name=graph_name,
                dataset=dataset,
                output_dir=output_dir,
                method="baseline-louvain-adjacency",
                eval_similarity=eval_similarity,
                clusterer=louvain_clusterer,
                target_k=target_k,
                k_selection=k_selection,
                num_nodes=num_nodes,
                enforce_dag=True,
            )
        )
    return rows


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    graph_paths = _discover_prune_graphs(
        args.input_path,
        node_threshold=args.node_threshold,
    )
    summary_rows: list[dict[str, Any]] = []
    for graph_path in graph_paths:
        summary_rows.extend(
            evaluate_prune_graph(
                graph_path=graph_path,
                input_paths=args.input_path,
                output_dir=output_dir,
                map_location=args.map_location,
                max_layer_span=args.max_layer_span,
                random_state=args.random_state,
                n_init=args.n_init,
            )
        )

    summary_path = output_dir / "summary.csv"
    results_path = output_dir / "results.json"
    manifest_path = output_dir / "manifest.json"
    _write_summary_csv(summary_path, summary_rows)
    _write_json(results_path, summary_rows)
    _write_json(
        manifest_path,
        {
            "input_paths": list(args.input_path),
            "node_threshold": args.node_threshold,
            "graph_paths": [str(path) for path in graph_paths],
            "output_dir": str(output_dir),
            "method_grid": METHOD_GRID,
            "baselines": [
                "baseline-random",
                "baseline-by-layer",
                "baseline-louvain-adjacency",
            ],
            "cluster_k_policy": (
                "Middle-node count n = |non-fixed nodes|. Run each method at "
                "k = round(n/2) and k = round(n/3), clamped to [1, n], except "
                "baseline-by-layer which uses one cluster per layer."
            ),
            "summary_csv": str(summary_path),
            "results_json": str(results_path),
            "n_graphs": len(graph_paths),
            "n_runs": len(summary_rows),
            "config": {
                "node_threshold": args.node_threshold,
                "max_layer_span": args.max_layer_span,
                "enforce_dag": True,
                "map_location": args.map_location,
                "random_state": args.random_state,
                "n_init": args.n_init,
            },
        },
    )
    return {
        "output_dir": str(output_dir),
        "summary_csv": str(summary_path),
        "results_json": str(results_path),
        "manifest_json": str(manifest_path),
        "n_graphs": len(graph_paths),
        "n_runs": len(summary_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate clustering variants and simple baselines on saved prune-graph .pt files."
        )
    )
    parser.add_argument(
        "--input-path",
        action="append",
        default=[],
        help=(
            "File or directory containing prune-graph .pt files. "
            "May be repeated; directories are searched recursively."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_outputs/clustering",
        help="Directory where summary CSV/JSON and per-run artifacts will be saved.",
    )
    parser.add_argument("--map-location", type=str, default="cpu")
    parser.add_argument(
        "--node-threshold",
        type=float,
        default=None,
        metavar="T",
        help=(
            "Only evaluate prune graphs whose path includes a threshold folder, e.g. "
            "'node_0.7' (…\\\\entmax15\\\\node_0.7\\\\…) or SHAP-style "
            "'node_inf_0.7_rel_0.7'. Omit to use every discovered *_prune_graph.pt."
        ),
    )
    parser.add_argument("--max-layer-span", type=int, default=4)
    parser.add_argument("--enforce-dag", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.input_path:
        args.input_path = _default_input_paths()
    result = run_evaluation(args)
    print("\n=== Clustering Evaluation (fixed k = n/2 and n/3) ===")
    print(f"output_dir: {result['output_dir']}")
    print(f"summary_csv: {result['summary_csv']}")
    print(f"results_json: {result['results_json']}")
    print(f"manifest_json: {result['manifest_json']}")
    print(f"n_graphs: {result['n_graphs']}")
    print(f"n_runs: {result['n_runs']}")
    if args.node_threshold is not None:
        print(f"node_threshold filter: {args.node_threshold:g}")


if __name__ == "__main__":
    main()
