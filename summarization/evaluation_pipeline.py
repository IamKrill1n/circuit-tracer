from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering

from summarization.auto_grouping import eigengap_analysis, find_best_k, score_k
from summarization.cluster import cluster_graph, compute_similarity
from summarization.flow_analysis import (
    build_supernode_graph,
    flow_faithfulness_report,
    supernodes_to_mapping,
)
from summarization.prune import PruneGraph, load_prune_graph
from summarization.utils import _is_embedding, _is_fixed, _is_logit

METHOD_GRID: list[dict[str, str]] = [
    {
        "mean_method": mean_method,
        "normalization": normalization,
        "similarity_mode": similarity_mode,
    }
    for mean_method in ("harm", "arith")
    for normalization in ("cos", "cos_relu")
    for similarity_mode in ("node", "edge")
]

SUMMARY_COLUMNS = [
    "graph_name",
    "dataset",
    "graph_path",
    "method",
    "method_family",
    "mean_method",
    "normalization",
    "similarity_mode",
    "best_k",
    "auto_k_candidates",
    "n_supernodes",
    "total",
    "total_base",
    "intra_sim",
    "dag_safety",
    "attr_balance",
    "size_score",
    "n_middle",
    "n_warnings",
    "inf_conservation",
    "edge_conservation",
    "within_cluster_weighted_edge_cosine_mean",
    "F_phi",
    "path_score",
    "residual_score",
    "shortcut_score",
    "D_phi",
    "R_phi",
    "R_phi_balance",
    "R_phi_suppressive",
    "R_phi_max",
    "shortcut_frac",
    "sigma_phi",
    "top_k_frac",
    "n_paths",
    "n_shortcuts",
    "n_direct",
    "n_suppressive",
    "n_balanced",
    "result_path",
    "supernode_map_path",
    "auto_k_sweep_path",
    "flow_report_path",
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


def _discover_prune_graphs(input_paths: Sequence[str]) -> list[Path]:
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
    return unique


def _default_input_paths() -> list[str]:
    candidates = [
        "demos/subgraph/clt",
        "demos/subgraph/clt-hp",
        "demos/subgraph/gemma-scope-16k",
        "demos/subgraph/gemmascope-transcoder-16k",
    ]
    existing = [path for path in candidates if Path(path).exists()]
    return existing or candidates


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
    return [i for i, nid in enumerate(prune_graph.kept_ids) if not _is_fixed(prune_graph.attr, nid)]


def _normalize_edge_weights(weights: torch.Tensor) -> torch.Tensor:
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = float(weights.abs().max().item()) if weights.numel() else 0.0
    if max_abs <= 0.0:
        return torch.zeros_like(weights)
    return weights / (max_abs + 1e-8)


def _edge_weighted_profile_features(prune_graph: PruneGraph) -> np.ndarray:
    adj_sender = prune_graph.pruned_adj.clone().float().T
    edge_rel = prune_graph.edge_relevance
    edge_inf = prune_graph.edge_influence
    if edge_rel is None or edge_inf is None:
        rel_sender = _normalize_edge_weights(adj_sender)
        inf_sender = rel_sender
    else:
        rel_sender = _normalize_edge_weights(edge_rel.float().T)
        inf_sender = _normalize_edge_weights(edge_inf.float().T)
    weighted_out = adj_sender * inf_sender
    weighted_in = (adj_sender * rel_sender).T
    return torch.cat([weighted_out, weighted_in], dim=1).detach().cpu().numpy()


def _node_profile_features(prune_graph: PruneGraph) -> np.ndarray:
    adj_sender = prune_graph.pruned_adj.clone().float().T
    return torch.cat([adj_sender, adj_sender.T], dim=1).detach().cpu().numpy()


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


def _labels_to_supernodes(
    prune_graph: PruneGraph,
    middle_ids: list[str],
    labels: np.ndarray,
) -> list[list[str]]:
    grouped: dict[int, list[str]] = {}
    for node_id, label in zip(middle_ids, labels):
        grouped.setdefault(int(label), []).append(node_id)

    middle_clusters = [grouped[label] for label in sorted(grouped)]
    emb_singletons = [[nid] for nid in prune_graph.kept_ids if _is_embedding(prune_graph.attr, nid)]
    logit_singletons = [[nid] for nid in prune_graph.kept_ids if _is_logit(prune_graph.attr, nid)]
    return middle_clusters + emb_singletons + logit_singletons


def _generic_auto_k(
    *,
    prune_graph: PruneGraph,
    similarity: np.ndarray,
    clusterer: Callable[[int], list[list[str]]],
    k_min_override: int | None,
    k_max_override: int | None,
    weights: dict[str, float] | None,
    enforce_dag: bool,
) -> tuple[int, dict[int, dict[str, Any]]]:
    n_middle = len(_middle_indices(prune_graph))
    target_n_middle = max(1, n_middle)
    if n_middle < 3:
        fallback_k = max(0, n_middle)
        clusters = clusterer(fallback_k)
        mapping = supernodes_to_mapping(prune_graph, clusters)
        result = score_k(
            mapping,
            prune_graph,
            similarity,
            target_n_middle=target_n_middle,
            w_intra=(weights or {}).get("w_intra", 0.30),
            w_dag=(weights or {}).get("w_dag", 0.25),
            w_attr=(weights or {}).get("w_attr", 0.25),
            w_size=(weights or {}).get("w_size", 0.20),
            use_flow_faithfulness=False,
            enforce_dag=enforce_dag,
        )
        result["final_supernodes"] = mapping
        return fallback_k, {fallback_k: result}

    eigengap = eigengap_analysis(similarity, prune_graph, max_k=min(20, n_middle - 1))
    k_min = k_min_override if k_min_override is not None else int(eigengap["search_range"][0])
    k_max = k_max_override if k_max_override is not None else int(eigengap["search_range"][1])
    k_min = max(2, min(k_min, n_middle))
    k_max = max(k_min, min(k_max, n_middle))

    results: dict[int, dict[str, Any]] = {}
    for target_k in range(k_min, k_max + 1):
        clusters = clusterer(target_k)
        mapping = supernodes_to_mapping(prune_graph, clusters)
        result = score_k(
            mapping,
            prune_graph,
            similarity,
            target_n_middle=target_n_middle,
            w_intra=(weights or {}).get("w_intra", 0.30),
            w_dag=(weights or {}).get("w_dag", 0.25),
            w_attr=(weights or {}).get("w_attr", 0.25),
            w_size=(weights or {}).get("w_size", 0.20),
            use_flow_faithfulness=False,
            enforce_dag=enforce_dag,
        )
        result["final_supernodes"] = mapping
        results[target_k] = result

    best_k = max(results, key=lambda k: float(results[k]["total"]))
    return best_k, results


def _within_cluster_mean_cosine(
    features: np.ndarray,
    final_supernodes: dict[str, list[str]],
    prune_graph: PruneGraph,
) -> float | None:
    similarity = _cosine_similarity(features, nonnegative=False)
    node_to_idx = {node_id: i for i, node_id in enumerate(prune_graph.kept_ids)}
    pair_values: list[float] = []
    for sn_name, members in final_supernodes.items():
        if "EMB" in sn_name or "LOGIT" in sn_name:
            continue
        member_idx = [node_to_idx[node_id] for node_id in members if node_id in node_to_idx]
        if len(member_idx) < 2:
            continue
        block = similarity[np.ix_(member_idx, member_idx)]
        upper = block[np.triu_indices(len(member_idx), k=1)]
        pair_values.extend(float(value) for value in upper.tolist())
    if not pair_values:
        return None
    return float(np.mean(pair_values))


def _flatten_metrics(
    *,
    graph_name: str,
    dataset: str,
    graph_path: Path,
    method: str,
    method_family: str,
    mean_method: str | None,
    normalization: str | None,
    similarity_mode: str | None,
    best_k: int,
    auto_k_candidates: int,
    final_supernodes: dict[str, list[str]],
    base_score: dict[str, Any],
    flow_report: dict[str, Any],
    weighted_edge_cosine_mean: float | None,
    result_path: Path,
    supernode_map_path: Path,
    auto_k_sweep_path: Path | None,
    flow_report_path: Path,
) -> dict[str, Any]:
    combined = flow_report["combined"]
    row = {
        "graph_name": graph_name,
        "dataset": dataset,
        "graph_path": str(graph_path),
        "method": method,
        "method_family": method_family,
        "mean_method": mean_method,
        "normalization": normalization,
        "similarity_mode": similarity_mode,
        "best_k": best_k,
        "auto_k_candidates": auto_k_candidates,
        "n_supernodes": len(final_supernodes),
        "total": base_score.get("total"),
        "total_base": base_score.get("total_base"),
        "intra_sim": base_score.get("intra_sim"),
        "dag_safety": base_score.get("dag_safety"),
        "attr_balance": base_score.get("attr_balance"),
        "size_score": base_score.get("size_score"),
        "n_middle": base_score.get("n_middle"),
        "n_warnings": base_score.get("n_warnings"),
        "inf_conservation": base_score.get("inf_conservation"),
        "edge_conservation": base_score.get("edge_conservation"),
        "within_cluster_weighted_edge_cosine_mean": weighted_edge_cosine_mean,
        "F_phi": combined.get("F_phi"),
        "path_score": combined.get("path_score"),
        "residual_score": combined.get("residual_score"),
        "shortcut_score": combined.get("shortcut_score"),
        "D_phi": combined.get("D_phi"),
        "R_phi": combined.get("R_phi"),
        "R_phi_balance": combined.get("R_phi_balance"),
        "R_phi_suppressive": combined.get("R_phi_suppressive"),
        "R_phi_max": combined.get("R_phi_max"),
        "shortcut_frac": combined.get("shortcut_frac"),
        "sigma_phi": combined.get("sigma_phi"),
        "top_k_frac": combined.get("top_k_frac"),
        "n_paths": combined.get("n_paths"),
        "n_shortcuts": combined.get("n_shortcuts"),
        "n_direct": combined.get("n_direct"),
        "n_suppressive": combined.get("n_suppressive"),
        "n_balanced": combined.get("n_balanced"),
        "result_path": str(result_path),
        "supernode_map_path": str(supernode_map_path),
        "auto_k_sweep_path": str(auto_k_sweep_path) if auto_k_sweep_path else "",
        "flow_report_path": str(flow_report_path),
    }
    return row


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def _evaluate_existing_method(
    *,
    prune_graph: PruneGraph,
    graph_path: Path,
    graph_name: str,
    dataset: str,
    output_dir: Path,
    method_config: dict[str, str],
    k_min_override: int | None,
    k_max_override: int | None,
    weights: dict[str, float] | None,
    max_layer_span: int,
    mediation_penalty: float,
    enforce_dag: bool,
    random_state: int,
    n_init: int,
) -> dict[str, Any]:
    similarity = compute_similarity(
        prune_graph,
        mean_method=method_config["mean_method"],
        mediation_penalty=mediation_penalty,
        similarity_mode=method_config["similarity_mode"],
        normalization=method_config["normalization"],
    )
    best_k, sweep = find_best_k(
        prune_graph=prune_graph,
        similarity=similarity,
        max_layer_span=max_layer_span,
        k_min_override=k_min_override,
        k_max_override=k_max_override,
        weights=weights,
        mean_method=method_config["mean_method"],
        mediation_penalty=mediation_penalty,
        similarity_mode=method_config["similarity_mode"],
        normalization=method_config["normalization"],
        use_flow_faithfulness=False,
        enforce_dag=enforce_dag,
        random_state=random_state,
        n_init=n_init,
    )

    if sweep:
        best_result = sweep[best_k]
        final_supernodes = best_result["final_supernodes"]
        base_score = {
            key: value for key, value in best_result.items() if key not in ("final_supernodes",)
        }
    else:
        clusters = cluster_graph(
            prune_graph,
            target_k=max(best_k, 1),
            max_layer_span=max_layer_span,
            mean_method=method_config["mean_method"],
            mediation_penalty=mediation_penalty,
            similarity_mode=method_config["similarity_mode"],
            normalization=method_config["normalization"],
            enforce_dag=enforce_dag,
            random_state=random_state,
            n_init=n_init,
        )
        final_supernodes = supernodes_to_mapping(prune_graph, clusters)
        base_score = score_k(
            final_supernodes,
            prune_graph,
            similarity,
            target_n_middle=max(1, len(_middle_indices(prune_graph))),
            w_intra=(weights or {}).get("w_intra", 0.30),
            w_dag=(weights or {}).get("w_dag", 0.25),
            w_attr=(weights or {}).get("w_attr", 0.25),
            w_size=(weights or {}).get("w_size", 0.20),
            use_flow_faithfulness=False,
            enforce_dag=enforce_dag,
        )

    method_slug = (
        f"ours-{method_config['mean_method']}-"
        f"{method_config['normalization']}-{method_config['similarity_mode']}"
    )
    run_dir = output_dir / "runs" / graph_name / method_slug
    supernode_map_path = run_dir / "supernode_map.json"
    auto_k_sweep_path = run_dir / "auto_k_sweep.json"
    flow_report_path = run_dir / "flow_report.json"
    result_path = run_dir / "result.json"

    edge_cosine_mean = _within_cluster_mean_cosine(
        _edge_weighted_profile_features(prune_graph),
        final_supernodes,
        prune_graph,
    )
    sng = build_supernode_graph(prune_graph, final_supernodes, enforce_dag=enforce_dag)
    flow_report = flow_faithfulness_report(sng, final_supernodes)

    _write_json(supernode_map_path, final_supernodes)
    _write_json(
        auto_k_sweep_path,
        {
            str(k): {key: value for key, value in result.items() if key != "final_supernodes"}
            for k, result in sweep.items()
        },
    )
    _write_json(flow_report_path, flow_report)

    summary_row = _flatten_metrics(
        graph_name=graph_name,
        dataset=dataset,
        graph_path=graph_path,
        method=method_slug,
        method_family="ours",
        mean_method=method_config["mean_method"],
        normalization=method_config["normalization"],
        similarity_mode=method_config["similarity_mode"],
        best_k=best_k,
        auto_k_candidates=len(sweep),
        final_supernodes=final_supernodes,
        base_score=base_score,
        flow_report=flow_report,
        weighted_edge_cosine_mean=edge_cosine_mean,
        result_path=result_path,
        supernode_map_path=supernode_map_path,
        auto_k_sweep_path=auto_k_sweep_path,
        flow_report_path=flow_report_path,
    )
    result_payload = {
        **summary_row,
        "final_supernodes": final_supernodes,
        "score_details": base_score.get("details", {}),
        "flow_report": flow_report,
    }
    _write_json(result_path, result_payload)
    return result_payload


def _evaluate_baseline(
    *,
    prune_graph: PruneGraph,
    graph_path: Path,
    graph_name: str,
    dataset: str,
    output_dir: Path,
    method: str,
    features: np.ndarray | None,
    affinity: np.ndarray,
    clusterer: Callable[[int], list[list[str]]],
    k_min_override: int | None,
    k_max_override: int | None,
    weights: dict[str, float] | None,
    enforce_dag: bool,
) -> dict[str, Any]:
    best_k, sweep = _generic_auto_k(
        prune_graph=prune_graph,
        similarity=affinity,
        clusterer=clusterer,
        k_min_override=k_min_override,
        k_max_override=k_max_override,
        weights=weights,
        enforce_dag=enforce_dag,
    )
    best_result = sweep[best_k]
    final_supernodes = best_result["final_supernodes"]
    base_score = {key: value for key, value in best_result.items() if key != "final_supernodes"}

    run_dir = output_dir / "runs" / graph_name / method
    supernode_map_path = run_dir / "supernode_map.json"
    auto_k_sweep_path = run_dir / "auto_k_sweep.json"
    flow_report_path = run_dir / "flow_report.json"
    result_path = run_dir / "result.json"

    edge_cosine_mean = _within_cluster_mean_cosine(
        _edge_weighted_profile_features(prune_graph),
        final_supernodes,
        prune_graph,
    )
    sng = build_supernode_graph(prune_graph, final_supernodes, enforce_dag=enforce_dag)
    flow_report = flow_faithfulness_report(sng, final_supernodes)

    _write_json(supernode_map_path, final_supernodes)
    _write_json(
        auto_k_sweep_path,
        {
            str(k): {key: value for key, value in result.items() if key != "final_supernodes"}
            for k, result in sweep.items()
        },
    )
    _write_json(flow_report_path, flow_report)

    summary_row = _flatten_metrics(
        graph_name=graph_name,
        dataset=dataset,
        graph_path=graph_path,
        method=method,
        method_family="baseline",
        mean_method=None,
        normalization=None,
        similarity_mode=None,
        best_k=best_k,
        auto_k_candidates=len(sweep),
        final_supernodes=final_supernodes,
        base_score=base_score,
        flow_report=flow_report,
        weighted_edge_cosine_mean=edge_cosine_mean,
        result_path=result_path,
        supernode_map_path=supernode_map_path,
        auto_k_sweep_path=auto_k_sweep_path,
        flow_report_path=flow_report_path,
    )
    result_payload = {
        **summary_row,
        "final_supernodes": final_supernodes,
        "score_details": base_score.get("details", {}),
        "flow_report": flow_report,
        "feature_dim": int(features.shape[1]) if features is not None else None,
    }
    _write_json(result_path, result_payload)
    return result_payload


def evaluate_prune_graph(
    *,
    graph_path: Path,
    input_paths: Sequence[str],
    output_dir: Path,
    map_location: str,
    k_min_override: int | None,
    k_max_override: int | None,
    weights: dict[str, float] | None,
    max_layer_span: int,
    mediation_penalty: float,
    enforce_dag: bool,
    random_state: int,
    n_init: int,
) -> list[dict[str, Any]]:
    prune_graph = load_prune_graph(str(graph_path), map_location=map_location)
    graph_name, dataset = _graph_identity(graph_path, input_paths)
    rows: list[dict[str, Any]] = []

    for method_config in METHOD_GRID:
        rows.append(
            _evaluate_existing_method(
                prune_graph=prune_graph,
                graph_path=graph_path,
                graph_name=graph_name,
                dataset=dataset,
                output_dir=output_dir,
                method_config=method_config,
                k_min_override=k_min_override,
                k_max_override=k_max_override,
                weights=weights,
                max_layer_span=max_layer_span,
                mediation_penalty=mediation_penalty,
                enforce_dag=enforce_dag,
                random_state=random_state,
                n_init=n_init,
            )
        )

    mid_idx = _middle_indices(prune_graph)
    middle_ids = [prune_graph.kept_ids[i] for i in mid_idx]
    node_features = _node_profile_features(prune_graph)
    node_features_mid = node_features[mid_idx]
    node_profile_similarity = _cosine_similarity(node_features, nonnegative=True)

    def kmeans_clusterer(target_k: int) -> list[list[str]]:
        if len(middle_ids) == 0:
            return _labels_to_supernodes(prune_graph, [], np.array([], dtype=np.int64))
        if target_k >= len(middle_ids):
            labels = np.arange(len(middle_ids), dtype=np.int64)
        elif target_k == 1:
            labels = np.zeros(len(middle_ids), dtype=np.int64)
        else:
            labels = KMeans(
                n_clusters=target_k,
                random_state=random_state,
                n_init=n_init,
            ).fit_predict(node_features_mid)
        return _labels_to_supernodes(prune_graph, middle_ids, labels)

    rows.append(
        _evaluate_baseline(
            prune_graph=prune_graph,
            graph_path=graph_path,
            graph_name=graph_name,
            dataset=dataset,
            output_dir=output_dir,
            method="baseline-kmeans-node-profile",
            features=node_features,
            affinity=node_profile_similarity,
            clusterer=kmeans_clusterer,
            k_min_override=k_min_override,
            k_max_override=k_max_override,
            weights=weights,
            enforce_dag=enforce_dag,
        )
    )

    adjacency_affinity = _adjacency_affinity(prune_graph)
    adjacency_mid = adjacency_affinity[np.ix_(mid_idx, mid_idx)]

    def adjacency_spectral_clusterer(target_k: int) -> list[list[str]]:
        if len(middle_ids) == 0:
            return _labels_to_supernodes(prune_graph, [], np.array([], dtype=np.int64))
        if target_k >= len(middle_ids):
            labels = np.arange(len(middle_ids), dtype=np.int64)
        elif target_k == 1:
            labels = np.zeros(len(middle_ids), dtype=np.int64)
        else:
            labels = SpectralClustering(
                n_clusters=target_k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=random_state,
                n_init=n_init,
            ).fit_predict(adjacency_mid)
        return _labels_to_supernodes(prune_graph, middle_ids, labels)

    rows.append(
        _evaluate_baseline(
            prune_graph=prune_graph,
            graph_path=graph_path,
            graph_name=graph_name,
            dataset=dataset,
            output_dir=output_dir,
            method="baseline-spectral-adjacency",
            features=None,
            affinity=adjacency_affinity,
            clusterer=adjacency_spectral_clusterer,
            k_min_override=k_min_override,
            k_max_override=k_max_override,
            weights=weights,
            enforce_dag=enforce_dag,
        )
    )
    return rows


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    graph_paths = _discover_prune_graphs(args.input_path)
    summary_rows: list[dict[str, Any]] = []
    for graph_path in graph_paths:
        summary_rows.extend(
            evaluate_prune_graph(
                graph_path=graph_path,
                input_paths=args.input_path,
                output_dir=output_dir,
                map_location=args.map_location,
                k_min_override=args.k_min,
                k_max_override=args.k_max,
                weights={
                    "w_intra": args.w_intra,
                    "w_dag": args.w_dag,
                    "w_attr": args.w_attr,
                    "w_size": args.w_size,
                },
                max_layer_span=args.max_layer_span,
                mediation_penalty=args.mediation_penalty,
                enforce_dag=args.enforce_dag,
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
            "graph_paths": [str(path) for path in graph_paths],
            "output_dir": str(output_dir),
            "method_grid": METHOD_GRID,
            "baselines": [
                "baseline-kmeans-node-profile",
                "baseline-spectral-adjacency",
            ],
            "summary_csv": str(summary_path),
            "results_json": str(results_path),
            "n_graphs": len(graph_paths),
            "n_runs": len(summary_rows),
            "config": {
                "k_min": args.k_min,
                "k_max": args.k_max,
                "max_layer_span": args.max_layer_span,
                "mediation_penalty": args.mediation_penalty,
                "enforce_dag": args.enforce_dag,
                "map_location": args.map_location,
                "random_state": args.random_state,
                "n_init": args.n_init,
                "weights": {
                    "w_intra": args.w_intra,
                    "w_dag": args.w_dag,
                    "w_attr": args.w_attr,
                    "w_size": args.w_size,
                },
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
    parser.add_argument("--k-min", type=int, default=None)
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--max-layer-span", type=int, default=4)
    parser.add_argument("--mediation-penalty", type=float, default=0.1)
    parser.add_argument("--enforce-dag", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--w-intra", type=float, default=0.30)
    parser.add_argument("--w-dag", type=float, default=0.25)
    parser.add_argument("--w-attr", type=float, default=0.25)
    parser.add_argument("--w-size", type=float, default=0.20)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.input_path:
        args.input_path = _default_input_paths()
    result = run_evaluation(args)
    print("\n=== Clustering Evaluation Pipeline ===")
    print(f"output_dir: {result['output_dir']}")
    print(f"summary_csv: {result['summary_csv']}")
    print(f"results_json: {result['results_json']}")
    print(f"manifest_json: {result['manifest_json']}")
    print(f"n_graphs: {result['n_graphs']}")
    print(f"n_runs: {result['n_runs']}")


if __name__ == "__main__":
    main()
