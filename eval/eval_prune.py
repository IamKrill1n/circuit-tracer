from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch

from summarization.prune import (
    LogitWeightMode,
    prune_graph_pipeline,
    save_prune_graph,
)
from summarization.token_attribution import (
    NormalizeMethod,
    _normalize_scores,
    _special_token_mask,
    get_token_attribution_from_graph,
)
from summarization.utils import _build_index_sets, get_data_from_json

DEFAULT_SOURCE_SETS = ("clt-hp", "gemmascope-transcoder-16k")
DEFAULT_SHAP_EVAL_NORMALIZATIONS: tuple[NormalizeMethod, ...] = (
    "softmax",
    "relu_l1",
    "entmax15",
)
DEFAULT_SHAP_VALUES_JSON = Path("demos") / "shap_values.json"


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


def _strip_bos_from_prompt(prompt: str) -> str:
    p = (prompt or "").strip()
    if p.lower().startswith("<bos>"):
        p = p[5:].lstrip()
    return p.strip()


def _load_shap_values_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_shap_lookup(payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_prompt: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt", "")).strip()
        key = _strip_bos_from_prompt(prompt)
        if key:
            by_prompt[key] = row
        idx = row.get("index")
        if isinstance(idx, int):
            by_index[idx] = row
    return by_prompt, by_index


def _match_shap_row(
    stem: str,
    metadata: dict[str, Any],
    by_prompt: dict[str, dict[str, Any]],
    by_index: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    meta_prompt = str(metadata.get("prompt", "")).strip()
    key = _strip_bos_from_prompt(meta_prompt)
    if key and key in by_prompt:
        return by_prompt[key]
    if meta_prompt and meta_prompt in by_prompt:
        return by_prompt[meta_prompt]
    m = re.search(r"-p(\d+)-", stem, flags=re.IGNORECASE)
    if m:
        return by_index.get(int(m.group(1)))
    return None


def _scatter_raw_shap_into_prompt_positions(
    prompt_tokens: list[str],
    raw_shap: list[float],
) -> torch.Tensor:
    """Map JSON raw_shap (no BOS) onto full graph prompt_tokens (may include BOS)."""
    special = _special_token_mask(prompt_tokens)
    n = len(prompt_tokens)
    values = torch.zeros(n, dtype=torch.float32)
    j = 0
    for i in range(n):
        if bool(special[i].item()):
            continue
        if j >= len(raw_shap):
            raise ValueError(
                f"raw_shap too short: need more than index {j} for {n} prompt tokens "
                f"({int((~special).sum().item())} non-special positions)."
            )
        values[i] = float(raw_shap[j])
        j += 1
    expected = int((~special).sum().item())
    if j != len(raw_shap) or j != expected:
        raise ValueError(
            f"raw_shap length {len(raw_shap)} does not match non-special token count {expected} "
            f"(consumed {j})."
        )
    return values


def normalize_shap_values_for_prune(
    prompt_tokens: list[str],
    raw_shap: list[float],
    normalize_method: NormalizeMethod,
    *,
    masker_keep_prefix: int | None = None,
) -> torch.Tensor:
    """Map ``raw_shap`` onto full ``prompt_tokens`` and apply token normalization.

    ``masker_keep_prefix`` mirrors SHAP's Text masker ``keep_prefix``: the first *k*
    tokens are treated as fixed (excluded from the normalized mass), matching
    :func:`summarization.token_attribution.get_token_attribution`.
    """
    values = _scatter_raw_shap_into_prompt_positions(
        prompt_tokens, [float(x) for x in raw_shap]
    )
    special = _special_token_mask(prompt_tokens)
    if masker_keep_prefix is not None and int(masker_keep_prefix) > 0:
        k = min(int(masker_keep_prefix), int(special.shape[0]))
        special = special.clone()
        special[:k] = True
    return _normalize_scores(values.clone(), normalize_method, special)


def _token_weights_for_embeddings(
    normalized: torch.Tensor,
    node_ids: list[str],
    emb_idx: list[int],
) -> list[float]:
    weights: list[float] = []
    for i in emb_idx:
        nid = node_ids[i]
        parts = nid.split("_")
        ctx_idx = int(parts[-1])
        if ctx_idx < 0 or ctx_idx >= normalized.shape[0]:
            raise ValueError(f"ctx_idx {ctx_idx} out of range for normalized len={normalized.shape[0]} ({nid=})")
        weights.append(float(normalized[ctx_idx].item()))
    return weights


def _node_threshold_sweep(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("sweep step must be positive")
    out: list[float] = []
    t = start
    # Include end within float tolerance (e.g. 0.3 + 0.1 * 7 = 1.0)
    while t <= end + 1e-9:
        out.append(round(t, 6))
        t = round(t + step, 6)
    return out


def _sweep_axis(
    start_override: float | None,
    end_override: float | None,
    step_override: float | None,
    *,
    fallback_start: float,
    fallback_end: float,
    fallback_step: float,
) -> list[float]:
    """Build one threshold axis; unset overrides use the shared --sweep-node-* fallbacks."""
    return _node_threshold_sweep(
        float(start_override if start_override is not None else fallback_start),
        float(end_override if end_override is not None else fallback_end),
        float(step_override if step_override is not None else fallback_step),
    )


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_shap_json_sweep(args: argparse.Namespace) -> None:
    graphs_root = Path(args.graphs_root)
    output_root = Path(args.output_root)
    source_sets = tuple(args.source_sets)
    shap_path = Path(args.shap_values_json)

    payload = _load_shap_values_json(shap_path)
    by_prompt, by_index = _build_shap_lookup(payload)
    json_keep = payload.get("masker_keep_prefix")
    if args.masker_keep_prefix is not None:
        eff_keep_prefix: int | None = int(args.masker_keep_prefix)
    elif isinstance(json_keep, (int, float)) and int(json_keep) > 0:
        eff_keep_prefix = int(json_keep)
    else:
        eff_keep_prefix = None

    normalizations: tuple[NormalizeMethod, ...] = tuple(args.eval_normalizations)  # type: ignore[assignment]
    fb_s, fb_e, fb_st = float(args.sweep_node_start), float(args.sweep_node_end), float(args.sweep_node_step)
    node_inf_thresholds = _sweep_axis(
        args.sweep_node_influence_start,
        args.sweep_node_influence_end,
        args.sweep_node_influence_step,
        fallback_start=fb_s,
        fallback_end=fb_e,
        fallback_step=fb_st,
    )
    node_rel_thresholds = _sweep_axis(
        args.sweep_node_relevance_start,
        args.sweep_node_relevance_end,
        args.sweep_node_relevance_step,
        fallback_start=fb_s,
        fallback_end=fb_e,
        fallback_step=fb_st,
    )
    edge_threshold = float(args.edge_threshold)

    discovered = _discover_graph_files(graphs_root, source_sets)
    rows_out: list[dict[str, Any]] = []
    failures: list[str] = []

    total_runs = 0
    ok_runs = 0

    for source_set, graph_paths in discovered.items():
        print(f"\n=== Source set: {source_set} ({len(graph_paths)} files) ===")
        if args.limit is not None:
            graph_paths = graph_paths[: args.limit]

        for graph_path in graph_paths:
            stem = graph_path.stem
            try:
                _adj, node_ids, attr, metadata = get_data_from_json(str(graph_path))
                idx = _build_index_sets(node_ids, attr)
                emb_idx = idx["embedding"]
                prompt_tokens = [str(t) for t in (metadata.get("prompt_tokens") or [])]
                if not prompt_tokens:
                    raise ValueError("metadata.prompt_tokens missing or empty")

                row = _match_shap_row(stem, metadata, by_prompt, by_index)
                if row is None:
                    raise ValueError("no matching SHAP row (prompt / pNN index)")
                raw_shap = row.get("raw_shap")
                if not isinstance(raw_shap, list) or not raw_shap:
                    raise ValueError("matched SHAP row has no raw_shap list")

                for norm_method in normalizations:
                    normalized = normalize_shap_values_for_prune(
                        prompt_tokens,
                        [float(x) for x in raw_shap],
                        norm_method,  # type: ignore[arg-type]
                        masker_keep_prefix=eff_keep_prefix,
                    )
                    token_weights = _token_weights_for_embeddings(normalized, node_ids, emb_idx)

                    norm_dir = output_root / source_set / norm_method
                    norm_dir.mkdir(parents=True, exist_ok=True)

                    for node_inf_thr in node_inf_thresholds:
                        for node_rel_thr in node_rel_thresholds:
                            total_runs += 1
                            try:
                                prune_graph = prune_graph_pipeline(
                                    json_path=str(graph_path),
                                    logit_weights=args.logit_weights,
                                    token_weights=token_weights,
                                    node_influence_threshold=node_inf_thr,
                                    node_relevance_threshold=node_rel_thr,
                                    edge_threshold=edge_threshold,
                                    keep_all_tokens_and_logits=args.keep_all_tokens_and_logits,
                                    filter_act_density=args.filter_act_density,
                                    act_density_lb=args.act_density_lb,
                                    act_density_ub=args.act_density_ub,
                                )
                                graph_scores = prune_graph.graph_scores
                                thr_dir = norm_dir / f"node_inf_{node_inf_thr:.1f}_rel_{node_rel_thr:.1f}"
                                thr_dir.mkdir(parents=True, exist_ok=True)
                                prune_graph_path = thr_dir / f"{stem}_prune_graph.pt"
                                save_prune_graph(prune_graph, str(prune_graph_path))

                                rec = {
                                    "source_set": source_set,
                                    "graph_file": graph_path.name,
                                    "graph_stem": stem,
                                    "shap_json": str(shap_path),
                                    "shap_row_index": row.get("index"),
                                    "masker_keep_prefix": eff_keep_prefix,
                                    "normalize_method": norm_method,
                                    "node_influence_threshold": node_inf_thr,
                                    "node_relevance_threshold": node_rel_thr,
                                    "edge_threshold": edge_threshold,
                                    "num_nodes": prune_graph.num_nodes,
                                    "num_edges": prune_graph.num_edges,
                                    "graph_scores": graph_scores,
                                    "prune_graph_path": str(prune_graph_path),
                                }
                                rows_out.append(rec)
                                ok_runs += 1
                            except Exception as inner_exc:
                                msg = (
                                    f"{source_set}/{graph_path.name} "
                                    f"norm={norm_method} "
                                    f"node_inf={node_inf_thr} node_rel={node_rel_thr}: {inner_exc}"
                                )
                                failures.append(msg)
                                print(f"[failed] {msg}")
            except Exception as exc:
                msg = f"{source_set}/{graph_path.name}: {exc}"
                failures.append(msg)
                print(f"[failed] {msg}")

    source_out = output_root / (source_sets[0] if len(source_sets) == 1 else "multi")
    if len(source_sets) == 1:
        source_out = output_root / source_sets[0]
    else:
        source_out = output_root / "multi"

    source_out.mkdir(parents=True, exist_ok=True)
    results_path = source_out / "results.json"
    summary_path = source_out / "summary.csv"
    manifest_path = source_out / "manifest.json"

    _write_json(results_path, rows_out)

    if rows_out:
        fieldnames = list(rows_out[0].keys())
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)

    manifest = {
        "graphs_root": str(graphs_root),
        "output_root": str(output_root),
        "shap_values_json": str(shap_path),
        "source_sets": list(source_sets),
        "masker_keep_prefix": eff_keep_prefix,
        "eval_normalizations": list(normalizations),
        "sweep_node_start": float(args.sweep_node_start),
        "sweep_node_end": float(args.sweep_node_end),
        "sweep_node_step": float(args.sweep_node_step),
        "sweep_node_influence_start": args.sweep_node_influence_start,
        "sweep_node_influence_end": args.sweep_node_influence_end,
        "sweep_node_influence_step": args.sweep_node_influence_step,
        "sweep_node_relevance_start": args.sweep_node_relevance_start,
        "sweep_node_relevance_end": args.sweep_node_relevance_end,
        "sweep_node_relevance_step": args.sweep_node_relevance_step,
        "node_influence_thresholds": node_inf_thresholds,
        "node_relevance_thresholds": node_rel_thresholds,
        "edge_threshold": edge_threshold,
        "logit_weights": args.logit_weights,
        "limit": args.limit,
        "total_grid_cells_attempted": total_runs,
        "successful_runs": ok_runs,
        "n_result_rows": len(rows_out),
        "results_json": str(results_path),
        "summary_csv": str(summary_path),
        "failures": failures,
    }
    _write_json(manifest_path, manifest)

    print("\n=== SHAP JSON sweep summary ===")
    print(f"shap_values_json: {shap_path}")
    print(f"output: {output_root}")
    print(f"result rows: {len(rows_out)} (ok cells: {ok_runs}, failures: {len(failures)})")
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")
    if failures:
        print("\nFailures (first 20):")
        for item in failures[:20]:
            print(f"- {item}")
    if ok_runs == 0:
        raise RuntimeError("No successful prune+score runs in SHAP JSON sweep mode.")


def run_legacy(args: argparse.Namespace) -> None:
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
                    f"[ok] {graph_path.name} -> nodes={prune_graph.num_nodes}, "
                    f"edges={prune_graph.num_edges}, graph_scores={prune_graph.graph_scores}"
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


def run(args: argparse.Namespace) -> None:
    if args.shap_values_json:
        run_shap_json_sweep(args)
    else:
        run_legacy(args)


def _parse_logit_weights(value: str) -> LogitWeightMode:
    if value not in ("probs", "target"):
        raise argparse.ArgumentTypeError("--logit-weights must be 'probs' or 'target'.")
    return value  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run prune_graph_pipeline with SHAP token weights for graph JSON files "
            "in demos/temp_graph_files/<source_set>. "
            "With --shap-values-json, sweeps node_influence_threshold × node_relevance_threshold "
            "(optional normalization grid) using precomputed raw_shap from JSON (no HF model). "
            "Rows include graph_scores from summarization.prune (influence×relevance mass retained vs full graph)."
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
        default=None,
        help=(
            "Source-set subdirectories under --graphs-root. "
            "Default: clt-hp when --shap-values-json is set, else clt-hp and gemmascope-transcoder-16k."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Output root. Default: demos/eval_shap_prune when --shap-values-json is set, "
            "else demos/subgraph."
        ),
    )
    parser.add_argument(
        "--shap-values-json",
        type=str,
        default=None,
        help=(
            "Path to shap_values.json (raw_shap per prompt). Enables sweep mode: "
            "normalization grid, 2D node influence/relevance threshold sweep, graph_scores in results.json / summary.csv."
        ),
    )
    parser.add_argument(
        "--eval-normalizations",
        nargs="+",
        choices=["softmax", "relu_l1", "entmax15", "sparsemax"],
        default=list(DEFAULT_SHAP_EVAL_NORMALIZATIONS),
        help="Token-weight normalizations in sweep mode (default: softmax relu_l1 entmax15).",
    )
    parser.add_argument(
        "--masker-keep-prefix",
        type=int,
        default=None,
        help=(
            "SHAP masker keep_prefix for normalization (first k tokens pinned). "
            "Default: use top-level masker_keep_prefix from --shap-values-json if set."
        ),
    )
    parser.add_argument(
        "--sweep-node-start",
        type=float,
        default=0.0,
        help=(
            "Default sweep start for both node influence and relevance axes (inclusive). "
            "Overridden per axis by --sweep-node-influence-* / --sweep-node-relevance-* when set."
        ),
    )
    parser.add_argument(
        "--sweep-node-end",
        type=float,
        default=1.0,
        help="Default sweep end for both node axes (inclusive); per-axis overrides available.",
    )
    parser.add_argument(
        "--sweep-node-step",
        type=float,
        default=0.1,
        help="Default sweep step for both node axes; per-axis overrides available.",
    )
    parser.add_argument(
        "--sweep-node-influence-start",
        type=float,
        default=None,
        help="Override sweep start for node_influence_threshold only (else --sweep-node-start).",
    )
    parser.add_argument(
        "--sweep-node-influence-end",
        type=float,
        default=None,
        help="Override sweep end for node_influence_threshold only (else --sweep-node-end).",
    )
    parser.add_argument(
        "--sweep-node-influence-step",
        type=float,
        default=None,
        help="Override sweep step for node_influence_threshold only (else --sweep-node-step).",
    )
    parser.add_argument(
        "--sweep-node-relevance-start",
        type=float,
        default=None,
        help="Override sweep start for node_relevance_threshold only (else --sweep-node-start).",
    )
    parser.add_argument(
        "--sweep-node-relevance-end",
        type=float,
        default=None,
        help="Override sweep end for node_relevance_threshold only (else --sweep-node-end).",
    )
    parser.add_argument(
        "--sweep-node-relevance-step",
        type=float,
        default=None,
        help="Override sweep step for node_relevance_threshold only (else --sweep-node-step).",
    )
    parser.add_argument(
        "--model-name",
        default="google/gemma-2-2b",
        help="HF model name for SHAP attribution (legacy mode only).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
        help="Device used for model/explainer initialization (legacy mode only).",
    )
    parser.add_argument(
        "--normalize-method",
        choices=["softmax", "sparsemax", "entmax15", "relu_l1"],
        default="sparsemax",
        help="Normalization applied to SHAP attribution scores (legacy mode only).",
    )
    parser.add_argument(
        "--logit-weights",
        type=_parse_logit_weights,
        default="target",
    )
    parser.add_argument("--node-threshold", type=float, default=0.7)
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=None,
        help="Edge threshold. Default: 0.95 with --shap-values-json, else 0.98 (legacy).",
    )
    parser.add_argument("--keep-all-tokens-and-logits", action="store_true")
    parser.add_argument("--filter-act-density", action="store_true")
    parser.add_argument("--act-density-lb", type=float, default=2e-5)
    parser.add_argument("--act-density-ub", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Optional max files per source set.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.shap_values_json is None:
        args.shap_values_json = str(DEFAULT_SHAP_VALUES_JSON)
    if args.source_sets is None:
        args.source_sets = ["clt-hp"] if args.shap_values_json else list(DEFAULT_SOURCE_SETS)
    if args.output_root is None:
        args.output_root = "demos/eval_shap_prune" if args.shap_values_json else "demos/subgraph"
    if args.edge_threshold is None:
        args.edge_threshold = 0.95 if args.shap_values_json else 0.98
    run(args)


if __name__ == "__main__":
    main()
