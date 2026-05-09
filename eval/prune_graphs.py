"""Mass-prune graph JSON files using SHAP-derived token weights.

Sweeps a single ``node_threshold`` (combined-score cutoff) x normalization,
saves PruneGraph .pt files plus a manifest.json describing every prune cell.
The manifest is consumed by ``eval/eval_prune.py`` to compute evaluation metrics.
"""

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
)
from summarization.utils import _build_index_sets, get_data_from_json

DEFAULT_SOURCE_SETS = ("clt-hp",)
DEFAULT_SHAP_EVAL_NORMALIZATIONS: tuple[NormalizeMethod, ...] = (
    "softmax",
    "entmax",
    "entmax15",
)
DEFAULT_SHAP_VALUES_JSON = Path("demos") / "shap_values.json"


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
    entmax_alpha: float | None = None,
) -> torch.Tensor:
    """Map ``raw_shap`` onto full ``prompt_tokens`` and apply token normalization."""
    values = _scatter_raw_shap_into_prompt_positions(
        prompt_tokens, [float(x) for x in raw_shap]
    )
    special = _special_token_mask(prompt_tokens)
    if masker_keep_prefix is not None and int(masker_keep_prefix) > 0:
        k = min(int(masker_keep_prefix), int(special.shape[0]))
        special = special.clone()
        special[:k] = True
    return _normalize_scores(
        values.clone(),
        normalize_method,
        special,
        entmax_alpha=entmax_alpha,
    )


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
    while t <= end + 1e-9:
        out.append(round(t, 6))
        t = round(t + step, 6)
    return out


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_prune_sweep(args: argparse.Namespace) -> None:
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
    node_thresholds = _node_threshold_sweep(
        float(args.sweep_node_start),
        float(args.sweep_node_end),
        float(args.sweep_node_step),
    )
    edge_threshold = float(args.edge_threshold)
    combine_method = str(args.combine_method)
    score_normalization = str(args.normalization)
    alpha = float(args.alpha)

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
                _adj, nodes, metadata = get_data_from_json(str(graph_path))
                node_ids = [n.node_id for n in nodes]
                idx = _build_index_sets(nodes)
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
                        entmax_alpha=args.entmax_alpha,
                    )
                    token_weights = _token_weights_for_embeddings(normalized, node_ids, emb_idx)

                    norm_dir = output_root / source_set / norm_method
                    norm_dir.mkdir(parents=True, exist_ok=True)

                    for node_thr in node_thresholds:
                        total_runs += 1
                        try:
                            prune_graph = prune_graph_pipeline(
                                json_path=str(graph_path),
                                logit_weights=args.logit_weights,
                                token_weights=token_weights,
                                node_threshold=node_thr,
                                edge_threshold=edge_threshold,
                                combine_method=combine_method,  # type: ignore[arg-type]
                                normalization=score_normalization,  # type: ignore[arg-type]
                                alpha=alpha,
                                keep_all_tokens_and_logits=args.keep_all_tokens_and_logits,
                                filter_act_density=args.filter_act_density,
                                act_density_lb=args.act_density_lb,
                                act_density_ub=args.act_density_ub,
                            )
                            thr_dir = norm_dir / f"node_{node_thr:.2f}"
                            thr_dir.mkdir(parents=True, exist_ok=True)
                            prune_graph_path = thr_dir / f"{stem}_prune_graph.pt"
                            save_prune_graph(prune_graph, str(prune_graph_path))

                            rec = {
                                "source_set": source_set,
                                "graph_file": graph_path.name,
                                "graph_stem": stem,
                                "graph_path": str(graph_path),
                                "shap_json": str(shap_path),
                                "shap_row_index": row.get("index"),
                                "masker_keep_prefix": eff_keep_prefix,
                                "normalize_method": norm_method,
                                "node_threshold": node_thr,
                                "edge_threshold": edge_threshold,
                                "combine_method": combine_method,
                                "score_normalization": score_normalization,
                                "alpha": alpha,
                                "logit_weights": args.logit_weights,
                                "keep_all_tokens_and_logits": bool(args.keep_all_tokens_and_logits),
                                "filter_act_density": bool(args.filter_act_density),
                                "act_density_lb": float(args.act_density_lb),
                                "act_density_ub": float(args.act_density_ub),
                                "token_weights": [float(w) for w in token_weights],
                                "num_nodes": prune_graph.num_nodes,
                                "num_edges": prune_graph.num_edges,
                                "prune_graph_path": str(prune_graph_path),
                            }
                            rows_out.append(rec)
                            ok_runs += 1
                        except Exception as inner_exc:
                            msg = (
                                f"{source_set}/{graph_path.name} "
                                f"norm={norm_method} node={node_thr}: {inner_exc}"
                            )
                            failures.append(msg)
                            print(f"[failed] {msg}")
            except Exception as exc:
                msg = f"{source_set}/{graph_path.name}: {exc}"
                failures.append(msg)
                print(f"[failed] {msg}")

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
        # token_weights is a variable-length list per graph; drop from CSV for stability.
        csv_fields = [k for k in rows_out[0].keys() if k != "token_weights"]
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
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
        "node_thresholds": node_thresholds,
        "edge_threshold": edge_threshold,
        "combine_method": combine_method,
        "score_normalization": score_normalization,
        "alpha": alpha,
        "logit_weights": args.logit_weights,
        "keep_all_tokens_and_logits": bool(args.keep_all_tokens_and_logits),
        "filter_act_density": bool(args.filter_act_density),
        "act_density_lb": float(args.act_density_lb),
        "act_density_ub": float(args.act_density_ub),
        "limit": args.limit,
        "total_grid_cells_attempted": total_runs,
        "successful_runs": ok_runs,
        "n_result_rows": len(rows_out),
        "results_json": str(results_path),
        "summary_csv": str(summary_path),
        "records": rows_out,
        "failures": failures,
    }
    _write_json(manifest_path, manifest)

    print("\n=== Prune sweep summary ===")
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
        raise RuntimeError("No successful prune runs.")


def _parse_logit_weights(value: str) -> LogitWeightMode:
    if value not in ("probs", "target"):
        raise argparse.ArgumentTypeError("--logit-weights must be 'probs' or 'target'.")
    return value  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mass-prune graph JSON files using SHAP token weights. Sweeps a single "
            "node_threshold (combined-score cutoff) x normalization. Produces "
            "PruneGraph .pt files plus a manifest.json consumed by eval/eval_prune.py."
        )
    )
    parser.add_argument("--graphs-root", default="demos/temp_graph_files")
    parser.add_argument("--source-sets", nargs="+", default=None)
    parser.add_argument("--output-root", default="eval_outputs/prune/subgraph")
    parser.add_argument("--shap-values-json", type=str, default=str(DEFAULT_SHAP_VALUES_JSON))
    parser.add_argument(
        "--eval-normalizations",
        nargs="+",
        choices=["softmax", "entmax", "entmax15", "sparsemax"],
        default=list(DEFAULT_SHAP_EVAL_NORMALIZATIONS),
    )
    parser.add_argument("--masker-keep-prefix", type=int, default=None)
    parser.add_argument("--sweep-node-start", type=float, default=0.0)
    parser.add_argument("--sweep-node-end", type=float, default=1.0)
    parser.add_argument("--sweep-node-step", type=float, default=0.1)
    parser.add_argument("--entmax-alpha", type=float, default=1.25)
    parser.add_argument("--logit-weights", type=_parse_logit_weights, default="target")
    parser.add_argument("--edge-threshold", type=float, default=0.95)
    parser.add_argument(
        "--combine-method",
        choices=["geometric", "arithmetic", "harmonic"],
        default="geometric",
    )
    parser.add_argument(
        "--normalization",
        choices=["rank", "min_max"],
        default="rank",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--keep-all-tokens-and-logits", action="store_true")
    parser.add_argument("--filter-act-density", action="store_true")
    parser.add_argument("--act-density-lb", type=float, default=2e-5)
    parser.add_argument("--act-density-ub", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.source_sets is None:
        args.source_sets = list(DEFAULT_SOURCE_SETS)
    run_prune_sweep(args)


if __name__ == "__main__":
    main()
