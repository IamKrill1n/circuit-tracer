"""Analyze pruning eval results: compare normalization schemes and baselines.

Usage:
    python eval/analyze_prune.py \
        --combined eval_outputs/prune/subgraph/clt-hp/eval/eval_results.json \
        --baseline eval_outputs/prune/subgraph_alpha1/clt-hp/eval/eval_results.json \
        --output eval_outputs/prune/analysis/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


METRICS = [
    "token_attribution_faithfulness",
    "relevance_conservation_rate",
    "asymmetric_pruning_divergence",
    "influence_relevance_agreement",
]

METHODS = ["softmax", "entmax", "entmax15"]
METHOD_LABELS = {"softmax": "Softmax", "entmax": "Entmax (α=1.25)", "entmax15": "Entmax-1.5"}
METHOD_COLORS = {"softmax": "#1f77b4", "entmax": "#ff7f0e", "entmax15": "#2ca02c"}
BASELINE_COLOR = "#d62728"


def load_results(path: str) -> pd.DataFrame:
    with open(path) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    for m in METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    df["node_threshold"] = pd.to_numeric(df["node_threshold"], errors="coerce")
    df["num_nodes"] = pd.to_numeric(df["num_nodes"], errors="coerce")
    df["num_edges"] = pd.to_numeric(df["num_edges"], errors="coerce")
    return df


def compression_ratio(df: pd.DataFrame) -> pd.Series:
    """num_nodes relative to max (full graph = threshold 1.0)."""
    max_nodes = df.groupby("graph_stem")["num_nodes"].transform("max")
    return df["num_nodes"] / max_nodes.clip(lower=1)


def plot_faithfulness_curves(
    df_combined: pd.DataFrame,
    df_baseline: pd.DataFrame | None,
    output_dir: Path,
    metric: str = "token_attribution_faithfulness",
    y_label: str = "Token attribution faithfulness",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Mean ± std across graphs for each (method, threshold)
    for method in METHODS:
        sub = df_combined[df_combined["normalize_method"] == method].copy()
        grouped = (
            sub.groupby("node_threshold")[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.plot(
            grouped["node_threshold"],
            grouped["mean"],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            marker="o",
            markersize=4,
        )
        ax.fill_between(
            grouped["node_threshold"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15,
            color=METHOD_COLORS[method],
        )

    if df_baseline is not None and metric in df_baseline.columns:
        # Baseline uses fixed alpha=1.0, show as dashed line
        # Baseline might only have one normalization method — aggregate all
        grouped_b = (
            df_baseline.groupby("node_threshold")[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.plot(
            grouped_b["node_threshold"],
            grouped_b["mean"],
            label="Influence-only (CLT baseline)",
            color=BASELINE_COLOR,
            linestyle="--",
            marker="s",
            markersize=4,
        )

    ax.set_xlabel("Node threshold", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f"{y_label} vs. pruning threshold", fontsize=12)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = output_dir / f"{metric}_vs_threshold.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


def summary_table(
    df_combined: pd.DataFrame,
    df_baseline: pd.DataFrame | None,
    threshold: float = 0.7,
) -> pd.DataFrame:
    rows = []
    target = df_combined[np.isclose(df_combined["node_threshold"], threshold)]
    for method in METHODS:
        sub = target[target["normalize_method"] == method]
        row = {"method": METHOD_LABELS[method], "alpha": 0.5}
        for m in METRICS:
            if m in sub.columns:
                row[m] = round(float(sub[m].mean()), 4)
        row["mean_nodes"] = int(sub["num_nodes"].mean())
        rows.append(row)

    if df_baseline is not None:
        btarget = df_baseline[np.isclose(df_baseline["node_threshold"], threshold)]
        row = {"method": "Influence-only (CLT)", "alpha": 1.0}
        for m in METRICS:
            if m in btarget.columns:
                row[m] = round(float(btarget[m].mean()), 4)
        row["mean_nodes"] = int(btarget["num_nodes"].mean()) if len(btarget) else None
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", required=True, help="eval_results.json for combined sweep")
    parser.add_argument("--baseline", default=None, help="eval_results.json for influence-only baseline")
    parser.add_argument("--output", default="eval_outputs/prune/analysis")
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.combined)
    df_b = load_results(args.baseline) if args.baseline else None

    print(f"\nLoaded {len(df)} combined records, {len(df_b) if df_b is not None else 0} baseline records")
    print(f"Methods: {sorted(df['normalize_method'].unique())}")
    print(f"Thresholds: {sorted(df['node_threshold'].unique())}")
    print(f"Graphs: {df['graph_stem'].nunique()}")

    # Summary table at target threshold
    print(f"\n=== Summary at node_threshold={args.threshold} ===")
    table = summary_table(df, df_b, threshold=args.threshold)
    print(table.to_string(index=False))
    table.to_csv(output_dir / f"summary_thr{args.threshold}.csv", index=False)

    # Faithfulness curve plots
    metric_labels = {
        "token_attribution_faithfulness": "Token attribution faithfulness",
        "relevance_conservation_rate": "Relevance conservation rate",
        "asymmetric_pruning_divergence": "Asymmetric pruning divergence",
        "influence_relevance_agreement": "Influence–relevance agreement",
    }
    for m, label in metric_labels.items():
        if m in df.columns:
            plot_faithfulness_curves(df, df_b, output_dir, metric=m, y_label=label)

    print(f"\nDone. Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
