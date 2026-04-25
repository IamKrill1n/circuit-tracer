#!/usr/bin/env python3
"""
Evaluate clustering on saved pruned graphs (e.g. demos/subgraph/clt/*.pt).

Example:

  python demos/run_cluster_eval.py \\
    --graph demos/subgraph/clt/example.pt \\
    --out-json demos/subgraph/cluster_eval_results.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from summarization.cluster_eval import run_hyperparam_grid, save_eval_json
from summarization.prune import load_prune_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Clustering evaluation grid + baselines")
    parser.add_argument(
        "--graph",
        type=str,
        required=True,
        help="Path to a .pt PruneGraph (e.g. demos/subgraph/clt/foo.pt)",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="demos/subgraph/cluster_eval.json",
        help="Where to write JSON results",
    )
    parser.add_argument("--max-layer-span", type=int, default=4)
    parser.add_argument("--mediation-penalty", type=float, default=0.1)
    parser.add_argument("--enforce-dag", action="store_true")
    parser.add_argument("--max-sn", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)
    args = parser.parse_args()

    graph_path = Path(args.graph)
    if not graph_path.is_file():
        raise SystemExit(f"Graph file not found: {graph_path}")

    prune_graph = load_prune_graph(str(graph_path))
    payload = run_hyperparam_grid(
        prune_graph,
        max_layer_span=args.max_layer_span,
        max_sn=args.max_sn,
        mediation_penalty=args.mediation_penalty,
        enforce_dag=args.enforce_dag,
        random_state=args.random_state,
        n_init=args.n_init,
    )
    payload["graph_path"] = str(graph_path.resolve())
    save_eval_json(payload, args.out_json)
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
