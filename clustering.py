from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

from circuit_tracer.subgraph.cluster import cluster_graph, compute_similarity
from circuit_tracer.subgraph.flow_analysis import build_supernode_graph, supernodes_to_mapping
from circuit_tracer.subgraph.prune import PruneGraph, load_prune_graph


def _parse_layer(node_id: str) -> int:
    if node_id.startswith("E"):
        return 0
    if node_id.startswith("27"):
        return 27
    try:
        return int(node_id.split("_")[0])
    except (ValueError, IndexError):
        return -1


def build_synthetic_prune_graph(seed: int = 42) -> PruneGraph:
    torch.manual_seed(seed)
    kept_ids = [
        "E_0_0",
        "E_1_0",
        "16_10_0",
        "16_11_0",
        "17_10_0",
        "17_11_0",
        "18_10_0",
        "18_11_0",
        "19_10_0",
        "19_11_0",
        "20_10_0",
        "20_11_0",
        "27_0_0",
    ]
    n = len(kept_ids)
    adj = torch.zeros((n, n), dtype=torch.float32)
    for i, src in enumerate(kept_ids):
        src_layer = _parse_layer(src)
        for j, dst in enumerate(kept_ids):
            if i == j:
                continue
            dst_layer = _parse_layer(dst)
            if src_layer <= dst_layer:
                w = torch.rand(1).item()
                if src.startswith("E"):
                    w *= 0.9
                if dst.startswith("27"):
                    w *= 1.1
                if w > 0.45:
                    adj[i, j] = round(w, 4)
    attr = {}
    for idx, nid in enumerate(kept_ids):
        layer = _parse_layer(nid)
        attr[nid] = {
            "relevance": float((layer + 1) / 30.0 + (idx % 3) * 0.05),
            "influence": float(max(0.0, (30 - layer) / 30.0 + ((idx + 1) % 4) * 0.03)),
            "activation": float(10 + layer),
            "is_target_logit": nid.startswith("27"),
        }
    node_scores = torch.tensor(
        [0.2 + (idx % 5) * 0.12 for idx in range(n)],
        dtype=torch.float32,
    ).clamp(0.0, 1.0)
    return PruneGraph(
        kept_ids=kept_ids,
        pruned_adj=adj.T.clone(),  # sender-indexed view inside clustering uses .T
        node_scores=node_scores,
        attr=attr,
        metadata={"source": "synthetic"},
    )


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "detach"):
        return obj.detach().cpu().tolist()
    return obj


def save_cluster_graph_visualization(
    final_supernodes: dict[str, list[str]],
    sng: dict[str, Any],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = nx.DiGraph()
    sn_names = sng["sn_names"]
    sn_adj = np.asarray(sng["sn_adj"])
    sn_inf = np.asarray(sng["sn_inf"])

    for i, sn in enumerate(sn_names):
        g.add_node(sn, influence=float(sn_inf[i]), size=max(300.0, 250.0 * len(final_supernodes.get(sn, []))))
    for i, src in enumerate(sn_names):
        for j, dst in enumerate(sn_names):
            if i == j or sn_adj[i, j] <= 0:
                continue
            g.add_edge(src, dst, weight=float(sn_adj[i, j]))

    if not g.nodes:
        return
    pos = nx.spring_layout(g, seed=42, k=1.2 / np.sqrt(max(len(g.nodes), 1)))
    node_sizes = [g.nodes[n]["size"] for n in g.nodes]
    node_colors = [g.nodes[n]["influence"] for n in g.nodes]
    edge_widths = [max(0.8, g.edges[e]["weight"] * 6) for e in g.edges]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="viridis",
        ax=ax,
    )
    nx.draw_networkx_labels(g, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(
        g,
        pos,
        arrows=True,
        arrowstyle="-|>",
        width=edge_widths,
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Clustered Supernode Graph")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_experiments(
    prune_graph: PruneGraph,
    out_dir: Path,
) -> dict[str, Any]:
    settings = [
        {"name": "scores_dag", "similarity_mode": "scores", "target_k": 4, "max_layer_span": 3, "mediation_penalty": 0.1, "enforce_dag": True},
        {"name": "relevance_dag", "similarity_mode": "relevance", "target_k": 4, "max_layer_span": 4, "mediation_penalty": 0.2, "enforce_dag": True},
        {"name": "influence_no_dag", "similarity_mode": "influence", "target_k": 5, "max_layer_span": 6, "mediation_penalty": 1.0, "enforce_dag": False},
        {"name": "uniform_no_dag", "similarity_mode": "uniform", "target_k": 5, "max_layer_span": 8, "mediation_penalty": 1.0, "enforce_dag": False},
    ]
    outputs: dict[str, Any] = {}
    for cfg in settings:
        sim = compute_similarity(
            prune_graph,
            mediation_penalty=cfg["mediation_penalty"],
            similarity_mode=cfg["similarity_mode"],
        )
        supernodes = cluster_graph(
            prune_graph,
            target_k=cfg["target_k"],
            max_layer_span=cfg["max_layer_span"],
            mediation_penalty=cfg["mediation_penalty"],
            similarity_mode=cfg["similarity_mode"],
            enforce_dag=cfg["enforce_dag"],
        )
        mapped = supernodes_to_mapping(prune_graph, supernodes)
        sng = build_supernode_graph(prune_graph, mapped)
        run_dir = out_dir / cfg["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "supernodes.json", "w") as f:
            json.dump(mapped, f, indent=2)
        with open(run_dir / "supernode_flow.json", "w") as f:
            json.dump(_to_jsonable(sng), f, indent=2)
        with open(run_dir / "similarity_matrix.json", "w") as f:
            json.dump(_to_jsonable(sim), f, indent=2)
        outputs[cfg["name"]] = {
            "config": cfg,
            "n_supernodes": len(mapped),
            "saved": {
                "supernodes": str(run_dir / "supernodes.json"),
                "supernode_flow": str(run_dir / "supernode_flow.json"),
                "similarity_matrix": str(run_dir / "similarity_matrix.json"),
            },
        }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clustering experiments and visualize clustered graph.")
    parser.add_argument("--prune-graph", type=str, default=None, help="Path to PruneGraph .pt file; if omitted, synthetic data is used.")
    parser.add_argument("--out-dir", type=str, default="outputs/clustering_runs")
    parser.add_argument("--final-run-name", type=str, default="influence_no_dag")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prune_graph = load_prune_graph(args.prune_graph) if args.prune_graph else build_synthetic_prune_graph()
    experiments = run_experiments(prune_graph, out_dir)

    if args.final_run_name not in experiments:
        raise ValueError(f"final-run-name {args.final_run_name!r} not found. Available: {sorted(experiments.keys())}")
    final_supernodes_path = Path(experiments[args.final_run_name]["saved"]["supernodes"])
    with open(final_supernodes_path) as f:
        final_supernodes = json.load(f)
    final_sng_path = Path(experiments[args.final_run_name]["saved"]["supernode_flow"])
    with open(final_sng_path) as f:
        final_sng = json.load(f)
    viz_path = out_dir / f"{args.final_run_name}_clustered_graph.png"
    save_cluster_graph_visualization(final_supernodes, final_sng, viz_path)

    summary = {
        "experiments": experiments,
        "final_run": args.final_run_name,
        "final_visualization": str(viz_path),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved experiment summary: {summary_path}")
    print(f"Saved final visualization: {viz_path}")


if __name__ == "__main__":
    main()
