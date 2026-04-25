from __future__ import annotations

import argparse
import json

import torch

from summarization.evaluation_pipeline import run_evaluation
from summarization.prune import PruneGraph, save_prune_graph


def _build_test_graph() -> PruneGraph:
    kept_ids = ["E_0_0", "1_0_0", "1_1_0", "2_0_0", "2_1_0", "27_0_0"]
    attr = {
        "E_0_0": {"feature_type": "embedding", "is_target_logit": False},
        "1_0_0": {"feature_type": "sae_feature", "is_target_logit": False},
        "1_1_0": {"feature_type": "sae_feature", "is_target_logit": False},
        "2_0_0": {"feature_type": "sae_feature", "is_target_logit": False},
        "2_1_0": {"feature_type": "sae_feature", "is_target_logit": False},
        "27_0_0": {"feature_type": "logit", "is_target_logit": True},
    }

    pruned_adj = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.6, 0.0, 0.2, 0.0],
            [0.0, 0.6, 0.7, 0.3, 0.0, 0.0],
            [0.0, 0.1, 0.2, 0.8, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )

    edge_relevance = pruned_adj.clone() * 0.8
    edge_influence = pruned_adj.clone() * 1.2
    node_relevance = torch.tensor([0.0, 0.4, 0.45, 0.7, 0.75, 0.0], dtype=torch.float32)
    node_influence = torch.tensor([0.0, 0.3, 0.35, 0.8, 0.85, 0.0], dtype=torch.float32)
    return PruneGraph(
        kept_ids=kept_ids,
        pruned_adj=pruned_adj,
        attr=attr,
        metadata={},
        node_influence=node_influence,
        node_relevance=node_relevance,
        edge_influence=edge_influence,
        edge_relevance=edge_relevance,
    )


def test_evaluation_pipeline_writes_summary_and_runs_all_methods(tmp_path) -> None:
    graph_path = tmp_path / "toy_prune_graph.pt"
    save_prune_graph(_build_test_graph(), str(graph_path))

    output_dir = tmp_path / "eval"
    args = argparse.Namespace(
        input_path=[str(graph_path)],
        output_dir=str(output_dir),
        map_location="cpu",
        k_min=2,
        k_max=3,
        max_layer_span=4,
        mediation_penalty=0.1,
        enforce_dag=False,
        random_state=42,
        n_init=5,
        w_intra=0.30,
        w_dag=0.25,
        w_attr=0.25,
        w_size=0.20,
    )

    result = run_evaluation(args)

    assert result["n_graphs"] == 1
    assert result["n_runs"] == 10
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "results.json").exists()
    assert (output_dir / "manifest.json").exists()

    rows = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert len(rows) == 10
    methods = {row["method"] for row in rows}
    assert "baseline-kmeans-node-profile" in methods
    assert "baseline-spectral-adjacency" in methods
    assert "ours-harm-cos-node" in methods
    assert "ours-arith-cos_relu-edge" in methods

    for row in rows:
        assert "within_cluster_weighted_edge_cosine_mean" in row
        assert "F_phi" in row
        assert row["result_path"]
