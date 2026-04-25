from __future__ import annotations

import numpy as np
import torch

from summarization.auto_grouping import (
    eigengap_analysis,
    find_best_k,
    score_k,
)
from summarization.cluster import compute_similarity
from summarization.flow_analysis import build_supernode_graph, supernodes_to_mapping
from summarization.prune import PruneGraph


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


def test_eigengap_analysis_outputs_expected_keys() -> None:
    prune_graph = _build_test_graph()
    similarity = compute_similarity(
        prune_graph,
        mean_method="arith",
        mediation_penalty=0.1,
        similarity_mode="edge",
        normalization="cos_relu",
    )
    result = eigengap_analysis(similarity, prune_graph, max_k=5)
    assert {"eigengap_k", "eigenvalues", "gaps", "search_range"} <= set(result.keys())
    assert result["search_range"][0] <= result["search_range"][1]


def test_score_k_includes_flow_metrics_when_enabled() -> None:
    prune_graph = _build_test_graph()
    similarity = compute_similarity(
        prune_graph,
        mean_method="arith",
        mediation_penalty=0.1,
        similarity_mode="edge",
        normalization="cos_relu",
    )
    supernodes = [["1_0_0", "1_1_0"], ["2_0_0", "2_1_0"], ["E_0_0"], ["27_0_0"]]
    mapping = supernodes_to_mapping(prune_graph, supernodes)
    score = score_k(
        mapping,
        prune_graph,
        similarity,
        target_n_middle=4,
        use_flow_faithfulness=True,
        w_flow=0.4,
        enforce_dag=False,
    )
    assert "F_phi" in score
    assert "sigma_phi" in score
    assert score["total"] >= 0.0


def test_score_k_includes_intra_edge_cosine() -> None:
    prune_graph = _build_test_graph()
    similarity = compute_similarity(
        prune_graph,
        mean_method="arith",
        mediation_penalty=0.1,
        similarity_mode="edge",
        normalization="cos_relu",
    )
    supernodes = [["1_0_0", "1_1_0"], ["2_0_0", "2_1_0"], ["E_0_0"], ["27_0_0"]]
    mapping = supernodes_to_mapping(prune_graph, supernodes)
    score = score_k(
        mapping,
        prune_graph,
        similarity,
        target_n_middle=4,
        use_flow_faithfulness=False,
        enforce_dag=False,
    )
    assert "intra_edge_cosine_mean" in score
    assert -1.0 <= float(score["intra_edge_cosine_mean"]) <= 1.0


def test_find_best_k_returns_flow_enhanced_results() -> None:
    prune_graph = _build_test_graph()
    similarity = compute_similarity(
        prune_graph,
        mean_method="arith",
        mediation_penalty=0.1,
        similarity_mode="edge",
        normalization="cos_relu",
    )
    best_k, results = find_best_k(
        prune_graph,
        similarity=similarity,
        max_layer_span=4,
        k_min_override=2,
        k_max_override=3,
        max_sn=None,
        use_flow_faithfulness=True,
        w_flow=0.4,
        mediation_penalty=0.1,
        enforce_dag=False,
    )
    assert best_k in results
    assert all("F_phi" in v for v in results.values())
