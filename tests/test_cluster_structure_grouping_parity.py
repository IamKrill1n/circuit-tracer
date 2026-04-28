from __future__ import annotations

import torch

from summarization.cluster import cluster_graph, compute_similarity
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
    # receiver-indexed adjacency
    pruned_adj = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.0, 0.2, 0.0, 0.0, 0.0],
            [0.7, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.8, 0.0, 0.3, 0.0],
            [0.0, 0.8, 0.9, 0.2, 0.0, 0.0],
            [0.0, 0.2, 0.3, 0.8, 0.9, 0.0],
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


def test_compute_similarity_uses_edge_channels() -> None:
    prune_graph = _build_test_graph()
    sim_out = compute_similarity(
        prune_graph,
        mean_method="arith",
        similarity_mode="edge",
        mediation_penalty=1.0,
    )
    sim_in = compute_similarity(
        prune_graph,
        mean_method="arith",
        similarity_mode="node",
        mediation_penalty=1.0,
    )

    assert sim_out.shape == sim_in.shape == (len(prune_graph.kept_ids), len(prune_graph.kept_ids))
    assert torch.all(sim_out >= 0.0) and torch.all(sim_out <= 1.0)
    assert torch.all(sim_in >= 0.0) and torch.all(sim_in <= 1.0)
    assert not torch.allclose(sim_out, sim_in)


def test_mediation_penalty_reduces_similarity() -> None:
    prune_graph = _build_test_graph()
    no_penalty = compute_similarity(
        prune_graph,
        mean_method="arith",
        similarity_mode="edge",
        mediation_penalty=1.0,
    )
    with_penalty = compute_similarity(
        prune_graph,
        mean_method="arith",
        similarity_mode="edge",
        mediation_penalty=0.1,
    )
    assert torch.all(with_penalty <= no_penalty + 1e-8)


def test_cluster_graph_spectral_output_shape() -> None:
    prune_graph = _build_test_graph()
    supernodes = cluster_graph(
        prune_graph,
        target_k=2,
        max_layer_span=4,
        max_sn=None,
        mediation_penalty=0.1,
        enforce_dag=False,
    )

    middle = [sn for sn in supernodes if not sn[0].startswith("E") and not sn[0].startswith("27")]
    fixed = [sn for sn in supernodes if sn[0].startswith("E") or sn[0].startswith("27")]
    assert len(middle) == 2
    assert len(fixed) == 2  # one embedding and one logit singleton in this fixture
