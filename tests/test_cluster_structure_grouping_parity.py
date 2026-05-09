from __future__ import annotations

import torch

from summarization.cluster import cluster_graph, compute_similarity
from summarization.prune import PruneGraph
from summarization.utils import _node_from_json_dict


def _build_test_graph() -> PruneGraph:
    node_specs: list[tuple[str, dict]] = [
        ("E_0_0", {"feature_type": "embedding", "is_target_logit": False, "layer": "E"}),
        ("1_0_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "1"}),
        ("1_1_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "1"}),
        ("2_0_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "2"}),
        ("2_1_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "2"}),
        ("27_0_0", {"feature_type": "logit", "is_target_logit": True, "layer": "27"}),
    ]
    nodes = [_node_from_json_dict({"node_id": nid, **spec}) for nid, spec in node_specs]
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
        nodes=nodes,
        pruned_adj=pruned_adj,
        metadata={},
        node_influence=node_influence,
        node_relevance=node_relevance,
        edge_influence=edge_influence,
        edge_relevance=edge_relevance,
    )


def test_compute_similarity_shape_and_range() -> None:
    prune_graph = _build_test_graph()
    sim = compute_similarity(prune_graph, mean_method="arith")
    assert sim.shape == (len(prune_graph.node_ids), len(prune_graph.node_ids))
    assert torch.all(sim >= 0.0) and torch.all(sim <= 1.0)


def test_cluster_graph_spectral_output_shape() -> None:
    prune_graph = _build_test_graph()
    supernodes = cluster_graph(
        prune_graph,
        target_k=2,
        max_layer_span=4,
        max_sn=None,
        enforce_dag=False,
    )

    middle = [sn for sn in supernodes if not sn[0].startswith("E") and not sn[0].startswith("27")]
    fixed = [sn for sn in supernodes if sn[0].startswith("E") or sn[0].startswith("27")]
    assert len(middle) == 2
    assert len(fixed) == 2  # one embedding and one logit singleton in this fixture
