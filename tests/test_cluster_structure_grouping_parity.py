from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

from circuit_tracer.subgraph.cluster import cluster_graph, compute_similarity as package_compute_similarity
from circuit_tracer.subgraph.prune import PruneGraph


def _load_demo_structure_grouping_module() -> types.ModuleType:
    """
    Load demos/structure_grouping.py while stubbing scipy imports used only for dendrogram mode.
    """
    scipy_module = types.ModuleType("scipy")
    scipy_cluster = types.ModuleType("scipy.cluster")
    scipy_cluster_hierarchy = types.ModuleType("scipy.cluster.hierarchy")
    scipy_spatial = types.ModuleType("scipy.spatial")
    scipy_spatial_distance = types.ModuleType("scipy.spatial.distance")

    setattr(scipy_cluster_hierarchy, "dendrogram", lambda *_args, **_kwargs: None)
    setattr(scipy_cluster_hierarchy, "fcluster", lambda *_args, **_kwargs: None)
    setattr(scipy_cluster_hierarchy, "linkage", lambda *_args, **_kwargs: None)
    setattr(scipy_spatial_distance, "squareform", lambda *_args, **_kwargs: None)

    original_modules = {
        name: sys.modules.get(name)
        for name in [
            "scipy",
            "scipy.cluster",
            "scipy.cluster.hierarchy",
            "scipy.spatial",
            "scipy.spatial.distance",
        ]
    }

    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.cluster"] = scipy_cluster
    sys.modules["scipy.cluster.hierarchy"] = scipy_cluster_hierarchy
    sys.modules["scipy.spatial"] = scipy_spatial
    sys.modules["scipy.spatial.distance"] = scipy_spatial_distance

    try:
        demo_path = Path(__file__).resolve().parents[1] / "demos" / "structure_grouping.py"
        module_name = "_demo_structure_grouping_for_parity_test"
        spec = importlib.util.spec_from_file_location(module_name, demo_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create import spec for {demo_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                del sys.modules[name]
            else:
                sys.modules[name] = original


def _build_test_graph() -> tuple[PruneGraph, dict]:
    kept_ids = ["E_0_0", "1_0_0", "1_1_0", "2_0_0", "2_1_0", "27_0_0"]
    attr = {
        "E_0_0": {"feature_type": "embedding", "activation": None, "influence": 0.0, "is_target_logit": False},
        "1_0_0": {"feature_type": "sae_feature", "activation": 1.0, "influence": 0.3, "is_target_logit": False},
        "1_1_0": {"feature_type": "sae_feature", "activation": 0.9, "influence": 0.35, "is_target_logit": False},
        "2_0_0": {"feature_type": "sae_feature", "activation": 1.2, "influence": 0.5, "is_target_logit": False},
        "2_1_0": {"feature_type": "sae_feature", "activation": 1.1, "influence": 0.45, "is_target_logit": False},
        "27_0_0": {"feature_type": "logit", "activation": None, "influence": None, "is_target_logit": True},
    }

    # receiver-indexed adjacency (both implementations transpose internally to sender-indexed)
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

    prune_graph = PruneGraph(
        kept_ids=kept_ids,
        pruned_adj=pruned_adj,
        node_influence=torch.zeros(len(kept_ids)),
        node_relevance=torch.zeros(len(kept_ids)),
        attr=attr,
        metadata={},
    )
    raw = {"kept_ids": kept_ids, "pruned_adj": pruned_adj, "attr": attr}
    return prune_graph, raw


def _normalize_partition(clusters: list[list[str]]) -> set[frozenset[str]]:
    return {frozenset(cluster) for cluster in clusters}


def test_compute_similarity_matches_demo_structure_grouping() -> None:
    demo_structure_grouping = _load_demo_structure_grouping_module()
    prune_graph, raw = _build_test_graph()

    package_similarity = package_compute_similarity(
        prune_graph,
        alpha=0.5,
        beta=0.5,
        mediation_penalty=0.1,
    )

    data = demo_structure_grouping.prepare_graph_data(raw)
    demo_similarity = demo_structure_grouping.compute_similarity(
        data,
        alpha=0.5,
        beta=0.5,
        mediation_penalty=0.1,
    )

    assert torch.allclose(package_similarity, demo_similarity, atol=1e-6, rtol=1e-6)


def test_cluster_partition_matches_demo_structure_grouping() -> None:
    # cluster_with_target_k imports sklearn lazily.
    import pytest

    pytest.importorskip("sklearn")

    demo_structure_grouping = _load_demo_structure_grouping_module()
    prune_graph, raw = _build_test_graph()

    package_clusters = cluster_graph(
        prune_graph,
        target_k=2,
        max_layer_span=4,
        max_sn=None,
        alpha=0.5,
        beta=0.5,
        mediation_penalty=0.1,
    )

    data = demo_structure_grouping.prepare_graph_data(raw)
    similarity = demo_structure_grouping.compute_similarity(data, alpha=0.5, beta=0.5, mediation_penalty=0.1)
    demo_supernodes = demo_structure_grouping.cluster_with_target_k(
        data,
        similarity,
        target_k=2,
        max_layer_span=4,
        max_sn=None,
    )
    demo_clusters = list(demo_supernodes.values())

    assert _normalize_partition(package_clusters) == _normalize_partition(demo_clusters)
