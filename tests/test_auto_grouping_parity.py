from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from circuit_tracer.subgraph.auto_grouping import (
    eigengap_analysis as package_eigengap_analysis,
    find_best_k as package_find_best_k,
    score_k as package_score_k,
)
from circuit_tracer.subgraph.cluster import compute_similarity as package_compute_similarity
from circuit_tracer.subgraph.prune import PruneGraph


def _install_scipy_import_stubs() -> dict[str, types.ModuleType | None]:
    """
    Provide lightweight scipy modules so demo files import without hard scipy dependency.
    """
    scipy_module = types.ModuleType("scipy")
    scipy_cluster = types.ModuleType("scipy.cluster")
    scipy_cluster_hierarchy = types.ModuleType("scipy.cluster.hierarchy")
    scipy_spatial = types.ModuleType("scipy.spatial")
    scipy_spatial_distance = types.ModuleType("scipy.spatial.distance")
    scipy_linalg = types.ModuleType("scipy.linalg")

    setattr(scipy_cluster_hierarchy, "dendrogram", lambda *_args, **_kwargs: None)
    setattr(scipy_cluster_hierarchy, "fcluster", lambda *_args, **_kwargs: None)
    setattr(scipy_cluster_hierarchy, "linkage", lambda *_args, **_kwargs: None)
    setattr(scipy_spatial_distance, "squareform", lambda *_args, **_kwargs: None)
    setattr(scipy_linalg, "eigvalsh", np.linalg.eigvalsh)

    original_modules = {
        name: sys.modules.get(name)
        for name in [
            "scipy",
            "scipy.cluster",
            "scipy.cluster.hierarchy",
            "scipy.spatial",
            "scipy.spatial.distance",
            "scipy.linalg",
        ]
    }

    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.cluster"] = scipy_cluster
    sys.modules["scipy.cluster.hierarchy"] = scipy_cluster_hierarchy
    sys.modules["scipy.spatial"] = scipy_spatial
    sys.modules["scipy.spatial.distance"] = scipy_spatial_distance
    sys.modules["scipy.linalg"] = scipy_linalg
    return original_modules


def _restore_modules(original_modules: dict[str, types.ModuleType | None]) -> None:
    for name, original in original_modules.items():
        if original is None:
            del sys.modules[name]
        else:
            sys.modules[name] = original


def _load_module(path: Path, module_name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_demo_auto_grouping_module() -> tuple[types.ModuleType, types.ModuleType]:
    """
    Load demos/structure_grouping.py and demos/auto_grouping.py with import stubs.
    """
    original_scipy = _install_scipy_import_stubs()
    original_structure_grouping = sys.modules.get("structure_grouping")
    try:
        root = Path(__file__).resolve().parents[1] / "demos"
        structure_grouping = _load_module(
            root / "structure_grouping.py", "_demo_structure_grouping_for_auto_grouping_parity_test"
        )
        sys.modules["structure_grouping"] = structure_grouping
        auto_grouping = _load_module(root / "auto_grouping.py", "_demo_auto_grouping_for_parity_test")
        return structure_grouping, auto_grouping
    finally:
        if original_structure_grouping is None:
            if "structure_grouping" in sys.modules:
                del sys.modules["structure_grouping"]
        else:
            sys.modules["structure_grouping"] = original_structure_grouping
        _restore_modules(original_scipy)


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

    # receiver-indexed adjacency (both pipelines transpose internally to sender-indexed)
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


def test_eigengap_analysis_matches_demo_auto_grouping() -> None:
    demo_structure_grouping, demo_auto_grouping = _load_demo_auto_grouping_module()
    prune_graph, raw = _build_test_graph()

    package_similarity = package_compute_similarity(prune_graph, alpha=0.5, beta=0.5, mediation_penalty=0.1)
    demo_data = demo_structure_grouping.prepare_graph_data(raw)
    demo_similarity = demo_structure_grouping.compute_similarity(
        demo_data, alpha=0.5, beta=0.5, mediation_penalty=0.1
    )

    package_result = package_eigengap_analysis(package_similarity, prune_graph, max_k=5)
    demo_result = demo_auto_grouping.eigengap_analysis(demo_similarity, demo_data["kept_ids"], max_k=5)

    assert package_result["eigengap_k"] == demo_result["eigengap_k"]
    assert package_result["search_range"] == demo_result["search_range"]
    assert np.allclose(package_result["eigenvalues"], demo_result["eigenvalues"], rtol=1e-8, atol=1e-8)
    assert np.allclose(package_result["gaps"], demo_result["gaps"], rtol=1e-8, atol=1e-8)


def test_score_k_matches_demo_auto_grouping() -> None:
    pytest.importorskip("sklearn")

    demo_structure_grouping, demo_auto_grouping = _load_demo_auto_grouping_module()
    prune_graph, raw = _build_test_graph()

    package_similarity = package_compute_similarity(prune_graph, alpha=0.5, beta=0.5, mediation_penalty=0.1)
    demo_data = demo_structure_grouping.prepare_graph_data(raw)
    demo_similarity = demo_structure_grouping.compute_similarity(
        demo_data, alpha=0.5, beta=0.5, mediation_penalty=0.1
    )
    demo_supernodes = demo_structure_grouping.cluster_with_target_k(
        demo_data, demo_similarity, target_k=2, max_layer_span=4, max_sn=None
    )

    package_score = package_score_k(
        demo_supernodes,
        prune_graph,
        package_similarity,
        target_n_middle=4,
    )
    demo_score = demo_auto_grouping.score_k(
        demo_supernodes,
        demo_data,
        demo_similarity,
        target_n_middle=4,
    )

    keys = [
        "total",
        "intra_sim",
        "dag_safety",
        "flow_balance",
        "attr_balance",
        "size_score",
        "inf_conservation",
        "edge_conservation",
    ]
    for key in keys:
        assert package_score[key] == pytest.approx(demo_score[key], rel=1e-8, abs=1e-10)

    assert package_score["n_middle"] == demo_score["n_middle"]
    assert package_score["n_warnings"] == demo_score["n_warnings"]


def test_find_best_k_matches_demo_auto_grouping() -> None:
    pytest.importorskip("sklearn")

    demo_structure_grouping, demo_auto_grouping = _load_demo_auto_grouping_module()
    prune_graph, raw = _build_test_graph()

    package_similarity = package_compute_similarity(prune_graph, alpha=0.5, beta=0.5, mediation_penalty=0.1)
    demo_data = demo_structure_grouping.prepare_graph_data(raw)
    demo_similarity = demo_structure_grouping.compute_similarity(
        demo_data, alpha=0.5, beta=0.5, mediation_penalty=0.1
    )

    package_best_k, package_results = package_find_best_k(
        prune_graph,
        similarity=package_similarity,
        max_layer_span=4,
        k_min_override=2,
        k_max_override=3,
        max_sn=None,
        alpha=0.5,
        beta=0.5,
        mediation_penalty=0.1,
    )
    demo_best_k, demo_results = demo_auto_grouping.find_best_k(
        demo_data,
        demo_similarity,
        max_layer_span=4,
        k_min_override=2,
        k_max_override=3,
        max_sn=None,
    )

    assert package_best_k == demo_best_k
    assert set(package_results.keys()) == set(demo_results.keys())
    for k in package_results:
        assert package_results[k]["total"] == pytest.approx(demo_results[k]["total"], rel=1e-8, abs=1e-10)
