from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from circuit_tracer.subgraph import flow_analysis as package_flow_analysis


def _load_demo_flow_analysis_module() -> types.ModuleType:
    """
    Load demos/flow_analysis.py with a lightweight structure_grouping stub.
    """
    structure_grouping_stub = types.ModuleType("structure_grouping")
    setattr(structure_grouping_stub, "load_snapshot", lambda *_args, **_kwargs: None)
    setattr(structure_grouping_stub, "prepare_graph_data", lambda *_args, **_kwargs: None)
    setattr(structure_grouping_stub, "compute_similarity", lambda *_args, **_kwargs: None)
    setattr(structure_grouping_stub, "cluster_with_target_k", lambda *_args, **_kwargs: None)
    setattr(structure_grouping_stub, "check_dag_safety", lambda *_args, **_kwargs: None)
    setattr(structure_grouping_stub, "evaluate_grouping", lambda *_args, **_kwargs: None)
    setattr(structure_grouping_stub, "build_supernode_graph", lambda *_args, **_kwargs: None)
    setattr(structure_grouping_stub, "build_synthetic_snapshot", lambda *_args, **_kwargs: None)
    setattr(
        structure_grouping_stub,
        "parse_layer",
        lambda node_id: 0 if node_id.startswith("E") else (27 if node_id.startswith("27") else 1),
    )
    setattr(
        structure_grouping_stub,
        "_is_fixed",
        lambda node_id: node_id.startswith("E") or node_id.startswith("27"),
    )

    original_structure_grouping = sys.modules.get("structure_grouping")
    sys.modules["structure_grouping"] = structure_grouping_stub
    try:
        demo_path = Path(__file__).resolve().parents[1] / "demos" / "flow_analysis.py"
        module_name = "_demo_flow_analysis_for_parity_test"
        spec = importlib.util.spec_from_file_location(module_name, demo_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create import spec for {demo_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if original_structure_grouping is None:
            del sys.modules["structure_grouping"]
        else:
            sys.modules["structure_grouping"] = original_structure_grouping


def test_flow_faithfulness_matches_demo_implementation() -> None:
    demo_flow_analysis = _load_demo_flow_analysis_module()

    sng = {
        "sn_names": ["SN_EMB_0", "SN_0", "SN_1", "SN_LOGIT_0"],
        "sn_adj": np.array(
            [
                [0.0, 2.0, 1.0, 0.0],
                [0.0, 0.0, 0.5, 0.2],
                [0.0, 0.0, 0.0, 0.7],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "sn_inf": np.array([0.3, 0.1, -0.2, 0.0], dtype=np.float64),
    }
    final_supernodes = {
        "SN_EMB_0": ["E0"],
        "SN_0": ["1_0"],
        "SN_1": ["2_0"],
        "SN_LOGIT_0": ["27_0"],
    }

    demo_report = demo_flow_analysis.flow_faithfulness_report(sng, final_supernodes, top_k=5)
    package_report = package_flow_analysis.flow_faithfulness_report(sng, final_supernodes, top_k=5)

    metric_keys = [
        "F_phi",
        "path_score",
        "residual_score",
        "shortcut_score",
        "D_phi",
        "R_phi",
        "R_phi_balance",
        "R_phi_suppressive",
        "R_phi_max",
        "shortcut_frac",
        "top_k_frac",
    ]

    mismatches: list[str] = []
    for key in metric_keys:
        demo_value = demo_report["combined"][key]
        package_value = package_report["combined"][key]
        if package_value != pytest.approx(demo_value, rel=1e-9):
            mismatches.append(f"{key}: package={package_value!r}, demo={demo_value!r}")

    count_keys = ["n_paths", "n_shortcuts", "n_direct", "n_suppressive", "n_balanced"]
    for key in count_keys:
        demo_value = demo_report["combined"][key]
        package_value = package_report["combined"][key]
        if package_value != demo_value:
            mismatches.append(f"{key}: package={package_value!r}, demo={demo_value!r}")

    assert not mismatches, "Flow analysis mismatch:\n" + "\n".join(mismatches)
