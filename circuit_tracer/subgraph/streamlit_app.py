from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import streamlit as st

from circuit_tracer.subgraph.auto_grouping import find_best_k
from circuit_tracer.subgraph.cluster import cluster_graph
from circuit_tracer.subgraph.cluster_viz import supernode_graph_figure
from circuit_tracer.subgraph.flow_analysis import build_supernode_graph, supernodes_to_mapping
from circuit_tracer.subgraph.prune import PruneGraph, load_prune_graph, prune_graph_pipeline

FULL_GRAPH_MODE = "Existing full graph JSON"
PRUNED_GRAPH_MODE = "Existing pruned graph (.pt)"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _list_files(paths: list[Path], suffix: str) -> list[Path]:
    files: list[Path] = []
    for directory in paths:
        if not directory.exists():
            continue
        files.extend(sorted(directory.glob(f"*{suffix}")))
    return sorted(files)


def _format_path_for_ui(path: Path) -> str:
    root = _repo_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _parse_token_weights(raw: str) -> list[float] | None:
    cleaned = raw.strip()
    if not cleaned:
        return None
    items = [part.strip() for part in cleaned.split(",") if part.strip()]
    return [float(item) for item in items]


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "detach"):
        return obj.detach().cpu().tolist()
    return obj


def _build_supernode_network(
    supernode_map: dict[str, list[str]],
    sng: dict[str, Any],
    edge_threshold: float,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    sn_names: list[str] = list(sng["sn_names"])
    sn_adj = np.asarray(sng["sn_adj"], dtype=np.float64)
    sn_inf = np.asarray(sng["sn_inf"], dtype=np.float64)

    for idx, name in enumerate(sn_names):
        members = supernode_map.get(name, [])
        if "EMB" in name:
            sn_type = "embedding"
        elif "LOGIT" in name:
            sn_type = "logit"
        else:
            sn_type = "middle"
        graph.add_node(
            name,
            n_members=len(members),
            influence=float(sn_inf[idx]),
            type=sn_type,
        )

    for i, src in enumerate(sn_names):
        for j, dst in enumerate(sn_names):
            if i == j:
                continue
            weight = float(sn_adj[i, j])
            if abs(weight) >= edge_threshold:
                graph.add_edge(src, dst, weight=weight)

    return graph


def _load_prune_graph(input_mode: str, input_path: Path, prune_cfg: dict[str, Any]) -> PruneGraph:
    if input_mode == FULL_GRAPH_MODE:
        return prune_graph_pipeline(
            json_path=str(input_path),
            logit_weights=prune_cfg["logit_weights"],
            token_weights=prune_cfg["token_weights"],
            node_threshold=prune_cfg["node_threshold"],
            edge_threshold=prune_cfg["edge_threshold"],
            alpha=prune_cfg["alpha"],
            keep_all_tokens_and_logits=prune_cfg["keep_all_tokens_and_logits"],
            filter_act_density=prune_cfg["filter_act_density"],
            combined_scores_method=prune_cfg["combined_scores_method"],
            normalization=prune_cfg["normalization"],
            act_density_lb=prune_cfg["act_density_lb"],
            act_density_ub=prune_cfg["act_density_ub"],
        )
    return load_prune_graph(str(input_path))


def _cluster_from_prune(
    prune_graph: PruneGraph,
    cluster_cfg: dict[str, Any],
    enforce_dag: bool,
    *,
    auto_k: bool,
    auto_k_cfg: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any], dict[str, Any]]:
    """
    Returns `(supernode_map, sng, run_meta)` where `run_meta` includes resolved `target_k`
    and optional auto-k diagnostics.
    """
    max_sn = cluster_cfg["max_sn"]
    target_k_user = int(cluster_cfg["target_k"])
    run_meta: dict[str, Any] = {"auto_k": bool(auto_k), "target_k_requested": target_k_user}

    target_k_eff = target_k_user
    if auto_k:
        best_k, sweep = find_best_k(
            prune_graph,
            max_layer_span=int(cluster_cfg["max_layer_span"]),
            k_min_override=auto_k_cfg["k_min_override"],
            k_max_override=auto_k_cfg["k_max_override"],
            weights=auto_k_cfg["weights"],
            max_sn=max_sn,
            gamma=float(cluster_cfg["gamma"]),
            mediation_penalty=float(cluster_cfg["mediation_penalty"]),
            similarity_mode=cluster_cfg["similarity_mode"],
            use_flow_faithfulness=bool(auto_k_cfg["use_flow_faithfulness"]),
            w_flow=float(auto_k_cfg["w_flow"]),
            enforce_dag=enforce_dag,
        )
        run_meta["sweep_size"] = len(sweep)
        if sweep:
            target_k_eff = int(best_k)
            run_meta["target_k_auto"] = target_k_eff
            run_meta["sweep_scores"] = {
                str(k): {kk: vv for kk, vv in v.items() if kk not in ("final_supernodes", "flow_report")}
                for k, v in sweep.items()
            }
        else:
            st.warning(
                "Auto-k search was skipped (too few middle nodes). Using the manual target_k "
                f"value ({target_k_user})."
            )
            run_meta["target_k_auto"] = None

    run_meta["target_k_used"] = int(target_k_eff)

    clusters = cluster_graph(
        prune_graph=prune_graph,
        target_k=int(target_k_eff),
        max_layer_span=int(cluster_cfg["max_layer_span"]),
        max_sn=max_sn,
        gamma=float(cluster_cfg["gamma"]),
        mediation_penalty=float(cluster_cfg["mediation_penalty"]),
        similarity_mode=cluster_cfg["similarity_mode"],
        enforce_dag=enforce_dag,
        random_state=int(cluster_cfg["random_state"]),
        n_init=int(cluster_cfg["n_init"]),
    )
    supernode_map = supernodes_to_mapping(prune_graph, clusters)
    sng = build_supernode_graph(prune_graph, supernode_map, enforce_dag=enforce_dag)
    return supernode_map, sng, run_meta


def main() -> None:
    st.set_page_config(page_title="Cluster Graph Visualizer", layout="wide")
    st.title("Cluster Graph Visualizer")
    st.caption(
        "Load a full graph JSON or a pruned graph .pt, run clustering, and visualize "
        "the final cluster graph with nodes and directed edges."
    )

    repo_root = _repo_root()
    default_full_files = _list_files(
        [repo_root / "demos" / "temp_graph_files", repo_root / "demos" / "graph_files"],
        ".json",
    )
    default_pruned_files = _list_files([repo_root / "demos" / "subgraph"], ".pt")

    input_mode = st.radio(
        "Input graph type",
        options=[FULL_GRAPH_MODE, PRUNED_GRAPH_MODE],
        horizontal=True,
    )

    if input_mode == FULL_GRAPH_MODE:
        default_files = default_full_files
        if not (repo_root / "demos" / "temp_graph_files").exists():
            st.info(
                "Directory `demos/temp_graph_files/` was not found here, so files from "
                "`demos/graph_files/` are also listed."
            )
    else:
        default_files = default_pruned_files

    file_options = ["<none>"] + [_format_path_for_ui(path) for path in default_files]
    selected_label = st.selectbox("Choose an input file", options=file_options, index=0)
    custom_path = st.text_input(
        "Or enter a path manually",
        value="",
        placeholder="e.g. demos/temp_graph_files/example.json",
    ).strip()

    selected_path: Path | None = None
    if custom_path:
        selected_path = Path(custom_path)
        if not selected_path.is_absolute():
            selected_path = repo_root / selected_path
    elif selected_label != "<none>":
        selected_path = repo_root / selected_label

    st.subheader("Display")
    edge_display_threshold = st.slider(
        "Display edge threshold (absolute weight)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )

    prune_cfg: dict[str, Any] = {
        "logit_weights": "target",
        "token_weights": None,
        "node_threshold": 0.8,
        "edge_threshold": 0.98,
        "alpha": 0.5,
        "keep_all_tokens_and_logits": True,
        "filter_act_density": False,
        "combined_scores_method": "geometric",
        "normalization": "min_max",
        "act_density_lb": 2e-5,
        "act_density_ub": 0.1,
    }

    prune_expander_title = (
        "Pruning parameters (full JSON only)"
        if input_mode == FULL_GRAPH_MODE
        else "Pruning parameters (optional overrides for .pt — not applied to loaded graph)"
    )
    with st.expander(prune_expander_title, expanded=input_mode == FULL_GRAPH_MODE):
        prune_col_a, prune_col_b, prune_col_c = st.columns(3)
        with prune_col_a:
            prune_cfg["logit_weights"] = st.selectbox("logit_weights", options=["target", "probs"])
            prune_cfg["node_threshold"] = st.slider(
                "node_threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.01,
            )
            prune_cfg["edge_threshold"] = st.slider(
                "edge_threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.98,
                step=0.01,
            )
        with prune_col_b:
            prune_cfg["alpha"] = st.slider("alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            prune_cfg["combined_scores_method"] = st.selectbox(
                "combined_scores_method",
                options=["geometric", "arithmetic", "harmonic"],
            )
            prune_cfg["normalization"] = st.selectbox(
                "normalization",
                options=["min_max", "rank"],
                index=0,
            )
        with prune_col_c:
            prune_cfg["keep_all_tokens_and_logits"] = st.checkbox(
                "keep_all_tokens_and_logits",
                value=True,
            )
            prune_cfg["filter_act_density"] = st.checkbox("filter_act_density", value=False)
            prune_cfg["act_density_lb"] = st.number_input("act_density_lb", value=2e-5, format="%.8f")
            prune_cfg["act_density_ub"] = st.number_input("act_density_ub", value=0.1, format="%.6f")

        token_weights_raw = st.text_area(
            "token_weights (comma-separated floats, optional)",
            value="",
            height=80,
            placeholder="0, 0, 0.33, 0, 0.67",
        )
        try:
            prune_cfg["token_weights"] = _parse_token_weights(token_weights_raw)
        except ValueError as exc:
            st.error(f"Invalid token_weights: {exc}")
            st.stop()
        if input_mode == PRUNED_GRAPH_MODE:
            st.caption(
                "These values match `prune_graph_pipeline` so you can reproduce pruning; "
                "they are not re-applied when loading an existing `.pt` file."
            )

    st.subheader("Clustering parameters")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        auto_k = st.checkbox("Pick k automatically (`find_best_k`)", value=False)
        target_k = st.slider(
            "target_k (ignored when auto-k finds a value)",
            min_value=1,
            max_value=50,
            value=7,
            step=1,
        )
        max_layer_span = st.slider("max_layer_span", min_value=1, max_value=32, value=4, step=1)
        max_sn_raw = st.number_input(
            "max_sn (0 means no cap)",
            min_value=0,
            value=0,
            step=1,
        )
        similarity_mode = st.selectbox(
            "similarity_mode",
            options=["edge", "scores", "relevance", "influence", "uniform"],
            index=0,
        )
    with col_b:
        gamma = st.slider("gamma", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        mediation_penalty = st.slider(
            "mediation_penalty",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
        )
        enforce_dag = st.checkbox("enforce_dag", value=True)
        random_state = st.number_input(
            "SpectralClustering random_state",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
        )
        n_init = st.number_input("SpectralClustering n_init (k-means)", min_value=1, max_value=100, value=20, step=1)
    with col_c:
        st.markdown("**Auto-k (`find_best_k`)**")
        use_flow_faithfulness = st.checkbox("use_flow_faithfulness", value=True)
        w_flow = st.slider("w_flow (blend with flow faithfulness)", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        k_min_override_raw = st.number_input(
            "k_min_override (0 = use eigengap range)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )
        k_max_override_raw = st.number_input(
            "k_max_override (0 = use eigengap range)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )
        w_intra = st.slider("score weight w_intra", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
        w_dag = st.slider("score weight w_dag", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
        w_attr = st.slider("score weight w_attr (attr balance)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
        w_size = st.slider("score weight w_size", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

    cluster_cfg = {
        "target_k": int(target_k),
        "max_layer_span": int(max_layer_span),
        "max_sn": None if int(max_sn_raw) == 0 else int(max_sn_raw),
        "gamma": float(gamma),
        "mediation_penalty": float(mediation_penalty),
        "similarity_mode": str(similarity_mode),
        "random_state": int(random_state),
        "n_init": int(n_init),
    }
    auto_k_cfg = {
        "use_flow_faithfulness": bool(use_flow_faithfulness),
        "w_flow": float(w_flow),
        "k_min_override": None if int(k_min_override_raw) == 0 else int(k_min_override_raw),
        "k_max_override": None if int(k_max_override_raw) == 0 else int(k_max_override_raw),
        "weights": {
            "w_intra": float(w_intra),
            "w_dag": float(w_dag),
            "w_flow": float(w_attr),
            "w_size": float(w_size),
        },
    }

    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None

    if st.button("Run clustering pipeline", type="primary"):
        if selected_path is None:
            st.error("Please select or enter an input path.")
            st.stop()
        if not selected_path.exists():
            st.error(f"Input file not found: {selected_path}")
            st.stop()

        with st.spinner("Running pipeline..."):
            try:
                prune_graph = _load_prune_graph(input_mode, selected_path, prune_cfg)
                supernode_map, sng, run_meta = _cluster_from_prune(
                    prune_graph,
                    cluster_cfg,
                    bool(enforce_dag),
                    auto_k=bool(auto_k),
                    auto_k_cfg=auto_k_cfg,
                )
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
                st.stop()

        st.session_state.pipeline_result = {
            "input_mode": input_mode,
            "input_path": str(selected_path),
            "prune_graph": prune_graph,
            "supernode_map": supernode_map,
            "sng": sng,
            "run_meta": run_meta,
        }

    result_payload = st.session_state.pipeline_result
    if result_payload is None:
        st.info("Choose an input and click **Run clustering pipeline**.")
        return

    prune_graph: PruneGraph = result_payload["prune_graph"]
    supernode_map: dict[str, list[str]] = result_payload["supernode_map"]
    sng: dict[str, Any] = result_payload["sng"]
    run_meta: dict[str, Any] = result_payload.get("run_meta", {})

    if run_meta.get("auto_k"):
        tk_auto = run_meta.get("target_k_auto")
        if tk_auto is not None:
            st.info(
                f"Auto k selected **{tk_auto}** (manual slider was {run_meta.get('target_k_requested')}; "
                f"sweep covered **{run_meta.get('sweep_size', 0)}** candidate values)."
            )
    st.caption(f"Clustering used **target_k = {run_meta.get('target_k_used', '?')}**.")

    supernode_graph = _build_supernode_network(
        supernode_map=supernode_map,
        sng=sng,
        edge_threshold=float(edge_display_threshold),
    )

    stat_col_1, stat_col_2, stat_col_3, stat_col_4 = st.columns(4)
    stat_col_1.metric("Pruned nodes", f"{prune_graph.num_nodes}")
    stat_col_2.metric("Pruned edges", f"{prune_graph.num_edges}")
    stat_col_3.metric("Final supernodes", f"{len(supernode_map)}")
    stat_col_4.metric("Supernode edges shown", f"{supernode_graph.number_of_edges()}")

    fig = supernode_graph_figure(
        sng=sng,
        final_supernodes=supernode_map,
        attr=prune_graph.attr,
        title="Final Cluster Graph",
    )
    st.plotly_chart(fig, use_container_width=True)

    sn_names: list[str] = list(sng["sn_names"])
    sn_adj = np.asarray(sng["sn_adj"], dtype=np.float64)
    sn_inf = np.asarray(sng["sn_inf"], dtype=np.float64)

    node_rows: list[dict[str, Any]] = []
    for idx, sn_name in enumerate(sn_names):
        members = supernode_map.get(sn_name, [])
        node_rows.append(
            {
                "supernode": sn_name,
                "type": supernode_graph.nodes[sn_name]["type"],
                "n_members": len(members),
                "influence": float(sn_inf[idx]),
                "members": ", ".join(members),
            }
        )

    edge_rows: list[dict[str, Any]] = []
    for i, src in enumerate(sn_names):
        for j, dst in enumerate(sn_names):
            if i == j:
                continue
            weight = float(sn_adj[i, j])
            if abs(weight) < float(edge_display_threshold):
                continue
            if abs(weight) > 0.0:
                edge_rows.append({"source": src, "target": dst, "weight": weight})

    st.subheader("Supernodes")
    st.dataframe(node_rows, use_container_width=True)

    st.subheader("Edges")
    st.dataframe(edge_rows, use_container_width=True)

    st.subheader("Downloads")
    st.download_button(
        "Download supernode map JSON",
        data=json.dumps(supernode_map, indent=2),
        file_name="supernodes.json",
        mime="application/json",
    )
    st.download_button(
        "Download supernode flow JSON",
        data=json.dumps(_to_jsonable(sng), indent=2),
        file_name="supernode_flow.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
