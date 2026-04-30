from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal, cast

import networkx as nx
import numpy as np
import streamlit as st

from api import save_subgraph
from summarization.auto_grouping import find_best_k
from summarization.cluster import build_supernode_graph, cluster_graph, supernodes_to_mapping
from summarization.cluster_viz import supernode_graph_figure
from summarization.flow_analysis import flow_faithfulness_report
from summarization.prune import PruneGraph, load_prune_graph, prune_graph_pipeline, save_prune_graph
from summarization.token_attribution import get_token_attribution_from_graph

FULL_GRAPH_MODE = "Existing full graph JSON"
PRUNED_GRAPH_MODE = "Existing pruned graph (.pt)"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _list_files(paths: list[Path], suffix: str) -> list[Path]:
    files: list[Path] = []
    for directory in paths:
        if not directory.exists():
            continue
        files.extend(sorted(directory.glob(f"*{suffix}")))
    return sorted(files)


def _list_pruned_graphs(root: Path) -> list[Path]:
    """List all pruned-graph .pt files under `demos/subgraph` recursively.

    Excludes the `cache/` subtree (auto-generated streamlit cache) and any
    companion `*_token_weights.pt` files which are not PruneGraph payloads.
    """
    if not root.exists():
        return []
    found: list[Path] = []
    for path in sorted(root.rglob("*.pt")):
        if not path.is_file():
            continue
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            rel_parts = path.parts
        if rel_parts and rel_parts[0] == "cache":
            continue
        if path.name.endswith("_token_weights.pt"):
            continue
        found.append(path)
    return found


def _format_path_for_ui(path: Path) -> str:
    root = _repo_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return cleaned or "graph"


def _prune_cache_path(repo_root: Path, input_path: Path, prune_cfg: dict[str, Any]) -> Path:
    try:
        rel_input = str(input_path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        rel_input = str(input_path.resolve())

    key_payload = {
        "input_path": rel_input,
        "logit_weights": prune_cfg["logit_weights"],
        "token_weights": prune_cfg["token_weights"],
        "node_influence_threshold": float(prune_cfg["node_influence_threshold"]),
        "node_relevance_threshold": float(prune_cfg["node_relevance_threshold"]),
        "edge_influence_threshold": float(prune_cfg["edge_influence_threshold"]),
        "edge_relevance_threshold": float(prune_cfg["edge_relevance_threshold"]),
        "keep_all_tokens_and_logits": bool(prune_cfg["keep_all_tokens_and_logits"]),
        "filter_act_density": bool(prune_cfg["filter_act_density"]),
        "act_density_lb": float(prune_cfg["act_density_lb"]),
        "act_density_ub": float(prune_cfg["act_density_ub"]),
        "token_attribution_model": prune_cfg.get("token_attribution_model"),
        "token_attribution_normalize": prune_cfg.get("token_attribution_normalize"),
        "token_attribution_masker_keep_prefix": prune_cfg.get("token_attribution_masker_keep_prefix"),
    }
    key_raw = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()[:16]

    cache_dir = repo_root / "demos" / "subgraph" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{_slugify(input_path.stem)}-{digest}.pt"


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


def _clustered_supernodes_for_upload(supernode_map: dict[str, list[str]]) -> list[list[str]]:
    """Convert clustered supernodes into Neuronpedia upload rows."""
    indexed_rows: list[tuple[int, str, list[str]]] = []
    for name, members in supernode_map.items():
        is_boundary = ("EMB" in name) or ("LOGIT" in name)
        if len(members) <= 1 or is_boundary:
            continue
        try:
            cluster_idx = int(name.rsplit("_", 1)[1])
        except ValueError:
            cluster_idx = 10**9
        indexed_rows.append((cluster_idx, name, members))
    indexed_rows.sort(key=lambda row: (row[0], row[1]))
    return [[f"cluster_{idx}", *members] for idx, (_, _, members) in enumerate(indexed_rows)]


def _pretty_response_body(body: str) -> str:
    try:
        return json.dumps(json.loads(body), indent=2)
    except json.JSONDecodeError:
        return body


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
            node_influence_threshold=prune_cfg["node_influence_threshold"],
            node_relevance_threshold=prune_cfg["node_relevance_threshold"],
            edge_influence_threshold=prune_cfg["edge_influence_threshold"],
            edge_relevance_threshold=prune_cfg["edge_relevance_threshold"],
            keep_all_tokens_and_logits=prune_cfg["keep_all_tokens_and_logits"],
            filter_act_density=prune_cfg["filter_act_density"],
            act_density_lb=prune_cfg["act_density_lb"],
            act_density_ub=prune_cfg["act_density_ub"],
        )
    return load_prune_graph(str(input_path))


def _resolve_device(device_flag: str) -> str:
    if device_flag == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"
    return device_flag


def _compute_shap_token_weights(input_path: Path, prune_cfg: dict[str, Any]) -> list[float]:
    device = _resolve_device(str(prune_cfg.get("token_attribution_device", "auto")))
    weights = get_token_attribution_from_graph(
        graph_path=input_path,
        model_name=str(prune_cfg["token_attribution_model"]),
        normalize_method=prune_cfg["token_attribution_normalize"],
        device=device,
        masker_keep_prefix=prune_cfg.get("token_attribution_masker_keep_prefix"),
    )
    return [float(value) for value in weights.detach().cpu().tolist()]


def _load_or_build_prune_graph(
    repo_root: Path,
    input_mode: str,
    input_path: Path,
    prune_cfg: dict[str, Any],
) -> tuple[PruneGraph, dict[str, Any]]:
    if input_mode != FULL_GRAPH_MODE:
        return load_prune_graph(str(input_path)), {
            "cache_hit": None,
            "prune_graph_path": str(input_path),
            "source": "provided_pt",
        }

    prune_cfg["token_weights"] = _compute_shap_token_weights(input_path, prune_cfg)

    cached_path = _prune_cache_path(repo_root, input_path, prune_cfg)
    if cached_path.exists():
        return load_prune_graph(str(cached_path)), {
            "cache_hit": True,
            "prune_graph_path": str(cached_path),
            "source": "cache",
            "token_weights": prune_cfg["token_weights"],
        }

    prune_graph = _load_prune_graph(input_mode, input_path, prune_cfg)
    save_prune_graph(prune_graph, str(cached_path))
    return prune_graph, {
        "cache_hit": False,
        "prune_graph_path": str(cached_path),
        "source": "fresh",
        "token_weights": prune_cfg["token_weights"],
    }


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
    mean_method = cast(Literal["geo", "harm", "arith"], cluster_cfg["mean_method"])
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
            mean_method=mean_method,
            mediation_penalty=float(cluster_cfg["mediation_penalty"]),
            similarity_mode=cluster_cfg["similarity_mode"],
            enforce_dag=enforce_dag,
            random_state=int(cluster_cfg["random_state"]),
            n_init=int(cluster_cfg["n_init"]),
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
        mean_method=mean_method,
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
        [
            repo_root / "demos" / "temp_graph_files" / "clt-hp",
            repo_root / "demos" / "temp_graph_files" / "gemmascope-transcoder-16k",
            repo_root / "demos" / "temp_graph_files",
            repo_root / "demos" / "graph_files",
        ],
        ".json",
    )
    default_pruned_files = _list_pruned_graphs(repo_root / "demos" / "subgraph")

    input_mode = st.radio(
        "Input graph type",
        options=[FULL_GRAPH_MODE, PRUNED_GRAPH_MODE],
        horizontal=True,
        help=(
            "Full JSON: run the pruning pipeline (with SHAP+sparsemax token weights), then cluster. "
            "Pruned .pt: load a saved `PruneGraph` directly."
        ),
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
    selected_label = st.selectbox(
        "Choose an input file",
        options=file_options,
        index=0,
        help="Pick a discovered file under the repo, or use the manual path field below.",
    )
    custom_path = st.text_input(
        "Or enter a path manually",
        value="",
        placeholder="e.g. demos/temp_graph_files/example.json",
        help="If set, this path wins over the dropdown. Relative paths are resolved from the repository root.",
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
        help=(
            "Visualization-only filter: supernode edges with |weight| below this are omitted from the "
            "networkx view and the edges table. Does not change clustering or saved outputs."
        ),
    )

    prune_cfg: dict[str, Any] = {
        "logit_weights": "target",
        "token_weights": None,
        "node_influence_threshold": 0.8,
        "node_relevance_threshold": 0.8,
        "edge_influence_threshold": 0.98,
        "edge_relevance_threshold": 0.98,
        "keep_all_tokens_and_logits": True,
        "filter_act_density": False,
        "act_density_lb": 2e-5,
        "act_density_ub": 0.1,
        "use_shap_token_weights": True,
        "token_attribution_model": "google/gemma-2-2b",
        "token_attribution_normalize": "sparsemax",
        "token_attribution_masker_keep_prefix": None,
        "token_attribution_device": "auto",
    }

    if input_mode == FULL_GRAPH_MODE:
        with st.expander("Pruning parameters", expanded=True):
            prune_col_a, prune_col_b = st.columns(2)
            with prune_col_a:
                prune_cfg["logit_weights"] = st.selectbox(
                    "logit_weights",
                    options=["target", "probs"],
                    help=(
                        "How logit nodes seed backward influence: **target** puts mass on the target-logit "
                        "node(s); **probs** uses each logit node’s `token_prob` from graph attributes."
                    ),
                )
                prune_cfg["node_influence_threshold"] = st.slider(
                    "node_influence_threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.01,
                    help="Quantile-style cutoff on node *influence* (target-logit backward flow).",
                )
                prune_cfg["node_relevance_threshold"] = st.slider(
                    "node_relevance_threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.01,
                    help="Quantile-style cutoff on node *relevance* (token forward flow).",
                )
                prune_cfg["edge_influence_threshold"] = st.slider(
                    "edge_influence_threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.98,
                    step=0.01,
                    help="Quantile-style cutoff on edge *influence*.",
                )
                prune_cfg["edge_relevance_threshold"] = st.slider(
                    "edge_relevance_threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.98,
                    step=0.01,
                    help="Quantile-style cutoff on edge *relevance*.",
                )
            with prune_col_b:
                prune_cfg["keep_all_tokens_and_logits"] = st.checkbox(
                    "keep_all_tokens_and_logits",
                    value=True,
                    help=(
                        "If true, every embedding and every logit node is forced into the kept set regardless "
                        "of score; if false, only target logit(s) are forced."
                    ),
                )
                prune_cfg["filter_act_density"] = st.checkbox(
                    "filter_act_density",
                    value=False,
                    help=(
                        "After pruning, optionally drop cross-layer transcoder features whose Neuronpedia "
                        "`frac_nonzero` lies outside [act_density_lb, act_density_ub]."
                    ),
                )
                prune_cfg["act_density_lb"] = st.number_input(
                    "act_density_lb",
                    value=2e-5,
                    format="%.8f",
                    help="Lower bound on activation density.",
                )
                prune_cfg["act_density_ub"] = st.number_input(
                    "act_density_ub",
                    value=0.1,
                    format="%.6f",
                    help="Upper bound on activation density.",
                )

            st.markdown(
                "**Token weights** are computed from SHAP teacher-forcing attribution and "
                "normalized via the selected method (see `summarization.token_attribution`)."
            )
            tok_col_a, tok_col_b, tok_col_c, tok_col_d = st.columns(4)
            with tok_col_a:
                prune_cfg["token_attribution_model"] = st.text_input(
                    "token_attribution_model",
                    value="google/gemma-2-2b",
                    help="HuggingFace model name used by SHAP to score per-token contribution.",
                )
            with tok_col_b:
                prune_cfg["token_attribution_normalize"] = st.selectbox(
                    "token_attribution_normalize",
                    options=["sparsemax", "softmax", "entmax15", "relu_l1"],
                    index=0,
                    help="Normalization applied to raw SHAP scores before pruning.",
                )
            with tok_col_c:
                keep_prefix_raw = st.number_input(
                    "token_attribution_masker_keep_prefix (-1 disables)",
                    min_value=-1,
                    value=-1,
                    step=1,
                    help=(
                        "Optionally pin the first k token segments during SHAP masking. "
                        "Set to -1 to disable."
                    ),
                )
                prune_cfg["token_attribution_masker_keep_prefix"] = (
                    None if int(keep_prefix_raw) < 0 else int(keep_prefix_raw)
                )
            with tok_col_d:
                prune_cfg["token_attribution_device"] = st.selectbox(
                    "token_attribution_device",
                    options=["auto", "cpu", "cuda"],
                    index=0,
                    help="Device used to run the SHAP explainer.",
                )

    st.subheader("Clustering parameters")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        auto_k = st.checkbox(
            "Pick k automatically (`find_best_k`)",
            value=False,
            help=(
                "Sweep k over an eigengap-derived range (or your overrides), cluster each k, and pick "
                "the k with the best composite score. The final clustering then uses that k."
            ),
        )
        target_k = st.slider(
            "target_k (ignored when auto-k finds a value)",
            min_value=1,
            max_value=50,
            value=7,
            step=1,
            help="Desired number of middle supernodes for spectral clustering (clamped to the number of middle nodes).",
        )
        max_layer_span = st.slider(
            "max_layer_span",
            min_value=1,
            max_value=32,
            value=4,
            step=1,
            help=(
                "Post-cluster repair: split any middle supernode whose member layers span more than this "
                "many layers (keeps supernodes layer-local)."
            ),
        )
        max_sn_raw = st.number_input(
            "max_sn (0 means no cap)",
            min_value=0,
            value=0,
            step=1,
            help=(
                "Optional hard cap on the count of middle supernodes: greedily merge adjacent layer clusters "
                "until at most this many remain. Zero disables the cap."
            ),
        )
        similarity_mode = st.selectbox(
            "similarity_mode",
            options=["edge", "node"],
            index=0,
            help=(
                "What feeds the affinity matrix for spectral clustering: **edge** uses normalized edge "
                "structure (influence/relevance channels), while **node** weights similarity using "
                "node influence and node relevance scores."
            ),
        )
    with col_b:
        mean_method = st.selectbox(
            "mean_method",
            options=["geo", "harm", "arith"],
            index=2,
            help=(
                "How output/input cosine similarities are combined to form affinity: "
                "**geo** (geometric), **harm** (harmonic), or **arith** (arithmetic)."
            ),
        )
        mediation_penalty = st.slider(
            "mediation_penalty",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help=(
                "Down-weights similarity for pairs that are only “close” via a mediator on a layer strictly "
                "between them (reduces merges that invite cycles). At 1.0 the penalty matrix is all ones "
                "(no down-weighting)."
            ),
        )
        enforce_dag = st.checkbox(
            "enforce_dag",
            value=True,
            help=(
                "When splitting/repairing clusters, enforce layer ordering so supernode DAG interpretations "
                "stay sensible. Passed through to supernode graph construction and auto-k scoring."
            ),
        )
        random_state = st.number_input(
            "SpectralClustering random_state",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
            help=(
                "RNG seed for sklearn’s k-means label assignment in the **final** `cluster_graph` run. "
                "The auto-k sweep calls `cluster_graph` without this argument and therefore uses library defaults."
            ),
        )
        n_init = st.number_input(
            "SpectralClustering n_init (k-means)",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
            help=(
                "k-means restarts for the **final** clustering step; the auto-k sweep uses `cluster_graph`’s "
                "default `n_init` unless you change the library call."
            ),
        )
    with col_c:
        st.markdown("**Auto-k (`find_best_k`)**")
        k_min_override_raw = st.number_input(
            "k_min_override (0 = use eigengap range)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            help="If non-zero, fixes the minimum k in the sweep instead of the eigengap-derived search_range lower bound.",
        )
        k_max_override_raw = st.number_input(
            "k_max_override (0 = use eigengap range)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            help="If non-zero, fixes the maximum k in the sweep instead of the eigengap-derived upper bound.",
        )
        w_intra = st.slider(
            "score weight w_intra",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05,
            help="Weight on mean within-supernode similarity (normalized against global middle-node similarity).",
        )
        w_dag = st.slider(
            "score weight w_dag",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Weight on DAG-style safety (fewer interleaving layer-range warnings between supernodes is better).",
        )
        w_attr = st.slider(
            "score weight w_attr (attr balance)",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help=(
                "Weight on how evenly supernode-level influence is spread (entropy of normalized supernode "
                "influences — the `attr_balance` term in `score_k`)."
            ),
        )
        w_size = st.slider(
            "score weight w_size",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.05,
            help="Weight on closeness of the number of middle supernodes to an ideal ~√(n_middle) target.",
        )

    cluster_cfg = {
        "target_k": int(target_k),
        "max_layer_span": int(max_layer_span),
        "max_sn": None if int(max_sn_raw) == 0 else int(max_sn_raw),
        "mean_method": str(mean_method),
        "mediation_penalty": float(mediation_penalty),
        "similarity_mode": str(similarity_mode),
        "random_state": int(random_state),
        "n_init": int(n_init),
    }
    auto_k_cfg = {
        "k_min_override": None if int(k_min_override_raw) == 0 else int(k_min_override_raw),
        "k_max_override": None if int(k_max_override_raw) == 0 else int(k_max_override_raw),
        "weights": {
            "w_intra": float(w_intra),
            "w_dag": float(w_dag),
            "w_attr": float(w_attr),
            "w_size": float(w_size),
        },
    }

    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None
    if "last_upload_result" not in st.session_state:
        st.session_state.last_upload_result = None

    if st.button("Run clustering pipeline", type="primary"):
        if selected_path is None:
            st.error("Please select or enter an input path.")
            st.stop()
        if not selected_path.exists():
            st.error(f"Input file not found: {selected_path}")
            st.stop()

        with st.spinner("Running pipeline..."):
            try:
                prune_graph, prune_meta = _load_or_build_prune_graph(
                    repo_root=repo_root,
                    input_mode=input_mode,
                    input_path=selected_path,
                    prune_cfg=prune_cfg,
                )
                supernode_map, sng, run_meta = _cluster_from_prune(
                    prune_graph,
                    cluster_cfg,
                    bool(enforce_dag),
                    auto_k=bool(auto_k),
                    auto_k_cfg=auto_k_cfg,
                )
                flow_report = flow_faithfulness_report(sng, supernode_map)
                print(
                    "[flow_analysis] report:\n"
                    + json.dumps(_to_jsonable(flow_report), indent=2)
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
            "flow_report": flow_report,
            "run_meta": run_meta,
            "prune_meta": prune_meta,
        }
        st.session_state.last_upload_result = None

    result_payload = st.session_state.pipeline_result
    if result_payload is None:
        st.info("Choose an input and click **Run clustering pipeline**.")
        return

    prune_graph: PruneGraph = result_payload["prune_graph"]
    result_input_mode: str = result_payload.get("input_mode", input_mode)
    supernode_map: dict[str, list[str]] = result_payload["supernode_map"]
    sng: dict[str, Any] = result_payload["sng"]
    flow_report: dict[str, Any] | None = result_payload.get("flow_report")
    if flow_report is None:
        flow_report = flow_faithfulness_report(sng, supernode_map)
        result_payload["flow_report"] = flow_report
        print(
            "[flow_analysis] report (backfilled):\n"
            + json.dumps(_to_jsonable(flow_report), indent=2)
        )
    flow_report = cast(dict[str, Any], flow_report)
    run_meta: dict[str, Any] = result_payload.get("run_meta", {})
    prune_meta: dict[str, Any] = result_payload.get("prune_meta", {})

    if result_input_mode == FULL_GRAPH_MODE:
        if prune_meta.get("cache_hit") is True:
            st.success("Reused cached pruned graph from previous run.")
        elif prune_meta.get("cache_hit") is False:
            st.info("Pruned graph computed and cached for future clustering reruns.")

    prune_graph_path = prune_meta.get("prune_graph_path")
    if prune_graph_path:
        st.caption(f"Pruned graph source: `{prune_graph_path}`")

    token_weights_used = prune_meta.get("token_weights")
    if token_weights_used:
        formatted = ", ".join(f"{w:.3f}" for w in token_weights_used)
        norm_name = str(prune_cfg.get("token_attribution_normalize", "unknown"))
        st.caption(f"SHAP+{norm_name} token weights: [{formatted}]")

    if run_meta.get("auto_k"):
        tk_auto = run_meta.get("target_k_auto")
        if tk_auto is not None:
            st.info(
                f"Auto k selected **{tk_auto}** (manual slider was {run_meta.get('target_k_requested')}; "
                f"sweep covered **{run_meta.get('sweep_size', 0)}** candidate values)."
            )
    st.caption(f"Clustering used **target_k = {run_meta.get('target_k_used', '?')}**.")

    flow_combined = cast(dict[str, Any], flow_report.get("combined", {}))
    st.subheader("Flow analysis summary")
    flow_col_1, flow_col_2, flow_col_3, flow_col_4 = st.columns(4)
    flow_col_1.metric("F_phi", f"{float(flow_combined.get('F_phi', 0.0)):.4f}")
    flow_col_2.metric("D_phi", f"{float(flow_combined.get('D_phi', 0.0)):.4f}")
    flow_col_3.metric("R_phi_balance", f"{float(flow_combined.get('R_phi_balance', 0.0)):.4f}")
    flow_col_4.metric("shortcut_frac", f"{float(flow_combined.get('shortcut_frac', 0.0)):.4f}")

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

    st.subheader("Upload to Neuronpedia")
    upload_col_1, upload_col_2 = st.columns(2)
    with upload_col_1:
        upload_model_id = st.text_input("model_id", value="gemma-2-2b")
        upload_slug = st.text_input("slug (parent graph slug)", value="")
        upload_display_name = st.text_input("display_name", value="")
    with upload_col_2:
        upload_pruning_threshold = st.number_input(
            "pruning_threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.01,
        )
        upload_density_threshold = st.number_input(
            "density_threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.99,
            step=0.01,
        )
        upload_overwrite_id = st.text_input("overwrite_id (optional)", value="")

    if st.button("Upload clustered graph to Neuronpedia"):
        if not upload_slug.strip():
            st.error("`slug` is required to upload.")
            st.stop()
        if not upload_display_name.strip():
            st.error("`display_name` is required to upload.")
            st.stop()

        upload_supernodes = _clustered_supernodes_for_upload(supernode_map)
        if not upload_supernodes:
            st.error("No clustered supernodes with more than one member were found to upload.")
            st.stop()

        with st.spinner("Uploading clustered graph to Neuronpedia..."):
            status, body = save_subgraph(
                modelId=upload_model_id.strip(),
                slug=upload_slug.strip(),
                displayName=upload_display_name.strip(),
                pinnedIds=prune_graph.kept_ids,
                supernodes=upload_supernodes,
                pruningThreshold=float(upload_pruning_threshold),
                densityThreshold=float(upload_density_threshold),
                overwriteId=upload_overwrite_id.strip(),
            )
        st.session_state.last_upload_result = {
            "status": int(status),
            "body": body,
            "model_id": upload_model_id.strip(),
            "slug": upload_slug.strip(),
            "display_name": upload_display_name.strip(),
            "supernode_count": len(upload_supernodes),
            "pinned_count": len(prune_graph.kept_ids),
        }

    last_upload = st.session_state.last_upload_result
    if last_upload is not None:
        status = int(last_upload["status"])
        if 200 <= status < 300:
            st.success(
                f"Upload succeeded (status {status}). Uploaded {last_upload['supernode_count']} clustered supernodes."
            )
        else:
            st.error(f"Upload failed with status {status}.")
        st.caption(
            f"model_id={last_upload['model_id']} slug={last_upload['slug']} "
            f"display_name={last_upload['display_name']} pinned_ids={last_upload['pinned_count']}"
        )
        st.code(_pretty_response_body(str(last_upload["body"])), language="json")


if __name__ == "__main__":
    main()
